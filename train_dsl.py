#!/usr/bin/env python3
"""
LLM-DSL Training Script

Train a Student model to generate compressed DSL code that a frozen Generator
model interprets back to Python. Uses GRPOTrainer with custom reward function.

Usage:
    python train_dsl.py [--student-model MODEL] [--generator-model MODEL]
"""

# Suppress warnings before importing libraries
import os
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*is deprecated.*")
warnings.filterwarnings("ignore", message=".*right-padding was detected.*")

import argparse
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from trl import GRPOConfig, GRPOTrainer

from src.callbacks import DSLVisualizationCallback, MetricsCallback
from src.models import load_generator_model, load_student_model, estimate_memory_usage, get_device_info
from src.prompts import apply_chat_template, format_student_prompt
from src.rewards import RewardConfig, create_reward_function

# Load environment variables
load_dotenv()


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Model settings (correct Qwen naming format)
    student_model: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    generator_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

    # Training hyperparameters
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_epochs: int = 3
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.1

    # GRPO settings
    num_generations: int = 4  # G in GRPO - samples per prompt
    max_prompt_length: int = 512
    max_completion_length: int = 256
    beta: float = 0.0  # KL penalty coefficient (0 = no KL penalty)

    # Reward settings
    pass_reward: float = 10.0
    fail_reward: float = -1.0
    length_penalty: float = 0.05

    # Logging
    log_steps: int = 10
    save_steps: int = 500
    visualization_steps: int = 50
    output_dir: str = "./outputs/llm-dsl"

    # ClearML
    clearml_project: str = "LLM-DSL"
    clearml_task: str = "GRPO-Training"
    use_clearml: bool = True

    # Debug
    max_samples: Optional[int] = None  # Limit dataset size for debugging


def init_clearml(config: TrainingConfig):
    """Initialize ClearML logging if enabled."""
    if not config.use_clearml:
        return None, None

    try:
        import clearml

        task = clearml.Task.init(
            project_name=config.clearml_project,
            task_name=config.clearml_task,
            auto_connect_frameworks=True,
        )
        logger = task.get_logger()
        print(f"ClearML initialized: {config.clearml_project}/{config.clearml_task}")
        return task, logger
    except Exception as e:
        print(f"ClearML initialization failed: {e}")
        print("Continuing without ClearML logging...")
        return None, None


def prepare_dataset(tokenizer, config: TrainingConfig):
    """Load and prepare MBPP dataset for GRPOTrainer."""
    print("Loading MBPP dataset (sanitized)...")
    dataset = load_dataset("mbpp", "sanitized", split="train")

    if config.max_samples:
        dataset = dataset.select(range(min(config.max_samples, len(dataset))))
        print(f"  Limited to {len(dataset)} samples for debugging")

    print(f"  Loaded {len(dataset)} training examples")

    def format_for_grpo(example):
        """Format example for GRPOTrainer."""
        # Create chat messages for student prompt
        messages = format_student_prompt(example["prompt"])

        # Apply chat template to create the input prompt
        formatted_prompt = apply_chat_template(
            tokenizer, messages, add_generation_prompt=True
        )

        # Store original prompt as task_prompt for reward function
        return {
            "prompt": formatted_prompt,
            "task_prompt": example["prompt"],  # Original for Generator
            "test_list": example["test_list"],
            "test_imports": example.get("test_imports", []),
        }

    dataset = dataset.map(format_for_grpo, remove_columns=["source_file", "code"])
    print(f"  Dataset prepared with columns: {dataset.column_names}")

    return dataset


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train LLM-DSL")
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode (small dataset)"
    )
    parser.add_argument(
        "--no-clearml", action="store_true", help="Disable ClearML logging"
    )
    parser.add_argument(
        "--student-model",
        type=str,
        default="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        help="Student model name (default: Qwen/Qwen2.5-Coder-0.5B-Instruct)"
    )
    parser.add_argument(
        "--generator-model",
        type=str,
        default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        help="Generator model name (default: Qwen/Qwen2.5-Coder-1.5B-Instruct)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size per device (default: 2)"
    )
    args = parser.parse_args()

    # Create config with CLI overrides
    config = TrainingConfig()
    config.student_model = args.student_model
    config.generator_model = args.generator_model
    config.batch_size = args.batch_size

    if args.debug:
        config.max_samples = 50
        config.save_steps = 10
        config.log_steps = 1
        print("Debug mode enabled: using 50 samples")
    if args.no_clearml:
        config.use_clearml = False

    print(f"\n=== Model Configuration ===")
    print(f"  Student: {config.student_model}")
    print(f"  Generator: {config.generator_model}")

    # Print device info
    print("\n=== Device Information ===")
    device_info = get_device_info()
    print(f"  Platform: {device_info['platform']}")
    print(f"  CUDA available: {device_info['cuda_available']}")
    print(f"  MPS available: {device_info['mps_available']}")
    print(f"  bitsandbytes available: {device_info['bnb_available']}")
    print(f"  Recommended device: {device_info['recommended_device']}")

    # Print memory estimate
    print("\n=== Memory Estimation ===")
    mem = estimate_memory_usage(student_size_b=0.5, generator_size_b=1.5, quantization="auto")
    print(f"  Quantization: {mem['quantization']}")
    print(f"  Student: {mem['student_model_gb']:.1f} GB")
    print(f"  Generator: {mem['generator_model_gb']:.1f} GB")
    print(f"  Total estimated: {mem['total_estimated_gb']:.1f} GB")

    # Initialize ClearML
    clearml_task, clearml_logger = init_clearml(config)

    # ==========================================================================
    # 1. Load Models
    # ==========================================================================
    print("\n=== Loading Models ===")

    student_model, student_tokenizer = load_student_model(
        model_name=config.student_model
    )

    generator_model, generator_tokenizer = load_generator_model(
        model_name=config.generator_model
    )

    # ==========================================================================
    # 2. Prepare Dataset
    # ==========================================================================
    print("\n=== Preparing Dataset ===")
    dataset = prepare_dataset(student_tokenizer, config)

    # ==========================================================================
    # 3. Create Reward Function
    # ==========================================================================
    print("\n=== Creating Reward Function ===")
    reward_config = RewardConfig(
        pass_reward=config.pass_reward,
        fail_reward=config.fail_reward,
        length_penalty_coef=config.length_penalty,
    )

    reward_fn = create_reward_function(
        generator_model=generator_model,
        generator_tokenizer=generator_tokenizer,
        student_tokenizer=student_tokenizer,
        config=reward_config,
    )
    print(f"  Pass reward: {config.pass_reward}")
    print(f"  Fail reward: {config.fail_reward}")
    print(f"  Length penalty: {config.length_penalty} per token")

    # ==========================================================================
    # 4. Configure GRPOTrainer
    # ==========================================================================
    print("\n=== Configuring GRPOTrainer ===")

    training_args = GRPOConfig(
        output_dir=config.output_dir,
        # Training
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        # GRPO specific
        num_generations=config.num_generations,
        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_completion_length,
        beta=config.beta,
        # Optimization
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        # Logging
        logging_steps=config.log_steps,
        save_steps=config.save_steps,
        save_total_limit=3,
        # Hub
        push_to_hub=False,
        report_to=["clearml"] if config.use_clearml and clearml_task else [],
    )

    print(f"  Batch size: {config.batch_size}")
    print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"  Effective batch: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"  Num generations (G): {config.num_generations}")
    print(f"  Learning rate: {config.learning_rate}")

    # ==========================================================================
    # 5. Create Callbacks
    # ==========================================================================
    print("\n=== Creating Callbacks ===")

    visualization_callback = DSLVisualizationCallback(
        reward_fn=reward_fn,
        student_tokenizer=student_tokenizer,
        log_every_n_steps=config.visualization_steps,
        num_examples=1,
        clearml_logger=clearml_logger,
    )

    metrics_callback = MetricsCallback(clearml_logger=clearml_logger)

    callbacks = [visualization_callback, metrics_callback]
    print(f"  Visualization every {config.visualization_steps} steps")

    # ==========================================================================
    # 6. Initialize Trainer
    # ==========================================================================
    print("\n=== Initializing Trainer ===")

    trainer = GRPOTrainer(
        model=student_model,
        args=training_args,
        reward_funcs=reward_fn,
        train_dataset=dataset,
        processing_class=student_tokenizer,
        callbacks=callbacks,
    )

    # ==========================================================================
    # 7. Train!
    # ==========================================================================
    print("\n" + "=" * 60)
    print("  STARTING TRAINING")
    print("=" * 60 + "\n")

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        # Save final model
        print("\n=== Saving Model ===")
        final_path = os.path.join(config.output_dir, "final")
        trainer.save_model(final_path)
        print(f"  Model saved to: {final_path}")

        # Cleanup
        if clearml_task:
            clearml_task.close()

    print("\n=== Training Complete ===")


if __name__ == "__main__":
    main()
