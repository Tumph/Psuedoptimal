"""Training callbacks for visualization and logging."""

from typing import Any, Dict, Optional

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_callback import TrainerCallback

from .prompts import apply_chat_template, format_student_prompt
from .rewards import DSLRewardFunction


class DSLVisualizationCallback(TrainerCallback):
    """
    Callback that visualizes Encoding→Python expansion every N steps.

    Prints examples showing:
    1. Original MBPP task prompt (for human reference)
    2. Generated encoding (Student output)
    3. Expanded Python (Generator output - from encoding ONLY)

    Note: The Generator sees ONLY the encoding, not the task prompt.
    """

    def __init__(
        self,
        reward_fn: DSLRewardFunction,
        student_tokenizer,
        log_every_n_steps: int = 50,
        num_examples: int = 1,
        clearml_logger=None,
    ):
        """
        Initialize the visualization callback.

        Args:
            reward_fn: DSLRewardFunction instance (has generator model)
            student_tokenizer: Student model's tokenizer
            log_every_n_steps: How often to log examples
            num_examples: How many examples to show per log
            clearml_logger: Optional ClearML logger instance
        """
        self.reward_fn = reward_fn
        self.student_tokenizer = student_tokenizer
        self.log_every_n_steps = log_every_n_steps
        self.num_examples = num_examples
        self.clearml_logger = clearml_logger

        # Cache for recent batch data
        self._recent_data = None

    def on_step_end(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        """Called at the end of each training step."""
        if state.global_step % self.log_every_n_steps != 0:
            return

        if state.global_step == 0:
            return

        # Try to get recent examples from various sources
        self._visualize_from_model(model, state)

    def _visualize_from_model(self, model, state: TrainerState):
        """Generate and visualize examples using the current model."""
        if model is None:
            return

        # Sample tasks for visualization
        sample_tasks = [
            "Write a function to find the maximum of two numbers.",
            "Write a function to check if a number is even.",
            "Write a function to reverse a string.",
        ]

        print("\n" + "=" * 80)
        print(f"  ENCODING VISUALIZATION - Step {state.global_step}")
        print("=" * 80)

        for i, task in enumerate(sample_tasks[: self.num_examples]):
            self._visualize_single(model, task, i + 1, state.global_step)

        print("=" * 80 + "\n")

    def _visualize_single(
        self, model, task_prompt: str, example_num: int, step: int
    ):
        """Visualize a single example."""
        # Format prompt for student
        messages = format_student_prompt(task_prompt)
        formatted_prompt = apply_chat_template(
            self.student_tokenizer, messages, add_generation_prompt=True
        )

        # Tokenize
        inputs = self.student_tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(model.device)

        # Generate DSL from student
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.student_tokenizer.pad_token_id,
                eos_token_id=self.student_tokenizer.eos_token_id,
            )

        # Decode encoding
        generated = outputs[0][inputs.input_ids.shape[1] :]
        encoding = self.student_tokenizer.decode(generated, skip_special_tokens=True)

        # Expand to Python via Generator (encoding only, no task context)
        python_code = self.reward_fn.expand_single(encoding)

        # Count tokens
        encoding_tokens = len(self.student_tokenizer.encode(encoding))

        # Print visualization
        print(f"\n--- Example {example_num} ---")
        print(f"TASK: {task_prompt}")
        print(f"\nENCODING ({encoding_tokens} tokens):")
        print("-" * 40)
        print(encoding[:500] if encoding else "(empty)")
        print("-" * 40)
        print(f"\nEXPANDED PYTHON (from encoding only):")
        print("-" * 40)
        print(python_code[:800] if python_code else "(empty)")
        print("-" * 40)

        # Log to ClearML if available
        if self.clearml_logger:
            self._log_to_clearml(
                task_prompt, encoding, python_code, encoding_tokens, step, example_num
            )

    def _log_to_clearml(
        self,
        task: str,
        encoding: str,
        python: str,
        tokens: int,
        step: int,
        example_num: int,
    ):
        """Log example to ClearML."""
        text = f"""
## Example {example_num} at Step {step}

**Task:** {task}

**Encoding ({tokens} tokens):**
```
{encoding[:500]}
```

**Expanded Python (from encoding only):**
```python
{python[:800]}
```
"""
        self.clearml_logger.report_text(
            msg=text,
            title="Encoding Expansion",
            series=f"Example_{example_num}",
            iteration=step,
        )


class MetricsCallback(TrainerCallback):
    """Callback for logging additional training metrics."""

    def __init__(self, clearml_logger=None):
        """
        Initialize the metrics callback.

        Args:
            clearml_logger: Optional ClearML logger instance
        """
        self.clearml_logger = clearml_logger
        self.reward_history = []
        self.dsl_length_history = []

        # Track reward distribution to detect lottery tickets
        self.pass_rate_history = []
        self.compression_ratio_history = []
        self.encoding_length_history = []

    def on_log(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Log metrics to ClearML."""
        if logs is None:
            return

        # Track reward if available
        if "reward" in logs:
            self.reward_history.append(logs["reward"])

        # Track additional metrics if available
        if "pass_rate" in logs:
            self.pass_rate_history.append(logs["pass_rate"])
        if "compression_ratio" in logs:
            self.compression_ratio_history.append(logs["compression_ratio"])
        if "encoding_length" in logs:
            self.encoding_length_history.append(logs["encoding_length"])

        # Log to ClearML
        if self.clearml_logger:
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    self.clearml_logger.report_scalar(
                        title="Training",
                        series=key,
                        value=value,
                        iteration=state.global_step,
                    )

            # Log reward statistics to detect lottery tickets
            if len(self.reward_history) > 0:
                import numpy as np
                recent_rewards = self.reward_history[-100:]  # Last 100 steps

                self.clearml_logger.report_scalar(
                    title="Reward Stats",
                    series="mean",
                    value=np.mean(recent_rewards),
                    iteration=state.global_step,
                )
                self.clearml_logger.report_scalar(
                    title="Reward Stats",
                    series="std",
                    value=np.std(recent_rewards),
                    iteration=state.global_step,
                )
                self.clearml_logger.report_scalar(
                    title="Reward Stats",
                    series="max",
                    value=np.max(recent_rewards),
                    iteration=state.global_step,
                )
                self.clearml_logger.report_scalar(
                    title="Reward Stats",
                    series="min",
                    value=np.min(recent_rewards),
                    iteration=state.global_step,
                )

            # Log GPU memory
            if torch.cuda.is_available():
                mem_gb = torch.cuda.memory_allocated() / 1e9
                self.clearml_logger.report_scalar(
                    title="GPU Memory",
                    series="allocated_gb",
                    value=mem_gb,
                    iteration=state.global_step,
                )

    def on_train_end(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Log final statistics."""
        if self.reward_history:
            avg_reward = sum(self.reward_history) / len(self.reward_history)
            print(f"\nTraining complete!")
            print(f"Average reward: {avg_reward:.3f}")
            print(f"Total steps: {state.global_step}")


class EarlyStoppingCallback(TrainerCallback):
    """Early stopping based on reward improvement."""

    def __init__(self, patience: int = 5, min_delta: float = 0.1):
        """
        Initialize early stopping.

        Args:
            patience: Number of evaluations without improvement before stopping
            min_delta: Minimum improvement required
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_reward = None
        self.wait_count = 0

    def on_evaluate(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        """Check for improvement after evaluation."""
        if metrics is None:
            return

        reward = metrics.get("eval_reward", metrics.get("reward"))
        if reward is None:
            return

        if self.best_reward is None:
            self.best_reward = reward
        elif reward > self.best_reward + self.min_delta:
            self.best_reward = reward
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                print(f"\nEarly stopping: No improvement for {self.patience} evaluations")
                control.should_training_stop = True
