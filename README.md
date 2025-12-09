# LLM-Golf: Learning Compressed Code Languages

Train a small LLM to invent a compressed domain-specific language (DSL) that a larger LLM can interpret back into executable Python code.

## Architecture

```
┌─────────────────┐         DSL          ┌─────────────────┐        Python        ┌───────────┐
│     Student     │ ───────────────────► │    Generator    │ ────────────────────►│  Sandbox  │
│  (0.5B, LoRA)   │                      │  (1.5B, Frozen) │                      │  (Tests)  │
│    Trainable    │ ◄─────────────────── │   Interpreter   │ ◄────────────────────│  Reward   │
└─────────────────┘       Reward         └─────────────────┘                      └───────────┘
```

**The Goal**: The Student learns to generate increasingly compressed "DSL" code that:
1. Is shorter than full Python (measured in tokens)
2. Still correctly solves programming problems when expanded by the Generator

## How It Works

1. **Student Model** receives an MBPP programming task and generates compressed DSL
2. **Generator Model** (frozen) interprets the DSL and expands it to Python
3. **Sandbox** executes the Python against MBPP test cases
4. **Reward** = `(+10 if tests pass, -1 if fail) - (0.05 × DSL_tokens)`

The length penalty incentivizes the Student to invent increasingly compressed representations while the pass/fail reward ensures the code remains correct.

## Requirements

- Python 3.10+
- GPU with sufficient memory:
  - **Linux + CUDA**: 8GB+ VRAM recommended (4-bit quantization enabled)
  - **macOS + MPS**: 16GB+ unified memory recommended (no quantization, float16)
  - **CPU**: Not recommended (very slow)
- ~6GB disk space for models

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/llm-golf.git
cd llm-golf

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .

# Linux with CUDA: Also install bitsandbytes for 4-bit quantization
pip install -e ".[cuda]"

# Optional: Install dev dependencies
pip install -e ".[dev]"
```

### Platform Notes

| Platform | Quantization | Memory Usage | Notes |
|----------|-------------|--------------|-------|
| Linux + CUDA | 4-bit (NF4) | ~6 GB VRAM | Best performance |
| macOS + MPS | None (float16) | ~8 GB unified | Works but slower, no quantization |
| macOS + CPU | None (float32) | ~12 GB RAM | Very slow, not recommended |

## Quick Start

### 1. Login to HuggingFace (required for Qwen models)

```bash
# Set your HuggingFace token
export HF_TOKEN="your_huggingface_token"

# Or login via CLI
huggingface-cli login
```

Get your token at: https://huggingface.co/settings/tokens

### 2. Basic Training

```bash
python train_dsl.py
```

### 3. Custom Models

```bash
# Use different model sizes
python train_dsl.py \
    --student-model "Qwen/Qwen2.5-Coder-3B-Instruct" \
    --generator-model "Qwen/Qwen2.5-Coder-7B-Instruct"
```

### Debug Mode (Small Dataset)

```bash
python train_dsl.py --debug
```

### Without ClearML Logging

```bash
python train_dsl.py --no-clearml
```

## Configuration

### CLI Arguments

```bash
python train_dsl.py --help

Options:
  --student-model    Student model (default: Qwen/Qwen2.5-Coder-0.5B-Instruct)
  --generator-model  Generator model (default: Qwen/Qwen2.5-Coder-1.5B-Instruct)
  --batch-size       Batch size per device (default: 2)
  --debug            Use small dataset for debugging
  --no-clearml       Disable ClearML logging
```

### Available Model Sizes

| Model | Parameters | VRAM (4-bit) |
|-------|------------|--------------|
| Qwen/Qwen2.5-Coder-0.5B-Instruct | 0.5B | ~0.5 GB |
| Qwen/Qwen2.5-Coder-1.5B-Instruct | 1.5B | ~1.0 GB |
| Qwen/Qwen2.5-Coder-3B-Instruct | 3B | ~1.5 GB |
| Qwen/Qwen2.5-Coder-7B-Instruct | 7B | ~3.5 GB |

### TrainingConfig

```python
@dataclass
class TrainingConfig:
    # Models (correct Qwen naming format)
    student_model: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    generator_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

    # Training
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_epochs: int = 3
    learning_rate: float = 1e-5

    # Reward
    pass_reward: float = 10.0
    fail_reward: float = -1.0
    length_penalty: float = 0.05  # Per token
```

## RunPod Deployment (Recommended for Serious Training)

For larger model configurations (3B Student + 7B Generator), use RunPod cloud GPUs.

### 1. Create RunPod Account

1. Sign up at https://runpod.io
2. Add payment method and credits ($10-20 for initial testing)

### 2. Deploy a Pod

1. Go to https://console.runpod.io/gpu-cloud
2. Deploy an **RTX 4090** (~$0.44/hr) or **A100 80GB** (~$1.89/hr)
3. Select template: **RunPod PyTorch 2.1**
4. Configure storage:
   - Container Disk: 20 GB
   - Volume Disk: 50 GB
   - Volume Mount: `/workspace`

### 3. Connect and Setup

```bash
# SSH into your pod (get command from RunPod console)
ssh root@<pod-ip> -p <port>

# Set your HuggingFace token
export HF_TOKEN="your_token"

# Clone and setup
cd /workspace
git clone https://github.com/YOUR_USERNAME/llm-golf.git
cd llm-golf
bash runpod/setup.sh
```

### 4. Run Training

```bash
# Foreground (see output directly)
bash runpod/run_training.sh

# Background (keeps running after disconnect)
bash runpod/run_training_background.sh
```

### 5. Monitor

- **Console**: `tail -f logs/training_*.log`
- **ClearML**: https://app.clear.ml (if configured)

### Estimated Costs

| GPU | Cost/hr | 10hr Training |
|-----|---------|---------------|
| RTX 4090 24GB | ~$0.44 | ~$4.40 |
| A100 80GB | ~$1.89 | ~$18.90 |

## ClearML Setup

1. Create a free account at [app.clear.ml](https://app.clear.ml)
2. Get your credentials from Settings → Workspace → Create credentials
3. Run `clearml-init` and paste your credentials
4. Training metrics will automatically appear in the ClearML dashboard

## Memory Requirements

### Linux + CUDA (with 4-bit quantization)

| Component | VRAM |
|-----------|------|
| Student (0.5B, 4-bit + LoRA) | ~0.5 GB |
| Generator (1.5B, 4-bit) | ~1.0 GB |
| Optimizer states | ~0.5 GB |
| Activations | ~2.0 GB |
| **Total** | **~4-6 GB** |

### macOS + MPS (float16, no quantization)

| Component | Memory |
|-----------|--------|
| Student (0.5B, float16 + LoRA) | ~1.0 GB |
| Generator (1.5B, float16) | ~3.0 GB |
| Optimizer states | ~0.5 GB |
| Activations | ~2.0 GB |
| **Total** | **~6-8 GB** |

For systems with less memory, reduce `batch_size` and `num_generations`.

## Project Structure

```
llm-golf/
├── pyproject.toml          # Dependencies and project config
├── README.md               # This file
├── train_dsl.py            # Main training script
├── runpod/                 # Cloud deployment scripts
│   ├── setup.sh            # One-time pod setup
│   ├── run_training.sh     # Run training (foreground)
│   └── run_training_background.sh  # Run training (background)
└── src/
    ├── __init__.py
    ├── models.py           # Model loading with 4-bit quantization
    ├── prompts.py          # Prompt templates for Student & Generator
    ├── rewards.py          # Custom reward function (core logic)
    ├── sandbox.py          # Safe code execution
    └── callbacks.py        # Visualization callback
```

## Key Components

### Reward Function (`src/rewards.py`)

The `DSLRewardFunction` class implements the core training signal:

```python
def __call__(self, completions, task_prompt, test_list, test_imports, **kwargs):
    # 1. Expand DSL → Python via Generator
    python_codes = self._batch_expand_dsl(completions, task_prompt)

    # 2. Execute tests in sandbox
    for dsl, python, tests in zip(completions, python_codes, test_list):
        passed = self._execute_tests(python, tests, imports)
        dsl_tokens = len(tokenizer.encode(dsl))

        # 3. Compute reward
        reward = (10.0 if passed else -1.0) - (0.05 * dsl_tokens)
```

### Sandbox (`src/sandbox.py`)

Executes generated Python safely using subprocess isolation:
- Resource limits (CPU time, memory)
- Restricted builtins (no file/network access)
- Timeout protection

### Visualization Callback (`src/callbacks.py`)

Every 50 steps, prints DSL→Python expansion examples:

```
================================================================================
  DSL VISUALIZATION - Step 150
================================================================================

--- Example 1 ---
TASK: Write a function to find the maximum of two numbers.

DSL (12 tokens):
----------------------------------------
fn max2(a,b)->a if a>b el b
----------------------------------------

EXPANDED PYTHON:
----------------------------------------
def max2(a, b):
    return a if a > b else b
----------------------------------------
================================================================================
```

## Troubleshooting

### CUDA Out of Memory

1. Reduce `batch_size` to 1
2. Reduce `num_generations` to 2
3. Reduce `max_completion_length` to 128
4. Enable `gradient_checkpointing` (already enabled by default)

### ClearML Not Connecting

1. Run `clearml-init` to reconfigure
2. Check your network/firewall settings
3. Use `--no-clearml` to disable

### Generator Produces Bad Python

The Generator may need warmup. Try:
1. Using a larger Generator model
2. Lowering the Generator's temperature (currently 0.1)
3. Adding few-shot examples to the Generator prompt

## How the DSL Evolves

Initially, the Student outputs verbose code. Over training, it learns patterns like:

| Training Step | Example DSL | Tokens |
|---------------|-------------|--------|
| 0 | `def is_prime(n): return n > 1 and all(n % i for i in range(2, int(n**0.5)+1))` | 35 |
| 500 | `fn prime(n)->n>1 and all(n%i for i in range(2,int(n**.5)+1))` | 28 |
| 2000 | `fn p(n)->n>1&all(n%i for i∈2..√n)` | 18 |

The language that emerges depends on what the Generator can reliably interpret.

## License

MIT

## Citation

If you use this work, please cite:

```bibtex
@software{llm_golf,
  title = {LLM-Golf: Learning Compressed Code Languages},
  year = {2025},
  url = {https://github.com/your-org/llm-golf}
}
```