# Psuedoptimal: Emergent Machine-to-Machine Communication

Train a small LLM to invent a compressed encoding that a larger LLM can interpret back into executable Python code — without seeing the original task.

## Research Question

**If forced to communicate under severe token constraints, will LLMs develop their own efficient encoding schemes?**

The key insight: by removing the original task description from the decoder, we force the encoder to embed ALL necessary information into a compressed representation. Over training, this creates pressure for emergent communication protocols.

## Architecture

```
┌─────────────────┐                      ┌─────────────────┐
│  MBPP Task      │                      │                 │
│  Description    │────┐                 │                 │
└─────────────────┘    │                 │                 │
                       ▼                 │                 │
┌─────────────────┐  Encoding only  ┌────┴────────────┐        Python        ┌───────────┐
│     Student     │ ──────────────► │    Generator    │ ────────────────────►│  Sandbox  │
│  (0.5B, LoRA)   │                 │  (1.5B, Frozen) │                      │  (Tests)  │
│     Encoder     │ ◄────────────── │     Decoder     │ ◄────────────────────│  Reward   │
└─────────────────┘     Reward      └─────────────────┘                      └───────────┘
                                           │
                                           │ NO task description!
                                           │ Encoding must be self-contained
                                           ▼
```

**Critical Design**: The Generator (decoder) receives ONLY the Student's encoding — NOT the original task. This forces the Student to encode all information needed to reconstruct working Python.

## How It Works

1. **Student Model** (encoder) receives an MBPP task and produces a compressed encoding
2. **Generator Model** (decoder, frozen) sees ONLY the encoding and must produce Python
3. **Sandbox** executes the Python against MBPP test cases
4. **Reward** = `pass_reward × (reference_tokens / encoding_tokens)` if tests pass

The compression ratio reward incentivizes shorter encodings. The correctness signal shapes which encodings work. Over training, an efficient encoding scheme emerges.

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
git clone https://github.com/your-org/Psuedoptimal.git
cd Psuedoptimal

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
    # Models (encoder-decoder pair)
    student_model: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"   # Encoder
    generator_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct" # Decoder (frozen)

    # Training
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_epochs: int = 3
    learning_rate: float = 1e-5

    # Reward (compression ratio based)
    pass_reward: float = 10.0   # Base reward, multiplied by compression ratio
    fail_reward: float = -1.0   # Penalty for failing tests
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
git clone https://github.com/YOUR_USERNAME/Psuedoptimal.git
cd Psuedoptimal
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
Psuedoptimal/
├── pyproject.toml          # Dependencies and project config
├── README.md               # This file
├── train_dsl.py            # Main training script
├── runpod/                 # Cloud deployment scripts
│   ├── setup.sh            # One-time pod setup
│   ├── run_training.sh     # Run training (foreground)
│   └── run_training_background.sh  # Run training (background)
└── src/
    ├── __init__.py
    ├── models.py           # Model loading (encoder + decoder)
    ├── prompts.py          # Prompts (encoder sees task, decoder sees encoding only)
    ├── rewards.py          # Reward function (compression ratio based)
    ├── sandbox.py          # Safe code execution
    └── callbacks.py        # Visualization callback
```

## Key Components

### Reward Function (`src/rewards.py`)

The `DSLRewardFunction` class implements the core training signal:

```python
def __call__(self, completions, test_list, reference_code, **kwargs):
    # 1. Expand Encoding → Python via Generator (NO task context!)
    python_codes = self._batch_expand_encodings(completions)

    # 2. Execute tests in sandbox
    for encoding, python, tests, ref in zip(completions, python_codes, test_list, reference_code):
        passed = self._execute_tests(python, tests, imports)
        encoding_tokens = len(tokenizer.encode(encoding))
        ref_tokens = len(tokenizer.encode(ref))

        # 3. Compute reward: compression ratio × pass reward
        if passed:
            reward = 10.0 * (ref_tokens / encoding_tokens)  # Higher compression = higher reward
        else:
            reward = -1.0
```

**Key**: The Generator never sees `task_prompt` — only the raw encoding.

### Sandbox (`src/sandbox.py`)

Executes generated Python safely using subprocess isolation:
- Resource limits (CPU time, memory)
- Restricted builtins (no file/network access)
- Timeout protection

### Visualization Callback (`src/callbacks.py`)

Every 50 steps, prints Encoding→Python expansion examples:

```
================================================================================
  ENCODING VISUALIZATION - Step 150
================================================================================

--- Example 1 ---
TASK: Write a function to find the maximum of two numbers.

ENCODING (12 tokens):
----------------------------------------
max2(a,b)->a>b?a:b
----------------------------------------

EXPANDED PYTHON (from encoding only):
----------------------------------------
def max2(a, b):
    return a if a > b else b
----------------------------------------
================================================================================
```

Note: The Generator produces Python from the encoding alone — it never sees "find the maximum of two numbers".

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

## How the Encoding Evolves

Initially, the Student outputs verbose, Python-like code. As training progresses, compression pressure and correctness signals shape the encoding:

| Training Step | Example Encoding | Tokens | Notes |
|---------------|------------------|--------|-------|
| 0 | `def is_prime(n): return n > 1 and all(n % i for i in range(2, int(n**0.5)+1))` | 35 | Full Python |
| 500 | `prime(n)->n>1&all(n%i for i in 2..√n)` | 22 | Syntax shortcuts |
| 2000 | `p:n→n>1∧∀i∈[2,√n]:n∤i` | 14 | Mathematical notation |
| 5000+ | `Ƥ(n)⊢n>1∧⊥∃d∈[2,√n]` | 10 | Novel symbols |

The encoding that emerges is whatever the Student-Generator pair can reliably use to pass tests. Since the Generator has no task context, the encoding must be self-explanatory.

### Why This Works

1. **Information Bottleneck**: The Student must compress all task information into minimal tokens
2. **Correctness Pressure**: Only encodings the Generator can decode to working code survive
3. **Co-adaptation**: Though only the Student trains, the pre-trained Generator's knowledge shapes viable encodings
4. **Emergent Protocol**: The final encoding scheme is not prescribed — it emerges from optimization

## License

MIT

## Citation

If you use this work, please cite:

```bibtex
@software{Psuedoptimal,
  title = {Psuedoptimal: Emergent Machine-to-Machine Communication for Code},
  year = {2025},
  url = {https://github.com/your-org/Psuedoptimal}
}
```