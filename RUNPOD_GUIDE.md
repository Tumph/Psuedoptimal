# RunPod Deployment Guide: Psuedoptimal Training

Complete guide for deploying and running full-scale training on RunPod cloud GPUs.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [RunPod Setup](#runpod-setup)
3. [Pod Deployment](#pod-deployment)
4. [Training Execution](#training-execution)
5. [Monitoring Training](#monitoring-training)
6. [Extracting Results](#extracting-results)
7. [Cost Management](#cost-management)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### 1. GitHub Repository

Ensure your code is pushed to GitHub:

```bash
# From your local machine
cd /path/to/llm-golf
git add -A
git commit -m "Prepare for RunPod deployment"
git push origin main
```

Your repo: `https://github.com/Tumph/Psuedoptimal.git`

### 2. HuggingFace Token

Get your token from: https://huggingface.co/settings/tokens

Required for downloading Qwen models. Copy this token - you'll need it on RunPod.

### 3. ClearML Account (Optional but Recommended)

Create a free account at: https://app.clear.ml

Get your credentials from: Settings → Workspace → Create new credentials

You'll need:
- `CLEARML_API_ACCESS_KEY`
- `CLEARML_API_SECRET_KEY`
- `CLEARML_API_HOST` (usually `https://api.clear.ml`)

---

## RunPod Setup

### 1. Create RunPod Account

1. Go to https://runpod.io
2. Sign up for an account
3. Add payment method
4. Add credits ($20-50 recommended for initial experiments)

### 2. Choose GPU

**Recommended configurations:**

| GPU | VRAM | Models | Cost/hr | Training Time (est.) |
|-----|------|--------|---------|----------------------|
| **RTX 4090** | 24GB | 3B + 7B | ~$0.44 | 3-4 hours |
| **RTX A6000** | 48GB | 3B + 7B | ~$0.79 | 3-4 hours |
| **A100 40GB** | 40GB | 3B + 7B | ~$1.39 | 2-3 hours |
| **A100 80GB** | 80GB | 7B + 14B | ~$1.89 | 2-3 hours |

**For this deployment, we recommend RTX 4090** (best price/performance).

---

## Pod Deployment

### Step 1: Launch Pod

1. Go to https://console.runpod.io/gpu-cloud
2. Click **"Deploy"** on an available RTX 4090
3. Select template: **RunPod Pytorch 2.4.0**
4. Configure storage:
   - **Container Disk**: 30 GB (minimum)
   - **Volume Disk**: 50 GB (for model caching)
   - **Volume Mount Path**: `/workspace`
5. Enable **SSH** and **Jupyter** ports
6. Click **"Deploy On-Demand"**

Wait for pod to spin up (~1-2 minutes).

### Step 2: Connect via SSH

Once pod is running, get SSH command from RunPod console:

```bash
# Example (your command will be different)
ssh root@X.X.X.X -p XXXXX -i ~/.ssh/id_ed25519
```

You should now be connected to your RunPod instance.

### Step 3: Set Environment Variables

```bash
# HuggingFace token (REQUIRED)
export HF_TOKEN='hf_xxxxxxxxxxxxxxxxxxxx'

# ClearML credentials (OPTIONAL but recommended)
export CLEARML_API_ACCESS_KEY='your_access_key'
export CLEARML_API_SECRET_KEY='your_secret_key'
export CLEARML_API_HOST='https://api.clear.ml'
```

**Pro tip**: Save these in `~/.bashrc` so they persist:

```bash
cat >> ~/.bashrc << 'EOF'
export HF_TOKEN='hf_xxxxxxxxxxxxxxxxxxxx'
export CLEARML_API_ACCESS_KEY='your_access_key'
export CLEARML_API_SECRET_KEY='your_secret_key'
export CLEARML_API_HOST='https://api.clear.ml'
EOF

source ~/.bashrc
```

### Step 4: Run Setup Script

```bash
cd /workspace
git clone https://github.com/Tumph/Psuedoptimal.git
cd Psuedoptimal
bash runpod/setup.sh
```

This will:
- Install all Python dependencies
- Install bitsandbytes for 4-bit quantization
- Login to HuggingFace
- Verify GPU is accessible

**Expected output:**
```
=== GPU Status ===
name, memory.total [MiB], memory.free [MiB]
NVIDIA GeForce RTX 4090, 24564, 23456
```

---

## Training Execution

### Configuration Overview

Default configuration (optimized for RTX 4090):
- **Student Model**: Qwen/Qwen2.5-Coder-3B-Instruct (encoder, trainable with LoRA)
- **Generator Model**: Qwen/Qwen2.5-Coder-7B-Instruct (decoder, frozen)
- **Batch Size**: 2 per device
- **Gradient Accumulation**: 8 steps (effective batch size: 16)
- **Training Epochs**: 3
- **Dataset**: MBPP sanitized (400 programming problems)
- **Quantization**: 4-bit NF4 (via bitsandbytes)

### Option A: Foreground Training (Interactive)

**Use this if you want to watch training live:**

```bash
cd /workspace/Psuedoptimal
bash runpod/run_training.sh
```

Training will run in your terminal. You'll see live updates every 10 steps.

**Limitations:**
- If SSH disconnects, training stops
- Terminal must stay open

### Option B: Background Training (Recommended)

**Use this to let training run even after you disconnect:**

```bash
cd /workspace/Psuedoptimal
bash runpod/run_training_background.sh
```

**Output:**
```
=== Starting LLM-Golf Full Training in Background ===
Configuration:
  Student Model: Qwen/Qwen2.5-Coder-3B-Instruct
  Generator Model: Qwen/Qwen2.5-Coder-7B-Instruct
  Batch Size: 2
  Log File: /workspace/Psuedoptimal/logs/training_20251216_180530.log

Training started in background with PID: 12345

To monitor training:
  tail -f /workspace/Psuedoptimal/logs/training_20251216_180530.log
```

Now you can disconnect SSH and training continues!

### Custom Configuration

Override models or batch size:

```bash
# Use smaller models (faster, less accurate)
STUDENT_MODEL="Qwen/Qwen2.5-Coder-1.5B-Instruct" \
GENERATOR_MODEL="Qwen/Qwen2.5-Coder-3B-Instruct" \
BATCH_SIZE=4 \
bash runpod/run_training.sh

# Use larger models (A100 80GB only)
STUDENT_MODEL="Qwen/Qwen2.5-Coder-7B-Instruct" \
GENERATOR_MODEL="Qwen/Qwen2.5-Coder-14B-Instruct" \
BATCH_SIZE=1 \
bash runpod/run_training.sh
```

---

## Monitoring Training

### 1. Watch Training Logs (SSH)

If running in background:

```bash
# Follow log file in real-time
tail -f /workspace/Psuedoptimal/logs/training_*.log

# Check if training is still running
ps aux | grep train_dsl

# View GPU usage
watch -n 1 nvidia-smi
```

### 2. ClearML Dashboard (Recommended)

Go to: https://app.clear.ml

Navigate to: **Projects → LLM-DSL → GRPO-Training**

**What you'll see:**
- Real-time loss curves
- Reward progression
- Compression ratio trends
- Example encodings every 50 steps
- GPU memory usage
- Hyperparameters

### 3. Check Training Progress

Example log output:

```
=== STARTING TRAINING ===

Step 10/900:
  loss: 2.456
  rewards/mean: -0.123
  rewards/margin: 1.234

================================================================================
  ENCODING VISUALIZATION - Step 50
================================================================================

--- Example 1 ---
TASK: Write a function to find the maximum of two numbers.

ENCODING (15 tokens):
----------------------------------------
max2(a,b)->a if a>b else b
----------------------------------------

EXPANDED PYTHON (from encoding only):
----------------------------------------
def max2(a, b):
    return a if a > b else b
----------------------------------------
================================================================================

Step 100/900:
  loss: 1.823
  rewards/mean: 2.456
  rewards/margin: 3.123
```

**Key metrics to watch:**
- **loss**: Should decrease over time
- **rewards/mean**: Should increase (higher compression + passing tests)
- **ENCODING length**: Should decrease over epochs
- **EXPANDED PYTHON**: Should remain correct

---

## Extracting Results

### 1. Download Trained Model

After training completes, the model is saved to:

```
/workspace/Psuedoptimal/outputs/llm-dsl/final/
```

**Download via SSH/SCP:**

```bash
# From your LOCAL machine
scp -P XXXXX -r root@X.X.X.X:/workspace/Psuedoptimal/outputs/llm-dsl/final ./trained_model_3b

# Or just the adapter weights (much smaller)
scp -P XXXXX -r root@X.X.X.X:/workspace/Psuedoptimal/outputs/llm-dsl/final/adapter_* ./adapter_weights
```

**File structure:**
```
final/
├── adapter_config.json       # LoRA configuration
├── adapter_model.safetensors # Trained LoRA weights (~35 MB)
├── special_tokens_map.json
├── tokenizer.json
├── tokenizer_config.json
└── training_args.bin
```

### 2. Download Training Logs

```bash
# From your LOCAL machine
scp -P XXXXX -r root@X.X.X.X:/workspace/Psuedoptimal/logs ./training_logs
scp -P XXXXX -r root@X.X.X.X:/workspace/Psuedoptimal/outputs/llm-dsl ./full_outputs
```

### 3. Export ClearML Data

**Via ClearML Web UI:**
1. Go to your experiment: https://app.clear.ml/projects/.../experiments/...
2. Click **"Scalars"** → **"Export to CSV"**
3. Click **"Plots"** → **"Download Plot Data"**
4. Click **"Debug Samples"** → View encoding examples

**Via ClearML API:**

```python
from clearml import Task

# Get your experiment
task = Task.get_task(task_id='your_task_id_here')

# Get all logged scalars
scalars = task.get_reported_scalars()
print(scalars.keys())  # ['loss', 'rewards/mean', 'rewards/margin', ...]

# Get scalar values
loss_values = scalars['loss']['loss']['values']
steps = scalars['loss']['loss']['iterations']
```

### 4. Test Your Trained Model

Create a test script on RunPod:

```python
# test_model.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-3B-Instruct",
    device_map="auto",
)

# Load trained adapter
model = PeftModel.from_pretrained(
    base_model,
    "/workspace/Psuedoptimal/outputs/llm-dsl/final"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")

# Test encoding generation
messages = [
    {"role": "system", "content": "You are an encoder. Compress programming tasks into minimal representations."},
    {"role": "user", "content": "Task: Write a function to check if a number is prime.\n\nEncode (minimal):"}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.1)
encoding = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

print(f"Generated encoding: {encoding}")
```

Run it:
```bash
python test_model.py
```

---

## Cost Management

### Estimated Costs

**Full training (3 epochs, 400 examples, 3B + 7B models):**

| GPU | Time | Cost |
|-----|------|------|
| RTX 4090 | ~3.5 hours | **~$1.54** |
| A100 40GB | ~2.5 hours | **~$3.48** |
| A100 80GB | ~2.5 hours | **~$4.73** |

### Cost Optimization Tips

1. **Use RTX 4090** - Best price/performance for this task
2. **Run in background** - Avoid accidental termination
3. **Stop pod immediately** after training completes
4. **Use spot instances** if available (cheaper but can be interrupted)
5. **Debug locally first** - Use `--debug` flag on local machine to test setup

### Stop Your Pod

**After training completes:**

```bash
# On RunPod console: Click "Stop" or "Terminate"
```

Or programmatically:
```bash
# This doesn't stop the pod, just exits SSH
exit
```

**Important:** Pods continue charging until you manually stop them in the RunPod console!

---

## Troubleshooting

### Issue: CUDA Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.50 GiB
```

**Solutions:**

```bash
# Reduce batch size
BATCH_SIZE=1 bash runpod/run_training.sh

# Or modify train_dsl.py temporarily:
# config.batch_size = 1
# config.gradient_accumulation_steps = 16  # Keep effective batch size
```

### Issue: HuggingFace Authentication Failed

**Symptoms:**
```
Cannot access gated repo for url https://huggingface.co/Qwen/...
```

**Solution:**
```bash
# Re-check your token
echo $HF_TOKEN  # Should print your token

# Re-login
huggingface-cli login --token $HF_TOKEN
```

### Issue: Generator Produces Invalid Python

**Symptoms:**
- Encodings logged but all tests fail
- `rewards/mean` stays negative

**Solutions:**

1. **Check Generator is receiving encodings correctly:**
   Look at visualization output - is the Generator getting task context?

2. **Increase Generator temperature** (in `src/rewards.py`):
   ```python
   generator_temperature: float = 0.3  # Was 0.1
   ```

3. **Use larger Generator model:**
   ```bash
   GENERATOR_MODEL="Qwen/Qwen2.5-Coder-14B-Instruct" bash runpod/run_training.sh
   ```

### Issue: Training Appears Stuck

**Symptoms:**
- No log output for >10 minutes
- GPU utilization at 0%

**Solutions:**

```bash
# Check if process is alive
ps aux | grep train_dsl

# Check GPU usage
nvidia-smi

# Check logs for errors
tail -100 /workspace/Psuedoptimal/logs/training_*.log

# Restart training if truly stuck
pkill -f train_dsl
bash runpod/run_training_background.sh
```

### Issue: Model Download Stuck

**Symptoms:**
```
Downloading model... (no progress bar movement)
```

**Solutions:**

```bash
# Check network connectivity
curl https://huggingface.co

# Pre-download models separately
huggingface-cli download Qwen/Qwen2.5-Coder-3B-Instruct
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct

# Then run training
bash runpod/run_training.sh
```

---

## Expected Timeline

**For RTX 4090, 3B + 7B models:**

| Stage | Duration | What's Happening |
|-------|----------|------------------|
| Pod startup | 1-2 min | RunPod provisions GPU |
| Setup script | 5-10 min | Install dependencies, download models (~20GB) |
| Training start | 1 min | Load models, prepare dataset |
| **Training** | **3-3.5 hours** | **900 steps total (3 epochs × 300 steps)** |
| Model saving | 1 min | Save final weights |

**Total: ~3.5-4 hours**

---

## Next Steps After Training

### 1. Analyze Compression Evolution

Compare encodings at different training steps:

```bash
# Check ClearML "Debug Samples" tab
# Or grep logs for "ENCODING VISUALIZATION"
grep -A 20 "ENCODING VISUALIZATION" logs/training_*.log
```

**Look for:**
- Initial encodings: Verbose, Python-like
- Mid-training: Shortened syntax
- Final: Highly compressed symbols

### 2. Measure Compression Ratio

```python
# analyze_compression.py
import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")

# Load your best encodings from logs
encodings = [
    "max2(a,b)->a>b?a:b",
    "prime(n)->all(n%i for i in range(2,int(n**.5)+1))",
    # ... more from your training
]

references = [
    "def max2(a, b):\n    return a if a > b else b",
    "def is_prime(n):\n    return n > 1 and all(n % i != 0 for i in range(2, int(n**0.5) + 1))",
    # ... corresponding reference implementations
]

for enc, ref in zip(encodings, references):
    enc_tokens = len(tokenizer.encode(enc))
    ref_tokens = len(tokenizer.encode(ref))
    ratio = ref_tokens / enc_tokens
    print(f"Compression ratio: {ratio:.2f}x ({ref_tokens} → {enc_tokens} tokens)")
```

### 3. Evaluate on Test Set

Test your trained model on unseen problems:

```python
# Load model and test on new MBPP problems
# Compare pass rate vs compression ratio
```

### 4. Visualize Training Curves

```python
import matplotlib.pyplot as plt
from clearml import Task

task = Task.get_task(task_id='your_task_id')
scalars = task.get_reported_scalars()

# Plot loss
plt.figure(figsize=(10, 6))
plt.plot(scalars['loss']['loss']['iterations'],
         scalars['loss']['loss']['values'])
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig('training_loss.png')

# Plot rewards
plt.figure(figsize=(10, 6))
plt.plot(scalars['rewards/mean']['rewards/mean']['iterations'],
         scalars['rewards/mean']['rewards/mean']['values'])
plt.xlabel('Step')
plt.ylabel('Mean Reward')
plt.title('Reward Progression (Higher = Better Compression)')
plt.savefig('rewards.png')
```

### 5. Share Results

**Create experiment report:**
- Initial vs final encodings
- Compression ratios achieved
- Test pass rates
- Novel symbols/patterns discovered
- Links to ClearML dashboard

---

## Quick Reference Commands

```bash
# === Initial Setup ===
export HF_TOKEN='your_token'
cd /workspace
git clone https://github.com/Tumph/Psuedoptimal.git
cd Psuedoptimal
bash runpod/setup.sh

# === Start Training (Background) ===
bash runpod/run_training_background.sh

# === Monitor ===
tail -f logs/training_*.log
watch -n 1 nvidia-smi

# === Download Results ===
# (From local machine)
scp -P XXXXX -r root@X.X.X.X:/workspace/Psuedoptimal/outputs/llm-dsl/final ./trained_model

# === Stop Pod ===
# Go to RunPod console → Stop pod
```

---

## Support

- **RunPod Issues**: https://docs.runpod.io/
- **HuggingFace**: https://huggingface.co/docs
- **ClearML**: https://clear.ml/docs/
- **Project Issues**: https://github.com/Tumph/Psuedoptimal/issues

---

**Ready to train? Let's discover what encoding emerges!** 🚀
