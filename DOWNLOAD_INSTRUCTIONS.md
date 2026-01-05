# Download Models for Offline Training

## Quick Setup

I've created download scripts to cache the models locally. Follow these steps:

### Step 1: Install Dependencies

```bash
# Install the package (this installs all dependencies from pyproject.toml)
pip install -e .

# Optional: Install CUDA dependencies if you have NVIDIA GPU on Linux
pip install -e ".[cuda]"
```

### Step 2: Set HuggingFace Token

Qwen models require authentication:

```bash
# Set your token (get it from https://huggingface.co/settings/tokens)
export HF_TOKEN="your_huggingface_token_here"

# Or login via CLI
huggingface-cli login
```

### Step 3: Download Models

```bash
# Download Student (0.5B) and Generator (1.5B) models
python3 download_models.py
```

This will download:
- **Student**: Qwen/Qwen2.5-Coder-0.5B-Instruct (~0.5 GB)
- **Generator**: Qwen/Qwen2.5-Coder-1.5B-Instruct (~1.5 GB)

Expected download time: 5-15 minutes depending on your connection.

### Step 4: Download Dataset

```bash
# Download MBPP dataset (~1 MB)
python3 download_dataset.py
```

### Step 5: Verify Setup

```bash
# Run a quick debug test (50 samples)
python3 train_dsl.py --debug --no-clearml
```

## Expected Output

After successful downloads:

```
✓ All models downloaded successfully!
  You can now train offline.

  Models cached in: ~/.cache/huggingface/hub/
  Dataset cached in: ~/.cache/huggingface/datasets/
```

## Disk Space Requirements

- Models: ~2 GB (0.5B + 1.5B models)
- Dataset: ~1 MB
- Training outputs: ~500 MB (checkpoints)
- **Total**: ~3 GB

## Troubleshooting

### "ModuleNotFoundError: No module named 'transformers'"

You need to install dependencies first:
```bash
pip install -e .
```

### "401 Unauthorized" or "HF_TOKEN not set"

You need to authenticate:
```bash
export HF_TOKEN="your_token"
# Get token from: https://huggingface.co/settings/tokens
```

### "Connection timeout"

Check your internet connection. The models are ~2GB total and may take time to download.

### Downloads interrupted

Just re-run the script. HuggingFace will resume from where it left off.

## What Gets Downloaded?

1. **Models** (cached in `~/.cache/huggingface/hub/`):
   - Model weights (PyTorch .bin or .safetensors files)
   - Tokenizer files (vocab, merges, config)
   - Model config files

2. **Dataset** (cached in `~/.cache/huggingface/datasets/`):
   - MBPP sanitized split (~400 training examples)
   - Parquet files with task descriptions, code, and tests

## Running Offline

Once downloaded, training will work offline:

```bash
# No internet needed after initial download
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 train_dsl.py
```

## For RunPod Deployment

On RunPod, the setup script handles this automatically:

```bash
bash runpod/setup.sh
```

This will:
1. Install dependencies
2. Download models
3. Download dataset
4. Set up environment

Then you can run training:

```bash
bash runpod/run_training_background.sh
```
