#!/bin/bash
# RunPod Setup Script for LLM-Golf
# Run this once when you first connect to a pod

set -e

echo "=== LLM-Golf RunPod Setup ==="

# Check for required environment variables
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN not set. You'll need to set it before training."
    echo "  export HF_TOKEN='your_huggingface_token'"
fi

# Update system packages
echo "Updating system packages..."
apt-get update -qq

# Install Python packages
echo "Installing Python packages..."
pip install --upgrade pip
pip install torch transformers trl peft bitsandbytes accelerate datasets clearml python-dotenv huggingface_hub

# Login to HuggingFace if token is set
if [ -n "$HF_TOKEN" ]; then
    echo "Logging into HuggingFace..."
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
fi

# Clone or update the repository
cd /workspace
if [ -d "llm-golf" ]; then
    echo "Updating existing llm-golf repository..."
    cd llm-golf
    git pull
else
    echo "Cloning llm-golf repository..."
    # Replace with your actual repo URL
    git clone https://github.com/YOUR_USERNAME/llm-golf.git
    cd llm-golf
fi

# Install the package
echo "Installing llm-golf package..."
pip install -e ".[cuda]"

# Verify GPU is available
echo ""
echo "=== GPU Status ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Set your HuggingFace token: export HF_TOKEN='your_token'"
echo "  2. Set ClearML credentials (optional):"
echo "     export CLEARML_API_ACCESS_KEY='your_key'"
echo "     export CLEARML_API_SECRET_KEY='your_secret'"
echo "  3. Run training: ./runpod/run_training.sh"
echo ""
