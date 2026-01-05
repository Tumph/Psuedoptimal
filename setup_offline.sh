#!/bin/bash
# Complete setup script for offline training
# This script installs dependencies and downloads all required models/datasets

set -e  # Exit on error

echo "================================================================"
echo "  Pseudoptimal - Offline Setup"
echo "================================================================"
echo ""

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  WARNING: HF_TOKEN environment variable not set"
    echo ""
    echo "Qwen models require HuggingFace authentication."
    echo "Get your token from: https://huggingface.co/settings/tokens"
    echo ""
    echo "Set it with:"
    echo "  export HF_TOKEN='your_token_here'"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# Step 1: Install dependencies
echo ""
echo "Step 1/3: Installing dependencies..."
echo "------------------------------------------------------"
pip install -e . --quiet
echo "✓ Dependencies installed"

# Step 2: Download models
echo ""
echo "Step 2/3: Downloading models (this may take 5-15 minutes)..."
echo "------------------------------------------------------"
python3 download_models.py

# Step 3: Download dataset
echo ""
echo "Step 3/3: Downloading dataset..."
echo "------------------------------------------------------"
python3 download_dataset.py

# Done
echo ""
echo "================================================================"
echo "  Setup Complete!"
echo "================================================================"
echo ""
echo "You can now run training offline:"
echo "  python3 train_dsl.py --debug          # Debug mode (50 samples)"
echo "  python3 train_dsl.py                  # Full training (400 samples)"
echo ""
echo "Files are cached in:"
echo "  Models:  ~/.cache/huggingface/hub/"
echo "  Dataset: ~/.cache/huggingface/datasets/"
echo ""
