#!/bin/bash
# Robust setup script with longer timeouts for slow connections

set -e  # Exit on error

echo "================================================================"
echo "  Pseudoptimal - Offline Setup (Robust)"
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

# Step 1: Upgrade pip with increased timeout
echo ""
echo "Step 0/4: Upgrading pip..."
echo "------------------------------------------------------"
pip install --upgrade pip --timeout=120 --quiet
echo "✓ pip upgraded"

# Step 1: Install build dependencies with longer timeout
echo ""
echo "Step 1/4: Installing build dependencies..."
echo "------------------------------------------------------"
pip install hatchling editables --timeout=120 --retries=5
echo "✓ Build dependencies installed"

# Step 2: Install package dependencies with longer timeout
echo ""
echo "Step 2/4: Installing package dependencies..."
echo "------------------------------------------------------"
echo "This may take a while (downloading PyTorch, transformers, etc.)..."

# Install core dependencies with long timeout
pip install --timeout=300 --retries=5 \
    torch \
    transformers \
    trl \
    peft \
    accelerate \
    datasets \
    bitsandbytes \
    huggingface-hub

echo "✓ Core dependencies installed"

# Install optional dependencies
pip install --timeout=120 --retries=5 \
    python-dotenv \
    clearml

echo "✓ Optional dependencies installed"

# Install package in editable mode (no build needed now)
pip install -e . --no-build-isolation --timeout=120

echo "✓ Package installed"

# Step 3: Download models
echo ""
echo "Step 3/4: Downloading models (this may take 5-15 minutes)..."
echo "------------------------------------------------------"
python3 download_models.py

# Step 4: Download dataset
echo ""
echo "Step 4/4: Downloading dataset..."
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
