#!/bin/bash
# Simplified installation - just install the package and download models
# Use this if setup_offline_robust.sh has issues

echo "================================================================"
echo "  Simple Installation"
echo "================================================================"
echo ""

# Upgrade pip first
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install all build dependencies
echo "Installing build dependencies..."
pip install -q hatchling editables setuptools wheel

# Install the package
echo "Installing Pseudoptimal package..."
pip install -e .

echo ""
echo "✓ Installation complete!"
echo ""
echo "Now run the model download script:"
echo "  python3 download_models.py"
echo "  python3 download_dataset.py"
echo ""
