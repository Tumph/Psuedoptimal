#!/bin/bash
# RunPod Training Script for LLM-Golf
# Runs full training with scaled models on GPU

set -e

cd /workspace/llm-golf

echo "=== LLM-Golf Full Training ==="

# Check for HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN not set. Required for Qwen models."
    echo "  export HF_TOKEN='your_huggingface_token'"
    exit 1
fi

# Suppress warnings
export TOKENIZERS_PARALLELISM=false

# Default model configuration (3B Student, 7B Generator for RTX 4090)
STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen2.5-Coder-3B-Instruct}"
GENERATOR_MODEL="${GENERATOR_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
BATCH_SIZE="${BATCH_SIZE:-2}"

echo "Configuration:"
echo "  Student Model: $STUDENT_MODEL"
echo "  Generator Model: $GENERATOR_MODEL"
echo "  Batch Size: $BATCH_SIZE"
echo ""

# Check GPU
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Run full training (no --debug flag)
echo "Starting full training..."
python train_dsl.py \
    --student-model "$STUDENT_MODEL" \
    --generator-model "$GENERATOR_MODEL" \
    --batch-size "$BATCH_SIZE"
