#!/bin/bash
# RunPod Background Training Script
# Runs full training in background so it continues after SSH disconnect

set -e

cd /workspace/llm-golf

echo "=== Starting LLM-Golf Full Training in Background ==="

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

# Create logs directory
mkdir -p /workspace/llm-golf/logs

# Get timestamp for log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/workspace/llm-golf/logs/training_${TIMESTAMP}.log"

echo "Configuration:"
echo "  Student Model: $STUDENT_MODEL"
echo "  Generator Model: $GENERATOR_MODEL"
echo "  Batch Size: $BATCH_SIZE"
echo "  Log File: $LOG_FILE"
echo ""

# Run full training in background with nohup (no --debug flag)
TOKENIZERS_PARALLELISM=false nohup python train_dsl.py \
    --student-model "$STUDENT_MODEL" \
    --generator-model "$GENERATOR_MODEL" \
    --batch-size "$BATCH_SIZE" \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "Training started in background with PID: $PID"
echo ""
echo "To monitor training:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To check if still running:"
echo "  ps aux | grep train_dsl"
echo ""
echo "To stop training:"
echo "  kill $PID"
