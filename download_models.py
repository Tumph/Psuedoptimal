#!/usr/bin/env python3
"""
Download models for offline training.

This script downloads the Student and Generator models to the local cache
so that training can run without internet access.

Usage:
    python download_models.py
"""

import os
from transformers import AutoTokenizer, AutoModelForCausalLM

def download_model(model_name: str):
    """Download a model and its tokenizer to cache."""
    print(f"\n{'='*60}")
    print(f"Downloading: {model_name}")
    print(f"{'='*60}\n")

    try:
        # Download tokenizer
        print(f"  [1/2] Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        print(f"  ✓ Tokenizer downloaded")

        # Download model
        print(f"  [2/2] Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )
        print(f"  ✓ Model downloaded")

        # Get model size
        param_count = sum(p.numel() for p in model.parameters())
        size_gb = param_count * 4 / 1e9  # 4 bytes per param (float32)

        print(f"\n  Model info:")
        print(f"    Parameters: {param_count / 1e9:.2f}B")
        print(f"    Size (float32): ~{size_gb:.2f} GB")
        print(f"    Cache location: {tokenizer.name_or_path}")

        return True

    except Exception as e:
        print(f"  ✗ Error downloading {model_name}:")
        print(f"    {e}")
        return False


def main():
    """Download all required models."""
    print("\n" + "="*60)
    print("  MODEL DOWNLOAD SCRIPT")
    print("="*60)

    # Check for HF token
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("\n⚠️  WARNING: HF_TOKEN not set!")
        print("  Qwen models require authentication.")
        print("  Set your token: export HF_TOKEN='your_token'")
        print("  Or login: huggingface-cli login")
        print()
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    # Models to download
    models = [
        "Qwen/Qwen2.5-Coder-0.5B-Instruct",  # Student
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",  # Generator
    ]

    results = {}
    for model_name in models:
        success = download_model(model_name)
        results[model_name] = success

    # Summary
    print("\n" + "="*60)
    print("  DOWNLOAD SUMMARY")
    print("="*60)

    for model_name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {status}: {model_name}")

    if all(results.values()):
        print("\n✓ All models downloaded successfully!")
        print("  You can now train offline.")
    else:
        print("\n✗ Some models failed to download.")
        print("  Check your internet connection and HF_TOKEN.")

    print()


if __name__ == "__main__":
    main()
