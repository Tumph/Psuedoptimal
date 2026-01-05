#!/usr/bin/env python3
"""
Download MBPP dataset for offline training.

This script downloads the MBPP dataset to the local cache.

Usage:
    python download_dataset.py
"""

from datasets import load_dataset

def main():
    """Download MBPP dataset."""
    print("\n" + "="*60)
    print("  DATASET DOWNLOAD SCRIPT")
    print("="*60)

    try:
        print("\nDownloading MBPP (sanitized) dataset...")
        dataset = load_dataset("mbpp", "sanitized", split="train")

        print(f"\n✓ Dataset downloaded successfully!")
        print(f"  Samples: {len(dataset)}")
        print(f"  Columns: {dataset.column_names}")
        print(f"  Cache location: ~/.cache/huggingface/datasets")

        # Show a sample
        print(f"\nSample entry:")
        example = dataset[0]
        print(f"  Task: {example['prompt'][:80]}...")
        print(f"  Code length: {len(example['code'])} chars")
        print(f"  Tests: {len(example['test_list'])} assertions")

    except Exception as e:
        print(f"\n✗ Error downloading dataset:")
        print(f"  {e}")
        return False

    print()
    return True


if __name__ == "__main__":
    main()
