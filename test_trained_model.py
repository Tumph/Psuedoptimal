#!/usr/bin/env python3
"""
Test script for trained Psuedoptimal model.

Usage:
    python test_trained_model.py [--adapter-path PATH]
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model(adapter_path: str, device: str = "auto"):
    """Load base model and trained adapter."""
    print(f"Loading model from: {adapter_path}")

    # Determine base model from adapter config
    base_model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"

    print(f"  Base model: {base_model_name}")
    print(f"  Loading adapter...")

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map=device,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    # Load trained adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("  Model loaded successfully!")
    return model, tokenizer


def generate_encoding(model, tokenizer, task: str, max_tokens: int = 50):
    """Generate compressed encoding for a task."""
    # Format prompt (Student encoder prompt)
    messages = [
        {
            "role": "system",
            "content": "You are an encoder. Compress programming tasks into minimal representations. "
                      "Your output is the ONLY information another model receives to write Python code. "
                      "You are heavily penalized for every token - be extremely concise. "
                      "Include: function name, parameters, core logic. Use any symbols or format you invent."
        },
        {
            "role": "user",
            "content": f"Task: {task}\n\nEncode (minimal):"
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode
    generated = outputs[0][inputs.input_ids.shape[1]:]
    encoding = tokenizer.decode(generated, skip_special_tokens=True)

    return encoding


def count_tokens(tokenizer, text: str) -> int:
    """Count tokens in text."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def main():
    parser = argparse.ArgumentParser(description="Test trained Psuedoptimal model")
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="./outputs/llm-dsl/final",
        help="Path to trained adapter (default: ./outputs/llm-dsl/final)"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("  Testing Trained Psuedoptimal Model")
    print("=" * 80)
    print()

    # Load model
    model, tokenizer = load_model(args.adapter_path)

    print()
    print("=" * 80)
    print("  Test Cases")
    print("=" * 80)

    # Test tasks
    test_tasks = [
        "Write a function to find the maximum of two numbers.",
        "Write a function to check if a number is even.",
        "Write a function to reverse a string.",
        "Write a function to check if a number is prime.",
        "Write a function to compute the factorial of a number.",
    ]

    for i, task in enumerate(test_tasks, 1):
        print(f"\n--- Test {i} ---")
        print(f"Task: {task}")

        # Generate encoding
        encoding = generate_encoding(model, tokenizer, task)
        encoding_tokens = count_tokens(tokenizer, encoding)

        print(f"\nGenerated Encoding ({encoding_tokens} tokens):")
        print("-" * 60)
        print(encoding)
        print("-" * 60)

        # Calculate compression vs naive reference
        naive_func_name = task.split("to ")[-1].replace(" ", "_").replace(".", "").lower()
        naive_code = f"def {naive_func_name}():\n    pass"
        naive_tokens = count_tokens(tokenizer, naive_code)

        if encoding_tokens < naive_tokens:
            print(f"✓ More compressed than naive ({naive_tokens} tokens)")
        else:
            print(f"⚠ Longer than naive ({naive_tokens} tokens)")

    print()
    print("=" * 80)
    print("  Testing Complete")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Check if encodings are more compressed than original Python")
    print("  2. Manually test if Generator can decode these encodings")
    print("  3. Compare to encodings from untrained model (baseline)")
    print()


if __name__ == "__main__":
    main()
