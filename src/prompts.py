"""Prompt templates for Student and Generator models."""

from typing import Union

# =============================================================================
# Student Model Prompts (Encoding Generation)
# =============================================================================

STUDENT_SYSTEM_PROMPT = """You are an encoder. Compress programming tasks into minimal representations.
Your output is the ONLY information another model receives to write Python code.
You are heavily penalized for every token - be extremely concise.
Include: function name, parameters, core logic. Use any symbols or format you invent."""

STUDENT_USER_TEMPLATE = """Task: {prompt}

Encode (minimal):"""


def format_student_prompt(task_prompt: str) -> list[dict[str, str]]:
    """
    Format an MBPP task for the Student model (chat format).

    Args:
        task_prompt: The MBPP task description

    Returns:
        List of message dicts for chat template
    """
    return [
        {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
        {"role": "user", "content": STUDENT_USER_TEMPLATE.format(prompt=task_prompt)},
    ]


def format_student_prompt_text(task_prompt: str) -> str:
    """
    Format an MBPP task for the Student model (plain text).

    Args:
        task_prompt: The MBPP task description

    Returns:
        Formatted prompt string
    """
    return f"{STUDENT_SYSTEM_PROMPT}\n\n{STUDENT_USER_TEMPLATE.format(prompt=task_prompt)}"


# =============================================================================
# Generator Model Prompts (Encoding to Python Expansion)
# =============================================================================

GENERATOR_SYSTEM_PROMPT = """You are a decoder. You receive a compressed encoding and must expand it into complete, valid, executable Python code.
The encoding contains all information needed: function name, parameters, and logic.
You have NO other context - interpret the encoding directly.
Expand to proper Python with correct syntax, indentation, and necessary imports.
Output ONLY the Python code."""

GENERATOR_USER_TEMPLATE = """{encoding}

Python:"""


def format_generator_prompt(encoding: str) -> list[dict[str, str]]:
    """
    Format an encoding expansion request for the Generator model (chat format).

    Args:
        encoding: The compressed encoding from the Student

    Returns:
        List of message dicts for chat template
    """
    return [
        {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": GENERATOR_USER_TEMPLATE.format(encoding=encoding),
        },
    ]


def format_generator_prompt_text(encoding: str) -> str:
    """
    Format an encoding expansion request for the Generator model (plain text).

    Args:
        encoding: The compressed encoding from the Student

    Returns:
        Formatted prompt string
    """
    return (
        f"{GENERATOR_SYSTEM_PROMPT}\n\n"
        f"{GENERATOR_USER_TEMPLATE.format(encoding=encoding)}"
    )


# =============================================================================
# Utility Functions
# =============================================================================


def apply_chat_template(
    tokenizer,
    messages: list[dict[str, str]],
    add_generation_prompt: bool = True,
) -> str:
    """
    Apply tokenizer's chat template to messages.

    Args:
        tokenizer: HuggingFace tokenizer with chat template
        messages: List of message dicts
        add_generation_prompt: Whether to add the assistant turn start

    Returns:
        Formatted prompt string
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def extract_python_code(text: str) -> str:
    """
    Extract Python code from text, handling markdown code blocks.

    Args:
        text: Generated text that may contain code blocks

    Returns:
        Extracted Python code
    """
    import re

    # Try to find ```python ... ``` block
    pattern = r"```python\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try generic ``` ... ``` block
    pattern = r"```\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Return as-is if no code block found
    return text.strip()
