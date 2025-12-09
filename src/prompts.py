"""Prompt templates for Student and Generator models."""

from typing import Union

# =============================================================================
# Student Model Prompts (DSL Generation)
# =============================================================================

STUDENT_SYSTEM_PROMPT = """You are a code compression expert. Your task is to write extremely concise pseudo-code (DSL) that captures the essential logic of a programming problem.

DSL Compression Rules:
- Use single letters for common patterns: L=list, D=dict, S=set, s=string
- Use arrow notation: -> for return
- Use shorthand: fn for function, lp for loop, if/el for if/else
- Omit type hints and docstrings
- Use math notation where clearer: ∈ for 'in', ∀ for 'for all'
- Focus on algorithm essence, minimize syntax

Your DSL will be expanded by another model into full Python code. Be as concise as possible while preserving the logic."""

STUDENT_USER_TEMPLATE = """Task: {prompt}

Write minimal DSL that solves this. Be extremely concise:"""


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
# Generator Model Prompts (DSL to Python Expansion)
# =============================================================================

GENERATOR_SYSTEM_PROMPT = """You are a code expansion expert. You receive compressed pseudo-code (DSL) and must expand it into complete, valid, executable Python code.

Expansion Rules:
- Expand all abbreviations to full Python syntax
- Add proper indentation and structure
- Include necessary imports at the top
- The function name MUST match what the task describes
- Output ONLY the Python code, no explanations

Common DSL patterns:
- fn = function definition (def)
- -> = return statement
- L/D/S = list/dict/set
- lp = loop (for/while)"""

GENERATOR_USER_TEMPLATE = """Original Task: {task_prompt}

DSL to expand:
{dsl_code}

Python code:"""


def format_generator_prompt(task_prompt: str, dsl_code: str) -> list[dict[str, str]]:
    """
    Format a DSL expansion request for the Generator model (chat format).

    Args:
        task_prompt: The original MBPP task description
        dsl_code: The compressed DSL from the Student

    Returns:
        List of message dicts for chat template
    """
    return [
        {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": GENERATOR_USER_TEMPLATE.format(
                task_prompt=task_prompt, dsl_code=dsl_code
            ),
        },
    ]


def format_generator_prompt_text(task_prompt: str, dsl_code: str) -> str:
    """
    Format a DSL expansion request for the Generator model (plain text).

    Args:
        task_prompt: The original MBPP task description
        dsl_code: The compressed DSL from the Student

    Returns:
        Formatted prompt string
    """
    return (
        f"{GENERATOR_SYSTEM_PROMPT}\n\n"
        f"{GENERATOR_USER_TEMPLATE.format(task_prompt=task_prompt, dsl_code=dsl_code)}"
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
