
from transformers import AutoTokenizer

model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

pairs = [
    ("def", "fn"),
    ("return", "->"),
    ("list", "L"),
    ("dict", "D"),
    ("for", "lp"),
    ("while", "wh"),
    ("range", "rn"),
    ("print", "pr"),
    ("if", "?"),
    ("else", ":"),
    ("import", "imp"),
    (" class", " cls"),
]

print(f"{'Python':<10} | {'Tokens':<6} || {'DSL':<10} | {'Tokens':<6} | {'Gain':<6}")
print("-" * 60)

for py, dsl in pairs:
    # We add a space prefix as they often appear in middle of code
    py_toks = len(tokenizer.encode(" " + py, add_special_tokens=False))
    dsl_toks = len(tokenizer.encode(" " + dsl, add_special_tokens=False))
    print(f"{py:<10} | {py_toks:<6} || {dsl:<10} | {dsl_toks:<6} | {py_toks - dsl_toks:<6}")

# Check specific sentences
s1 = "def is_even(n): return n % 2 == 0"
s2 = "fn is_even(n)->n%2==0"
t1 = len(tokenizer.encode(s1, add_special_tokens=False))
t2 = len(tokenizer.encode(s2, add_special_tokens=False))
print(f"\nSentence 1: {t1} tokens")
print(f"Sentence 2: {t2} tokens")
print(f"Gain: {t1-t2}")
