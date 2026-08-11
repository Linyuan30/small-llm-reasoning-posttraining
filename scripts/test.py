"""Quick tokenizer sanity check for special tokens used in this project."""

import os

from transformers import AutoTokenizer

MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen3-0.6B-Base")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print(tokenizer.encode(" νοέω"))
print(tokenizer.encode(">"))
print(tokenizer.encode("<answer>"))
print(tokenizer.encode("</answer>"))
