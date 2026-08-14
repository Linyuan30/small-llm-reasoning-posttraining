# Copyright (c) 2026
#
# Shared utilities for building <think>/<answer> formatted prompts and responses.
# Used by prepare_gsm8k.py, prepare_math.py, and build_sft_mix.py.

from __future__ import annotations

import re

SYSTEM_PROMPT = (
    "You are a helpful math assistant. "
    "For every problem, first think step by step inside <think>...</think>, "
    "then give the final answer inside <answer>...</answer>. "
    "The final answer in <answer></answer> should be a concise value only "
    "(a number, fraction, or short expression), without extra words."
)

INSTRUCTION_SUFFIX = (
    "Please reason step by step, and put your final answer within "
    "<answer></answer>. Respond in the following format strictly:\n"
    "<think>\n(your reasoning process)\n</think>\n<answer>\n(final answer only)\n</answer>"
)


def build_prompt_messages(question: str, add_system_prompt: bool = True) -> list[dict]:
    """Build a chat-format prompt for RL (GRPO/PPO) training and evaluation."""
    messages = []
    if add_system_prompt:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append(
        {
            "role": "user",
            "content": f"{question}\n\n{INSTRUCTION_SUFFIX}",
        }
    )
    return messages


def format_think_answer(reasoning: str, final_answer: str) -> str:
    """Combine (reasoning, answer) into a <think>/<answer> assistant reply for SFT."""
    reasoning = reasoning.strip()
    final_answer = str(final_answer).strip()
    return f"<think>\n{reasoning}\n</think>\n<answer>\n{final_answer}\n</answer>"


_GSM8K_CALC_ANNOTATION = re.compile(r"<<[^>]*>>")


def clean_gsm8k_rationale(answer_raw: str) -> tuple[str, str]:
    """Split a raw GSM8K answer field (with <<calc>> annotations and '#### X' suffix)
    into (rationale, final_answer).
    """
    # strip calculator annotations like "<<48/2=24>>"
    text = _GSM8K_CALC_ANNOTATION.sub("", answer_raw)

    if "####" in text:
        rationale, final_answer = text.split("####", 1)
    else:
        rationale, final_answer = text, ""

    rationale = rationale.strip()
    final_answer = final_answer.strip().replace(",", "")
    return rationale, final_answer


def extract_boxed_answer(solution: str) -> str | None:
    """Extract the content of the last \\boxed{...} in a MATH solution string."""
    idx = solution.rfind("\\boxed")
    if idx < 0:
        idx = solution.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    # advance to the first '{'
    while i < len(solution) and solution[i] != "{":
        i += 1
    if i >= len(solution):
        return None

    depth = 0
    start = i
    for j in range(i, len(solution)):
        if solution[j] == "{":
            depth += 1
        elif solution[j] == "}":
            depth -= 1
            if depth == 0:
                return solution[start + 1 : j]
    return None
