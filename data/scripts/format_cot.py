# Copyright (c) 2026
#
# 统一 <think>/<answer> 输出格式的公共工具函数
# 对应 docs/流程.md 4.1 节：
#   <think>
#   （推理过程）
#   </think>
#   <answer>
#   （最终数值答案）
#   </answer>
#
# 供 prepare_gsm8k.py / prepare_math.py / build_sft_coldstart.py 共用。

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
    """构造统一的 chat 格式 prompt，用于 RL(GRPO/PPO) 训练与评估阶段。"""
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
    """把 (推理过程, 最终答案) 拼接成统一的 <think>/<answer> 格式字符串。

    用于构造 SFT 冷启动数据的 assistant 回复。
    """
    reasoning = reasoning.strip()
    final_answer = str(final_answer).strip()
    return f"<think>\n{reasoning}\n</think>\n<answer>\n{final_answer}\n</answer>"


_GSM8K_CALC_ANNOTATION = re.compile(r"<<[^>]*>>")


def clean_gsm8k_rationale(answer_raw: str) -> tuple[str, str]:
    """把 GSM8K 原始 answer 字段（含 <<calc>> 标注和 '#### X' 结尾）
    拆分为 (纯推理过程文本, 最终答案)。
    """
    # 去掉计算器标注，如 "<<48/2=24>>"
    text = _GSM8K_CALC_ANNOTATION.sub("", answer_raw)

    if "####" in text:
        rationale, final_answer = text.split("####", 1)
    else:
        rationale, final_answer = text, ""

    rationale = rationale.strip()
    final_answer = final_answer.strip().replace(",", "")
    return rationale, final_answer


def extract_boxed_answer(solution: str) -> str | None:
    """从 MATH 数据集官方 solution 文本中提取 \\boxed{...} 内的最终答案。"""
    idx = solution.rfind("\\boxed")
    if idx < 0:
        idx = solution.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    # 跳到第一个 '{'
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
