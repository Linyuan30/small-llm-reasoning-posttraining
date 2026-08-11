# Copyright (c) 2026
#
# 【消融实验变体 v2】不用 <answer>...</answer> XML 标签的输出格式。
#
# 背景（详见 docs/问题记录.md）：
#   Qwen3 tokenizer 里 <think>/</think> 是原生 special token（各编码为1个
#   token），但 <answer>/</answer> 会被 BPE 拆成 3 个普通子词
#   （如 '<' 'answer' '>'），学习难度明显更高。除了"给 answer 也加 special
#   token"（见 add_answer_special_tokens.py）之外，另一种思路是干脆不用
#   <answer> 标签，直接沿用 Qwen3 原生的 <think>...</think> 机制 + 纯文本
#   最终答案，格式更贴近 Qwen3 预训练时见过的分布。
#
# 本文件与主线 format_cot.py 完全独立，不修改、不影响主线的
# <think>...</think><answer>...</answer> 格式实验结果（该格式是
# docs/流程.md 4.1节明确的项目主设计，已跑通 SFT/GRPO/PPO/DPO 全流程）。
# 本文件仅用于对比消融实验："XML answer 标签 + special token" vs
# "无 answer 标签的原生格式"，哪个格式合规率/pass@k 更好。
#
# 输出格式：
#   <think>
#   （推理过程）
#   </think>
#   （最终答案，纯文本，无任何包裹标签）
#
# 供 prepare_gsm8k_v2_plain_answer.py / prepare_math_v2_plain_answer.py 共用。

from __future__ import annotations

import re

SYSTEM_PROMPT = (
    "You are a helpful math assistant. "
    "For every problem, first think step by step inside <think>...</think>, "
    "then write the final answer directly after </think>. "
    "The final answer should be a concise value only "
    "(a number, fraction, or short expression), without extra words."
)

INSTRUCTION_SUFFIX = (
    "Please reason step by step. Respond in the following format strictly:\n"
    "<think>\n(your reasoning process)\n</think>\n(final answer only, no tags)"
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
    """把 (推理过程, 最终答案) 拼接成 <think>...</think>+纯文本答案 格式字符串。

    用于构造 SFT 冷启动数据的 assistant 回复（v2 变体，无 <answer> 标签）。
    """
    reasoning = reasoning.strip()
    final_answer = str(final_answer).strip()
    return f"<think>\n{reasoning}\n</think>\n{final_answer}"


# 以下两个函数与主线 format_cot.py 完全相同（不涉及 answer 标签变化），
# 直接复用同样的实现，避免重复维护两份一致的逻辑。
_GSM8K_CALC_ANNOTATION = re.compile(r"<<[^>]*>>")


def clean_gsm8k_rationale(answer_raw: str) -> tuple[str, str]:
    """把 GSM8K 原始 answer 字段（含 <<calc>> 标注和 '#### X' 结尾）
    拆分为 (纯推理过程文本, 最终答案)。
    """
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
