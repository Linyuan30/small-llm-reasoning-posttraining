# Copyright (c) 2026
#
# 【消融实验变体 v2】基于规则的可验证 reward 函数 —— 无 <answer> XML 标签版本。
#
# 与主线 rule_reward.py 完全独立、不修改，仅供 "<think>...</think>+纯文本答案"
# 格式（见 data/scripts/format_cot_v2_plain_answer.py）的 SFT/RL 训练与评估使用。
#
# 格式定义变化：
#   主线：<think>...</think><answer>...</answer>   （answer 需要 XML 闭合标签）
#   v2  ：<think>...</think>\n最终答案文本            （</think> 后的剩余文本即为答案）
#
# 其余的答案规范化/等价判断逻辑（LaTeX/分数/根号等）与主线完全一致，直接复用。

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# 复用主线的答案规范化与等价判断逻辑，避免重复维护
from rule_reward import is_equivalent  # noqa: E402


# --------------------------------------------------------------------------
# 1. 输出格式定义（v2：无 answer 标签，</think> 后的剩余文本即为最终答案）
# --------------------------------------------------------------------------

THINK_PLAIN_ANSWER_PATTERN = re.compile(
    r"^\s*<think>(?P<think>(?:(?!</think>).)*?)</think>\s*(?P<answer>\S.*?)\s*$",
    re.DOTALL,
)

LOOSE_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)


@dataclass
class FormatCheckResult:
    strict_ok: bool  # 是否严格符合 <think>...</think>最终答案 完整格式
    has_think: bool  # 是否存在 <think>...</think>
    has_answer: bool  # 是否存在非空的最终答案文本
    answer_text: Optional[str]  # 抽取出的 answer 内容


def check_format(solution_str: str) -> FormatCheckResult:
    """检查模型输出是否符合 <think>...</think>+纯文本答案 格式，并抽取 answer 内容。"""
    stripped = solution_str.strip()
    strict_match = THINK_PLAIN_ANSWER_PATTERN.match(stripped)
    if strict_match is not None:
        answer_text = strict_match.group("answer").strip()
        return FormatCheckResult(
            strict_ok=True,
            has_think=True,
            has_answer=bool(answer_text),
            answer_text=answer_text or None,
        )

    think_matches = LOOSE_THINK_PATTERN.findall(solution_str)
    has_think = len(think_matches) > 0

    # 宽松抽取：取最后一个 </think> 之后的剩余文本作为 answer（若存在）
    answer_text = None
    last_think_end = solution_str.rfind("</think>")
    if last_think_end >= 0:
        remainder = solution_str[last_think_end + len("</think>") :].strip()
        answer_text = remainder if remainder else None

    return FormatCheckResult(
        strict_ok=False,
        has_think=has_think,
        has_answer=answer_text is not None,
        answer_text=answer_text,
    )


# --------------------------------------------------------------------------
# 2. Reward 组合：正确性 + 格式（与主线 RuleReward 接口保持一致，便于复用训练脚本）
# --------------------------------------------------------------------------


@dataclass
class RewardResult:
    score: float
    correct: bool
    format_ok: bool
    extracted_answer: Optional[str]


class RuleRewardV2PlainAnswer:
    """规则 reward（v2 变体）：正确性 reward + 格式 reward 加权组合。

    接口与主线 `rule_reward.RuleReward` 保持一致，参数含义相同，
    便于复用 eval_pass_at_k.py 等评估脚本（切换 import 即可）。
    """

    def __init__(
        self,
        correct_score: float = 1.0,
        wrong_score: float = 0.0,
        format_bonus: float = 0.1,
        format_penalty: float = -0.1,
        no_answer_score: float = -1.0,
        truncated_extra_penalty: float = 0.0,
        length_penalty_start_ratio: float = 0.9,
    ) -> None:
        self.correct_score = correct_score
        self.wrong_score = wrong_score
        self.format_bonus = format_bonus
        self.format_penalty = format_penalty
        self.no_answer_score = no_answer_score
        self.truncated_extra_penalty = truncated_extra_penalty
        self.length_penalty_start_ratio = length_penalty_start_ratio

    def score(
        self,
        solution_str: str,
        ground_truth: str,
        response_length_ratio: Optional[float] = None,
    ) -> RewardResult:
        fmt = check_format(solution_str)

        if fmt.answer_text is None:
            score = self.no_answer_score
            if (
                self.truncated_extra_penalty > 0
                and response_length_ratio is not None
                and response_length_ratio > self.length_penalty_start_ratio
            ):
                span = max(1e-6, 1.0 - self.length_penalty_start_ratio)
                progress = min(1.0, (response_length_ratio - self.length_penalty_start_ratio) / span)
                score -= self.truncated_extra_penalty * progress
            return RewardResult(
                score=score,
                correct=False,
                format_ok=fmt.strict_ok,
                extracted_answer=None,
            )

        correct = is_equivalent(fmt.answer_text, ground_truth)
        base = self.correct_score if correct else self.wrong_score
        base += self.format_bonus if fmt.strict_ok else self.format_penalty

        return RewardResult(
            score=base,
            correct=correct,
            format_ok=fmt.strict_ok,
            extracted_answer=fmt.answer_text,
        )


_default_reward = RuleRewardV2PlainAnswer()


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> float:
    """兼容 verl `custom_reward_function` 接口的打分函数（v2 变体）。

    在 veRL 的训练配置中通过如下方式接入：
        custom_reward_function.path=reward/rule_reward_v2_plain_answer.py
        custom_reward_function.name=compute_score
    """
    result = _default_reward.score(solution_str=solution_str, ground_truth=ground_truth)
    return result.score
