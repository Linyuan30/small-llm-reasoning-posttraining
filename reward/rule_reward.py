# Copyright (c) 2026
#
# 基于规则的可验证 reward 函数（数学推理任务）
#
# 设计目标（对应 docs/流程.md 4.3节 与 docs/方案计划.md P0 任务）：
#   1. 正确性 reward：从模型输出的 <answer>...</answer> 中提取最终答案，
#      与标准答案做规范化匹配（数值、分数、根号等 LaTeX 表达式的等价判断）
#   2. 格式 reward：是否严格符合 `<think>...</think><answer>...</answer>` 格式
#   3. 两者加权组合成最终 reward，权重可配置，便于做消融实验
#
# 本文件同时提供两类接口：
#   - compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs)
#     签名与 verl.utils.reward_score.default_compute_score 对齐，
#     可以直接通过 veRL 的 `custom_reward_function.path` 配置接入 GRPO/PPO 训练。
#   - RuleReward 类：更细粒度的答案正确性 / 格式合规性拆分接口，
#     用于 SFT 格式合规率评估、reward 消融实验、单元测试等场景。

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# --------------------------------------------------------------------------
# 1. 输出格式定义
# --------------------------------------------------------------------------

THINK_ANSWER_PATTERN = re.compile(
    r"^\s*<think>(?P<think>(?:(?!</think>).)*?)</think>\s*"
    r"<answer>(?P<answer>(?:(?!</answer>).)*?)</answer>\s*$",
    re.DOTALL,
)

# 宽松模式：只要求同时出现一对 <think>...</think> 和 <answer>...</answer>，
# 不强制要求二者之间没有多余文本、也不强制在字符串首尾。
# 用于在严格格式不满足时，仍然尝试抽取答案计算“是否算对了但格式不对”，
# 便于做 reward hacking / 格式合规率分析。
LOOSE_ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
LOOSE_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)


@dataclass
class FormatCheckResult:
    strict_ok: bool  # 是否严格符合 <think>...</think><answer>...</answer> 完整格式
    has_think: bool  # 是否存在 <think>...</think>
    has_answer: bool  # 是否存在 <answer>...</answer>
    answer_text: Optional[str]  # 抽取出的 answer 内容（宽松抽取，取最后一个 <answer> 块）


def check_format(solution_str: str) -> FormatCheckResult:
    """检查模型输出是否符合 <think>/<answer> 格式，并抽取 answer 内容。"""
    strict_match = THINK_ANSWER_PATTERN.match(solution_str.strip())
    if strict_match is not None:
        return FormatCheckResult(
            strict_ok=True,
            has_think=True,
            has_answer=True,
            answer_text=strict_match.group("answer").strip(),
        )

    think_matches = LOOSE_THINK_PATTERN.findall(solution_str)
    answer_matches = LOOSE_ANSWER_PATTERN.findall(solution_str)

    return FormatCheckResult(
        strict_ok=False,
        has_think=len(think_matches) > 0,
        has_answer=len(answer_matches) > 0,
        answer_text=answer_matches[-1].strip() if answer_matches else None,
    )


# --------------------------------------------------------------------------
# 2. 答案规范化与等价判断（数值 / 分数 / 根号等 LaTeX 表达式）
#    复用 MATH 数据集评测的经典规范化规则（Hendrycks MATH / lm-evaluation-harness）
# --------------------------------------------------------------------------


def _remove_boxed(s: str) -> str:
    """去掉 \\boxed{...} 或 \\boxed ... 包裹，取内部内容。"""
    if "\\boxed " in s:
        left = "\\boxed "
        return s[len(left):] if s.startswith(left) else s
    left = "\\boxed{"
    if s.startswith(left) and s.endswith("}"):
        return s[len(left):-1]
    return s


def _last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return None if right_brace_idx is None else string[idx: right_brace_idx + 1]


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if substr and substr[0] == "{":
                new_str += substr
            else:
                if len(substr) < 2:
                    return string
                a, b = substr[0], substr[1]
                if b != "{":
                    post = substr[2:]
                    new_str += "{" + a + "}{" + b + "}" + post
                else:
                    post = substr[2:]
                    new_str += "{" + a + "}" + b + post
    return new_str


def _fix_a_slash_b(string: str) -> str:
    parts = string.split("/")
    if len(parts) != 2:
        return string
    a, b = parts
    try:
        a_int, b_int = int(a), int(b)
        if string == f"{a_int}/{b_int}":
            return f"\\frac{{{a_int}}}{{{b_int}}}"
        return string
    except ValueError:
        return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if not split or split[0] != "{":
            a = split[0] if split else ""
            new_string += "\\sqrt{" + a + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def normalize_answer(raw: str) -> str:
    """把数学答案字符串规范化，用于做等价字符串比较。

    覆盖：LaTeX \\boxed 包裹、单位/多余文字、frac/sqrt 简写、空格、逗号分隔的千分位等。
    """
    if raw is None:
        return ""
    string = raw.strip()

    # 如果整体是 \boxed{...}，先拆箱
    boxed = _last_boxed_only_string(string)
    if boxed is not None:
        string = _remove_boxed(boxed)

    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "").replace("$", "")
    string = string.replace("\\%", "").replace("%", "")
    string = string.replace(",", "")  # 千分位逗号 / MATH answer 中的分隔逗号

    if "\\text{ " in string:
        string = string.split("\\text{ ")[0]

    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if string.startswith("."):
        string = "0" + string

    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)

    if string == "0.5":
        string = "\\frac{1}{2}"

    string = _fix_a_slash_b(string)

    return string.strip()


def _try_numeric_equal(a: str, b: str, atol: float = 1e-4) -> Optional[bool]:
    """尝试把两个字符串都解析为浮点数进行数值比较，失败返回 None。"""
    try:
        fa = float(a)
        fb = float(b)
        return abs(fa - fb) <= atol * max(1.0, abs(fb))
    except (TypeError, ValueError):
        return None


def is_equivalent(pred: str, ground_truth: str) -> bool:
    """判断预测答案与标准答案是否等价（数学意义下）。"""
    if pred is None or ground_truth is None:
        return False

    norm_pred = normalize_answer(pred)
    norm_gt = normalize_answer(ground_truth)

    if norm_pred == norm_gt:
        return True

    # 数值等价兜底（如 "0.5" vs "1/2" 归一化后不同字符串，但数值相等；
    # 或者小数位数、正负号书写差异等）
    numeric_result = _try_numeric_equal(norm_pred.replace("\\frac", "").replace("{", "").replace("}", ""),
                                         norm_gt.replace("\\frac", "").replace("{", "").replace("}", ""))
    if numeric_result is True:
        return True

    return False


# --------------------------------------------------------------------------
# 3. Reward 组合：正确性 + 格式
# --------------------------------------------------------------------------


@dataclass
class RewardResult:
    score: float  # 最终加权 reward
    correct: bool  # 答案是否正确
    format_ok: bool  # 是否严格符合输出格式
    extracted_answer: Optional[str]  # 抽取出的答案文本


class RuleReward:
    """规则 reward：正确性 reward + 格式 reward 加权组合。

    Args:
        correct_score: 答案正确时的基础得分
        wrong_score: 答案错误（但格式正确）时的得分
        format_bonus: 格式合规的额外加分（叠加在正确性得分之上）
        format_penalty: 格式不合规时的惩罚（叠加/扣减）
        no_answer_score: 完全提取不到 answer 时的得分（一般为最低分，防止空输出蒙混）
        truncated_extra_penalty: 在 `no_answer_score` 基础上，对"疑似被截断"
            （提取不到 answer 且 response_length_ratio 超过 length_penalty_start_ratio）
            的样本叠加的额外惩罚系数。默认为 0，即不启用，行为与旧版完全一致
            （不影响已验证稳定的 GRPO 结果）。仅当训练侧（如 PPO）显式传入
            `response_length_ratio` 且该值较高时才会生效，用于修复 PPO 训练中
            "整条轨迹被截断也只有和普通答错一样的惩罚，早期反馈不够陡峭"导致的
            response length 爆炸问题（详见 docs/实验结果.md 实验④ PPO 崩溃分析）。
        length_penalty_start_ratio: 长度惩罚开始生效的比例阈值（相对
            max_response_length），超过该比例后惩罚随长度线性增加，到1.0时达到
            no_answer_score - truncated_extra_penalty 的最低分。
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
        """计算 reward。

        Args:
            solution_str: 模型输出文本
            ground_truth: 标准答案
            response_length_ratio: 可选，当前 response 长度 / max_response_length
                的比例（[0, 1]）。仅当 `truncated_extra_penalty > 0` 且提取不到
                answer 时才会用于计算额外的长度惩罚，默认 None 表示不启用。
        """
        fmt = check_format(solution_str)

        if fmt.answer_text is None:
            score = self.no_answer_score
            if (
                self.truncated_extra_penalty > 0
                and response_length_ratio is not None
                and response_length_ratio > self.length_penalty_start_ratio
            ):
                # 超过阈值后惩罚随长度线性增长，在 length_ratio=1.0（即打满
                # max_response_length，几乎必然是截断）时达到最大额外惩罚，
                # 目的是让"截断"比"生成到一半但没截断的普通错误"惩罚更陡峭，
                # 给 critic/actor 更早、更明确的负反馈，抑制长度失控的正反馈循环。
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


# 默认单例，供 compute_score 使用；如需做权重消融实验，直接实例化 RuleReward 传参即可。
_default_reward = RuleReward()


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    response_length_ratio: Optional[float] = None,
    truncated_extra_penalty: float = 0.0,
    length_penalty_start_ratio: float = 0.9,
    **kwargs,
) -> float:
    """兼容 verl `custom_reward_function` 接口的打分函数。

    在 veRL 的训练配置中通过如下方式接入：
        reward.custom_reward_function.path=reward/rule_reward.py
        reward.custom_reward_function.name=compute_score

    Args:
        response_length_ratio: 可选，当前 response 有效长度 / max_response_length。
            由 verl 的 NaiveRewardManager 自动探测并传入（见该文件的
            inspect.signature 检测逻辑）。
        truncated_extra_penalty / length_penalty_start_ratio: PPO 崩溃修复用的
            截断额外惩罚参数，默认 0（不启用，不影响 GRPO/SFT 评估等既有调用
            路径），由 verl 训练配置的 `custom_reward_function.reward_kwargs`
            传入（比环境变量更可靠，不依赖 Ray 分布式进程的环境继承行为）。
            详见 RuleReward 类文档与 docs/实验结果.md 实验④ PPO 崩溃分析。
    """
    reward = _default_reward
    if truncated_extra_penalty != reward.truncated_extra_penalty or length_penalty_start_ratio != reward.length_penalty_start_ratio:
        # 仅当调用方显式传入非默认值时才临时构造一个新实例，避免每次打分都
        # 重新分配对象；正常情况下（未开启截断惩罚）走的是同一个默认单例。
        reward = RuleReward(
            truncated_extra_penalty=truncated_extra_penalty,
            length_penalty_start_ratio=length_penalty_start_ratio,
        )
    result = reward.score(
        solution_str=solution_str,
        ground_truth=ground_truth,
        response_length_ratio=response_length_ratio,
    )
    return result.score
