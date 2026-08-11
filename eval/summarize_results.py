# Copyright (c) 2026
#
# 汇总各阶段（Base/SFT/GRPO/PPO/DPO）的 pass@k 评估结果 json，
# 自动回写 docs/03_results.md 的"方法对比总表"和 docs/01_motivation_and_design.md 的"5.6 方法对比总表"。
#
# 用法：
#   python summarize_results.py
#
# 设计说明：
#   - 本脚本只负责"读取 eval/results/*.json 中的 metrics 并更新两个 md 文件里的表格行"，
#     不重新生成整份文档，避免破坏人工撰写的分析文字。
#   - 匹配逻辑：在 md 文件中找到表格的表头行，之后按"方法名"关键字匹配对应行，
#     替换该行里的数值列，其余文字（备注等）保持不变。

from __future__ import annotations

import json
import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "eval", "results")
EXPERIMENT_LOG_PATH = os.path.join(REPO_ROOT, "docs", "03_results.md")
LIUCHENG_PATH = os.path.join(REPO_ROOT, "docs", "01_motivation_and_design.md")

# method_key -> (结果 json 文件名, 实验结果.md 中的行首关键字)
METHOD_RESULT_FILES = {
    "SFT only": "sft_coldstart_qwen3_0.6b.json",
    "SFT+GRPO": "grpo_qwen3_0.6b.json",
    "SFT+PPO": "ppo_qwen3_0.6b.json",
    "SFT+DPO": "dpo_qwen3_0.6b.json",
}


def load_metrics(json_path: str):
    if not os.path.exists(json_path):
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = {}
    for data_name, payload in data.items():
        result[data_name] = payload.get("metrics", {})
    return result


def fmt_pct(x) -> str:
    if x is None:
        return "?"
    return f"{x * 100:.1f}%"


def build_row_values(metrics: dict):
    gsm8k = metrics.get("gsm8k", {}) if metrics else {}
    math = metrics.get("math", {}) if metrics else {}
    return {
        "gsm8k_pass1": fmt_pct(gsm8k.get("pass@1")),
        "gsm8k_pass8": fmt_pct(gsm8k.get("pass@8")),
        "math_pass1": fmt_pct(math.get("pass@1")),
        "math_pass8": fmt_pct(math.get("pass@8")),
        "gsm8k_format": fmt_pct(gsm8k.get("strict_format_rate")),
        "math_format": fmt_pct(math.get("strict_format_rate")),
    }


def update_experiment_log_table(content: str, method: str, values: dict) -> str:
    """更新 实验结果.md 里"方法对比总表"中对应方法那一行的四个百分比列。"""
    pattern = re.compile(
        r"^(\|\s*" + re.escape(method) + r"\s*\|\s*Qwen3-0\.6B-Base\s*\|)"
        r"\s*[^|]*\|\s*[^|]*\|\s*[^|]*\|\s*[^|]*\|(\s*[^|]*\|\s*[^|]*\|)$",
        re.MULTILINE,
    )

    def _replace(m: re.Match) -> str:
        prefix = m.group(1)
        suffix = m.group(2)
        return (
            f"{prefix} {values['gsm8k_pass1']} | {values['gsm8k_pass8']} | "
            f"{values['math_pass1']} | {values['math_pass8']} |{suffix}"
        )

    new_content, n = pattern.subn(_replace, content)
    if n == 0:
        print(f"[WARN] 未在 实验结果.md 中找到方法行：{method}")
    return new_content


def update_liucheng_table(content: str, method: str, values: dict) -> str:
    """更新 流程.md 5.6 节表格中对应方法那一行的 GSM8K/MATH pass@1 列。"""
    pattern = re.compile(
        r"^(\|\s*" + re.escape(method) + r"\s*\|[^|]*\|[^|]*\|[^|]*\|)"
        r"\s*[^|]*\|\s*[^|]*\|(\s*[^|]*\|\s*[^|]*\|)$",
        re.MULTILINE,
    )

    def _replace(m: re.Match) -> str:
        prefix = m.group(1)
        suffix = m.group(2)
        return f"{prefix} {values['gsm8k_pass1']} | {values['math_pass1']} |{suffix}"

    new_content, n = pattern.subn(_replace, content)
    if n == 0:
        print(f"[WARN] 未在 流程.md 中找到方法行：{method}")
    return new_content


def main():
    if os.path.exists(EXPERIMENT_LOG_PATH):
        with open(EXPERIMENT_LOG_PATH, "r", encoding="utf-8") as f:
            exp_content = f.read()
    else:
        exp_content = ""

    if os.path.exists(LIUCHENG_PATH):
        with open(LIUCHENG_PATH, "r", encoding="utf-8") as f:
            liucheng_content = f.read()
    else:
        liucheng_content = ""

    summary_lines = []
    for method, filename in METHOD_RESULT_FILES.items():
        json_path = os.path.join(RESULTS_DIR, filename)
        metrics = load_metrics(json_path)
        if metrics is None:
            summary_lines.append(f"[SKIP] {method}: 找不到结果文件 {json_path}，跳过")
            continue

        values = build_row_values(metrics)
        exp_content = update_experiment_log_table(exp_content, method, values)
        liucheng_content = update_liucheng_table(liucheng_content, method, values)
        summary_lines.append(
            f"[OK] {method}: GSM8K pass@1={values['gsm8k_pass1']} pass@8={values['gsm8k_pass8']} "
            f"(format={values['gsm8k_format']}) | "
            f"MATH pass@1={values['math_pass1']} pass@8={values['math_pass8']} (format={values['math_format']})"
        )

    with open(EXPERIMENT_LOG_PATH, "w", encoding="utf-8") as f:
        f.write(exp_content)
    with open(LIUCHENG_PATH, "w", encoding="utf-8") as f:
        f.write(liucheng_content)

    print("\n".join(summary_lines))
    print("\n已更新 docs/实验结果.md 和 docs/流程.md 的方法对比表。")


if __name__ == "__main__":
    main()
