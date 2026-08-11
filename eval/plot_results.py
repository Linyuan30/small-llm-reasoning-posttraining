# Copyright (c) 2026
#
# 从 eval/results/*.json（pass@k 评估结果，含逐样本生成明细）批量生成可视化图表，
# 补齐 docs/03_results.md、docs/04_repo_overview.md 里"只有表格、没有图"的短板。
#
# 设计原则：
#   - 只读取已有的 results/*.json，不重新触发任何推理/评估，运行成本极低（纯 CPU，秒级完成）
#   - 每张图对应文档里已经写好的一个结论/表格，图和文字结论一一呼应，而不是另起炉灶
#   - 某个方法的结果文件缺失时自动跳过该方法，不中断整体绘图流程
#   - 图表文字统一使用英文（标题/坐标轴/图例），避免服务器无中文字体导致的乱码方框问题；
#     中文解读放在 markdown 正文里，图和文字配合阅读
#
# 用法：
#   python plot_results.py                  # 生成全部图表到 ../docs/images/
#   python plot_results.py --output_dir xxx # 自定义输出目录
#
# 产出图表清单（与 docs/03_results.md「方法对比总表」「消融实验」章节一一对应）：
#   1. method_comparison_pass1.png       方法对比总表 -> pass@1 分组柱状图（GSM8K/MATH）
#   2. pass_at_k_curves.png              各方法 pass@1/4/8 折线图（含 Base，展示"蒙对->稳定做对"）
#   3. grpo_group_size_ablation.png      GRPO group size (n=4/8/16) 消融柱状图
#   4. ppo_kl_coef_ablation.png          PPO KL 系数 (0.005/0.01x2/0.02) 消融柱状图
#   5. sft_epoch_ablation.png            SFT 冷启动 epoch 数 (3/7/15) 消融折线图 + 模型规模对比
#   6. response_length_distribution.png  不同方法响应长度（token近似值）分布直方图

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "eval", "results")
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "docs", "images")

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.axisbelow": True,
    }
)

COLOR_PALETTE = {
    "Base": "#9e9e9e",
    "SFT": "#4c72b0",
    "SFT+DPO": "#55a868",
    "SFT+PPO": "#c44e52",
    "SFT+GRPO": "#8172b2",
}


def load_metrics(filename: str) -> Optional[dict]:
    """读取 results/*.json，返回 {dataset: metrics_dict}；文件不存在时返回 None。"""
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        print(f"[SKIP] 找不到结果文件：{path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {name: payload.get("metrics", {}) for name, payload in data.items()}


def load_response_lengths(filename: str, tokenizer=None, max_samples: int = 200) -> Optional[dict]:
    """从 per_sample[].generations[].text 估算响应长度（近似 token 数），用于长度分布图。

    为控制内存/耗时，每个数据集最多抽取 max_samples 条 prompt 的全部采样。
    没有传入 tokenizer 时，用近似公式 len(text) / 3.2（英文+LaTeX混排下的经验值）估算 token 数，
    仅用于粗粒度对比不同方法之间的相对长度差异，不追求绝对精确。
    """
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lengths_by_dataset = {}
    for name, payload in data.items():
        per_sample = payload.get("per_sample", [])[:max_samples]
        lengths = []
        for record in per_sample:
            for gen in record.get("generations", []):
                text = gen.get("text", "")
                if tokenizer is not None:
                    lengths.append(len(tokenizer.encode(text)))
                else:
                    lengths.append(len(text) / 3.2)
        if lengths:
            lengths_by_dataset[name] = np.array(lengths)
    return lengths_by_dataset or None


# --------------------------------------------------------------------------
# 图1：方法对比总表 -> pass@1 分组柱状图
# --------------------------------------------------------------------------


def plot_method_comparison(output_dir: str):
    methods = [
        ("Base", "baseline_qwen3_0.6b_base.json"),
        ("SFT", "sft_coldstart_qwen3_0.6b.json"),
        ("SFT+DPO", "dpo_qwen3_0.6b.json"),
        ("SFT+PPO", "ppo_qwen3_0.6b.json"),
        ("SFT+GRPO", "grpo_qwen3_0.6b.json"),
    ]

    labels, gsm8k_p1, math_p1 = [], [], []
    for label, filename in methods:
        metrics = load_metrics(filename)
        if metrics is None:
            continue
        labels.append(label)
        gsm8k_p1.append(metrics.get("gsm8k", {}).get("pass@1", 0) * 100)
        math_p1.append(metrics.get("math", {}).get("pass@1", 0) * 100)

    if not labels:
        print("[SKIP] plot_method_comparison: 无可用数据")
        return

    x = np.arange(len(labels))
    width = 0.35
    colors = [COLOR_PALETTE.get(l, "#333333") for l in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, gsm8k_p1, width, label="GSM8K pass@1", color=colors, alpha=0.95)
    bars2 = ax.bar(x + width / 2, math_p1, width, label="MATH pass@1", color=colors, alpha=0.55, hatch="//")

    for bars in (bars1, bars2):
        for b in bars:
            h = b.get_height()
            ax.annotate(
                f"{h:.1f}",
                xy=(b.get_x() + b.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_ylabel("pass@1 (%)")
    ax.set_title("Method Comparison: GSM8K / MATH pass@1 (Qwen3-0.6B-Base)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(max(gsm8k_p1, default=0), max(math_p1, default=0)) * 1.2 + 5)
    ax.legend(loc="upper left")
    fig.tight_layout()

    out_path = os.path.join(output_dir, "method_comparison_pass1.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[OK] 已生成 {out_path}")


# --------------------------------------------------------------------------
# 图2：各方法 pass@1/4/8 折线图
# --------------------------------------------------------------------------


def plot_pass_at_k_curves(output_dir: str):
    methods = [
        ("Base", "baseline_qwen3_0.6b_base.json"),
        ("SFT", "sft_coldstart_qwen3_0.6b.json"),
        ("SFT+DPO", "dpo_qwen3_0.6b.json"),
        ("SFT+PPO", "ppo_qwen3_0.6b.json"),
        ("SFT+GRPO", "grpo_qwen3_0.6b.json"),
    ]
    k_values = [1, 4, 8]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for dataset_idx, dataset in enumerate(["gsm8k", "math"]):
        ax = axes[dataset_idx]
        plotted = False
        for label, filename in methods:
            metrics = load_metrics(filename)
            if metrics is None or dataset not in metrics:
                continue
            ys = [metrics[dataset].get(f"pass@{k}", None) for k in k_values]
            if any(y is None for y in ys):
                continue
            ys = [y * 100 for y in ys]
            ax.plot(
                k_values,
                ys,
                marker="o",
                label=label,
                color=COLOR_PALETTE.get(label, None),
                linewidth=2,
            )
            plotted = True
        if not plotted:
            continue
        ax.set_xticks(k_values)
        ax.set_xlabel("k")
        ax.set_title(dataset.upper())
        if dataset_idx == 0:
            ax.set_ylabel("pass@k (%)")

    axes[-1].legend(loc="lower right", fontsize=9)
    fig.suptitle("pass@k Scaling Across Methods (k = 1, 4, 8)")
    fig.tight_layout()

    out_path = os.path.join(output_dir, "pass_at_k_curves.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[OK] 已生成 {out_path}")


# --------------------------------------------------------------------------
# 图3：GRPO group size 消融
# --------------------------------------------------------------------------


def plot_grpo_group_size_ablation(output_dir: str):
    variants = [
        ("n=4", "grpo_qwen3_0.6b_n4.json"),
        ("n=8 (main)", "grpo_qwen3_0.6b.json"),
        ("n=16", "grpo_qwen3_0.6b_n16.json"),
    ]

    labels, gsm8k_p1, math_p1 = [], [], []
    for label, filename in variants:
        metrics = load_metrics(filename)
        if metrics is None:
            continue
        labels.append(label)
        gsm8k_p1.append(metrics.get("gsm8k", {}).get("pass@1", 0) * 100)
        math_p1.append(metrics.get("math", {}).get("pass@1", 0) * 100)

    if not labels:
        print("[SKIP] plot_grpo_group_size_ablation: 无可用数据")
        return

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, gsm8k_p1, width, label="GSM8K pass@1", color="#4c72b0")
    ax.bar(x + width / 2, math_p1, width, label="MATH pass@1", color="#dd8452")

    for i, (g, m) in enumerate(zip(gsm8k_p1, math_p1)):
        ax.annotate(f"{g:.1f}", (i - width / 2, g), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
        ax.annotate(f"{m:.1f}", (i + width / 2, m), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("pass@1 (%)")
    ax.set_ylim(0, max(max(gsm8k_p1, default=0), max(math_p1, default=0)) * 1.2 + 5)
    ax.set_title("GRPO Group Size Ablation: GSM8K vs MATH pass@1")
    ax.legend(loc="upper center", ncol=2)
    fig.tight_layout()

    out_path = os.path.join(output_dir, "grpo_group_size_ablation.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[OK] 已生成 {out_path}")


# --------------------------------------------------------------------------
# 图4：PPO KL 系数消融
# --------------------------------------------------------------------------


def plot_ppo_kl_ablation(output_dir: str):
    variants = [
        ("KL=0.005\n(2GPU)", "ppo_qwen3_0.6b_kl0.005.json"),
        ("KL=0.01\n(4GPU, main)", "ppo_qwen3_0.6b.json"),
        ("KL=0.01\n(2GPU, controlled)", "ppo_qwen3_0.6b_kl0.01_2gpu.json"),
        ("KL=0.02\n(2GPU)", "ppo_qwen3_0.6b_kl0.02.json"),
    ]

    labels, gsm8k_p1, math_p1, crashed = [], [], [], []
    for label, filename in variants:
        metrics = load_metrics(filename)
        if metrics is None:
            continue
        labels.append(label)
        g = metrics.get("gsm8k", {}).get("pass@1", 0) * 100
        m = metrics.get("math", {}).get("pass@1", 0) * 100
        gsm8k_p1.append(g)
        math_p1.append(m)
        crashed.append(g < 10)  # 崩溃组 pass@1 极低，用不同颜色标出

    if not labels:
        print("[SKIP] plot_ppo_kl_ablation: 无可用数据")
        return

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bar_colors_g = ["#c44e52" if c else "#4c72b0" for c in crashed]
    bar_colors_m = ["#c44e52" if c else "#dd8452" for c in crashed]
    ax.bar(x - width / 2, gsm8k_p1, width, label="GSM8K pass@1", color=bar_colors_g)
    ax.bar(x + width / 2, math_p1, width, label="MATH pass@1", color=bar_colors_m, alpha=0.75, hatch="//")

    for i, (g, m) in enumerate(zip(gsm8k_p1, math_p1)):
        ax.annotate(f"{g:.1f}", (i - width / 2, g), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
        ax.annotate(f"{m:.1f}", (i + width / 2, m), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("pass@1 (%)")
    ax.set_title("PPO KL Coefficient Ablation (red = training collapsed)")
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(output_dir, "ppo_kl_coef_ablation.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[OK] 已生成 {out_path}")


# --------------------------------------------------------------------------
# 图5：SFT epoch 数消融 + 模型规模对比
# --------------------------------------------------------------------------


def plot_sft_ablation(output_dir: str):
    epoch_variants = [
        (3, "sft_coldstart_qwen3_0.6b.json"),
        (7, "sft_coldstart_qwen3_0.6b_v3.json"),
        (15, "sft_coldstart_qwen3_0.6b_v2.json"),
    ]

    epochs, gsm8k_p1, math_p1 = [], [], []
    for epoch, filename in epoch_variants:
        metrics = load_metrics(filename)
        if metrics is None:
            continue
        epochs.append(epoch)
        gsm8k_p1.append(metrics.get("gsm8k", {}).get("pass@1", 0) * 100)
        math_p1.append(metrics.get("math", {}).get("pass@1", 0) * 100)

    scale_variants = [
        ("0.6B\n(7ep)", "sft_coldstart_qwen3_0.6b_v3.json"),
        ("1.7B\n(7ep)", "sft_coldstart_qwen3_1.7b_v3.json"),
    ]
    scale_labels, scale_gsm8k, scale_math = [], [], []
    for label, filename in scale_variants:
        metrics = load_metrics(filename)
        if metrics is None:
            continue
        scale_labels.append(label)
        scale_gsm8k.append(metrics.get("gsm8k", {}).get("pass@1", 0) * 100)
        scale_math.append(metrics.get("math", {}).get("pass@1", 0) * 100)

    if not epochs and not scale_labels:
        print("[SKIP] plot_sft_ablation: 无可用数据")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    if epochs:
        ax.plot(epochs, gsm8k_p1, marker="o", label="GSM8K pass@1", color="#4c72b0", linewidth=2)
        ax.plot(epochs, math_p1, marker="s", label="MATH pass@1", color="#dd8452", linewidth=2)
        for e, g, m in zip(epochs, gsm8k_p1, math_p1):
            ax.annotate(f"{g:.1f}", (e, g), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=9)
            ax.annotate(f"{m:.1f}", (e, m), xytext=(0, -12), textcoords="offset points", ha="center", fontsize=9)
        ax.set_xticks(epochs)
        ax.set_xlabel("SFT total_epochs")
        ax.set_ylabel("pass@1 (%)")
        ax.set_title("SFT Cold-start: Epoch Count Ablation (Qwen3-0.6B)")
        ax.legend()

    ax = axes[1]
    if scale_labels:
        x = np.arange(len(scale_labels))
        width = 0.35
        ax.bar(x - width / 2, scale_gsm8k, width, label="GSM8K pass@1", color="#4c72b0")
        ax.bar(x + width / 2, scale_math, width, label="MATH pass@1", color="#dd8452")
        for i, (g, m) in enumerate(zip(scale_gsm8k, scale_math)):
            ax.annotate(f"{g:.1f}", (i - width / 2, g), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
            ax.annotate(f"{m:.1f}", (i + width / 2, m), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(scale_labels)
        ax.set_ylabel("pass@1 (%)")
        ax.set_title("Model Scale Ablation: 0.6B vs 1.7B (same data/epoch)")
        ax.legend()

    fig.tight_layout()
    out_path = os.path.join(output_dir, "sft_epoch_ablation.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[OK] 已生成 {out_path}")


# --------------------------------------------------------------------------
# 图6：响应长度分布
# --------------------------------------------------------------------------


def plot_response_length_distribution(output_dir: str):
    methods = [
        ("SFT", "sft_coldstart_qwen3_0.6b.json"),
        ("SFT+DPO", "dpo_qwen3_0.6b.json"),
        ("SFT+PPO", "ppo_qwen3_0.6b.json"),
        ("SFT+GRPO", "grpo_qwen3_0.6b.json"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for dataset_idx, dataset in enumerate(["gsm8k", "math"]):
        ax = axes[dataset_idx]
        plotted = False
        for label, filename in methods:
            lengths_by_dataset = load_response_lengths(filename)
            if lengths_by_dataset is None or dataset not in lengths_by_dataset:
                continue
            lengths = lengths_by_dataset[dataset]
            ax.hist(
                lengths,
                bins=30,
                histtype="step",
                linewidth=2,
                label=f"{label} (median={np.median(lengths):.0f})",
                color=COLOR_PALETTE.get(label, None),
                density=True,
            )
            plotted = True
        if not plotted:
            continue
        ax.set_xlabel("approx. response length (tokens)")
        ax.set_title(dataset.upper())
        if dataset_idx == 0:
            ax.set_ylabel("density")
        ax.legend(fontsize=8)

    fig.suptitle("Response Length Distribution by Method (approx. token count)")
    fig.tight_layout()

    out_path = os.path.join(output_dir, "response_length_distribution.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[OK] 已生成 {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    plot_method_comparison(args.output_dir)
    plot_pass_at_k_curves(args.output_dir)
    plot_grpo_group_size_ablation(args.output_dir)
    plot_ppo_kl_ablation(args.output_dir)
    plot_sft_ablation(args.output_dir)
    plot_response_length_distribution(args.output_dir)

    print(f"\n全部图表已生成到 {args.output_dir}")


if __name__ == "__main__":
    main()
