# Copyright (c) 2026
#
# 从 veRL 训练日志（logs/pipeline/*.log、logs/ablation/*.log）中解析逐 step 指标，
# 绘制训练过程曲线图，补齐 docs/03_results.md 里"只有关键 step 快照表格、没有完整曲线"的短板。
#
# veRL 训练日志每个 step 打印一行形如：
#   (TaskRunner pid=xxx) step:5 - response_length/mean:114.098 - critic/vf_explained_var:-0.771 - ...
# 本脚本用正则把每行的 `key:value` 对提取出来，按 step 聚合成时间序列后画图。
#
# 设计原则：
#   - 只读取已有 log 文件，不重新训练，秒级出图
#   - 某个日志文件缺失/无匹配行时自动跳过对应曲线，不中断整体流程
#   - 图表文字统一使用英文，避免无中文字体环境下的乱码
#
# 用法：
#   python plot_training_curves.py                  # 生成全部图表到 ../docs/images/
#   python plot_training_curves.py --output_dir xxx
#
# 产出图表清单：
#   1. grpo_vs_ppo_training_curves.png   GRPO(主实验) vs PPO(v3修复) 训练过程对比：
#                                        val reward@1 / response_length/mean 双面板
#   2. ppo_collapse_vs_fixed.png         PPO v1(崩溃) vs v3(修复) 对比：
#                                        response_length/mean / val reward@1 双面板，
#                                        这是全项目最有故事性的一张图（崩溃到自愈）
#   3. grpo_group_size_training_curves.png  GRPO group size (n=4/8/16) 训练曲线对比
#   4. dpo_training_curve.png            DPO loss / rewards accuracy 训练曲线（数据点较少）

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(REPO_ROOT, "logs")
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

# 匹配日志行里所有 "key:value" 形式的 metric（key 允许字母/数字/下划线/斜杠/@/连字符，
# 注意 "val-core/..." 这个 key 本身带连字符，必须包含 "-"，否则会被错误截断成
# "core/..." 丢失 "val-" 前缀；value 允许负号、小数点的数字）。
METRIC_PATTERN = re.compile(r"([\w/@-]+):(-?[0-9]+\.?[0-9]*)")
STEP_PATTERN = re.compile(r"\bstep:(\d+)\b")


def parse_step_log(path: str) -> Optional[Dict[str, List[float]]]:
    """解析 veRL 训练日志，返回 {metric_name: [values in step order]}，附带 'step' 键。

    只保留同时包含 `step:` 前缀的行（每个训练/验证 step 打印一次），
    用 dict 去重同一 step 内的重复 key（正常情况下每个 step 只打印一次）。
    """
    if not os.path.exists(path):
        print(f"[SKIP] 找不到日志文件：{path}")
        return None

    series: Dict[str, List[float]] = {}
    steps: List[int] = []

    with open(path, "r", errors="ignore") as f:
        for line in f:
            step_match = STEP_PATTERN.search(line)
            if step_match is None:
                continue
            step = int(step_match.group(1))
            metrics = dict(METRIC_PATTERN.findall(line))
            if not metrics:
                continue
            steps.append(step)
            for k, v in metrics.items():
                series.setdefault(k, []).append(float(v))
            # step 本身也被 METRIC_PATTERN 匹配进 metrics['step']，统一以 steps 列表为准
            series.setdefault("step", []).append(step)

    if not steps:
        print(f"[SKIP] {path} 未解析到任何 step 指标行")
        return None
    return series


def _get_series(series: Dict[str, List[float]], key: str):
    """按 step 顺序返回 (steps, values)，过滤掉该 key 未出现的 step（如 val 指标只在部分 step 打印）。"""
    if key not in series:
        return [], []
    # val-core 系列指标不是每个 step 都有，行数会少于 steps 总数，
    # 这里简单按其自身出现顺序配对 step（假设日志中 val 指标与其所在行的 step 一一对应）。
    return None, series[key]


def parse_step_log_with_alignment(path: str) -> Optional[Dict[str, List[tuple]]]:
    """解析训练日志，返回 {metric_name: [(step, value), ...]}，每个指标各自对齐自己出现的 step。"""
    if not os.path.exists(path):
        print(f"[SKIP] 找不到日志文件：{path}")
        return None

    series: Dict[str, List[tuple]] = {}
    with open(path, "r", errors="ignore") as f:
        for line in f:
            step_match = STEP_PATTERN.search(line)
            if step_match is None:
                continue
            step = int(step_match.group(1))
            metrics = dict(METRIC_PATTERN.findall(line))
            for k, v in metrics.items():
                if k == "step":
                    continue
                series.setdefault(k, []).append((step, float(v)))

    if not series:
        print(f"[SKIP] {path} 未解析到任何 step 指标")
        return None
    return series


def _unzip(pairs):
    if not pairs:
        return [], []
    xs, ys = zip(*pairs)
    return list(xs), list(ys)


# --------------------------------------------------------------------------
# 图1：GRPO vs PPO 训练过程对比
# --------------------------------------------------------------------------


def plot_grpo_vs_ppo(output_dir: str):
    grpo_log = os.path.join(LOGS_DIR, "pipeline", "stage2_train_grpo.log")
    ppo_log = os.path.join(LOGS_DIR, "pipeline", "stage4_train_ppo_retry_v3.log")

    grpo_series = parse_step_log_with_alignment(grpo_log)
    ppo_series = parse_step_log_with_alignment(ppo_log)
    if grpo_series is None and ppo_series is None:
        print("[SKIP] plot_grpo_vs_ppo: 无可用数据")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    if grpo_series and "val-core/gsm8k/reward/mean@1" in grpo_series:
        xs, ys = _unzip(sorted(grpo_series["val-core/gsm8k/reward/mean@1"]))
        ax.plot(xs, ys, marker="o", label="GRPO", color="#8172b2", linewidth=2)
    if ppo_series and "val-core/gsm8k/reward/mean@1" in ppo_series:
        xs, ys = _unzip(sorted(ppo_series["val-core/gsm8k/reward/mean@1"]))
        ax.plot(xs, ys, marker="s", label="PPO (v3 fixed)", color="#c44e52", linewidth=2)
    ax.set_xlabel("training step")
    ax.set_ylabel("val reward@1 (GSM8K)")
    ax.set_title("Validation Reward: GRPO vs PPO")
    ax.legend()

    ax = axes[1]
    if grpo_series and "response_length/mean" in grpo_series:
        xs, ys = _unzip(sorted(grpo_series["response_length/mean"]))
        ax.plot(xs, ys, label="GRPO", color="#8172b2", linewidth=1.5)
    if ppo_series and "response_length/mean" in ppo_series:
        xs, ys = _unzip(sorted(ppo_series["response_length/mean"]))
        ax.plot(xs, ys, label="PPO (v3 fixed)", color="#c44e52", linewidth=1.5)
    ax.set_xlabel("training step")
    ax.set_ylabel("response_length / mean")
    ax.set_title("Response Length: GRPO (stable) vs PPO (spike & self-correct)")
    ax.legend()

    fig.suptitle("GRPO vs PPO Training Dynamics (Qwen3-0.6B, 4 GPU, 116 steps)")
    fig.tight_layout()

    out_path = os.path.join(output_dir, "grpo_vs_ppo_training_curves.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[OK] 已生成 {out_path}")


# --------------------------------------------------------------------------
# 图2：PPO 崩溃(v1) vs 修复(v3) 对比 —— 全项目最有故事性的图
# --------------------------------------------------------------------------


def plot_ppo_collapse_vs_fixed(output_dir: str):
    v1_log = os.path.join(LOGS_DIR, "pipeline", "stage4_train_ppo.log")
    v3_log = os.path.join(LOGS_DIR, "pipeline", "stage4_train_ppo_retry_v3.log")

    v1_series = parse_step_log_with_alignment(v1_log)
    v3_series = parse_step_log_with_alignment(v3_log)
    if v1_series is None and v3_series is None:
        print("[SKIP] plot_ppo_collapse_vs_fixed: 无可用数据")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    if v1_series and "response_length/mean" in v1_series:
        xs, ys = _unzip(sorted(v1_series["response_length/mean"]))
        ax.plot(xs, ys, label="v1 (crashed, KL=0.001, critic_warmup=0)", color="#c44e52", linewidth=2)
    if v3_series and "response_length/mean" in v3_series:
        xs, ys = _unzip(sorted(v3_series["response_length/mean"]))
        ax.plot(xs, ys, label="v3 (fixed, KL=0.01, critic_warmup=20)", color="#55a868", linewidth=2)
    ax.axhline(1024, color="gray", linestyle="--", linewidth=1, label="max_response_length=1024")
    ax.set_xlabel("training step")
    ax.set_ylabel("response_length / mean")
    ax.set_title("Response Length: Collapse vs Recovery")
    ax.legend(fontsize=8)

    ax = axes[1]
    if v1_series and "val-core/gsm8k/reward/mean@1" in v1_series:
        xs, ys = _unzip(sorted(v1_series["val-core/gsm8k/reward/mean@1"]))
        ax.plot(xs, ys, marker="o", label="v1 (crashed)", color="#c44e52", linewidth=2)
    if v3_series and "val-core/gsm8k/reward/mean@1" in v3_series:
        xs, ys = _unzip(sorted(v3_series["val-core/gsm8k/reward/mean@1"]))
        ax.plot(xs, ys, marker="s", label="v3 (fixed)", color="#55a868", linewidth=2)
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("training step")
    ax.set_ylabel("val reward@1 (GSM8K)")
    ax.set_title("Validation Reward: Collapse vs Recovery")
    ax.legend(fontsize=8)

    fig.suptitle("PPO Training Collapse (v1) vs Fixed (v3): critic_warmup + truncation penalty")
    fig.tight_layout()

    out_path = os.path.join(output_dir, "ppo_collapse_vs_fixed.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[OK] 已生成 {out_path}")


# --------------------------------------------------------------------------
# 图3：GRPO group size 消融训练曲线
# --------------------------------------------------------------------------


def plot_grpo_group_size_curves(output_dir: str):
    variants = [
        ("n=4", os.path.join(LOGS_DIR, "ablation", "grpo_n4.log"), "#4c72b0"),
        ("n=8 (main)", os.path.join(LOGS_DIR, "pipeline", "stage2_train_grpo.log"), "#8172b2"),
        ("n=16", os.path.join(LOGS_DIR, "ablation", "grpo_n16_resume.log"), "#dd8452"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    plotted = False
    for label, log_path, color in variants:
        series = parse_step_log_with_alignment(log_path)
        if series is None:
            continue
        if "val-core/gsm8k/reward/mean@1" in series:
            xs, ys = _unzip(sorted(series["val-core/gsm8k/reward/mean@1"]))
            axes[0].plot(xs, ys, marker="o", label=label, color=color, linewidth=2, markersize=4)
            plotted = True
        if "response_length/mean" in series:
            xs, ys = _unzip(sorted(series["response_length/mean"]))
            axes[1].plot(xs, ys, label=label, color=color, linewidth=1.5)

    if not plotted:
        print("[SKIP] plot_grpo_group_size_curves: 无可用数据")
        plt.close(fig)
        return

    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("val reward@1 (GSM8K)")
    axes[0].set_title("Validation Reward by Group Size")
    axes[0].legend()

    axes[1].set_xlabel("training step")
    axes[1].set_ylabel("response_length / mean")
    axes[1].set_title("Response Length by Group Size")
    axes[1].legend()

    fig.suptitle("GRPO Group Size Ablation: Training Dynamics (n=4/8/16)")
    fig.tight_layout()

    out_path = os.path.join(output_dir, "grpo_group_size_training_curves.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[OK] 已生成 {out_path}")


# --------------------------------------------------------------------------
# 图4：DPO 训练曲线
# --------------------------------------------------------------------------

DPO_METRIC_PATTERN = re.compile(r"'(loss|rewards/accuracies|rewards/margins|epoch)':\s*(-?[0-9]+\.?[0-9]*)")


def parse_dpo_log(path: str) -> Optional[Dict[str, List[float]]]:
    if not os.path.exists(path):
        print(f"[SKIP] 找不到日志文件：{path}")
        return None
    series: Dict[str, List[float]] = {}
    cur_epoch = None
    with open(path, "r", errors="ignore") as f:
        for line in f:
            matches = DPO_METRIC_PATTERN.findall(line)
            if not matches:
                continue
            row = dict(matches)
            if "epoch" in row:
                cur_epoch = float(row["epoch"])
            for key in ("loss", "rewards/accuracies", "rewards/margins"):
                if key in row and cur_epoch is not None:
                    series.setdefault(key, []).append((cur_epoch, float(row[key])))
    return series or None


def plot_dpo_training_curve(output_dir: str):
    dpo_log = os.path.join(LOGS_DIR, "pipeline", "stage7_train_dpo_retry2.log")
    series = parse_dpo_log(dpo_log)
    if series is None:
        print("[SKIP] plot_dpo_training_curve: 无可用数据")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    if "loss" in series:
        xs, ys = _unzip(series["loss"])
        ax.plot(xs, ys, marker="o", color="#c44e52", linewidth=2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("train loss")
    ax.set_title("DPO Training Loss")

    ax = axes[1]
    if "rewards/accuracies" in series:
        xs, ys = _unzip(series["rewards/accuracies"])
        ax.plot(xs, [y * 100 for y in ys], marker="o", color="#55a868", linewidth=2, label="rewards/accuracies")
    ax.axhline(50, color="gray", linestyle="--", linewidth=1, label="random guess (50%)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("chosen > rejected accuracy (%)")
    ax.set_title("DPO Preference Accuracy")
    ax.legend(fontsize=8)

    fig.suptitle("DPO Training Curve (SFT-init, 3171 preference pairs, 1 epoch)")
    fig.tight_layout()

    out_path = os.path.join(output_dir, "dpo_training_curve.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[OK] 已生成 {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    plot_grpo_vs_ppo(args.output_dir)
    plot_ppo_collapse_vs_fixed(args.output_dir)
    plot_grpo_group_size_curves(args.output_dir)
    plot_dpo_training_curve(args.output_dir)

    print(f"\n全部图表已生成到 {args.output_dir}")


if __name__ == "__main__":
    main()
