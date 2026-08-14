# Copyright (c) 2026
#
# Parse per-step metrics from veRL training logs (logs/pipeline/*.log,
# logs/ablation/*.log) and plot training-dynamics figures.
#
# veRL prints one line per step in the form:
#   (TaskRunner pid=xxx) step:5 - response_length/mean:114.098 - critic/vf_explained_var:-0.771 - ...
# This script extracts every key:value pair with a regex, aggregates them
# into time series indexed by step, and draws the plots.
#
# Design principles:
#   - Reads existing log files only; no re-training required; output is instant.
#   - Missing or empty log files are silently skipped; other plots still run.
#   - All chart text is in English to avoid font issues in headless environments.
#
# Usage:
#   python plot_training_curves.py                  # write all figures to ../docs/images/
#   python plot_training_curves.py --output_dir xxx
#
# Output figures:
#   1. grpo_vs_ppo_training_curves.png     GRPO (main) vs PPO (v3 fixed):
#                                          val reward@1 / response_length/mean dual-panel
#   2. ppo_collapse_vs_fixed.png           PPO v1 (crashed) vs v3 (fixed):
#                                          response_length/mean / val reward@1 dual-panel
#   3. grpo_group_size_training_curves.png GRPO group size ablation (n=4/8/16)
#   4. dpo_training_curve.png              DPO loss / rewards accuracy (few data points)

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

# Match all "key:value" metric pairs in a log line. Key allows letters,
# digits, underscores, slashes, @, and hyphens ("val-core/..." keys carry a
# hyphen and must be captured in full; omitting "-" would silently drop the
# "val-" prefix). Value allows a leading minus sign and a decimal point.
METRIC_PATTERN = re.compile(r"([\w/@-]+):(-?[0-9]+\.?[0-9]*)")
STEP_PATTERN = re.compile(r"\bstep:(\d+)\b")


def parse_step_log(path: str) -> Optional[Dict[str, List[float]]]:
    """Parse a veRL training log; return {metric_name: [values in step order]},
    including a synthetic 'step' key.

    Only lines that contain a `step:` token are kept (one per training/val step);
    duplicate keys within the same step row are de-duplicated with dict.
    """
    if not os.path.exists(path):
        print(f"[SKIP] log file not found: {path}")
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
            # 'step' itself is also matched by METRIC_PATTERN into metrics['step'];
            # use the steps list as the authoritative source.
            series.setdefault("step", []).append(step)

    if not steps:
        print(f"[SKIP] {path}: no step-metric lines found")
        return None
    return series


def _get_series(series: Dict[str, List[float]], key: str):
    """Return (steps, values) in step order, filtering steps where key is absent
    (e.g. val metrics are only logged on a subset of steps)."""
    if key not in series:
        return [], []
    # val-core metrics are not present on every step; their row count is less
    # than len(steps). Match them positionally against their own appearance
    # order (assumes each val-metric occurrence is paired with its step row).
    return None, series[key]


def parse_step_log_with_alignment(path: str) -> Optional[Dict[str, List[tuple]]]:
    """Parse a training log; return {metric_name: [(step, value), ...]},
    each metric aligned to its own step occurrences."""
    if not os.path.exists(path):
        print(f"[SKIP] log file not found: {path}")
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
        print(f"[SKIP] {path}: no step metrics found")
        return None
    return series


def _unzip(pairs):
    if not pairs:
        return [], []
    xs, ys = zip(*pairs)
    return list(xs), list(ys)


# --------------------------------------------------------------------------
# Figure 1: GRPO vs PPO training dynamics
# --------------------------------------------------------------------------


def plot_grpo_vs_ppo(output_dir: str):
    grpo_log = os.path.join(LOGS_DIR, "pipeline", "stage2_train_grpo.log")
    ppo_log = os.path.join(LOGS_DIR, "pipeline", "stage4_train_ppo_retry_v3.log")

    grpo_series = parse_step_log_with_alignment(grpo_log)
    ppo_series = parse_step_log_with_alignment(ppo_log)
    if grpo_series is None and ppo_series is None:
        print("[SKIP] plot_grpo_vs_ppo: no data available")
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
    print(f"[OK] saved {out_path}")


# --------------------------------------------------------------------------
# Figure 2: PPO collapse (v1) vs fixed (v3)
# --------------------------------------------------------------------------


def plot_ppo_collapse_vs_fixed(output_dir: str):
    v1_log = os.path.join(LOGS_DIR, "pipeline", "stage4_train_ppo.log")
    v3_log = os.path.join(LOGS_DIR, "pipeline", "stage4_train_ppo_retry_v3.log")

    v1_series = parse_step_log_with_alignment(v1_log)
    v3_series = parse_step_log_with_alignment(v3_log)
    if v1_series is None and v3_series is None:
        print("[SKIP] plot_ppo_collapse_vs_fixed: no data available")
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
    print(f"[OK] saved {out_path}")


# --------------------------------------------------------------------------
# Figure 3: GRPO group-size ablation training curves
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
        print("[SKIP] plot_grpo_group_size_curves: no data available")
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
    print(f"[OK] saved {out_path}")


# --------------------------------------------------------------------------
# Figure 4: DPO training curve
# --------------------------------------------------------------------------

DPO_METRIC_PATTERN = re.compile(r"'(loss|rewards/accuracies|rewards/margins|epoch)':\s*(-?[0-9]+\.?[0-9]*)")


def parse_dpo_log(path: str) -> Optional[Dict[str, List[float]]]:
    if not os.path.exists(path):
        print(f"[SKIP] log file not found: {path}")
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
        print("[SKIP] plot_dpo_training_curve: no data available")
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
    print(f"[OK] saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    plot_grpo_vs_ppo(args.output_dir)
    plot_ppo_collapse_vs_fixed(args.output_dir)
    plot_grpo_group_size_curves(args.output_dir)
    plot_dpo_training_curve(args.output_dir)

    print(f"\nAll figures written to {args.output_dir}")


if __name__ == "__main__":
    main()
