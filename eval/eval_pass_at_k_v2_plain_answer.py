# Copyright (c) 2026
#
# 【消融实验变体 v2】训练同源指标评估：GSM8K/MATH 测试集上的 pass@1 / pass@k
# 与主线 eval_pass_at_k.py 逻辑完全一致，唯一差异是使用
# reward/rule_reward_v2_plain_answer.py（无 <answer> XML 标签格式）
# 而非主线的 reward/rule_reward.py 进行格式判定与打分。
#
# 用法：
#   python eval_pass_at_k_v2_plain_answer.py \
#       --model_path ../models/sft_coldstart/qwen3-0.6b-v2-plain-answer \
#       --test_parquet ../data/processed/gsm8k_v2_plain_answer/test.parquet \
#       --k_list 1,4,8 \
#       --output_path ../eval/results/sft_v2_plain_answer_qwen3_0.6b.json

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reward"))
from rule_reward_v2_plain_answer import RuleRewardV2PlainAnswer, check_format  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_parquet", required=True, nargs="+", help="一个或多个测试集 parquet 路径")
    parser.add_argument("--k_list", default="1,4,8", help="逗号分隔的 k 值列表，如 1,4,8")
    parser.add_argument("--max_samples", type=int, default=None, help="限制测试样本数（调试用）")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", required=True)
    parser.add_argument(
        "--save_generations",
        action="store_true",
        default=True,
        help="是否把每条样本的完整生成文本（含所有采样）保存到输出 json（默认保存）",
    )
    parser.add_argument(
        "--no_save_generations",
        dest="save_generations",
        action="store_false",
        help="不保存完整生成文本，仅保存统计指标（结果文件更小）",
    )
    return parser.parse_args()


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - float(_comb_ratio_product(n - c, k, n))


def _comb_ratio_product(n_minus_c: int, k: int, n: int) -> float:
    result = 1.0
    for i in range(k):
        result *= (n_minus_c - i) / (n - i)
    return result


def main():
    args = parse_args()
    k_list = [int(k) for k in args.k_list.split(",")]
    max_k = max(k_list)

    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    reward_fn = RuleRewardV2PlainAnswer()

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
    )
    sampling_params = SamplingParams(
        n=max_k,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    all_results = {}

    for test_path in args.test_parquet:
        data_name = os.path.basename(os.path.dirname(test_path)) or os.path.basename(test_path)
        df = pd.read_parquet(test_path)
        if args.max_samples is not None:
            df = df.iloc[: args.max_samples]

        chat_prompts = []
        ground_truths = []
        raw_questions = []
        for _, row in df.iterrows():
            messages = list(row["prompt"])
            chat_prompts.append(
                tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )
            ground_truths.append(row["reward_model"]["ground_truth"])
            raw_questions.append(messages[-1]["content"] if messages else None)

        print(f"[{data_name}] Generating {max_k} samples for {len(chat_prompts)} prompts ...")
        outputs = llm.generate(chat_prompts, sampling_params)

        pass_at_k_sums = {k: 0.0 for k in k_list}
        per_sample_records = []
        total_strict_format_ok = 0
        total_has_answer = 0
        total_samples = 0

        for prompt_text, question, output, gt in zip(chat_prompts, raw_questions, outputs, ground_truths):
            candidates = [o.text for o in output.outputs]
            scored = [reward_fn.score(text, gt) for text in candidates]
            correctness = [r.correct for r in scored]
            format_oks = [r.format_ok for r in scored]
            n = len(correctness)
            c = sum(correctness)

            total_strict_format_ok += sum(format_oks)
            total_has_answer += sum(1 for text in candidates if check_format(text).has_answer)
            total_samples += n

            for k in k_list:
                pass_at_k_sums[k] += compute_pass_at_k(n, c, k)

            record = {
                "question": question,
                "ground_truth": gt,
                "num_samples": n,
                "num_correct": c,
                "num_format_ok": sum(format_oks),
            }
            if args.save_generations:
                record["generations"] = [
                    {
                        "text": text,
                        "correct": r.correct,
                        "format_ok": r.format_ok,
                        "extracted_answer": r.extracted_answer,
                    }
                    for text, r in zip(candidates, scored)
                ]
            per_sample_records.append(record)

        num_prompts = len(chat_prompts)
        metrics = {f"pass@{k}": pass_at_k_sums[k] / num_prompts for k in k_list}
        metrics["num_prompts"] = num_prompts
        metrics["num_samples_per_prompt"] = max_k
        metrics["strict_format_rate"] = total_strict_format_ok / total_samples if total_samples else 0.0
        metrics["has_answer_rate"] = total_has_answer / total_samples if total_samples else 0.0

        print(f"[{data_name}] {metrics}")
        all_results[data_name] = {
            "metrics": metrics,
            "per_sample": per_sample_records,
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"Saved evaluation results to {args.output_path}")


if __name__ == "__main__":
    main()
