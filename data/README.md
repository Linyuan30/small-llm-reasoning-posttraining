# 数据构造说明

所有数据来自公开学术数据集（GSM8K、MATH），无需任何脱敏处理。原始数据和处理产物不纳入本仓库（体积问题，且都可通过下面的脚本重新生成），只保留 `scripts/` 下的处理逻辑。

## 输出格式

统一要求模型按以下格式输出，便于程序化提取答案计算 reward（定义见 `scripts/format_cot.py`）：

```text
<think>
（推理过程）
</think>
<answer>
（最终数值答案）
</answer>
```

## 构造流程

```
GSM8K / MATH（HuggingFace datasets）
        │
        ├─ prepare_gsm8k.py / prepare_math.py
        │       │
        │       ├─→ data/processed/{gsm8k,math}/{train,test}.parquet
        │       │   （RL 训练用，prompt 已套用统一格式要求，供 GRPO/PPO 直接读取）
        │       │
        │       └─→ data/processed/{gsm8k,math}_sft_coldstart.jsonl
        │           （SFT 冷启动用，官方带推理过程的答案改写为 <think>/<answer> 格式）
        │
        ├─ build_sft_mix.py
        │       └─→ 合并 gsm8k/math 的 SFT 数据，切分 train/val parquet
        │           （各来源约 2000 条，按 val_size_per_source 抽验证集，其余合并为训练集）
        │
        └─ build_dpo_pairs.py（依赖 SFT 阶段产出的 checkpoint）
                └─→ 用 SFT 模型对训练 prompt 做拒绝采样（每题采样 N 次），
                    按 reward/rule_reward.py 打分排序，取最优/最差组成 chosen/rejected 偏好对
```

## 脚本说明

| 脚本 | 作用 |
| --- | --- |
| `scripts/format_cot.py` | 统一 `<think>/<answer>` 格式的公共工具函数（system prompt、格式拼接），被其他脚本复用 |
| `scripts/prepare_gsm8k.py` | 下载 GSM8K，产出 RL 训练用 parquet 和 SFT 冷启动 jsonl |
| `scripts/prepare_math.py` | 同上，数据源为 MATH |
| `scripts/build_sft_mix.py` | 合并多个来源的 SFT 冷启动数据，切分训练/验证集 |
| `scripts/build_dpo_pairs.py` | 用 SFT 模型做拒绝采样，构造 DPO 偏好对 |
| `scripts/format_cot_v2_plain_answer.py` / `prepare_*_v2_plain_answer.py` | 一组格式变体实验：不用 `<answer>` XML 标签、改用纯文本答案，用于排查 SFT 格式合规率问题（见 `docs/sft_analysis.md`），非主线格式 |

## 用法示例

```bash
cd data/scripts

python prepare_gsm8k.py \
    --local_save_dir ../processed/gsm8k \
    --sft_output ../processed/gsm8k_sft_coldstart.jsonl \
    --sft_sample_size 2000

python prepare_math.py \
    --local_save_dir ../processed/math \
    --sft_output ../processed/math_sft_coldstart.jsonl \
    --sft_sample_size 2000

python build_sft_mix.py \
    --inputs ../processed/gsm8k_sft_coldstart.jsonl ../processed/math_sft_coldstart.jsonl \
    --train_output ../processed/sft_coldstart_mix_train.parquet \
    --val_output ../processed/sft_coldstart_mix_val.parquet

# SFT 训练完成后，用得到的 checkpoint 构造 DPO 偏好对
python build_dpo_pairs.py \
    --model_path ../../models/sft_coldstart/qwen3-0.6b \
    --input_parquet ../processed/gsm8k/train.parquet \
    --output_path ../processed/gsm8k_dpo_pairs.jsonl \
    --num_samples 8 \
    --max_prompts 4000
```
