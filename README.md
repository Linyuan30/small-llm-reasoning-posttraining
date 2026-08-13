# Small LLM Reasoning Post-Training

A systematic study of enhancing reasoning capabilities in small language models through post-training techniques, including:

- Cold-start Supervised Fine-Tuning (SFT)
- Direct Preference Optimization (DPO)
- Proximal Policy Optimization (PPO)
- Group Relative Policy Optimization (GRPO)
- Reinforcement Learning with Verifiable Rewards (RLVR)

This project investigates how small-scale language models acquire mathematical reasoning abilities through supervised learning and reinforcement learning. Experiments cover two routes — Base models (Qwen3-0.6B/1.7B-Base) and Instruct models (Qwen3-0.6B/1.7B-Instruct) — evaluated on GSM8K and MATH benchmarks.

Inspired by the DeepSeek-R1 training paradigm, this project explores reasoning-oriented post-training pipelines using verifiable rewards.

**Models:** Qwen3-0.6B/1.7B-Base · Qwen3-0.6B/1.7B-Instruct  
**Benchmarks:** GSM8K, MATH  
**Frameworks:** [veRL](https://github.com/volcengine/verl) + vLLM

🤗 Trained checkpoints on the Hub:
- [Linyuana/qwen3-1.7b-instruct-grpo-math-reasoning](https://huggingface.co/Linyuana/qwen3-1.7b-instruct-grpo-math-reasoning) — **best overall** 
- [Linyuana/qwen3-0.6b-instruct-grpo-math-reasoning](https://huggingface.co/Linyuana/qwen3-0.6b-instruct-grpo-math-reasoning) — 0.6B Instruct + GRPO 
- [Linyuana/qwen3-0.6b-grpo-math-reasoning](https://huggingface.co/Linyuana/qwen3-0.6b-grpo-math-reasoning) — 0.6B Base + GRPO 

---

## Overview

DeepSeek-R1's report makes a striking claim: a base model with no reasoning training can acquire structured, self-correcting reasoning behavior through RL alone, given a verifiable reward. That's easy to accept on paper and hard to have real intuition for without running it yourself. This project reproduces the recipe end-to-end on a small model — cold-start SFT, then a controlled comparison of GRPO / PPO / DPO — and documents what actually happens along the way, including a full PPO training collapse and the fix.

Math reasoning (GSM8K/MATH) was chosen deliberately: the final answer is auto-gradable, but the reasoning path is not unique, which makes it one of the few tasks where RL can plausibly add value on top of SFT (as opposed to classification-style tasks where the SFT and RL signal are effectively the same information).

```
Qwen3-Base     ──SFT (cold-start)──► unified <think>/<answer> format ─-─┐
Qwen3-Instruct ──SFT (fmt adapt) ──► unified <think>/<answer> format ──-┤
                                                                        │
                                             ┌──────────────────────────┤
                                             ▼                          │
                                  ┌──────────┴──────────┐               │
                                GRPO        PPO         DPO             │
                         (group-relative, (actor+    (offline           │
                          no critic)   critic+ref)   preference)        │
                                  └──────────┬──────────┘               │
                                             ▼                          │
                                      pass@k eval ◄─────────────────────┘
                                   (GSM8K / MATH, same rule reward)
```

Reward is purely rule-based (regex-extract `<answer>`, normalize, compare to ground truth) — no reward model is trained. Implementation: [reward/rule_reward.py](reward/rule_reward.py).

## Results

**Base route (Qwen3-0.6B-Base):**

| Method | GSM8K pass@1 | GSM8K pass@8 | MATH pass@1 | MATH pass@8 | Cost |
| --- | ---: | ---: | ---: | ---: | --- |
| Base (no post-training) | 4.8 | 30.6 | 3.7 | 24.8 | — |
| SFT only | 38.9 | 78.4 | 29.0 | 66.4 | low |
| SFT + DPO | 39.0 | 78.0 | 29.8 | 69.0 | low |
| SFT + PPO | 42.6 | 76.8 | 31.2 | 69.0 | high |
| **SFT + GRPO** | **67.7** | **85.0** | **48.8** | **79.2** | mid |

**Instruct route (same SFT data/hyperparameters, Instruct base model):**

| Method | Model | GSM8K pass@1 | GSM8K pass@8 | MATH pass@1 | MATH pass@8 | Cost |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Instruct baseline | 0.6B | 65.2 | 84.0 | 33.0 | 61.8 | — |
| SFT (format adapt) | 0.6B | 37.5 | 74.5 | 18.5 | 48.9 | low |
| SFT + GRPO | 0.6B | **77.6** | 90.2 | **69.3** | 91.4 | mid |
| SFT + PPO | 0.6B | 64.5 | 88.0 | 48.1 | 82.6 | high |
| SFT + DPO | 0.6B | 37.7 | 73.2 | 27.7 | 67.4 | low |
| Instruct baseline | 1.7B | 71.1 | 90.4 | 49.2 | 72.2 | — |
| SFT (format adapt) | 1.7B | 55.7 | 88.2 | 28.3 | 62.0 | low |
| **SFT + GRPO** | **1.7B** | **86.1** | **95.6** | **73.6** | **93.8** | mid |
| SFT + PPO | 1.7B | 73.8 | 91.6 | 55.0 | 87.2 | high |
| SFT + DPO | 1.7B | 56.0 | 87.6 | 40.8 | 79.0 | low |

*Eval protocol: 500 held-out samples per dataset, temperature=0.8/top_p=0.95, 8 samples/question. Same rule-reward scorer used in training and eval. Full breakdown incl. all ablations: [docs/experiments.md](docs/experiments.md).*

All three trained checkpoints are on Hugging Face. Load with `transformers`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Best overall — 1.7B Instruct + GRPO (GSM8K 86.1% / MATH 73.6%)
model_id = "Linyuana/qwen3-1.7b-instruct-grpo-math-reasoning"

# 0.6B Instruct + GRPO (GSM8K 77.6% / MATH 69.3%)
# model_id = "Linyuana/qwen3-0.6b-instruct-grpo-math-reasoning"

# 0.6B Base + GRPO (GSM8K 67.7% / MATH 48.8%)
# model_id = "Linyuana/qwen3-0.6b-grpo-math-reasoning"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
```

![Method comparison](docs/images/method_comparison_pass1.png)

Each group of three bars is one training stage (Base/no-PT, SFT, GRPO, PPO, DPO); colors distinguish the three routes — blue = Base 0.6B, purple = 0.6B Instruct, orange = 1.7B Instruct. Note that the SFT bars for Base and Instruct converge to a similar height (~38%), despite very different starting points — and then the GRPO bars diverge sharply, with the Instruct routes gaining far more.

The pass@1 → pass@8 gap is also informative. The Base (no PT) line has the steepest climb across k — it can already stumble onto a correct answer with enough tries, just not reliably under greedy decoding. Post-training narrows this gap by converting "occasionally right" into "reliably right":

![pass@k curves](docs/images/pass_at_k_curves.png)

## Key Findings

**Base route:**

- **GRPO clearly outperforms PPO and DPO** under matched initialization and reward — and trains far more stably. Its group-relative advantage estimation removes the need for a critic entirely, which appears to be a key reason it avoided the instability PPO ran into (see below).
- **PPO collapsed on the first attempt** and had to be debugged and fixed — see the case study right below, and the full postmortem in [docs/ppo_analysis.md](docs/ppo_analysis.md).
- **PPO's stability w.r.t. the KL coefficient is non-monotonic**: 0.005 and 0.02 both trained cleanly, but 0.01 — the value in between — collapsed reproducibly across two independent runs. Stability here isn't simply "more KL = safer."
- **Model scale beats longer SFT training**: swapping the 0.6B base for 1.7B (identical data/epochs) gained +18.8pp GSM8K pass@1 — several times the gain from tuning SFT epochs (3/7/15).
- **DPO barely moves the needle** (39.0 vs 38.9 SFT-only) because its preference pairs are self-sampled from the same SFT model — if it gets a question wrong 8/8 times, "chosen" is just the least-wrong wrong answer. Off-policy methods are bounded by the sampling policy's existing capability this way — details in [docs/dpo_analysis.md](docs/dpo_analysis.md).

**Instruct route:**

- **SFT on an Instruct model is a format adapter, not a cold-start.** The Instruct model already reasons (0.6B pass@1 = 65.2%), but its free-form output makes the rule reward unreliable (`strict_format_rate` as low as 0.3% on MATH). SFT with the same data and hyperparameters forces the output into `<think>/<answer>` format — pass@1 temporarily drops to ~38% while `strict_format_rate` jumps to 96–99.9%, giving RL a stable reward extraction interface. Details: [docs/sft_analysis.md](docs/sft_analysis.md).
- **Instruct route unlocks substantially larger RL gains from the same SFT starting point.** Both Base and 0.6B Instruct enter GRPO at ~38% pass@1, but the RL gain diverges sharply: Base +28.8pp / +19.7pp (GSM8K/MATH), 0.6B Instruct +40.1pp / +50.8pp. The Instruct model carries more latent reasoning knowledge into RL — SFT compressed it into a new format without erasing it, and GRPO releases it. Details: [docs/grpo_analysis.md](docs/grpo_analysis.md).
- **DPO collapses back to the SFT starting point on every route.** Base DPO: 39.0% (SFT was 38.9%). 0.6B Instruct DPO: 37.7% (SFT was 37.5%). 1.7B Instruct DPO: 56.0% (SFT was 55.7%). The structural ceiling — preference pairs bounded by the same policy that generated them — holds regardless of whether the base model is a Base or Instruct variant.

## PPO Failure Case: Response Length Explosion

The first PPO run used hyperparameters matched to GRPO (same batch size, same `kl_loss_coef=0.001`, `critic_warmup=0`) to isolate the effect of adding a critic. Within 20 steps, `response_length` went from ~180 to 1024 (the hard cap — every rollout got truncated) and validation reward turned negative and never recovered:

![PPO v1 collapse vs v3 fix](docs/images/ppo_collapse_vs_fixed.png)

**Hypothesized mechanism** (consistent with the observed metrics, not directly instrumented): the critic starts from scratch with no predictive signal yet (`vf_explained_var` starts at -2.27), so early GAE advantages are largely noise. If a noisy update happens to assign positive advantage to "generate longer," policy gradient can reinforce that direction, and the reward function's flat -1 truncation penalty likely wasn't steep enough to correct it. GRPO's group-relative advantage doesn't depend on a learned value function, which removes this specific failure mode — though that's a statement about this setup, not a general "PPO is unstable" claim.

**Fix** (three complementary changes, not one silver bullet): critic warmup (20 steps of critic-only updates before the actor starts), a progressive length penalty starting at 90% of the token budget, tighter critic clipping and a lower sampling temperature. The fixed run (v3, green above) still shows an exploration spike around step 21-28, but recovers on its own instead of escalating — that "spike then self-correct" pattern is the clearest sign the fix addressed the mechanism rather than just the symptom.

A follow-up KL-coefficient ablation then found something less intuitive: 0.005 and 0.02 both trained cleanly, but 0.01 — the value in between — collapsed reproducibly across two independent runs. Full analysis and the non-monotonicity discussion: [docs/ppo_analysis.md](docs/ppo_analysis.md).

## Repository Structure

```
├── data/scripts/       data prep: GSM8K/MATH download, CoT formatting, DPO pair construction
├── reward/             rule_reward.py — verifiable reward (answer match + format check)
├── training/           run_sft.sh / run_grpo.sh / run_ppo.sh / run_dpo.py, ablation/ scripts
├── eval/                pass@k evaluation, result plotting
├── patches/            infra-only patch on top of official veRL v0.4.0 (no algorithm changes)
└── docs/
    ├── experiments.md      full experiment log — every run, config, and number
    ├── sft_analysis.md     cold-start SFT: format, epoch ablation, scale comparison
    ├── grpo_analysis.md    GRPO setup, group-size ablation, why it avoided PPO's failure mode
    ├── ppo_analysis.md     the collapse → fix → KL ablation story
    └── dpo_analysis.md     why an off-policy baseline underperforms on-policy RL here
```

## Quick Start

```bash
pip install -r requirements.txt
pip install verl==0.4.0
git apply patches/verl-v0.4.0.patch   # infra-only patch, see docs/experiments.md

# 1. data
python data/scripts/prepare_gsm8k.py && python data/scripts/prepare_math.py
python data/scripts/build_sft_mix.py

# 2. cold-start SFT -> GRPO / PPO / DPO
bash training/run_sft.sh 4
bash training/run_grpo.sh 4          # recommended: most stable, best results
bash training/run_ppo.sh 4
accelerate launch training/run_dpo.py

# 3. eval
python eval/eval_pass_at_k.py --model <ckpt_path>
python eval/plot_results.py
```

Ablations are driven by env vars, e.g. `ROLLOUT_N=16 bash training/run_grpo.sh 2` or `KL_LOSS_COEF=0.005 bash training/run_ppo.sh 2` — see `training/ablation/` for the full set.

## Open Question

`strict_format_rate` (exact `<think>...</think><answer>...</answer>` closure) stays near 0 on the **Base route** across every stage, despite `has_answer_rate` >94%. On the **Instruct route**, the same SFT data and evaluation script brings `strict_format_rate` to ~99% — the only systematic difference is the chat template each model uses. This strongly suggests the issue is a mismatch between the Base model's chat template and the eval regex, rather than anything in the training logic. A suspected off-by-one in the SFT loss mask was tried and reverted with no effect. Root cause for the Base route remains open. Documented as-is in [docs/sft_analysis.md](docs/sft_analysis.md).

## Acknowledgements

[veRL](https://github.com/volcengine/verl) · [vLLM](https://github.com/vllm-project/vllm) · [trl](https://github.com/huggingface/trl) · [DeepSeek-R1](https://arxiv.org/abs/2501.12948) · [Qwen3](https://huggingface.co/Qwen)

## License

[MIT](LICENSE)
