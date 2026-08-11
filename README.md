# Small LLM Reasoning Post-Training

A systematic study of enhancing reasoning capabilities in small language models through post-training techniques, including:

- Cold-start Supervised Fine-Tuning (SFT)
- Direct Preference Optimization (DPO)
- Proximal Policy Optimization (PPO)
- Group Relative Policy Optimization (GRPO)
- Reinforcement Learning with Verifiable Rewards (RLVR)

This project investigates how small-scale language models acquire mathematical reasoning abilities through supervised learning and reinforcement learning. Experiments are conducted on Qwen3-0.6B/1.7B-Base models with GSM8K and MATH benchmarks.

Inspired by the DeepSeek-R1 training paradigm, this project explores reasoning-oriented post-training pipelines using verifiable rewards.

**Base Models:** Qwen3-0.6B/1.7B-Base  
**Benchmarks:** GSM8K, MATH  
**Frameworks:** [veRL](https://github.com/volcengine/verl) + vLLM

---

## Overview

DeepSeek-R1's report makes a striking claim: a base model with no reasoning training can acquire structured, self-correcting reasoning behavior through RL alone, given a verifiable reward. That's easy to accept on paper and hard to have real intuition for without running it yourself. This project reproduces the recipe end-to-end on a small model — cold-start SFT, then a controlled comparison of GRPO / PPO / DPO — and documents what actually happens along the way, including a full PPO training collapse and the fix.

Math reasoning (GSM8K/MATH) was chosen deliberately: the final answer is auto-gradable, but the reasoning path is not unique, which makes it one of the few tasks where RL can plausibly add value on top of SFT (as opposed to classification-style tasks where the SFT and RL signal are effectively the same information).

```
Qwen3-Base ──SFT (cold start)──► unified <think>/<answer> format
                                          │
                          ┌───────────────┼───────────────┐
                          ▼               ▼               ▼
                        GRPO             PPO             DPO
                 (group-relative,   (actor+critic+   (offline preference
                  no critic)         ref, on-policy)   pairs, off-policy)
                          │               │               │
                          └───────►  pass@k eval  ◄───────┘
                                 (GSM8K / MATH, same rule reward)
```

Reward is purely rule-based (regex-extract `<answer>`, normalize, compare to ground truth) — no reward model is trained. Implementation: [reward/rule_reward.py](reward/rule_reward.py).

## Results

| Method | GSM8K pass@1 | GSM8K pass@8 | MATH pass@1 | MATH pass@8 | Cost |
| --- | ---: | ---: | ---: | ---: | --- |
| Base (no post-training) | 4.8 | 30.6 | 3.7 | 24.8 | - |
| SFT only | 38.9 | 78.4 | 29.0 | 66.4 | low |
| SFT + DPO | 39.0 | 78.0 | 29.8 | 69.0 | low |
| SFT + PPO | 42.6 | 76.8 | 31.2 | 69.0 | high |
| **SFT + GRPO** | **67.7** | **85.0** | **48.8** | **79.2** | mid |

*Eval protocol: 500 held-out samples per dataset, temperature=0.8/top_p=0.95, 8 samples/question. Same rule-reward scorer used in training and eval. Full breakdown incl. all ablations: [docs/experiments.md](docs/experiments.md).*

![Method comparison](docs/images/method_comparison_pass1.png)

The pass@1 → pass@8 gap is itself informative: the Base model has the steepest climb (4.8→30.6 on GSM8K), meaning it can already stumble onto correct answers with enough tries — it just can't do it reliably under greedy decoding. Post-training narrows this gap rather than only raising the ceiling, i.e. it converts "occasionally right" into "reliably right":

![pass@k curves](docs/images/pass_at_k_curves.png)

## Key Findings

- **GRPO clearly outperforms PPO and DPO** under matched initialization and reward — and trains far more stably. Its group-relative advantage estimation removes the need for a critic entirely, which appears to be a key reason it avoided the instability PPO ran into (see below).
- **PPO collapsed on the first attempt** and had to be debugged and fixed — see the case study right below, and the full postmortem in [docs/ppo_analysis.md](docs/ppo_analysis.md).
- **PPO's stability w.r.t. the KL coefficient is non-monotonic**: 0.005 and 0.02 both trained cleanly, but 0.01 — the value in between — collapsed reproducibly across two independent runs. Stability here isn't simply "more KL = safer."
- **Model scale beats longer SFT training**: swapping the 0.6B base for 1.7B (identical data/epochs) gained +18.8pp GSM8K pass@1 — several times the gain from tuning SFT epochs (3/7/15).
- **DPO barely moves the needle** (39.0 vs 38.9 SFT-only) because its preference pairs are self-sampled from the same SFT model — if it gets a question wrong 8/8 times, "chosen" is just the least-wrong wrong answer. Off-policy methods are bounded by the sampling policy's existing capability this way — details in [docs/dpo_analysis.md](docs/dpo_analysis.md).

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

`strict_format_rate` (exact `<think>...</think><answer>...</answer>` closure) stays near 0 across every stage despite `has_answer_rate` >94%. A suspected off-by-one in the SFT loss mask was tried and reverted after it showed no effect — root cause is still open, most likely in the eval regex or chat template rather than the model. Documented as-is in [docs/sft_analysis.md](docs/sft_analysis.md).

## Acknowledgements

[veRL](https://github.com/volcengine/verl) · [vLLM](https://github.com/vllm-project/vllm) · [trl](https://github.com/huggingface/trl) · [DeepSeek-R1](https://arxiv.org/abs/2501.12948) · [Qwen3](https://huggingface.co/Qwen)

## License

[MIT](LICENSE)
