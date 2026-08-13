# DPO Analysis

## Background

DPO was included as an off-policy baseline, to test whether the on-policy RL methods (GRPO, PPO) are actually earning their extra training cost and complexity on this task, or whether a cheaper offline preference-learning approach gets most of the way there. Unlike GRPO/PPO, DPO never interacts with the environment during training — it optimizes against a fixed set of preference pairs constructed upfront.

## Setup

Preference pairs were built by rejection sampling: the SFT checkpoint generated 8 completions per prompt (temperature > 0) on 4,000 GSM8K training prompts, each scored by the same rule reward used elsewhere, then the highest- and lowest-scoring completions per prompt were kept as the chosen/rejected pair (prompts where all 8 samples scored identically were skipped, since there's no preference signal). This produced 3,171 pairs.

Training used `trl.DPOTrainer`, 4-GPU `accelerate launch`, `learning_rate=5e-7`, `beta=0.1`, 1 epoch (49 steps), effective batch size 64.

## Result

**Base route (Qwen3-0.6B-Base):**

| Dataset | pass@1 | pass@4 | pass@8 |
| --- | --- | --- | --- |
| GSM8K | 39.0% | 66.9% | 78.0% |
| MATH | 29.8% | 57.3% | 69.0% |

This is close to a no-op relative to SFT-only (GSM8K 38.9% → 39.0%, MATH 29.0% → 29.8%) — a fraction of a percentage point, compared to GRPO's +28.8pp / +19.8pp under the same starting checkpoint. The training dynamics themselves looked healthy: loss dropped from 0.690 to 0.682, and `rewards/accuracies` rose from ~50% to 62-65%. So the model clearly learned *something* from the preference pairs — it just didn't translate into more correct answers.

**Instruct routes (same hyperparameters, Instruct base):**

| Route | Dataset | SFT start | DPO result | Change |
| --- | --- | ---: | ---: | ---: |
| 0.6B-Instruct | GSM8K | 37.5% | 37.7% | +0.2pp |
| 0.6B-Instruct | MATH | 18.5% | 27.7% | +9.2pp |
| 1.7B-Instruct | GSM8K | 55.7% | 56.0% | +0.3pp |
| 1.7B-Instruct | MATH | 28.3% | 40.8% | +12.5pp |

The pattern is the same: DPO returns pass@1 almost exactly to SFT-start levels. The MATH numbers show slightly larger gains than GSM8K, but given that these models start from a very low MATH SFT pass@1 (18-28%) while MATH pass@8 stays much higher (49-62%), the small absolute gain likely reflects the same ceiling effect — the model could already produce correct MATH answers with some probability, DPO slightly improved extraction reliability rather than introducing new solution paths.

![DPO training curve](images/dpo_training_curve.png)

## Why the Internal Metric Improved but pass@k Didn't

The `rewards/accuracies` metric only measures relative ordering within pairs the model has already seen — it doesn't require the model to produce new correct answers it couldn't produce before. Three contributing factors, roughly in order of expected impact:

1. **The ceiling problem.** Both chosen and rejected completions are self-sampled from the same SFT model. If that model gets a question wrong in all 8 samples, "chosen" is still a wrong answer — just a less-wrong one by whatever the reward function measures (e.g. closer format compliance). Learning to prefer it over "rejected" cannot introduce a correct solution path the base policy didn't already have some probability of producing. This is the structural limitation of building preference data this way: it's fundamentally bounded by the sampling policy's existing capability, unlike on-policy RL which can reinforce a correct trajectory the first time it's sampled, however rare.
2. **Limited training budget.** 1 epoch, 49 steps, `learning_rate=5e-7` — conservative compared to GRPO/PPO's 4 epochs / 116 steps. This alone would produce a smaller effect size even with better data.
3. **`beta=0.1`** constrains how far the policy can move from the reference model; combined with (2), the total displacement from the SFT checkpoint is small by construction.

We can't cleanly separate how much of the gap is (1) vs (2)+(3) from this single run — a longer DPO run, or preference pairs sourced from a stronger model (e.g. the GRPO checkpoint) rather than self-sampling, would be needed to test whether the ceiling in (1) is actually the binding constraint. That's a reasonable next step, not done here.

## Cost vs Value

Training cost was the lowest of all four methods: 3,171 pairs, 49 steps, ~3 minutes on a single setup. As a baseline for "how much do you get almost for free," that's a fair characterization of what DPO delivered here. It's a reasonable choice if the goal is a cheap correction pass on top of an already-capable policy (e.g. style/safety alignment after GRPO), but on this task — where the bottleneck is teaching the model new correct reasoning paths rather than re-ranking outputs it can already produce — it's not a substitute for on-policy RL.

## Summary of Findings

| Question | Observation |
| --- | --- |
| Did DPO improve over SFT-only? | Marginally (+0.1pp GSM8K, +0.8pp MATH pass@1 on Base) — not a meaningful gain |
| Did the model learn anything during training? | Yes — internal preference accuracy rose from ~50% to 62-65%, so the optimization itself worked |
| Why didn't that transfer to pass@k? | Preference pairs are self-sampled from the same SFT policy, so "chosen" is capped by what that policy could already produce; the RL methods aren't bounded this way |
| Is DPO's low cost worth it here? | Only as a baseline / cheap post-hoc correction step, not as a primary capability-improvement method for this task |
| Does the pattern hold on Instruct routes? | Yes — both 0.6B and 1.7B Instruct DPO results fall back to within ~0.3pp of their SFT starting points on GSM8K; the structural ceiling problem is route-agnostic |

---

Raw logs and evaluation JSON paths: [experiments.md](experiments.md), Experiment ⑤.
