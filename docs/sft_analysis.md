# SFT Cold-Start Analysis

## Why SFT Before RL

The base model (Qwen3-0.6B-Base) has never seen any instruction or format training. Starting RL from this point runs into two problems: the output format is entirely unconstrained, making it hard to reliably extract answers for the rule reward; and the noisy, high-variance outputs in early training produce low-quality reward signal. The goal of SFT cold-start is not to teach the model to reason — that is RL's job — but to establish a stable `<think>...</think><answer>...</answer>` output habit before RL begins.

The base model's own pass@k curve makes this concrete. On GSM8K, pass@1 is 4.8% but pass@8 is 30.6%; on MATH, pass@1 is 3.7% while pass@8 reaches 24.8% — roughly a 6-7x gap. The model isn't completely unable to reason; it just can't reliably reproduce a correct answer under greedy decoding. The acceptance criterion for SFT cold-start is to close this gap: pull pass@1 up toward the current pass@8 level. It shouldn't be expected to raise the pass@8 ceiling itself.

## Data and Training Configuration

Cold-start data: approximately 2,000 GSM8K + 2,000 MATH examples, mixed and formatted as `<think>/<answer>`. Training: veRL `fsdp_sft_trainer`, `lr=1e-5`, `micro_batch_size_per_gpu=4`, `total_epochs=3` (42 steps), `max_length=1536`.

## Main Result

| Dataset | pass@1 | pass@4 | pass@8 | has_answer_rate |
| --- | --- | --- | --- | --- |
| GSM8K | 38.9% | 66.8% | 78.4% | 94.4% |
| MATH | 29.0% | 55.8% | 66.4% | 97.3% |

Pass@1 jumped by an order of magnitude relative to the base model (GSM8K 4.8% → 38.9%, MATH 3.7% → 29.0%), and `has_answer_rate` reached 94%+, confirming the model can now consistently produce an extractable answer. The cold-start objective is met.

## Ablation: Training Duration (3 / 7 / 15 Epochs)

Same data and hyperparameters, only `total_epochs` varied — to check whether more training monotonically improves downstream pass@1 or whether overfitting sets in.

| Epochs | GSM8K pass@1 | MATH pass@1 |
| --- | --- | --- |
| 3 (main, used for all RL runs) | 38.9% | 29.0% |
| 7 | **42.8%** | **31.6%** |
| 15 | 40.9% | 27.9% |

7 epochs improves over 3 on both benchmarks — so the main run is not fully converged. But going to 15 epochs reverses that: MATH pass@1 drops below even the 3-epoch baseline (27.9% vs 29.0%), and GSM8K also falls back from the 7-epoch peak. 15 epochs appears to overfit the training distribution at the expense of MATH, which has a higher difficulty and more varied problem types. The 7-epoch point looks like a sweet spot.

One important note: neither the 7-epoch nor the 15-epoch checkpoint has been fed into any downstream GRPO/PPO/DPO training — all RL experiments use the 3-epoch checkpoint as initialization. Swapping to the 7-epoch version as the RL starting point is a reasonable next experiment, not done here.

## Ablation: Model Scale (Qwen3-0.6B vs Qwen3-1.7B)

Same data, format, hyperparameters, and epoch count (7) as the 7-epoch ablation above. Only the base model changed.

| Base model | GSM8K pass@1 | GSM8K pass@8 | MATH pass@1 | MATH pass@8 |
| --- | --- | --- | --- | --- |
| Qwen3-0.6B-Base | 42.8% | 79.2% | 31.6% | 69.6% |
| **Qwen3-1.7B-Base** | **61.6%** | **89.0%** | **48.4%** | **85.4%** |

The scale gain (GSM8K +18.8pp, MATH +16.8pp) dwarfs what was achievable by tuning epoch count on the 0.6B model (which varied by at most a few percentage points). Pass@8 is also strictly higher on 1.7B, meaning this isn't just the larger model getting lucky more often — the reasoning ceiling is genuinely higher. Under compute constraints, spending the same effort on a larger base model is a better leverage point than hyperparameter search on a smaller one. (1.7B Base was not taken into downstream RL due to time limits; 1.7B Instruct was — see the Instruct section below.)

![SFT epoch ablation and scale comparison](images/sft_epoch_ablation.png)

## Open Question: `strict_format_rate` Stays Near Zero

The evaluation script measures two format metrics: `has_answer_rate` (loose: can an answer be extracted at all) and `strict_format_rate` (strict: output exactly matches the closed `<think>...</think><answer>...</answer>` form with no extra text). `has_answer_rate` reaches 94-99% across all SFT checkpoints and all downstream stages (GRPO, PPO, DPO), but `strict_format_rate` stays near 0 at every stage on the Base route.

The most suspicious hypothesis was a loss-mask alignment bug in veRL's `multiturn_sft_dataset.py` — specifically, that the mask interval might be off by one token in a multi-turn setting, causing the first token of the assistant response to receive no supervision. A patched version was trained for 7 epochs; `strict_format_rate` was unchanged. The patch was fully reverted (`git checkout`), and all SFT/GRPO/PPO runs are 100% unmodified veRL logic with no algorithmic customization.

The root cause is not yet identified. The most plausible remaining directions are: the evaluation regex is too strict (e.g. sensitive to trailing whitespace or newlines), or there is a subtle difference between the chat template applied at training vs inference time. Importantly, since the same evaluation script is used across all stages, the `strict_format_rate ≈ 0` result is consistent and doesn't affect cross-stage comparisons. The Instruct route comparison below provides additional diagnostic signal.

---

## Instruct Models: SFT as Format Adapter, Not Cold-Start

Running the same SFT procedure on Instruct variants (Qwen3-0.6B-Instruct and 1.7B-Instruct, same data, same hyperparameters, same 3 epochs) reveals that SFT serves a fundamentally different purpose depending on the base model's starting state.

### What changes

| | Base route | Instruct route |
| --- | --- | --- |
| Before SFT | pass@1 ~5%, `strict_format_rate` ~0% | pass@1 ~65-71%, `strict_format_rate` ~0.3-71% |
| After SFT | pass@1 jumps to ~39-56% | pass@1 *drops* to ~38-56% |
| `strict_format_rate` after SFT | stays ~0% | jumps to ~96-99.9% |
| Interpretation | SFT cold-starts both format and reasoning | SFT adapts output format only — reasoning was already there |

On the Base route, `has_answer_rate` (loose: can we extract *any* answer) reaches 94%+ while `strict_format_rate` (stricter: exact `<think>...</think><answer>...</answer>` closure) stays near 0 — suggesting the Base model's output structure is still loose even after SFT, producing correct answers but not always inside the expected tags. On the Instruct route, both metrics converge to ~99% after SFT, consistent with the Instruct model's instruction-following capability making it easier to adopt a strictly formatted template.

### Why pass@1 drops on the Instruct route

The Instruct model's existing reasoning is expressed in its natural free-form output style. SFT forces it into the `<think>/<answer>` template, a constrained format it wasn't using for math. During this format-switching period, some questions that the model could answer in its natural style get answered incorrectly in the constrained format — a short-term cost of the transition, not an erasure of underlying knowledge.

The key observation: after SFT, **both Base and Instruct routes arrive at roughly the same pass@1 (~38% for 0.6B)**. This near-convergence makes the subsequent RL comparison clean: both enter GRPO from the same numeric starting point, but the Instruct route has substantially more latent knowledge to unlock.

### What this means for RL

The pass@1 vs pass@8 gap is the relevant signal. After SFT, the 1.7B Instruct model has GSM8K pass@8 = 88.2% (close to its pre-SFT baseline of 90.4%) but pass@1 = 55.7% — a 32.5pp gap. This is exactly the regime on-policy RL is designed for: the model "knows how" but can't reliably demonstrate it under greedy decoding. Instruct+GRPO subsequently reaches 86.1% pass@1, confirming this: the SFT format cost was temporary; the RL payoff was substantial.

### The `strict_format_rate` contrast as a diagnostic

The Base route's persistent `strict_format_rate ≈ 0` across all stages — despite `has_answer_rate` > 94% — has an uncertain root cause (see the open question above). The Instruct route's jump to ~99% after SFT, using the *same SFT data and the same evaluation script*, strongly suggests the issue is in how the Base model's output is tokenized or wrapped by its chat template at eval time, rather than in the training logic. Two identical SFT runs with different base models produce identical training dynamics but very different `strict_format_rate` outcomes — the only systematic difference is the chat template.

![SFT as format adapter](images/instruct_sft_format_effect.png)

---

Raw data and training log paths: [experiments.md](experiments.md), Experiments ② and ⑧.
