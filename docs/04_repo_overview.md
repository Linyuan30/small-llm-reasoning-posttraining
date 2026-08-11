# 框架与实验总览

> 本文档回答三个问题：**用的什么框架、目录里每个文件是干什么的、目前跑过哪些实验**。
> 与 `01_motivation_and_design.md`（目标与方案设计）、`02_experiment_plan.md`（排期）、`03_results.md`（详细实验数据）互补，
> 本文档是"从代码/脚本视角"的一份索引和现状快照，写作时间：2026-08-07，最后更新：2026-08-07（PPO v3 修复成功后同步）。

---

## 1. 训练框架：veRL（官方原版 + 少量基础设施改动，无算法级魔改）

**结论：不是自研框架，主体是直接 clone 的 [volcengine/verl](https://github.com/volcengine/verl) 官方仓库（tag/commit 对应 v0.4.0），SFT 用它的 `fsdp_sft_trainer`，GRPO/PPO 用它的 `main_ppo`（`algorithm.adv_estimator` 分别设为 `grpo`/`gae`）。DPO 单独用 `trl.DPOTrainer`（不在 verl 内）。**

**本仓库不直接携带 verl 源码**：请按 `README.md` 的说明 `pip install verl==0.4.0`（或 `git clone` 官方仓库并 `checkout` 到 v0.4.0 tag），再用 `git apply patches/verl-v0.4.0.patch` 打上下表列出的基础设施小补丁后使用。（历史开发环境中代码位于 `train/verl/`，是官方仓库的本地 clone。）

`git remote -v` 确认为 `https://github.com/volcengine/verl.git`，`git diff` 显示对官方代码只做了以下改动，**均为基础设施/工程健壮性改动，没有任何算法逻辑改动**：

| 文件 | 改动内容 | 性质 |
| --- | --- | --- |
| `verl/trainer/config/ppo_trainer.yaml` | 新增 `trainer.wandb_proxy` 配置项 | 基础设施（wandb 需走代理连外网，且不能污染全局 `http_proxy` 影响 vLLM 等其他 HTTP 请求） |
| `verl/trainer/config/sft_trainer.yaml` | 同上 | 基础设施 |
| `verl/trainer/fsdp_sft_trainer.py` | `Tracking(...)` 初始化时传入 `config=OmegaConf.to_container(...)`，让 wandb 记录完整超参 | 基础设施（原版这里没传 config，wandb 面板看不到超参） |
| `verl/utils/tracking.py` | `wandb_proxy` 取值方式从 `config["trainer"][...]` 改为 `.get()` 链式取值，避免 `config=None` 时报错 | 基础设施（配合上面两处的健壮性修复） |
| `verl/workers/reward_manager/naive.py` | 用 `inspect.signature` 检测自定义 `compute_score` 是否支持 `response_length_ratio` 参数，支持则传入，供 reward 函数识别截断样本、抑制 PPO 的 response length 爆炸 | 基础设施（PPO 崩溃修复的关键一环，已验证有效，见 `03_results.md`实验④ v3） |

也就是说：**SFT/GRPO/PPO 用的都是官方 verl 的原生算法实现，没有做任何算法级魔改**。

> **曾经尝试过、已回退的改动**：`verl/utils/dataset/multiturn_sft_dataset.py` 的 loss_mask 曾怀疑存在 off-by-one 对齐问题，一度改动过（把 assistant turn 的 mask 区间整体左移一位）。但经过修复后的重新训练+评估验证，**该改动对 `strict_format_rate` 没有起到预期效果，判断为无效改动**，已用 `git checkout` 还原为官方原版。详见第4节。

---

## 2. 目录结构与各脚本作用

> 下面是本开源仓库（`llm-rl-reasoning/`）的实际目录结构。原始开发环境中还有 `verl/`（官方框架本地 clone，见第1节）、`models/`（1.6T checkpoint）、`logs/`/`wandb/`/`outputs/`（训练运行产物）、`data/processed/`（数据脚本产出的中间文件）等目录，体积过大或属于可重新生成的产物，未纳入本仓库，均已加入 `.gitignore`。

```text
llm-rl-reasoning/
├── data/
│   └── scripts/             # 数据处理脚本
│       ├── prepare_gsm8k.py / prepare_math.py           # 下载+格式化 GSM8K/MATH（主线：<think>+<answer>格式）
│       ├── format_cot.py                                 # 主线格式：<think>...</think><answer>...</answer>
│       ├── prepare_gsm8k_v2_plain_answer.py / prepare_math_v2_plain_answer.py  # 消融变体：不用<answer>标签
│       ├── format_cot_v2_plain_answer.py                 # 变体格式：<think>...</think>+纯文本答案
│       ├── build_sft_mix.py                              # 合并 GSM8K/MATH 冷启动数据，切分 train/val
│       └── build_dpo_pairs.py                            # 用 SFT 模型做 rejection sampling 构造 DPO 偏好对
├── reward/
│   ├── rule_reward.py                  # 主线规则 reward（答案规范化匹配 + 严格/宽松格式校验）
│   ├── rule_reward_v2_plain_answer.py  # plain_answer 变体对应的 reward（不校验 <answer> 标签）
│   └── test_rule_reward.py             # 单元测试
├── training/
│   ├── run_sft.sh            # SFT 冷启动，封装 `verl.trainer.fsdp_sft_trainer`
│   ├── run_grpo.sh           # GRPO，封装 `verl.trainer.main_ppo`（adv_estimator=grpo）+ vLLM rollout
│   ├── run_ppo.sh            # PPO，封装 `verl.trainer.main_ppo`（adv_estimator=gae）+ Critic + vLLM rollout
│   ├── run_dpo.py            # DPO，基于 `trl.DPOTrainer`（不在 verl 内）
│   ├── run_full_pipeline.sh  # 一键跑 SFT评估→GRPO→PPO→DPO→评估→汇总，全程无需人工介入
│   ├── plot_training_curves.py  # 【可视化】解析训练日志，绘制 GRPO/PPO 训练过程曲线
│   │                             #   （reward、response_length等），含 PPO v1崩溃 vs v3修复 对比图，
│   │                             #   产出到 docs/images/
│   └── ablation/             # 消融实验启动脚本
│       ├── ablation_grpo_n4.sh / ablation_grpo_n16.sh              # GRPO group size 消融
│       ├── ablation_ppo_kl005.sh / ablation_ppo_kl01_2gpu.sh / ablation_ppo_kl02.sh  # PPO KL系数消融
│       ├── ablation_sft_scheme1_answer_token.sh   # 方案1：<answer> special token 格式消融
│       └── ablation_sft_scheme2_plain_answer.sh   # 方案2：plain answer 格式消融
├── scripts/
│   ├── add_answer_special_tokens.py     # 方案1：给 tokenizer 新增 <answer>/</answer> special token 并 resize embedding
│   └── test.py                          # 零散测试脚本
├── eval/
│   ├── eval_pass_at_k.py                # 主线：vLLM批量采样 + pass@1/4/8 + strict_format_rate/has_answer_rate
│   ├── eval_pass_at_k_v2_plain_answer.py # plain_answer 变体对应的评估脚本
│   ├── summarize_results.py             # 汇总各 results/*.json，回写 docs/03_results.md
│   ├── plot_results.py                  # 【可视化】从 results/*.json 批量生成方法对比/pass@k曲线/
│   │                                     #   消融实验柱状图/响应长度分布图，产出到 docs/images/，纯CPU秒级完成
│   └── results/                         # 评估结果汇总（逐样本明细体积过大，未纳入仓库，跑脚本自动生成）
├── patches/
│   └── verl-v0.4.0.patch     # 对官方 verl v0.4.0 的基础设施小补丁（见第1节表格）
└── docs/
    ├── 01_motivation_and_design.md  # 项目目标、能力清单、技术方案设计（最上层的"为什么"）
    ├── 02_experiment_plan.md        # 8卡资源下的详细排期、P0/P1/P2优先级
    ├── 03_results.md                # 每个阶段的详细实验配置+结果+分析（最详细的"跑得怎么样"），
    │                                 #   已配图：方法对比、pass@k曲线、各消融实验柱状图、训练过程曲线、响应长度分布
    ├── 04_repo_overview.md          # 本文档：框架说明 + 目录索引 + 实验现状快照
    ├── 05_debugging_notes.md        # SFT格式合规率问题的诊断分析（前期分析笔记，部分结论后续被证伪，见第4节）
    └── images/                      # plot_*.py 脚本的图表产出目录，被 03_results.md 引用
```

---

## 3. 已产出的模型 checkpoint 清单

| 路径 | 对应实验 | 状态 |
| --- | --- | --- |
| `models/base_with_answer_token/qwen3-0.6b` | 方案1：加了 `<answer>`/`</answer>` special token 后的 base 模型（未训练） | 中间产物 |
| `models/sft_coldstart/qwen3-0.6b/global_step_42` | 最初版冷启动 SFT（3 epoch，主线 `<think><answer>` 格式，被 GRPO/PPO/DPO 用作初始化） | **仍在用（下游RL的基座）** |
| `models/sft_coldstart/qwen3-0.6b_v2/global_step_210` | 【epoch数消融】同数据/格式，15 epoch 重训练（0.6B） | 已评估：GSM8K pass@1=40.9%/MATH pass@1=27.9%，未接入下游RL |
| `models/sft_coldstart/qwen3-0.6b_v3/global_step_98` | 【epoch数消融】同数据/格式，7 epoch 重训练（0.6B），当前 pass@1 最优的 SFT 版本 | 已评估：GSM8K pass@1=42.8%/MATH pass@1=31.6%，未接入下游RL |
| `models/sft_coldstart/qwen3-1.7b_v3/global_step_98` | SFT 冷启动（1.7B 规模对比，同数据/格式，7 epoch） | 已补做评估：GSM8K pass@1=61.6%/MATH pass@1=48.4%，明显优于所有 0.6B 版本，未接入下游RL |
| `models/sft_coldstart/qwen3-0.6b-answer-token/global_step_98` | 方案1（special token）+ 曾尝试的loss_mask改动（已验证无效并回退，见第4节）的 7-epoch 重训练 | 已评估，未见改善 |
| `models/sft_coldstart/qwen3-0.6b-answer-token.buggy_20260807_173410/global_step_98` | 方案1、更早的旧 checkpoint | 已废弃，仅作备份 |
| `models/sft_coldstart/qwen3-0.6b-v2-plain-answer/global_step_98` | 方案2（plain answer，不用XML标签）+ 曾尝试的loss_mask改动（已验证无效并回退）的 7-epoch 重训练 | 已评估，未见改善 |
| `models/sft_coldstart/qwen3-0.6b-v2-plain-answer.buggy_20260807_173410/global_step_98` | 方案2、更早的旧 checkpoint | 已废弃，仅作备份 |
| `models/grpo_ckpt/grpo_qwen3_0.6b/global_step_116` | GRPO 主实验最终 checkpoint（4 epoch，116 steps） | 已评估，见 `03_results.md`实验③ |
| `models/grpo_ckpt/grpo_qwen3_0.6b_n4/global_step_116` | 【P1消融】GRPO group size=4 最终 checkpoint（2卡，4 epoch，116 steps） | 已评估，见 `03_results.md`实验③消融 |
| `models/grpo_ckpt/grpo_qwen3_0.6b_n16/` | 【P1消融】GRPO group size=16（2卡，训练中，从 step70 恢复） | 训练中，待补评估 |
| `models/ppo_ckpt/ppo_qwen3_0.6b/global_step_116` | PPO v3 修复后成功训练的最终 checkpoint（critic_warmup=20+截断惩罚+KL=0.01） | **已评估，训练稳定收敛，见 `03_results.md`实验④ v3** |
| `models/ppo_ckpt/ppo_qwen3_0.6b_kl0.005/global_step_116` | 【P1消融】PPO KL系数=0.005 最终 checkpoint（2卡，4 epoch，116 steps） | 已评估，见 `03_results.md`实验④消融 |
| `models/ppo_ckpt/ppo_qwen3_0.6b_kl0.02/global_step_116` | 【P1消融】PPO KL系数=0.02 最终 checkpoint（2卡，4 epoch，116 steps） | 已评估，见 `03_results.md`实验④消融 |
| `models/ppo_ckpt/ppo_qwen3_0.6b_failed_20260807_111909/` | PPO v1 崩溃尝试（response length 爆炸） | 已废弃，仅作复盘备份 |
| `models/ppo_ckpt/ppo_qwen3_0.6b_failed_20260807_200357_run2/` | PPO v2 崩溃尝试（延迟到 step12 崩溃） | 已废弃，仅作复盘备份 |
| `models/dpo_ckpt/qwen3-0.6b` | DPO（基于最初版SFT模型 rejection sampling构造偏好对训练） | 已评估，见实验⑤ |

> 注：`qwen3-1.7b_v3` 目前只完成了 SFT 评估（未做 RL），见下方第5节。

---

## 4. 曾排查的疑似 bug：SFT loss_mask 对齐（已验证无效并回退）

### 4.1 现象
所有版本（最初版冷启动、方案1、方案2）SFT 后模型的 `strict_format_rate`（严格 `<think>...</think><answer>...</answer>` 格式合规率）都接近 0，即使 `has_answer_rate` 高达 94%+、pass@1 也有明显提升，说明模型学会了"做题"但没学会"严格按格式收尾"。

### 4.2 当时的疑点（未能成立）
veRL 官方 `verl/utils/dataset/multiturn_sft_dataset.py` 在多轮消息（`data.multiturn.enable=true`，本项目 SFT 数据用的是这个模式）场景下构造 `loss_mask` 时，用 `loss_mask[start_pos:end_pos] = 1` 标记 assistant 消息所在的 token 区间。

当时推测：训练时 `fsdp_sft_trainer.py` 实际的 loss 计算方式是标准的"错一位"语言模型损失（`shift_logits`/`shift_labels`），因此怀疑 `loss_mask[j]` 应该对齐到"预测 input_ids[j+1]"而不是"input_ids[j] 本身属于 assistant"，从而推测多轮版本漏做了这个对齐，导致 assistant turn 的第一个 token（如 `<think>`）丢失监督。

### 4.3 实际做过的改动（已回退）
按上述疑点，曾把每个 assistant turn 的 mask 区间整体左移一位：

```python
# 【已回退】曾尝试过的改动，不再使用
mask_start = max(start_pos - 1, 0)
is_last_message = i == len(messages) - 1
mask_end = end_pos if is_last_message else end_pos - 1
loss_mask[mask_start:mask_end] = 1
```

并基于这个改动重新跑了方案1、方案2的 7-epoch SFT（见下方 4.4）。

### 4.4 验证结果：改动无效，已回退
1. 把改动前的两份 checkpoint（方案1、方案2）分别重命名备份为 `*.buggy_20260807_173410/`。
2. 用同样的数据、同样的 7 epoch 配置，基于上述改动重新跑了两版 SFT：
   - `scripts/_run_sft_scheme1_fixed.sh` → `models/sft_coldstart/qwen3-0.6b-answer-token/global_step_98`
   - `scripts/_run_sft_scheme2_fixed.sh` → `models/sft_coldstart/qwen3-0.6b-v2-plain-answer/global_step_98`
   （训练日志：`logs/sft_qwen3_0.6b_answer_token_fixed.log`、`logs/sft_qwen3_0.6b_v2_plain_answer_fixed.log`）。
3. 对两版重训练后的模型跑了 `eval_pass_at_k.py` 评估，**`strict_format_rate` 依然没有回升**，说明这个改动并没有命中真正的根因，判断为无效改动。
4. 已用 `git checkout -- verl/utils/dataset/multiturn_sft_dataset.py` 将该文件完全还原为官方原版（当前 `git diff` 已确认为空）。因此 SFT/GRPO/PPO 当前均为 **100% 官方 verl 算法逻辑**，无任何算法层面的自定义。
5. `strict_format_rate≈0` 的真正根因仍未定位，**属于开放问题**，后续需要重新排查方向（例如：评估评分脚本 `rule_reward.py` 中的严格正则匹配规则是否本身过于苛刻、chat template 是否与训练/推理一致、数据本身的 assistant 回答是否真的严格遵守该格式等）。

> 注意：`models/sft_coldstart/qwen3-0.6b/global_step_42`（GRPO/PPO/DPO 使用的初始化权重）未受这次回退影响，因为它从头到尾都是用官方 `multiturn_sft_dataset.py` 训的，与本次回退后的代码状态完全一致。

---

## 5. 已完成实验一览（简表，详细数据见 `03_results.md`）

| # | 实验 | 框架/方法 | 基座 | 状态 | 详情 |
| --- | --- | --- | --- | --- | --- |
| ① | Base baseline | 无 post-training | Qwen3-0.6B-Base | 完成 | 实验① |
| ② | SFT 冷启动（最初版） | veRL `fsdp_sft_trainer` | Qwen3-0.6B-Base | 完成（`strict_format_rate=0`，根因仍开放，见第4节） | 实验② |
| ②' | SFT 冷启动（1.7B 规模对比） | veRL `fsdp_sft_trainer` | Qwen3-1.7B-Base | 完成，已补做评估：**GSM8K pass@1=61.6%/MATH pass@1=48.4%**，全面显著优于同数据/格式训练的所有 0.6B 版本（含 v3 的 42.8%/31.6%），验证了模型规模本身对该任务的巨大收益 | `sft_qwen3_1.7b_v3.log` / `eval_sft_1.7b_v3.log` |
| ②-v2 | 【文档新增】SFT 冷启动（epoch数消融，15 epoch） | veRL `fsdp_sft_trainer` | Qwen3-0.6B-Base | 完成：GSM8K pass@1=40.9%/MATH pass@1=27.9% | `sft_qwen3_0.6b_v2.log` / `eval_sft_v2.log` |
| ②-v3 | 【文档新增】SFT 冷启动（epoch数消融，7 epoch，当前最优SFT） | veRL `fsdp_sft_trainer` | Qwen3-0.6B-Base | 完成：GSM8K pass@1=42.8%/MATH pass@1=31.6% | `sft_qwen3_0.6b_v3.log` / `eval_sft_v3.log` |
| ③ | GRPO | veRL `main_ppo`(adv_estimator=grpo) + vLLM | SFT②权重 | 完成，效果最好最稳定 | 实验③ |
| ③-n4 | 【P1消融】GRPO group size=4 | veRL `main_ppo`(adv_estimator=grpo) + vLLM | SFT②权重 | 完成：GSM8K pass@1=68.0%/MATH pass@1=47.9%，与n=8基本持平甚至略优 | 实验③消融 |
| ③-n16 | 【P1消融】GRPO group size=16 | veRL `main_ppo`(adv_estimator=grpo) + vLLM | SFT②权重 | 训练中（从step70恢复，预计还需1小时左右），待补 | 实验③消融 |
| ④ | PPO | veRL `main_ppo`(adv_estimator=gae) + Critic + vLLM | SFT②权重 | v1/v2**训练崩溃**（response length爆炸）→ **v3 修复成功**：GSM8K pass@1=42.6%/MATH pass@1=31.2% | 实验④ |
| ④-kl0.005 | 【P1消融】PPO KL系数=0.005 | veRL `main_ppo`(adv_estimator=gae) + Critic + vLLM | SFT②权重 | 完成：GSM8K pass@1=61.9%/MATH pass@1=39.7%，全程无length上冲，明显优于v3对照组（但2卡vs4卡存在未受控变量，见实验④消融分析） | 实验④消融 |
| ④-kl0.02 | 【P1消融】PPO KL系数=0.02 | veRL `main_ppo`(adv_estimator=gae) + Critic + vLLM | SFT②权重 | 完成：GSM8K pass@1=62.1%/MATH pass@1=42.9%，全程无length上冲，明显优于v3对照组（但2卡vs4卡存在未受控变量，见实验④消融分析） | 实验④消融 |
| ⑤ | DPO | `trl.DPOTrainer` | SFT②权重 | 完成，提升有限 | 实验⑤ |
| ⑥ | 方案1：`<answer>` special token | veRL `fsdp_sft_trainer` | `base_with_answer_token` | 已评估：`strict_format_rate` 仍为0；曾用 loss_mask 改动重训（见第4节）后依然为0，改动已回退 | `05_debugging_notes.md` + 本文档第4节 |
| ⑦ | 方案2：plain answer（去掉XML标签） | veRL `fsdp_sft_trainer` | Qwen3-0.6B-Base | 已评估：异常低分（has_answer_rate仅1.3%）；曾用 loss_mask 改动重训后未见改善，改动已回退 | 本文档第4节 |
| - | GRPO 冒烟测试 | veRL `main_ppo` | Qwen2.5-0.5B | 仅验证链路可用性，不计入正式对比 | 附录 |

**下一步待办**（未完成，`strict_format_rate≈0` 仍是开放问题）：
1. 重新审视 `strict_format_rate` 评估口径本身：检查 `rule_reward.py` 里 `THINK_ANSWER_PATTERN` 严格正则是否过于苛刻（如空白符、结尾换行等细节导致误判为不合规）。
2. 检查 SFT 训练数据本身：抽样确认 `data/processed/*.jsonl` 中的 assistant 回答是否真的每一条都严格闭合 `<think>...</think><answer>...</answer>`（或 plain answer 变体的对应格式），排除数据本身不干净的可能。
3. 检查生成/评估阶段使用的 chat template 是否与训练时 `verl` 构造 messages 用的模板完全一致（system prompt、特殊token、换行符等）。
4. 待根因定位后再决定是否需要重新修改 `multiturn_sft_dataset.py` 或其他环节，并重新训练评估方案1/方案2。
