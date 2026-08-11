# GRPO 分析

## 为什么选它作为重点算法

GRPO（Group Relative Policy Optimization）是 DeepSeek-Math / DeepSeek-R1 系列工作里用的核心算法，思路是对同一个 prompt 采样一组输出（这里组内大小取 8），用组内 reward 的相对排名/归一化值直接作为 advantage，不需要额外训练一个 critic 网络。相比 PPO，实现更简单，也不用担心 value 函数训练不稳定的问题——这一点在 [ppo_analysis.md](ppo_analysis.md) 里的崩溃复盘中有更详细的对比分析。

## 配置

以 SFT 冷启动 checkpoint（3 epoch 版本）为初始化权重，用 veRL 的 `main_ppo` 入口，把 `algorithm.adv_estimator` 设为 `grpo`，4×GPU FSDP + vLLM 做 rollout。关键超参：`train_batch_size=256`，`ppo_mini_batch_size=64`，`rollout.n=8`（组内采样数），`actor_lr=1e-6`，`kl_loss_coef=0.001`（`low_var_kl`），`max_prompt_length=512`，`max_response_length=1024`，训练 4 个 epoch（116 steps）。Reward 用 `reward/rule_reward.py` 的规则打分（答案正确性 + 格式）。

## 结果

| 数据集 | pass@1 | pass@4 | pass@8 |
| --- | --- | --- | --- |
| GSM8K | 67.7% | 80.4% | 85.0% |
| MATH | 48.8% | 71.5% | 79.2% |

训练过程中 validation reward 从初始 0.129 单调爬升到最终 0.563，`response_length` 全程稳定在 100-125 区间的窄幅波动，没有出现长度失控或 reward 塌陷。相比 SFT only，pass@1 提升幅度是所有方法里最大的（GSM8K +28.8pp，MATH +19.8pp），也是训练最稳定的一个。

训练耗时：4 张卡、4 个 epoch、116 steps，约 2 小时，介于 DPO（低）和 PPO（高，含 critic 前向反向）之间。

## 消融：组内采样数（group size）

其余超参不变，只调 `rollout.n`（4 / 8 / 16），2 卡训练。

| group size | GSM8K pass@1 | GSM8K pass@8 | MATH pass@1 | MATH pass@8 |
| --- | --- | --- | --- | --- |
| 4 | **68.0%** | 86.2% | **47.9%** | 79.8% |
| 8（主实验） | 67.7% | 85.0% | 48.8% | 79.2% |
| 16 | 69.0% | 84.3% | 32.1% | 59.0% |

n=4 跟 n=8 几乎持平，GSM8K 上甚至还略高一点，说明在 GSM8K 这种 reward 信号比较清晰的任务上，4 个样本就足够估计出组内相对排名，不需要堆到 8。而 n=16 在 GSM8K 上继续小幅提升的同时，MATH 上却明显下降（48.8%→32.1%），且训练过程中的 `val reward@1` 是三组里最高的（0.596）——这个组合看起来像是在 GSM8K（训练用的数据集）上过拟合得更充分，但没能等比例迁移到分布不同、难度更高的 MATH 上。

这个结果的实践含义是：更小的 group size 意味着更低的 rollout 开销（生成样本数减半），在算力有限时适当降低 group size 是划算的；而一味增大 group size 未必有收益，还可能以牺牲跨任务泛化为代价，需要结合多个评估集综合判断，不能只看训练任务本身的 reward 曲线。

（n=16 那组训练过程中曾经中断过一次，疑似 OOM 或会话终止，从 `global_step_70` 恢复后继续跑完，不是一次连续无中断的训练，这是需要如实说明的局限性。）

训练曲线见 `docs/images/grpo_group_size_training_curves.png`，柱状图见 `docs/images/grpo_group_size_ablation.png`。

---

原始数据与训练日志路径见 [experiments.md](experiments.md) 实验③及其消融小节。GRPO vs PPO 的完整对比分析见 [ppo_analysis.md](ppo_analysis.md)。
