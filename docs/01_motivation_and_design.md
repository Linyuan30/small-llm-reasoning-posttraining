# LLM Post-Training 核心能力实战项目（数学推理方向）

## 0. 项目的真正目标（先钉死，后面所有设计都服务于它）

**这不是一个"做出好用的产品"的项目，也不是任何垂直业务场景的落地项目。**

**唯一目标**：通过一个完整、真实、可复现的项目，系统性地锻炼并展示自己具备工业界大模型岗位（Seed、青云计划等）对 **post-training 工程师**的核心能力要求。载体任务本身做得好不好是次要的，**能力点是否被扎实覆盖、能否清晰展示出来、任务和方法是否真正匹配（fit），才是唯一评价标准**。

所以后面每一个设计决策，判断标准只有一个：**这样做是否能体现/锻炼一个真实的 post-training 核心能力点？** 不为了"省事"绕开难点——难点恰恰是能力体现的地方。

---

## 1. 载体任务选择：为什么是数学推理，而不是内容安全分类

项目最初考虑用已有的中文内容安全分类数据集（`sft/v2/by_label`）作为载体，但经过分析发现一个根本性问题：

> **内容安全分类是"有唯一确定标准答案"的分类任务，SFT 的监督信号和 RL 的 reward 信号本质上是同一个信息源，RL 很难带来真正的增量价值。** 如果载体任务本身对 RL"不 fit"，那么无论 RLHF 链路做得多完整，都只是"为了做而做"，练出来的经验含金量会打折扣，也很难在面试中讲出一个自洽的故事。

既然本项目的目的是**纯粹的能力练习和展示**，不受限于任何具体业务场景，就应该直接选择**业界公认真正适合 RL 优化、且已被标杆工作验证过**的任务。按"reward 是否可靠、最优路径是否非唯一"这个标准筛选：

| 方向 | 是否适合RL | 代表数据集 | 评价 |
| --- | --- | --- | --- |
| **数学推理** | 高：最终答案唯一可自动判分，但解题路径/推理链不唯一，RL能push模型探索更优路径 | GSM8K、MATH、AIME | **首选**：reward最干净，验证成本≈0，是DeepSeek-R1-Zero/Qwen2.5-Math等标杆工作的同款范式，业界认可度最高，故事性强 |
| 代码生成 | 高：单测通过与否可自动验证 | MBPP、HumanEval、APPS | 很好的备选/扩展方向，但需要额外搭建代码执行沙箱 |
| 内容安全分类（原方案） | 低：标准答案唯一，RL与SFT信息源重叠 | 内部数据 | 已放弃，且有数据合规负担 |

**最终选择：数学推理，对标 DeepSeek-R1 的技术路线（SFT冷启动 → RL训练 → 观察推理能力涌现）。**

这个选择额外带来两个好处：
1. **完全公开数据集，无合规负担**：GSM8K/MATH 都是标准学术数据集，可以直接把数据、代码、结果全部发布到 GitHub，不需要任何脱敏和内部合规审批流程
2. **能观察到RL真正的价值现象**：如推理长度随训练增长、自我验证/回溯行为涌现、pass@k提升——这些是分类任务不可能观察到的，也是目前RL for LLM领域最受关注的话题，能让项目更有"研究感"而不只是"调库跑通"

---

## 2. 目标能力清单（项目必须覆盖的核心能力点，不变）

| 模块 | 能力点 | 本项目如何体现 |
| --- | --- | --- |
| **数据能力** | 数据处理、SFT冷启动数据构造（含CoT格式化）、rollout数据管理 | 第4节 |
| **训练框架/分布式系统** | ZeRO-1/2/3、混合精度、gradient checkpointing、显存与吞吐分析 | 第6节 |
| **SFT** | loss mask、packing、CoT格式统一 | 第5.1节 |
| **Reward Model / 规则Reward** | 基于规则的可验证reward设计、（可选）RM训练对比 | 第5.2节 |
| **PPO（完整RLHF）** | actor/critic/reward/reference 四模型协同、KL惩罚、reward hacking识别 | 第5.3节 |
| **GRPO** | 无需Critic的组内相对优势估计，DeepSeek-R1同款算法 | 第5.4节 |
| **DPO对比** | off-policy vs on-policy 方法对比 | 第5.5节 |
| **OPD（On-Policy Distillation，扩展）** | 在学生自采样状态上用教师token级分布做稠密监督，融合KD与on-policy RL的优点 | 第5.7节 |
| **评估方法论** | pass@k、推理长度/涌现行为分析、消融实验设计 | 第7节 |
| **推理工程认知** | vLLM加速PPO/GRPO的rollout采样 | 第6.3节 |
| **工程规范** | 代码结构、配置管理、实验记录、可复现性、开源发布 | 第8节 |

---

## 3. 基座模型与数据集

- **基座模型**：Qwen3-0.6B/1.7B（Base 版本，非 Instruct，更能体现"从无推理能力到有"的post-training价值）。也可选 Qwen2.5-Math-1.5B 作为对比（该模型已经过数学语料继续预训练，可以对比"是否有领域CPT基础"对最终RL效果的影响，这本身也是个好实验）。
- **数据集**：
  - **GSM8K**（小学应用题，约7.5K训练样本）：入门级，SFT和RL都先在这个数据集上跑通
  - **MATH**（竞赛数学，约7.5K/12K训练样本，难度分级）：验证方法在更难任务上的表现，且可以按难度分层做实验
  - 两者都是HuggingFace `datasets`可直接下载的标准公开数据集

---

## 4. 数据处理与格式设计（对应"数据能力"模块）

### 4.1 统一输出格式（对标DeepSeek-R1）
要求模型按固定格式输出，便于程序化提取答案计算reward：

```text
<think>
（推理过程）
</think>
<answer>
（最终数值答案）
</answer>
```

### 4.2 SFT冷启动数据（可选但建议做，对标R1两阶段范式）
- 用少量（几百到几千条）人工/模型生成的高质量CoT解题过程，做格式统一化的SFT，让模型在进入RL前就具备稳定的输出格式和基本推理习惯
- 目的不是让模型"学会做题"（这是RL阶段的任务），而是解决"冷启动"问题：如果直接对base模型做RL，可能连基本格式都输出不稳定，导致reward信号噪声很大、训练不稳定
- 这一步本身是一个值得记录的对比实验：**有无SFT冷启动，对RL阶段收敛速度/稳定性的影响**（DeepSeek-R1的论文里，R1-Zero就是跳过这一步直接RL，出现了输出可读性差等问题，你可以复现类似的对比观察）

### 4.3 Reward 设计（核心，规则可验证）
- **正确性 reward**：提取 `<answer>` 中内容，与标准答案做数值/字符串匹配，正确+1，错误0（或按MATH的复杂答案做规范化匹配，比如分数、根号表达式等价判断）
- **格式 reward**：是否严格符合 `<think>...</think><answer>...</answer>` 格式，符合给小额加分，不符合给惩罚
- 两者加权组合成最终 reward，权重设计本身也是一个值得记录的消融点

---

## 5. 训练阶段设计（核心，逐个模块对应能力点）

### 5.1 阶段一：SFT冷启动
- 用 `trl` 的 `SFTTrainer` 或 `LLaMA-Factory` 跑通
- 关键工程点：loss mask（只在think+answer部分算loss）、packing、格式一致性检查

### 5.2 阶段二：Reward 机制
- 主方案：**基于规则的reward**（见4.3节），不需要训练神经网络RM，直接程序化判断，这是数学任务相比内容安全任务的一大优势——reward完全客观、无需近似
- **可选进阶实验**：额外训练一个神经网络Reward Model（用规则reward自动生成的偏好对来训练），对比"规则reward" vs "学习到的RM"在下游PPO效果上的差异，这个对比能体现你对RM存在意义的深入理解（规则reward在数学任务上通常已经足够好，但复杂任务往往需要学习型RM，这个认知本身值得写进报告）

### 5.3 阶段三：PPO（完整RLHF核心）
- 四个模型：Actor、Critic、Reward（规则或RM）、Reference（KL基准，通常是SFT模型）
- 框架：`trl`的`PPOTrainer`，或`OpenRLHF`/`veRL`（工业级框架，架构本身值得学习）
- 核心工程细节（务必逐一实验记录）：
  - **KL惩罚系数**：至少3组对比，观察对训练稳定性和最终准确率的影响
  - **reward hacking识别**：数学任务上典型的hacking现象——比如模型学会只输出格式正确但答案敷衍、或者复制题目中的数字凑答案，需要主动去找、去展示至少一个真实案例
  - **advantage estimation（GAE）**：gamma/lambda对训练的影响
  - **训练稳定性**：KL divergence、reward曲线、value loss曲线，记录是否崩溃及排查过程
  - LoRA-PPO vs 全参PPO的显存/效果对比

### 5.4 阶段四：GRPO（DeepSeek-R1同款算法，重点）
- 原理：对同一个prompt采样一组（如8个）输出，用组内reward的相对排名/归一化值作为advantage，**不需要Critic模型**，比PPO更省显存、实现更简单
- 这是当前最值得深入实践的算法，直接对标DeepSeek-R1-Zero的核心方法
- 实验点：
  - 组内采样数量（group size）对效果的影响（如4/8/16对比）
  - 观察训练过程中**推理长度是否自发增长**（R1论文中的标志性现象——"aha moment"，模型自发学会更长的思考和自我修正）
  - 与PPO对比：相同计算预算下，GRPO和PPO谁收敛更快、最终效果谁更好

### 5.5 阶段五：DPO 对比（作为off-policy基线）
- 用SFT模型对训练集做rejection sampling（多次采样，按reward排序取chosen/rejected）构造偏好对，训练DPO
- 与PPO/GRPO对比，验证"on-policy RL 在这类可验证reward任务上是否显著优于off-policy DPO"——这次的任务是真正适合RL的场景，这个对比应该能看到比之前（内容安全任务上）更明显的差距，这本身是很好的呼应和验证

### 5.6 方法对比总表（本项目最核心的产出）

| 方法 | 是否需要RM | 是否需要Critic | on-policy | GSM8K pass@1 | MATH pass@1 | 平均推理长度 | 训练成本 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Base（无post-training，Qwen3-0.6B-Base） | - | - | - | 4.8% | 3.7% | ? | - |
| SFT only | 否 | 否 | - | 38.9% | 29.0% | ? | 低 |
| SFT+DPO | 否 | 否 | 否 | 39.0% | 29.8% | ? | 低 |
| SFT+PPO | 是 | 是 | 是 | 0.0% | 0.0% | ? | 高 |
| SFT+GRPO | 是/规则 | 否 | 是 | 67.7% | 48.8% | ? | 中 |
| SFT+OPD（需强teacher，见5.7节） | 否（teacher替代RM） | 否 | 是 | ? | ? | ? | 中 |

**Base 行数据来源**：`eval/results/baseline_qwen3_0.6b_base.json`，GSM8K/MATH 各测试集抽样500条、temperature=0.8/top_p=0.95、每题采样8次。完整数据：GSM8K pass@1=4.8% / pass@4=17.4% / pass@8=30.6%；MATH pass@1=3.7% / pass@4=13.8% / pass@8=24.8%。pass@8远高于pass@1，说明Base模型已经具备一定概率生成正确答案的能力，但格式/稳定性不足以在greedy解码下稳定复现——这正是SFT冷启动和RL阶段要解决的问题，也是本项目post-training价值的起点参照。

跑完把"?"填上，这张表 + 背后的分析就是项目最有含金量的成果展示。

### 5.7 阶段六：OPD（On-Policy Distillation，扩展实验，对应P2）

- **背景与动机**：SFT/标准KD 在教师采样的轨迹上训练，存在 exposure bias（训练时学生看到的是教师的状态分布，推理时学生走自己的轨迹，两者不一致）；PPO/GRPO 等 on-policy RL 虽然状态分布对齐，但 reward 是稀疏的序列级信号。**OPD 的核心思路**：让学生在自己采样的轨迹（state）上，用更强的 teacher 模型提供 token 级 log-prob 作为稠密监督信号，兼具"状态对齐"和"密集监督"两个优点。这是 Qwen3、GLM 等近期技术报告中出现的后训练新范式，值得作为本项目 GRPO 的一个平行对比实验。
- **与本项目已有方法的关系**：
  - vs SFT/标准KD：OPD在学生自己的rollout上训练，而不是teacher的demonstration上，减少exposure bias
  - vs GRPO/PPO（RLVR）：OPD用teacher的token级分布替代稀疏的规则reward，信号更稠密，样本效率通常更高，但依赖是否存在一个足够强的teacher模型
- **实验设计（可选，需要一个更强的teacher模型，如 Qwen2.5-Math-7B/72B 或更大参数模型）**：
  - 主实验：以SFT后的Qwen3-0.6B/1.7B模型为student，Qwen2.5-Math-7B（或更强模型）为teacher，在GSM8K/MATH上做OPD训练
  - Loss变体对比：GKD OPD（直接最小化teacher/student分布KL，蒸馏信号更充分）vs PG OPD（用reverse KL的单样本估计作为reward，走policy gradient更新，实现更轻量）
  - 与GRPO对比：相同计算预算下，OPD和GRPO谁收敛更快、最终pass@1/pass@k谁更高，这是最有故事性的对比点
  - （可选）Multi-Teacher OPD：如果同时保留GSM8K和MATH两个不同风格的teacher/专家模型，观察多teacher蒸馏对模型综合能力的影响
- **工程实现**：`verl` 已内置OPD训练器（`examples/on_policy_distillation_trainer/`，配置见 `verl/docs/algo/opd.md`），可直接复用其teacher资源池、vLLM teacher推理服务、`distillation_ppo_loss` 等能力，无需从零实现

---

## 6. 分布式训练与系统能力（对应"训练框架"模块）

1. **ZeRO对比**：单卡baseline vs DeepSpeed ZeRO-2 vs ZeRO-3，记录显存、吞吐、最大batch size
2. **混合精度与显存优化**：bf16、gradient checkpointing的显存-速度权衡
3. **PPO/GRPO专项**：分析多模型同时驻留显存的分布，LoRA优化对比
4. **推理加速（重点）**：GRPO/PPO的rollout阶段用**vLLM**加速采样——这一点在数学推理RL任务上格外重要，因为每个prompt要采样多次（GRPO的group sampling），rollout是主要耗时环节，用vLLM相比HF原生generate能大幅提速，这是目前工业界RLHF训练加速的标准实践（OpenRLHF/veRL等框架都内置了vLLM集成），务必实测对比耗时差异

所有实验用wandb/tensorboard记录loss、reward、KL、推理长度、显存、吞吐曲线。

---

## 7. 评估体系（对应"评估方法论"能力）

只在训练同源的数据集（GSM8K/MATH自身测试集）上评估说服力有限，容易被质疑"自证"或过拟合训练分布。完整的评估体系需要三层：**训练同源指标 + 标准化评测框架 + 外部零样本benchmark**。

### 7.1 训练同源指标（基础验证）
- GSM8K/MATH 测试集上的 **pass@1**、**pass@k**（k=4/8），衡量模型"是否有能力生成正确答案"而不只是"greedy解码是否正确"，这是数学推理评估的标准做法

### 7.2 标准化评测框架（不要自己从零写评测逻辑）
- 优先使用 **`lm-evaluation-harness`**（EleutherAI）：业界最通用的LLM评测框架，内置GSM8K/MATH/MMLU等标准任务的答案抽取和评分逻辑，经过社区广泛验证，用它评测比自己手搓脚本更可信、更省事
- 数学答案匹配建议使用 **Math-Verify** 或参照 DeepSeek/Qwen2.5-Math 官方开源的评测脚本，处理分数、根号、多种等价表达式的规范化匹配（简单字符串匹配在数学任务上误差很大，会直接影响pass@k的准确性）
- 生成阶段统一用 **vLLM** 批量采样，兼顾评测效率和与训练时rollout逻辑的一致性

### 7.3 外部零样本 Benchmark（核心：证明泛化能力，不是只在自己练习的题目上考得好）

| Benchmark | 说明 | 用途 |
| --- | --- | --- |
| **AIME 2024/2025** | 美国数学邀请赛真题，难度远高于GSM8K/MATH，题量小（约30-90题）但区分度极高 | **零样本外部验证的黄金标准**，DeepSeek-R1、OpenAI o1的技术报告都以此为核心汇报指标，务必测 |
| **AMC** | 难度介于GSM8K和AIME之间 | 补充难度梯度，观察方法在不同难度上的提升是否一致 |
| **SVAMP / MathQA** | 应用题的结构变体（换个说法/换个数字顺序） | 鲁棒性测试：验证是否只是记住了GSM8K的题目模式，而非真正学会推理 |
| **MMLU-STEM子集** | 通用知识中的数理部分 | 验证数学能力是否能泛化到知识型题目，而不局限于应用题范式 |

### 7.4 通用能力回归测试（防止灾难性遗忘）
- **MMLU / MMLU-Pro**：验证长期做数学RL训练后，模型的通用知识能力是否受损
- **IFEval**：验证指令遵循能力是否因为长期被约束输出固定的`<think>/<answer>`格式而退化

### 7.5 推理行为与训练动态分析
- 推理长度分布随训练的变化曲线
- 抽样人工检查：是否出现自我验证/回溯/重新计算等"类推理"行为（对应R1的"aha moment"现象），找到具体案例展示
- reward hacking 案例分析

### 7.6 方法对比总表
第5.6节的核心对比表，**建议把AIME/AMC等外部benchmark结果也并入这张表**，而不只是GSM8K/MATH的同源指标，这样对比结论才有充分说服力。

---

## 8. 工程规范与发布

```text
llm-rl-reasoning/
├── README.md                     # 项目介绍、能力清单对照表、结果对比表、复现步骤
├── configs/
│   ├── sft.yaml
│   ├── ppo.yaml
│   ├── grpo.yaml
│   └── dpo.yaml
├── data/
│   └── scripts/
│       ├── prepare_gsm8k.py
│       ├── prepare_math.py
│       ├── format_cot.py            # 统一 <think>/<answer> 格式
│       └── build_dpo_pairs.py       # rejection sampling 构造偏好对
├── reward/
│   └── rule_reward.py               # 答案匹配 + 格式校验
├── training/
│   ├── run_sft.py
│   ├── run_ppo.py
│   ├── run_grpo.py
│   ├── run_dpo.py
│   └── ds_config/
├── eval/
│   ├── eval_pass_at_k.py
│   ├── analyze_reasoning_length.py  # 推理长度/涌现行为分析
│   └── analyze_reward_hacking.py
├── scripts/                          # 各阶段启动脚本
└── docs/
    ├── 01_motivation_and_design.md                       # 本文档
    └── experiment_log.md             # 所有实验记录与结论
```

数据集和代码均可完全公开，无需任何脱敏处理。

---

## 9. 里程碑（建议6-8周）

| 周次 | 目标 |
| --- | --- |
| Week 1 | 数据准备（GSM8K/MATH下载与格式化）、reward函数开发与单测、环境搭建（trl/OpenRLHF/veRL选型） |
| Week 2 | SFT冷启动跑通，验证格式稳定性；对比"有无SFT冷启动"对后续RL的影响 |
| Week 3-4 | GRPO 跑通（优先于PPO，实现更简单），完成group size消融、推理长度涌现现象观察 |
| Week 5 | PPO 跑通，完成KL系数消融、reward hacking案例分析，与GRPO做对比 |
| Week 6 | DPO 跑通，完成on-policy vs off-policy的完整对比（第5.6节总表） |
| Week 7 | 分布式训练专项实验（ZeRO对比、vLLM加速rollout实测）、pass@k与泛化性评估 |
| Week 8 | 整理README（能力清单逐条对应）、清理代码、开源发布 |

---

## 10. 需要掌握/查阅的关键资料

- 框架：`trl`（入门）、`OpenRLHF`、`veRL`（工业级，内置vLLM加速rollout，强烈建议至少用一次）、`LLaMA-Factory`（SFT便捷工具）
- 论文：
  - InstructGPT（PPO范式奠基）
  - DeepSeekMath（GRPO算法原始出处）
  - DeepSeek-R1技术报告（本项目直接对标的范式：冷启动SFT→RL→观察推理涌现）
  - DPO原论文
  - On-Policy Distillation相关：Agarwal et al. "On-policy distillation of language models"（GKD，ICLR 2024）；Thinking Machines Lab博客"On-Policy Distillation"（2025）；`verl` 文档 `docs/algo/opd.md`（本项目OPD扩展实验的直接工程参考）
- 系统：DeepSpeed ZeRO论文、vLLM PagedAttention原理
- 数据集：GSM8K、MATH（HuggingFace datasets 可直接加载）

---

## 11. 与之前版本的差异说明

之前版本以内部内容安全分类数据集为载体任务，经过讨论发现该任务对RL"不fit"（标准答案唯一，SFT和RL信息源重叠），且存在数据合规负担。本版本做出以下核心调整：

1. **载体任务从"内容安全分类"切换为"数学推理"（GSM8K/MATH）**：选择业界公认真正适合RL优化、已被DeepSeek-R1等标杆工作验证过的任务类型，让RLHF/RL链路的每一步都有真实的价值支撑，而不是"为了做而做"。
2. **完全放弃内部数据集**：不再有任何数据脱敏、合规审批负担，数据、代码、结果可以100%开源。
3. **新增GRPO作为重点算法**（对标DeepSeek-R1-Zero），并强调观察"推理长度涌现"等RL特有现象，这是分类任务上完全观察不到、但目前学界/工业界最受关注的现象。
4. **新增SFT冷启动 vs 直接RL的对比实验**，复现R1论文中讨论过的关键设计选择。
5. **评估体系改为pass@1/pass@k标准数学推理评估范式**，并增加泛化性验证（跨数据集测试），比单纯的分类准确率更能体现真实推理能力。
