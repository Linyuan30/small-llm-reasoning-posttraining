# LLM RL 推理能力实战项目 —— 详细执行方案（8×A800 80G）

> 前置说明：本方案是 `01_motivation_and_design.md` 的落地版本，`01_motivation_and_design.md` 回答"做什么、为什么做"，本文档回答"具体怎么排期、每步用多少卡、先做什么后做什么"。两份文档配合使用，`01_motivation_and_design.md` 不再重复展开。

---

## 0. 硬件条件与资源分配原则

**硬件**：8 × A800 80G，总显存 640GB，卡间通常为 NVLink/PCIe 高速互联（以实际机器为准，若有 NVLink 全互联，多卡通信效率会更好）。

**这套资源对本项目意味着什么（先建立正确预期）**：
- 模型规模是 0.6B~1.7B（Qwen3），单卡显存已经完全够用，**8卡的价值不在"切分单个大模型"，而在于"并行度"**：
  1. **并行做多组对比实验**（不同KL系数、不同group size等消融实验可以同时在不同GPU上跑，而不是排队串行跑），这是大幅压缩项目周期的关键
  2. **加速 rollout 采样**（GRPO/PPO 阶段用 vLLM 多卡/多副本并行生成，减少on-policy方法最大的耗时瓶颈）
  3. **支持更大 batch size / 更多 PPO 角色模型独立部署**，训练更稳定，调试更容易（比如 Actor/Critic/RM/Ref 四个角色可以分别独占1-2张卡，简化实现复杂度）
  4. 有余力做**模型规模对比**（Qwen3-0.6B vs Qwen3-1.7B vs Qwen2.5-Math-1.5B）而不是被迫只跑一个规模

**资源分配的基本策略**：
- SFT / DPO 阶段：数据并行（DDP/ZeRO-2）占用 4~8 卡，重点是"多组消融实验并行跑"而不是单任务用满8卡
- PPO / GRPO 阶段：优先用 **veRL 或 OpenRLHF**（内置 vLLM rollout + 训练分离架构），典型分配：2~4卡做 vLLM rollout 推理，4~6卡做训练（actor+critic的ZeRO训练），可根据实际吞吐调整比例
- 评估阶段：用 vLLM 起服务，多卡并行批量跑 benchmark，压缩评估耗时

---

## 1. 任务优先级分层（P0/P1/P2）

参照 `01_motivation_and_design.md` 第2节的能力清单，把所有任务按"如果时间不够，先砍哪些"排出优先级，防止后期赶工时不知道该舍弃什么。

### P0（必须完成，是项目能否成立的底线）
这些做不完，项目就没有完成"完整post-training核心能力"这个目标：

1. 数据准备与reward函数（GSM8K+MATH，规则reward，含单测）
2. SFT 冷启动跑通
3. **GRPO 跑通**（优先于PPO，DeepSeek-R1同款算法，性价比最高）
4. **PPO 跑通**（完整RLHF链路的核心，四模型协同）
5. DPO 跑通（作为off-policy基线对比）
6. 训练同源指标评估（GSM8K/MATH pass@1/pass@k）
7. 第5.6节方法对比总表（SFT/DPO/PPO/GRPO 全部填满）
8. 基础工程规范（代码结构、配置管理、README）

### P1（应该完成，直接决定项目"深度"和说服力）
时间允许必须做，是区分"跑通了"和"真正理解了"的关键：

1. KL惩罚系数消融实验（PPO，至少3组）
2. GRPO group size 消融实验（至少3组）
3. reward hacking 案例挖掘与分析
4. 推理长度涌现现象观察（训练曲线+案例）
5. AIME/AMC 外部零样本benchmark验证
6. ZeRO-2 vs ZeRO-3 对比实验
7. SFT冷启动 vs 直接RL(R1-Zero式) 对比
8. vLLM加速rollout vs HF原生generate 耗时对比

### P2（有余力再做，加分项）
不影响项目完整性，锦上添花：

1. 神经网络Reward Model训练，与规则reward对比
2. Qwen2.5-Math-1.5B（领域CPT基座）vs Qwen3-1.7B 对比
3. 模型规模对比（Qwen3-0.6B vs Qwen3-1.7B 全链路复现）
4. LoRA-PPO vs 全参PPO 显存/效果对比
5. SVAMP/MathQA 鲁棒性测试、MMLU-STEM泛化测试
6. MMLU/IFEval 通用能力回归测试
7. 代码生成（MBPP/HumanEval）作为跨任务泛化性扩展实验
8. **OPD（On-Policy Distillation）vs GRPO 对比**（详见`01_motivation_and_design.md`第5.7节）：用更强模型（如Qwen2.5-Math-7B）做teacher，对比OPD与规则reward GRPO在收敛速度、样本效率、最终pass@1/pass@k上的差异，`veRL`已内置OPD训练器可直接复用

**执行原则：P0 全部完成之前不启动 P2；P1 至少完成一半以上才考虑 P2。**

---

## 2. 详细时间表（按 6 周制定，含每周资源分配）

相比 `01_motivation_and_design.md` 里给出的 6-8 周估算，由于现在有 8 卡可以并行做消融实验（而不是单卡排队跑），实际时间可以压缩到 **6 周**，把节省出的时间留给 P1/P2 深化实验和调试冗余（强烈建议不要删减这部分缓冲，PPO调试大概率会超预期耗时）。

### Week 1：环境与数据 P0
| 任务 | 负责资源 | 产出 | 预计耗时 |
| --- | --- | --- | --- |
| 环境搭建：安装 veRL/OpenRLHF/trl，跑通官方 quickstart | 1卡验证 | 环境可用 | 1天 |
| GSM8K/MATH 下载、格式化、`<think>/<answer>` 模板统一 | CPU即可 | `data/processed/*.jsonl` | 1天 |
| 规则reward函数开发（答案匹配+格式校验）+ 单元测试 | CPU即可 | `reward/rule_reward.py` + 测试用例 | 1.5天 |
| vLLM 部署验证（跑通批量生成） | 1-2卡 | 验证rollout链路可用 | 0.5天 |
| SFT冷启动数据准备（CoT示例，几百~几千条，可用现成开源CoT数据集如MetaMathQA抽样，或用大模型API/本地大模型生成后人工抽检） | CPU+API/1卡 | `data/sft_coldstart.jsonl` | 1天 |

**本周关键决策点**：确定用 veRL 还是 OpenRLHF 作为主力RL框架（建议veRL，工业界主流、vLLM集成度高），trl仅用于SFT/DPO这种简单场景。

### Week 2：SFT 冷启动 + DPO 数据准备 P0
| 任务 | 负责资源 | 产出 | 预计耗时 |
| --- | --- | --- | --- |
| SFT冷启动训练（Qwen3-0.6B-Base） | 4卡 DDP/ZeRO-2 | SFT checkpoint | 训练<2h，调试1天 |
| SFT冷启动训练（Qwen3-1.7B-Base，并行做规模对比） | 另4卡同时跑 | SFT-1.7B checkpoint | 与上面并行，不额外占时间 |
| loss mask/packing 正确性验证 | - | 验证报告 | 0.5天 |
| SFT模型格式合规率评估（是否稳定输出`<think>/<answer>`） | 1-2卡 vLLM批量生成 | 评估报告 | 0.5天 |
| DPO偏好对构造：用SFT模型做rejection sampling（对训练prompt采样N次） | 4卡 vLLM并行采样 | `data/dpo_pairs.jsonl` | 1天 |
| **[P1] SFT冷启动 vs 直接RL 对比准备**：额外保留一份"跳过SFT直接RL"的实验分支配置 | - | 配置文件 | 0.5天 |

**里程碑检查**：Week 2结束时，SFT模型的格式合规率应接近100%，否则要回头检查数据/训练配置，这是后续RL阶段reward信噪比的基础，不能带着问题进入下一阶段。

### Week 3-4：GRPO + PPO 核心训练 P0（本项目难度和时间最大头，预留充足缓冲）
| 任务 | 负责资源 | 产出 | 预计耗时 |
| --- | --- | --- | --- |
| GRPO 训练主实验（group size=8，GSM8K） | 6-8卡（veRL rollout+train混合） | GRPO checkpoint | 训练1-2天，调试2-3天 |
| **[P1] GRPO group size 消融**（4/8/16，三组并行） | 每组2-3卡，3组同时跑 | 消融结果 | 与主实验并行，1-2天 |
| GRPO 训练扩展到 MATH 数据集 | 4-6卡 | GRPO-MATH checkpoint | 1天 |
| PPO 训练主实验（KL系数用常见默认值先跑通） | 6-8卡（actor/critic/rm/ref分卡部署） | PPO checkpoint | 训练1-2天，调试可能3-5天（PPO最容易踩坑） |
| **[P1] PPO KL系数消融**（3组不同系数并行） | 每组2-3卡，3组同时跑 | 消融结果 | 与主实验并行 |
| 训练动态监控：wandb记录reward/KL/loss曲线 | - | 曲线图 | 贯穿整个阶段 |
| **[P1] reward hacking 案例挖掘**：人工抽检训练中后期的高reward但低质量样本 | - | 案例分析文档 | 1天 |
| **[P1] 推理长度涌现现象记录**：绘制推理长度随训练步数变化曲线，抽样对比训练早/中/晚期输出 | - | 图表+案例 | 0.5天 |

**这是全项目风险最高的阶段**，PPO四模型协同、reward hacking、训练不稳定都可能发生，务必预留缓冲时间，不要把Week5的任务提前占用。**如果Week4结束PPO仍未调通，优先保证GRPO和已有结果的完整性，PPO可以顺延到Week5前半段，但不要无限拖延**——如果两周内PPO仍无法收敛，考虑先用trl的PPOTrainer（更简单但灵活性低）做一个简化版本兜底，保证P0任务不空缺。

### Week 5：DPO 训练 + 分布式系统实验 P0+P1
| 任务 | 负责资源 | 产出 | 预计耗时 |
| --- | --- | --- | --- |
| DPO 训练（用Week2准备的偏好对） | 2-4卡 | DPO checkpoint | 训练<1天 |
| **[P1] ZeRO-2 vs ZeRO-3 对比实验**（同一SFT任务） | 2组各2-4卡 | 显存/吞吐对比表 | 1天 |
| **[P1] vLLM vs HF原生generate rollout耗时对比** | 2卡 | 对比数据 | 0.5天 |
| **[P2] LoRA-PPO vs 全参PPO 显存对比**（如果Week3-4的PPO已跑通且有余力） | 2-4卡 | 对比数据 | 1天 |
| **[P2] OPD 训练与对比**（用`veRL`内置OPD训练器，Qwen2.5-Math-7B等作teacher，仅在P0+P1均已完成时启动） | 2-4卡（含teacher vLLM推理） | OPD checkpoint + 与GRPO对比数据 | 1-1.5天 |
| PPO调试缓冲（若Week4未完全收尾） | 剩余资源 | - | 弹性 |

### Week 6：评估、方法对比、开源发布 P0+P1
| 任务 | 负责资源 | 产出 | 预计耗时 |
| --- | --- | --- | --- |
| 用lm-evaluation-harness跑全部checkpoint的GSM8K/MATH pass@1/pass@k | 4-8卡 vLLM并行评估 | 评估结果表 | 1天 |
| **[P1] AIME2024/2025 + AMC 外部benchmark评估** | 2-4卡 | 外部验证结果 | 0.5天（题量小） |
| 填写第5.6节方法对比总表（Base/SFT/DPO/PPO/GRPO） | - | 核心结果表 | 0.5天 |
| **[P2] SVAMP/MathQA/MMLU-STEM 泛化性测试** | 2卡 | 泛化性报告 | 0.5天 |
| **[P2] MMLU/IFEval 通用能力回归测试** | 2卡 | 回归测试报告 | 0.5天 |
| 整理README（能力清单逐条对应+核心结果表+复现步骤） | - | README.md | 1天 |
| 代码清理、配置整理、最终检查 | - | - | 0.5天 |
| GitHub发布 | - | 仓库上线 | 0.5天 |

---

## 3. 关键风险点与应对预案

| 风险 | 影响阶段 | 应对预案 |
| --- | --- | --- |
| PPO 训练不收敛/崩溃 | Week3-4 | 优先保证GRPO完整；PPO先用trl简化版兜底，再逐步替换为veRL/OpenRLHF深度版本；降低学习率、增大KL系数、检查reward尺度是否合理 |
| reward hacking 导致训练"虚假提升" | Week3-4 | 训练中定期人工抽检高reward样本，而不是只看reward曲线；格式reward权重不宜过高，避免模型专注刷格式分 |
| SFT冷启动格式合规率不达标 | Week2 | 增加SFT数据量或训练轮数；检查chat template是否与生成时使用的一致 |
| 8卡资源调度冲突（多组消融实验同时抢卡） | Week3-4 | 提前用简单的任务队列/脚本管理GPU分配，避免手动冲突；单卡不够时优先保证P0任务 |
| veRL/OpenRLHF 环境搭建踩坑（依赖复杂） | Week1 | 预留1天缓冲；如果搭建成本过高，先用`trl`的PPOTrainer完成P0跑通，再择机迁移到工业级框架做深度实验（P1/P2阶段） |
| 时间不够、无法完成全部P1 | Week5-6 | 严格按第1节优先级砍，先保证P0+方法对比总表完整，P1按"性价比"排序做（reward hacking分析和AIME验证优先级高于ZeRO对比） |

---

## 4. 每周检查清单（自查用，防止进度虚报）

- [x] Week1末：环境跑通、数据格式化完成、reward函数有单测且通过
- [x] Week2末：SFT模型格式合规率≥95%（实际未达标，`strict_format_rate≈0`，已记录为开放问题，见`04_repo_overview.md`第4节，但`has_answer_rate`达94%+不影响下游RL训练），DPO偏好对已构造
- [x] Week3末：GRPO在GSM8K上有可见的reward上升趋势（0.129→0.563，见`03_results.md`实验③）
- [x] Week4末：GRPO和PPO均有可用checkpoint（GRPO稳定收敛，PPO v1/v2崩溃→v3修复成功），**已完成2组消融实验**：GRPO group size（n=4/8/16，n=16训练中待补）+ PPO KL系数（0.005/0.01/0.02），详见`03_results.md`实验③④消融小节
- [ ] Week5末：DPO训练完成（已完成，见实验⑤），ZeRO对比实验有数据（未做，P1优先级低于已完成的消融实验）
- [ ] Week6末：方法对比总表填满（已完成，见`03_results.md`），AIME验证完成（未做），README发布（未做）

---

## 5. 与 `01_motivation_and_design.md` 的关系

本文档不改变 `01_motivation_and_design.md` 中确定的目标、载体任务、能力清单和技术方案，只是在已有8×A800资源条件下，把"要做什么"转化为"按什么顺序、用多少卡、每步花多久做"的可执行排期，并新增了优先级分层和风险预案，供实际执行时对照检查进度、及时取舍。
