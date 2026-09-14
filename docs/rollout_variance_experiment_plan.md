# G1 rollout 方差验证后的实验计划

日期：2026-09-14。状态：实验设计；逐文档反事实 baseline 已实现并提供[使用说明](document_counterfactual_baseline.md)，正式 G1 的 baseline/reward 对比诊断尚未执行。

## 1. 目标与当前进度

下一阶段要回答：**如果 rollout 随机性确实造成了较大的训练波动，能否通过更丰富的 reward 或更精细的文档信用分配，提高梯度信噪比，并改善完整训练的稳定性和检索质量？**

优先顺序为：完成当前随机性隔离对照 → 固定状态比较 reward → 固定状态比较逐文档 baseline → 对有证据支持的方案做完整训练重复。前两类诊断不需要重新训练模型。

| 工作 | 当前状态 | 能回答的问题 |
|---|---|---|
| 同时改变训练/data seed 的三个历史 run | 已完成，最终 BRIGHT 差距较大 | 原配方对整体训练随机性敏感 |
| 固定模型和 batch，重复 rollout 梯度诊断 | 用户反馈支持高噪声怀疑；具体 JSON 尚未导入本地核对 | 给定状态下，rollout 带来多少梯度噪声 |
| 固定训练/data seed 42，仅改变 rollout seed 42/3407/2026 | 用户反馈正在运行 | rollout 随机性是否足以造成最终训练结果差距 |
| Reward 与逐文档 baseline 的受控比较 | 本文规划 | 哪种改动能改善估计，以及是否转化为训练收益 |

完整训练的 seed 差距与固定状态的梯度方差是两个不同层面的证据。前者大，不能自动证明后者是唯一机制；后者大，也不必然意味着最终成绩不稳定。具体已有证据及其限制见[讨论复盘](rollout_variance_discussion.md)。

## 2. 当前对照结束后如何判断

当前三个 run 都从 E0 开始，训练/data seed 固定为 42，使用新隔离 RNG 的 rollout seed 42/3407/2026。历史全局 RNG 的 seed-42 成绩不能作为其中一行复用。

结果汇总至少包括：三个 seed 各自的 BRIGHT 宏平均、均值、样本标准差、最小值与最大值；12 个 subset 逐列展示，沿用 [G1_RESULTS.md](../paper/G1_RESULTS.md) 的顺序。训练曲线用于辅助定位差异何时出现，不按 BRIGHT 结果事后挑 checkpoint。

| 当前对照结果 | 可以推进的判断 | 后续行动 |
|---|---|---|
| 固定 data 后，最终结果仍明显分散；固定状态诊断也显示较强相对噪声 | rollout 随机性足以造成训练分化，降低估计噪声值得优先尝试 | 按第 3–5 节筛选改进 |
| 最终结果接近，但固定状态噪声较大 | 局部梯度噪声存在，其下游影响可能被平均或训练过程缓解 | 仍可比较 reward；降低“噪声导致最终不稳定”的主张强度 |
| 固定 data 后，最终差距明显小于历史重复 | rollout 单因素不足以复现原有差距 | 后续考虑数据顺序及其与 rollout 的交互，不能先认定 rollout 是主要来源 |

这里只用三个 seed，不设事后挑选的“显著波动”阈值，也不将两组三 seed 的标准差之比解释为 rollout 的方差贡献率。当前对照能检验 rollout 是否足以造成分化；量化各随机性来源的贡献需要更完整的交叉设计。

## 3. 第一阶段：固定状态比较 reward

### 3.1 比较矩阵

保持当前 group-level leave-one-out（LOO）估计器，比较三种 reward：

| 条件 | Reward | 标签 | 主要比较意义 |
|---|---|---|---|
| R0 | MRR@10 | binary | 当前控制 |
| R1 | nDCG@10 | binary | 与 R0 比较指标本身带来的变化 |
| R2 | nDCG@10 | graded | 与 R1 比较 teacher 分级标签方案带来的变化 |

不能仅比较 R0/R2 后将差异全部归因于“reward 档位更多”：当前 graded 标签同时改变了相关性信息和非零标签覆盖。Binary nDCG 是用于解释机制的必要中间对照，不要求先为它重跑完整训练。

### 3.2 固定哪些条件

- 模型状态先选 **E0 和一个预先指定的最终模型**。最终模型默认选择当前 `G1-A-MRR090-Rollout42` 的最终权重，不按三个 seed 的成绩选模型；训练结束前可以先做 E0 诊断。最终 run 根目录为 113 步权重，`checkpoint-100` 没有特殊理论地位。
- 每个状态使用相同的样本 ID、query/document token 输入、文档顺序、候选 mask 和 batch 划分；graded 条件仅按设计改变标签。包含标签的整体 batch hash 可以不同。
- 固定 G=32、alignment=0.90、product rollout、双侧动作、无 advantage normalization、document log-prob sum、frozen-candidate rescaling、KL=0，以及精度和 dropout 设置。
- 每个条件使用相同的诊断 step 和 rollout seed 列表。优先复用同一批实际采样动作；若分别运行，须核对 RNG 调用路径或动作摘要一致，仅 seed 相同不保证动作一致。
- 默认使用 16 次独立 rollout 重复。先复用已有诊断的 batch；没有预定 batch 时，沿用脚本的 microbatch probe 0/1/2。主要筛选还要检查八个 microbatch 平均的 probe 0/8/16，保留各自 size-16 的候选池。

16 次重复是固定状态下的采样次数，不是完整训练次数。八 microbatch 平均用于接近 global batch 128 的平均结构，仍不等同于逐位重放八卡训练。命令及现有脚本限制见[诊断使用文档](rollout_rng_diagnostics.md)。

### 3.3 记录与比较

设固定状态下有 N 个完整参数梯度估计，定义：

\[
\bar g=\frac1N\sum_{s=1}^N g_s,\qquad
\widehat V=\frac1{N-1}\sum_{s=1}^N\|g_s-\bar g\|^2.
\]

核心报告 `noise_rms` = √V̂、`mean_gradient_norm` = ‖ḡ‖、`noise_to_mean_ratio` = √V̂/‖ḡ‖，以及 `mean_pairwise_cosine`、零梯度采样数。按 checkpoint 和 probe 分别报告，再汇总条件间变化，不先混合不同 batch 的梯度来计算 rollout 方差。

同时报告 reward 的 `n_distinct`、query/document `degenerate_frac` 和边际 reward 标准差；有需要时增加档位频率或熵。这些解释反馈结构，不能替代梯度噪声测量。

判断时同时检查以下问题：

1. 绝对噪声下降是否只是 reward/梯度尺度变小？相对噪声与方向一致性是否也改善？
2. 改善是否跨多个 probe 出现，并保留在八 microbatch 平均之后？
3. E0 与最终模型是否一致，还是只改善训练的某一阶段？
4. 均值是否接近零，使噪声比值不稳定？有限 N 下 ‖ḡ‖² 本身也含有采样噪声，不能把这个比值当作精确的总体信噪比。

若筛选结果模糊，可在同一预定状态增加独立 rollout 重复以检查稳定性，不靠更换 batch 寻找有利结果。不同 reward 的期望梯度本来可能不同，不要求 R0/R1/R2 的平均方向相同。Graded nDCG 的 reward 更丰富是已有线索，降低梯度相对噪声仍是待检验假设。

## 4. 第二阶段：逐文档反事实 baseline

### 4.1 要改变的是哪一层

当前 query/document 两侧已分别对另一动作轴取平均，再计算 LOO advantage。这部分保留。候选改动针对文档组内部：当前一个 document advantage 加权整个文档组的 log-prob 之和，拟改为每个文档使用自己的反事实 reward 差值。

记 qᵢ 为第 i 个 query 动作，Dⱼ=(eⱼ₁,…,eⱼₘ,…,eⱼM) 为第 j 组文档动作，Rᵢⱼ=r(qᵢ,Dⱼ)。其他固定候选及 mask 在记号中省略。

首个候选版本：只把文档 m 的动作替换成该文档的确定性单位方向 μₘ，其他动作不变：

\[
B_{ijm}=r(q_i,e_{j1},\ldots,\mu_m,\ldots,e_{jM}),
\qquad
A^D_{jm}=\frac1G\sum_i(R_{ij}-B_{ijm}).
\]

文档侧使用：

\[
\hat g_D=\frac1G\sum_j\sum_m
\operatorname{stopgrad}(A^D_{jm})\nabla_\theta\log\pi_\theta(e_{jm}).
\]

上式省略现有 batch 与组件 loss 的公共系数，实现时保持其约定。Query 侧继续使用原有边际化 LOO；文档侧替换原 LOO advantage，不能直接在已经中心化的 advantage 上再减一次 B。

当采样分布按文档独立分解时，Bᵢⱼₘ 不依赖当前文档自己的动作，因此条件期望满足 E[Bₘ∇ log π(eₘ)]=0。对 baseline/advantage 停止梯度后，这种减法保持原文档 score-function 项的期望；共享 encoder 参数不破坏该条件恒等式。这里讨论的是当前采样策略的估计器，不额外声称它等同于对整个训练计算图完全求导。

### 4.2 首版需要明确的边界

- μₘ 是 vMF 的单位均值方向，实际期望动作是 A_d(κ)μₘ。采用单位方向是一个具体 baseline 选择，并不等于计算了条件期望 reward，也不保证最小方差。
- 只替换目标文档对应的分数；保留其他 sampled/frozen 候选、标签、mask 和现有 rescaling。需明确目标文档替换分数的尺度，不能因切换代码分支而顺带改变其他候选的校准。
- 当前默认 in-batch 文档使用 frozen 表示。首版按现有动作影响范围实现，不在这次改动中切换为 sampled in-batch 文档；若扩大动作影响范围，需要重新检查 reward 与梯度归属。
- 正/零/负差值表示相对该参考动作的影响，不是文档之间唯一的因果贡献分解；文档交互仍然存在。
- 可复用 encoder 输出和未变化的分数，但仍有额外 reward 重算及张量开销。记录实际耗时、峰值显存和 reward 评估成本，不能将“无额外 encoder forward”写成零成本。
- MRR 未表达的排序改善，baseline 也无法恢复。该方案是否降低方差须实测。

如果单位方向参考效果不好，再考虑独立替代动作或条件均值近似，不在首轮同时扩展多个 baseline 家族。理论背景与之前的 control variate 讨论见[讨论复盘](rollout_variance_discussion.md#6-已提出方案的取舍)。

### 4.3 固定状态的 2 × 2 对照

| Reward | 当前 group-level LOO | 逐文档反事实 baseline |
|---|---|---|
| Binary MRR | R0，可复用第一阶段诊断 | C0，检查同一 MRR 目标下的估计改进 |
| Graded nDCG | R2，可复用第一阶段诊断 | C2，检查两种改动是否互补 |

沿用第一阶段相同的状态、输入和实际动作。核心比较是 **完整共享 encoder 的总梯度**；可补充 query/doc 分支统计，但文档分支方差下降并不保证两者相加后的方差下降，因为存在协方差。

实现验证应覆盖：固定其他动作时 baseline 对自身采样动作不变；小规模可枚举例子中，文档梯度期望与原估计器一致；真实小模型的前后向、mask 和默认路径保持正确。检查期望一致不等于要求有限样本的两组均值完全相同。完成这些检查后，再进入真实模型诊断。

## 5. 第三阶段：完整训练验证

进入完整训练的依据是：固定状态下相对噪声、方向一致性等指标有跨 probe 的改善，平均梯度没有仅靠缩小到接近零获得好看的绝对方差，且额外成本可接受。局部诊断用于筛选，不能保证最终质量提升；如果 graded nDCG 只改善 reward 信息而没有改善噪声，仍可作为目标函数对照，但应单独说明动机。

按诊断结果选择新增行：

| 训练条件 | 何时安排 | 重复与复用 |
|---|---|---|
| Binary MRR + 当前 LOO | 当前控制 | 复用正在运行的三个独立 rollout-seed run |
| Graded nDCG + 当前 LOO | reward 路线值得继续时 | 新跑 rollout seed 42/3407/2026 |
| Binary MRR + 逐文档 baseline | baseline 在同一 MRR 目标下有效时 | 新跑 rollout seed 42/3407/2026 |
| Graded nDCG + 逐文档 baseline | 组合在诊断中有额外收益时 | 新跑 rollout seed 42/3407/2026 |

每个选中条件都从 E0 开始，固定训练/data seed 42、113 steps、LR 5e-6、global batch 128 / microbatch 16、GPU 数量及其他配方。只改变表中指定的 reward/标签或文档 baseline，不同时调 LR、G、alignment、clipping 或 advantage normalization。全部执行时新增九次训练；无需在固定状态诊断前承诺这笔预算。

最终报告包括：各 seed 原始结果、BRIGHT 宏平均的均值与样本标准差、12 subset 完整表、相同 seed 相对当前控制的差值、训练时间和显存。三个 seed 的比较是初步稳定性证据；质量均值、稳定性与成本分别报告，不能只展示方差更小或挑最好 seed。

若 baseline 在固定状态下有效、完整训练却无收益，结论应限定为“改善了所测状态下的估计”，再检查新训练轨迹上的梯度噪声。若 graded nDCG 提升质量但未降低噪声，应优先考虑目标/标签信息变化的解释。

## 6. 交付物与执行边界

1. 导入当前固定状态诊断和三个完整隔离 run 的结果，保留模型、输入、配置及 RNG 标识。
2. 形成 R0/R1/R2 的固定状态对比表，分别呈现 reward 结构与完整梯度统计。
3. 实现并验证逐文档 baseline，形成 C0/C2 与控制的比较，记录新增成本。
4. 根据诊断选择完整训练条件，保留选择理由，再汇总多 seed 质量与稳定性。

G=64 已有训练结果；必要时可以在同一 checkpoint 增加 G=32/64 的噪声—成本比较，无需为此先重复完整 G=64 训练。多组独立 rollout 平均也属于增加采样预算的候选，暂不作为首轮主改动。

本计划补充 G1 的机制验证，不改变既定 G2 配置。反事实 baseline 的代码和诊断开关已实现，目前没有新增完整训练配置或启动正式对比实验。结果成熟后再决定如何组织论文中的机制证据与方法贡献。
