# G1 rollout 方差讨论复盘：证据、假设与下一步

本记录整理 seed 重复、rollout 隔离、reward 分辨率及 advantage 归因的讨论。当前问题是：**固定模型和 batch 时，现有梯度估计器的信噪比是否受到 MRR 粗粒度反馈的限制，以及如何改善。** 不用三个 seed 的 BRIGHT 排序来验证这个局部机制。

## 1. 当前可以保留的结论

| 信息 | 证据状态与适用范围 |
|---|---|
| 原始完整训练对 seed 敏感 | 三个总 seed 的 BRIGHT 为 22.014、16.193、19.383；均值 19.197，样本标准差 2.915。原始重复同时改变了数据流程和动作采样，不能单独归因于 rollout。 |
| 固定状态的 rollout 梯度诊断支持高噪声的怀疑 | 用户已在训练机器完成诊断并反馈支持该怀疑。本次本地没有具体诊断 JSON，尚不能独立核对噪声数值、所用 checkpoint、batch 和 microbatch 平均口径。 |
| MRR 的实际反馈分辨率低，后期退化增加 | 已复核 W&B 完整历史，见第 3 节。它是 reward 层面的证据，不能单独证明参数梯度方差大。 |
| Graded nDCG 提供更丰富、较少退化的 reward 反馈 | 当前配置下，后期约有 256 个不同 reward，query 轴退化比例约 0.35%。这是已测运行的描述；是否降低固定状态下的梯度相对噪声仍需直接比较。 |
| 当前 query/document advantage 已经边际化 | Product reward 矩阵按另一动作轴取均值，然后分别计算 leave-one-out advantage。不能再将“两侧尚未分离”作为实现缺陷。 |
| 独立 rollout seed 的完整训练对照已经配置 | 代码提交 `418bd5c`；按用户最近反馈，正在训练。此处没有导入其最终结果，不视为因果验证已完成。 |

原始 seed 结果见 [训练机器分析](seed_variance_analysis.md)；诊断与训练命令见 [rollout RNG 使用文档](rollout_rng_diagnostics.md)。原报告中若有更强的机制判断，以本记录的证据边界为准。

## 2. 当前估计器实际上做了什么

G=32 时，每个 query 采样 32 个 query 动作和 32 组文档动作；每组文档动作覆盖自有候选文档。交叉得到：

\[
R_{ij}=r(q_i,D_j),\qquad i,j=1,\ldots,32.
\]

它有 **1,024 次 reward 组合评估**，但这些组合共享 query 或文档动作，不是 1,024 个独立的联合动作样本。

Query 侧先边际化文档轴，再做 leave-one-out：

\[
\bar R_i^q=\frac1G\sum_jR_{ij},\qquad
A_i^q=\bar R_i^q-\frac1{G-1}\sum_{k\ne i}\bar R_k^q.
\]

Document 侧对 query 轴做同样的计算。两侧分别用自己的 advantage 加权自己的 log-prob；当前不做 advantage 标准化，document log-prob 求和。

这意味着 query/document 交互已经通过对另一侧动作分布的平均进入估计器。剩下的准确问题是：**有限采样下，边际均值还有多大的估计噪声；粗粒度 reward 能否充分区分动作方向。** “两侧同时改变，所以根本无法区分贡献”应撤回。

文档内部则不同：当前整个候选文档组共享一个 document advantage，形式为：

\[
\hat g_D=A_D\sum_j\nabla_\theta\log\pi_\theta(e_j).
\]

因此，即使某次 reward 变化主要由一个文档引起，其余被采样文档也会收到同一 advantage 加权的 score-function 梯度。这是一个可检查的方差来源，不代表该联合 score-function 形式本身有偏。

实现依据：[GRPO](../src/grpo.py)、[group advantages](../src/policy_math.py)、[reward](../src/rewards.py)。

## 3. 日志提供了哪些有效信息

分析数据来自 `defaultGroup/E2Rank-RL-v2` 的未抽样完整历史。已检查 9 个 finished run，每个均有连续、唯一的 optimizer step 1–113，共 1,017 个训练 step。下表只保留与当前机制有关的对照。

早期 = step 1–20；后期 = step 94–113；均为窗口内逐 step 等权均值。除 G 外，表中声明的设置均为 alignment 0.90、训练 seed 42。

| Reward / G | Query `degenerate_frac` 早→晚 | `n_distinct` 早→晚 |
|---|---:|---:|
| [Binary MRR / 32](https://wandb.ai/defaultGroup/E2Rank-RL-v2/runs/staphpap) | 22.23% → 32.19% | 4.18 → 3.13 |
| [Binary nDCG / 32](https://wandb.ai/defaultGroup/E2Rank-RL-v2/runs/eqqkyxor) | 6.25% → 11.05% | 32.45 → 24.33 |
| [Graded nDCG / 32](https://wandb.ai/defaultGroup/E2Rank-RL-v2/runs/6x03erra) | 0.27% → 0.35% | 256.16 → 256.47 |
| [Binary MRR / 64](https://wandb.ai/defaultGroup/E2Rank-RL-v2/runs/126j42no) | 20.59% → 30.43% | 4.48 → 3.31 |

**数值订正：**此前口头回复中的 binary nDCG 后期 `n_distinct=24.14` 不准确，原始日志重算为 **24.33**。

### 两个指标的含义

- `degenerate_frac`：先对另一动作轴平均 reward，再统计当前轴组内标准差不超过阈值的 query 比例；对当前 [0,1] reward，阈值为 1e-4。它不是文档比例、参数比例或“多少参数梯度为零”。当前未标准化 advantage 路径不会按此阈值强制清零。
- `n_distinct`：将每个 query 的完整 product reward grid 展平，统计相邻排序值差大于 1e-4 的取值数，再跨 query 平均。它统计的是原始 reward grid，和前者的边际化口径不同；也没有记录各 reward 档位的频率或熵。

当前 MRR 与 graded nDCG run 的 query/document 退化比例逐步相同。Binary nDCG 的两轴略有不同：document 轴为 6.29% → 11.09%；表格统一使用 query 轴。

### 能支持什么，不能支持什么

这些曲线支持：MRR 的许多 rollout 组近乎没有 reward 差异，实际覆盖的 reward 档位较少，且后期进一步减少；graded nDCG 的反馈明显更丰富。

它们**不能直接证明梯度方差大**：所有 reward 相同时，当前 group-centered advantage 甚至可以全部为零，梯度也为零。需要固定模型、固定 batch 的重复梯度来直接测量 `noise_rms`、相对平均梯度的噪声和方向一致性。

时间窗口中的 batch 在变化，各 reward run 的模型训练轨迹也不同。因此历史曲线是重要线索，但不是“固定同一模型，仅换 reward”的控制实验。无需用它们解释或预测三个 seed 的最终 BRIGHT 排序。

完整本地缓存位于 `paper/g1_gradient_analysis/raw/`，窗口统计位于 `reward_resolution_analysis.json`；这些具体分析产物按现有规则被 Git 忽略。上表保留直接 W&B 来源，便于独立追溯。

## 4. Reward 档位数量的结构性限制

对固定 query、候选池和标签：

\[
n_{\mathrm{distinct}}\leq\min(G_qG_D,\;|\mathcal R|),
\]

其中 \(|\mathcal R|\) 是该指标在这些候选标签下可取得的 reward 数量；有限探索覆盖和日志中的数值阈值还会进一步减少实际计数。

- **MRR@10 最多 11 个档位**：0、1、1/2、…、1/10。有些候选池甚至达不到全部 11 档。增加 G 不会增加这个上限。
- **Binary nDCG@10** 由 top-10 的正负标签位置决定。若候选池只有 3 个正例且负例足够，标签排列数上限为 Σⱼ₌₀³ C(10,j)=176；不同排列还可能映射到同一 reward，因此这是上界。
- **当前 graded nDCG** 使用 teacher 排序：第 1 名为 grade 3，第 2–5 名为 grade 2，第 6–10 名为 grade 1，其余为 0。候选足够时有 1/4/5 个文档分属三个非零等级，允许的标签排列远多于 1,024；不能认为所有离散 nDCG 都像 MRR 一样先被少量档位卡住。

因此，MRR 的 `n_distinct≈3.13` 应解释为“在最多 11 档中实际覆盖约 3 档”，不能解释为“1,024 次评估只有 3 次有效”。重复取到同一 reward 仍能帮助估计其概率以及 reward 与 score-function 的相关性。

还需保留一个实验区别：binary 标签是已知正例身份，graded 标签来自 teacher 排序。当前 4,963 条训练记录中，4,544 条有 10 个非零 graded 标签。切换到 graded nDCG 不仅增加数值等级，还改变了相关性标签信息；不能将全部变化归结为档位数量一个因素。标签构造见 [prepare_reasonrank.py](../scripts/prepare_reasonrank.py)。

## 5. 最有价值的机制假设

**工作假设：MRR 对高维联合动作的反馈过于粗糙，有限采样的边际化梯度中，有效方向信号相对随机扰动太弱。**

这里有两个需要分开的含义：

1. **目标不区分某些改善。** 只要第一个正例名次不变，正负 margin、后续正例位置等改善可能不影响 MRR；这些方向对该 reward 本身就是等价的。
2. **估计器噪声掩盖可区分的改善。** 改变 MRR 的少数边界事件伴随许多同时发生的随机扰动；在有限采样下，难以充分抵消无关方向。这需要梯度诊断和动作贡献诊断验证。

“大量 embedding 变化不改变 reward，某个关键正负翻转才改变 reward”与指标结构一致。但现有 `degenerate_frac` 和 `n_distinct` 不记录是哪对文档造成差异，尚不能确认每次主要由单个 pair 驱动。现有 `own_boundary_flip_rate` 也不是 MRR 变化的因果归因统计。

“梯度就是这两个文档的差方向”也要收紧：固定文档时，query 跨越某个排序边界的有效方向与文档差向量的投影有关；实际 score-function 梯度还包含采样扰动、归一化和 encoder 的 Jacobian，文档侧与 query 侧的路径不同。

## 6. 已提出方案的取舍

| 方案 | 保留的有效信息 | 当前定位 |
|---|---|---|
| 换用 graded nDCG | 直接丰富排序反馈，已有日志支持更少退化 | 优先做固定状态梯度比较；尚未证明降低梯度相对噪声 |
| 增大 G | 同一目标下增加采样，可能改善梯度估计 | G=64 已有训练；不能当作未做过的新方向，也不能用其单 seed 成绩判断方差是否降低 |
| 多组独立 rollout 后平均 | 同一模型/batch 下 R 个独立估计平均，期望不变、条件方差为原来的 1/R | 本质仍是增加采样预算，不能包装成已经解决 reward 粗粒度的新机制；是否划算要与增大 G 比较 |
| 再做 query/doc 边际化 | 当前已经实现 | 不作为新增改进 |
| 平滑排序 surrogate 的 control variate | 可尝试减少可预测的采样波动 | 需解析补偿和正确停止梯度；没有实测效果，不优先于现有 reward 对照 |
| 逐文档反事实 baseline | 检查某文档扰动对 reward 的影响，细化文档内部贡献 | 可检验候选；需保持 baseline 对该文档自身采样独立，不能直接按“翻转 pair”随意屏蔽梯度 |
| 监督辅助项或参考表示约束 | 可以提供确定性约束 | 会改变训练目标；目前没有证据要求立即加入 |
| Advantage normalization、减小 LR、whitening | 分别改变权重、更新幅度或表示 | 不能仅凭当前讨论认为它们解决了梯度方向噪声 |

平滑 surrogate 的参考思路见 [Q-Prop](https://arxiv.org/abs/1611.02247)；分解策略的 baseline 思路见 [action-dependent factorized baselines](https://arxiv.org/abs/1803.07246)。讨论中的 vMF 排序适配和逐文档实现是候选设计，并非这些论文已验证当前方法有效。

## 7. 最小的后续证据链

后续讨论已将 graded nDCG 与逐文档反事实 baseline 确定为优先候选。具体对照矩阵、baseline 定义、筛选标准和完整训练重复安排见[后续实验计划](rollout_variance_experiment_plan.md)；本节保留证据链概述，执行顺序以该计划为准。

1. **保留正在执行的完整 rollout 隔离对照。** 训练/data seed 均为 42，rollout seed 为 42/3407/2026；三行使用新的独立 RNG，不能复用历史 seed-42 成绩。它检验 rollout 随机性是否足以造成完整训练差距。
2. **汇入已经完成的固定状态诊断。** 核对模型、step、样本、采样数和精度，特别区分单个 microbatch 梯度与八个 microbatch 平均后的梯度。本次不假定未见到的数值。
3. **在同一 checkpoint、同一组输入和 rollout 随机数上比较 binary MRR、binary nDCG、graded nDCG。** 前两者隔离 reward 指标；后两者体现相关性标签方案变化。比较绝对噪声、平均梯度范数和二者比值，避免把目标尺度缩小误认为信噪比改善。不同目标的平均梯度方向本来可以不同，不要求它们一致。
4. **需要时补固定状态的 G=32/64 比较。** 单个 checkpoint 就够；不需要先重复训练 G=64。比较时固定模型参数，不能分别加载 G=32 和 G=64 各自训练出的模型后将差异全归因于 G。
5. **若仍需定位文档内部贡献，再记录 reward 档位频率和逐文档反事实影响。** 这些数据用于区分少数边界事件与大量伴随扰动；取得证据后再选择 control variate 的具体形式。

固定状态可以先选择 E0 和最终 checkpoint。`checkpoint-100` 只是现有保存间隔下的方便示例，没有特殊理论意义；最终权重在 run 根目录，113 步。操作细节沿用 [诊断文档](rollout_rng_diagnostics.md)。

切换 graded 标签后，包含 `relevance_labels` 的整份 batch hash 会变化，这是预期的；应核对样本 ID、query/document 输入、顺序及候选 mask 的一致性，而不是要求整份 hash 不变。

## 8. 早期几何分析中暂不沿用的说法

训练机器报告的评测复现和权重比较保留为原始证据；以下机制判断需要收紧：

- “几何指标排序与 BRIGHT 严格一致”不成立：有效维度与 raw gap 的三个 seed 排序并不都相同。
- “pooling 放大 100 倍”未被证明：相对权重范数和 1−cosine 不能直接相除得到因果放大倍数，也没有 pooling 对照。
- 几何脚本默认使用训练集 biology 前 200 条记录、长度 384 和通用 instruction；并非完整 BRIGHT 评测口径。本地实际为 3,718 个候选，数据共 4,963 条。
- 不能从上述几何相关性直接得出“各向异性已是根因”，也不能由三个相近的训练 reward 得出“reward 与表示质量几乎无关”。

本记录更新的是讨论与证据口径，没有改变训练配置、运行中的实验或论文正文。
