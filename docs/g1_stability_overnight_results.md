# G1 过夜稳定性实验结果：7 个配方 × 3 个 seed

结果快照：2026-09-15。本文汇总 [G1 过夜稳定性实验计划](g1_stability_overnight_plan.md)中的 21 次全新训练，目标是判断哪些配方既保持 BRIGHT 质量，又能降低完整训练对 seed 的敏感性。

## 摘要

本批最清楚的结果是：**将 group size 从 32 增至 64，是唯一同时改善平均分、最差 seed 和宏平均稳定性的改动。** 在 binary MRR 和 graded nDCG 两条路线中，G=64 相对 G=32 都在 3/3 个配对 seed 上提升，平均增益分别为 **+1.18** 和 **+1.19** 分。

MRR64 与 GradedNDCG64 的宏平均几乎相同，分别为 **18.89 ± 0.41** 和 **18.85 ± 0.52**。两者的 seed 排序会翻转，因此当前证据不能判定 MRR 或 graded nDCG reward 更优。MRR64 的宏平均略高且最差 seed 略好；GradedNDCG64 的逐领域波动更小。

BinaryNDCG32 的标准差达到 **1.40**，明显高于其他主要配方；学习率从 5e-6 减半到 2.5e-6 也没有形成有价值的质量—稳定性折中。历史 MRR32 的单次 22.01 没有在本批复现；新的独立 rollout RNG 结果集中在约 18 分，因此 22.01 应保留为历史配置搜索中的最佳单次结果，而不能继续代表该配方的典型表现。

## 1. 数据完整性与统计口径

本批 21 个 run 均有完整的 12 个 BRIGHT subset，汇总中没有缺失任务或错误，共得到 252 条领域结果。结果来源为：

- [`run_summary.csv`](../paper/_summary/g1_bright/run_summary.csv)：逐 run 宏平均；
- [`subset_summary.csv`](../paper/_summary/g1_bright/subset_summary.csv)：逐 run、逐领域分数。

本文报告 BRIGHT nDCG@10 × 100。每个配方的均值、样本标准差、最差值和最好值由三个 seed 的宏平均计算；配方对比使用相同 seed 配对。由于源汇总只保留两位小数，本文的派生统计也以这些舍入值为输入。

三个 seed 既改变 training seed 和 data seed，也改变独立 rollout seed，因此本批衡量的是**完整训练的总体 seed 稳定性**，不是单一随机源的方差分解。n=3 只适合配置开发和发现大效应，不足以可靠估计尾部失败概率或做确认性显著性结论。

## 2. 配方级主结果

| 排名 | 配方 | Seed 42 | Seed 3407 | Seed 2026 | 均值 | 样本 SD | 最差 | 最好 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | MRR64 | 18.44 | 19.24 | 18.98 | **18.89** | **0.41** | **18.44** | 19.24 |
| 2 | GradedNDCG64 | 18.98 | 19.30 | 18.28 | 18.85 | 0.52 | 18.28 | **19.30** |
| 3 | MRR64LRHalf | 18.52 | 18.58 | 17.78 | 18.29 | 0.45 | 17.78 | 18.58 |
| 4 | MRR32 | 16.86 | 18.33 | 17.94 | 17.71 | 0.76 | 16.86 | 18.33 |
| 5 | GradedNDCG32 | 16.89 | 18.12 | 17.99 | 17.67 | 0.68 | 16.89 | 18.12 |
| 6 | GradedNDCG64LRHalf | 17.32 | 17.94 | 17.57 | 17.61 | 0.31 | 17.32 | 17.94 |
| 7 | BinaryNDCG32 | 17.15 | 15.86 | 18.66 | 17.22 | 1.40 | 15.86 | 18.66 |

按实验计划预设的选择标准，MRR64 相对 MRR32 同时满足均值提高、标准差降低、最差 seed 提高：

- 均值：17.71 → **18.89**；
- 样本标准差：0.76 → **0.41**；
- 最差 seed：16.86 → **18.44**。

GradedNDCG64 相对 MRR32 也同时改善这三项，但它同时改变了 reward/标签和 G，不能把全部收益归因于某一个因素。

七个配方在各 seed 上的平均值分别为 seed 42 的 17.74、seed 3407 的 18.20 和 seed 2026 的 18.17。seed 42 在本批整体偏低，说明使用相同 seed 配对比较配方很重要；这三个平均值共享配方和 seed，不能视作独立样本来估计一般性的 seed 主效应。

## 3. Group size 是最稳定的改进

| 配对比较 | Seed 42 | Seed 3407 | Seed 2026 | 平均差 | 改善 seed 数 |
|---|---:|---:|---:|---:|---:|
| MRR64 − MRR32 | +1.58 | +0.91 | +1.04 | **+1.18** | **3/3** |
| GradedNDCG64 − GradedNDCG32 | +2.09 | +1.18 | +0.29 | **+1.19** | **3/3** |

G=64 的收益在两个 reward 下几乎相同，而且每个配对 seed 都为正。这比任一 reward 变化或学习率变化都更一致。它也与此前固定状态梯度诊断的结果吻合：增加 G 是当时唯一能在 E0 和 final checkpoint 上同时降低相对梯度噪声、提高方向一致性的超参数。

逐领域看，G 的收益并非完全均匀：

- MRR64 相对 MRR32 在 8/12 个领域提高，最大增益来自 biology（+5.42）、psychology（+4.82）和 earth_science（+1.89）；主要退化为 leetcode（−1.82）、robotics（−0.90）和 theoremqa_questions（−0.65）。
- GradedNDCG64 相对 GradedNDCG32 在 9/12 个领域提高，最大增益来自 biology（+3.67）、sustainable_living（+2.34）和 theoremqa_theorems（+2.15）；三个退化领域的平均差都很小，分别为 leetcode（−0.22）、pony（−0.07）和 theoremqa_questions（−0.06）。

因此，G=64 的宏平均收益有跨 reward、跨 seed 的直接证据，但不能宣称所有领域都会改善。

## 4. Reward 和标签没有决出胜负

### 4.1 MRR 与 graded nDCG

| G | MRR 均值 | Graded nDCG 均值 | Graded − MRR |
|---:|---:|---:|---:|
| 32 | 17.71 | 17.67 | −0.04 |
| 64 | 18.89 | 18.85 | −0.03 |

两个 G 下的宏平均差都不足 0.05 分。G=64 时，MRR64 − GradedNDCG64 在 seed 42/3407/2026 上分别为 −0.54、−0.06、+0.70，方向随 seed 翻转。因此，原先基于单 seed 的 MRR 优势没有得到稳定性实验支持。

两条路线的领域侧重也不同。相对 GradedNDCG64，MRR64 在 biology、psychology 和 stackoverflow 上平均更高；GradedNDCG64 在 earth_science、robotics 和 theoremqa_questions 上平均更高。这更像领域权衡，而不是一条路线全面支配另一条。

### 4.2 Binary nDCG 与 graded 标签

| 配对比较 | Seed 42 | Seed 3407 | Seed 2026 | 平均差 |
|---|---:|---:|---:|---:|
| BinaryNDCG32 − MRR32 | +0.29 | −2.47 | +0.72 | −0.49 |
| GradedNDCG32 − BinaryNDCG32 | −0.26 | +2.26 | −0.67 | +0.44 |
| GradedNDCG32 − MRR32 | +0.03 | −0.21 | +0.05 | −0.04 |

BinaryNDCG32 的极差为 2.80 分，标准差为 1.40。引入 graded 标签提高了它的三 seed 均值，但增益主要由 seed 3407 的一次大幅恢复产生，在另外两个 seed 上反而略降。最终 GradedNDCG32 与 MRR32 几乎完全打平。

这些结果不支持“将 MRR 换为 nDCG 指标”或“增加 graded 标签信息”本身可以改善最终 BRIGHT；它们也与固定状态诊断相符：reward 分辨率增加并没有稳定改善相对梯度信噪比。

## 5. 学习率减半不值得采用

| 配对比较 | Seed 42 | Seed 3407 | Seed 2026 | 平均差 | SD 变化 |
|---|---:|---:|---:|---:|---:|
| GradedNDCG64LRHalf − GradedNDCG64 | −1.66 | −1.36 | −0.71 | **−1.24** | 0.52 → 0.31 |
| MRR64LRHalf − MRR64 | +0.08 | −0.66 | −1.20 | **−0.59** | 0.41 → 0.45 |

GradedNDCG64 的半学习率虽然降低了标准差，但均值和最差 seed 都明显下降，更像稳定地训练不足。MRR64 的半学习率在均值、标准差和最差 seed 上都更差。当前没有理由用 2.5e-6 替代 5e-6。

## 6. 宏平均稳定不等于每个领域稳定

| 配方 | 宏平均 SD | 12 个领域 SD 的平均值 | 波动最大的领域 |
|---|---:|---:|---|
| MRR64 | **0.41** | 1.34 | biology 3.23；leetcode 1.96；psychology 1.70 |
| GradedNDCG64 | 0.52 | **1.00** | economics 1.96；sustainable_living 1.66；biology 1.63 |
| MRR64LRHalf | 0.45 | 0.81 | biology 1.57；earth_science 1.49；economics 1.03 |
| MRR32 | 0.76 | 1.83 | earth_science 3.74；biology 3.30；robotics 3.28 |
| GradedNDCG32 | 0.68 | 1.09 | sustainable_living 2.68；earth_science 2.26；robotics 2.16 |
| GradedNDCG64LRHalf | 0.31 | 0.76 | theoremqa_theorems 1.49；economics 1.21；stackoverflow 1.18 |
| BinaryNDCG32 | 1.40 | 1.86 | stackoverflow 3.86；theoremqa_theorems 3.12；biology 3.03 |

MRR64 的宏平均最稳，但其领域波动比 GradedNDCG64 大，说明不同领域的涨跌存在一定抵消。GradedNDCG64 在宏平均上只低 0.03 分，却有更平稳的领域表现。因此：

- 若选择标准严格以 BRIGHT 宏平均、宏平均 SD 和最差 seed 为主，MRR64 略占优；
- 若更重视不同领域分别稳定，GradedNDCG64 是同等可信的候选。

跨全部七个配方，平均波动最大的领域是 biology、earth_science、robotics 和 sustainable_living。这些领域应在后续新 seed 验证中继续单独报告，不能只看宏平均。

## 7. 历史 22.01 的重新解释

| 实验组 | 三个结果 | 均值 | 样本 SD | 极差 |
|---|---|---:|---:|---:|
| 历史总 seed 重复 | 22.01 / 16.19 / 19.38 | 19.19 | 2.91 | 5.82 |
| 固定 training/data、只换独立 rollout seed | 16.99 / 18.52 / 18.59 | 18.03 | 0.90 | 1.60 |
| 本批全新 MRR32 | 16.86 / 18.33 / 17.94 | 17.71 | 0.76 | 1.47 |

历史 22.01 来自使用全局 PyTorch RNG 的旧路径，不能等同于新增独立 rollout RNG 后的 seed 42。本批 MRR32 seed 42 为 16.86，先前独立 Rollout42 为 16.99；两次结果只差 0.13。独立 rollout 对照与本批 MRR32 的分布范围也接近，而历史组的方差主要由 22.01 高点拉大。

这些结果支持以下解释：

1. 22.01 是真实发生过的历史单次结果，但在当前独立 RNG 协议下没有复现；
2. 它更适合作为配置搜索中的最好单次轨迹，而不是 MRR32 的典型效果；
3. 当前协议下，MRR32 的代表性水平约为 17.7–18.0，增加到 G=64 后提高到约 18.9。

这里不能把全部差异严格归因于某一种 RNG 或运行环境变化；历史总 seed、本批总体 seed 和 rollout 单因素实验改变的随机源不同。但三组证据共同否定了“MRR32 稳定达到 22.01”的解释。更详细的独立 rollout 证据见 [G1 rollout 方差实验结果](rollout_variance_results.md)。

## 8. 与现有基线及论文主张的关系

原始 E0 为 15.07。本批 21/21 个训练 run 都超过 E0；最低的 BinaryNDCG32 seed 3407 仍为 15.86。因此，ReasonRank 适配总体带来 BRIGHT 增益的结论仍然稳定。

已有三个 seed 的 CL 分别为 16.96、17.62 和 17.48，均值 **17.35 ± 0.35**。使用相同 seed 配对：

| 比较 | Seed 42 | Seed 3407 | Seed 2026 | 平均差 |
|---|---:|---:|---:|---:|
| MRR64 − CL | +1.48 | +1.62 | +1.50 | **+1.53** |
| GradedNDCG64 − CL | +2.02 | +1.68 | +0.80 | **+1.50** |

因此，“G64 RL 在本组三个 seed 上稳定超过普通 CL”有直接证据。

最强监督对照 Graded LambdaLoss-Scaled 当前只有 seed 42 的 19.50。它高于本批所有单次 run，也比 MRR64 和 GradedNDCG64 的三 seed 均值分别高约 0.61 和 0.65 分。因为双方重复数不同，不能据此确认 LambdaLoss-Scaled 的总体期望更高；但现有数据已经不足以支持“RL 稳定超过最强监督对照”的论文主张。

相应地，论文结果应区分：

- **历史开发结果**：最佳单次 RL 为 22.01；
- **当前稳定性结果**：最佳三 seed 宏平均为 MRR64 的 18.89 ± 0.41；
- **已有稳定比较**：两个 G64 RL 配方均在 3/3 个配对 seed 上超过 CL；
- **尚未确认的比较**：RL 与 Graded LambdaLoss-Scaled 的多 seed 优劣。

## 9. 决策与后续实验

### 当前配方决策

1. **保留 G=64。** 这是本批唯一跨 reward、跨 seed 一致改善质量和稳定性的改动。
2. **不采用半学习率。** 2.5e-6 没有提供可接受的质量—稳定性折中。
3. **暂不宣布 reward 胜者。** MRR64 与 GradedNDCG64 的宏平均差只有 0.03 分，远小于当前 seed 波动。
4. **不继续使用 BinaryNDCG32。** 它的均值较低且 seed 方差最大。

若必须立即选择一个默认配方，按计划预设的宏平均、宏平均 SD 和最差 seed 三项标准，选择 **MRR64 / LR 5e-6**。若目标更重视逐领域一致性，则同时保留 **GradedNDCG64 / LR 5e-6** 进入确认阶段。

### 最小后续证据链

1. 用未参与本批配置选择的新 seed 验证 MRR64 和 GradedNDCG64；不要用本轮三个 seed 的最好点估计代替确认结果。
2. 若论文需要保留“RL 超过最强监督方法”的主张，为 Graded LambdaLoss-Scaled 补 seed 3407 和 2026，并按 seed 与两个 G64 RL 配方配对比较。
3. 最终表同时报告宏平均均值 ± SD、最差 seed，以及 biology、earth_science、robotics、sustainable_living 等高波动领域，避免宏平均抵消掩盖领域不稳定。

## 10. 最终结论

这批实验没有复现 22.01，但提供了比单次最好成绩更可靠的配方结论：**当前收益主要来自增加 rollout group size，而不是 reward 类型或降低学习率。** MRR64 和 GradedNDCG64 都把三 seed 平均提高到约 18.9，并明显缩小 MRR32 的宏平均波动；两者之间的差异不足以选出 reward 胜者。

因此，G1 的主要论点应从“某个 MRR 单 seed 配方达到 22.01”调整为“G=64 在两种 reward 下均带来约 +1.2 分的配对增益，并改善完整训练稳定性”。RL 相对 CL 的优势得到三 seed 支持；相对最强 LambdaLoss 监督对照的优势则需要补齐监督重复后再判断。
