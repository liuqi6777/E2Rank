# G1-R2 实验结果分析

分析日期：2026-09-16。对应 [实验计划](EXPERIMENT_PLAN.md) 与 [夜跑说明](../docs/g1_r2_overnight.md)。

## 结论

本批最值得推进的是 **graded nDCG64 + conditional projection（CP）**：BRIGHT 三 seed 均值 **20.72 ± 0.44**，三个 seed 均为九方法第一；相对同 reward 的 SF 提升 **2.19 分**，相对标签匹配的 LL-Graded 提升 **1.36 分**，两项比较均为 3/3 seed 正收益。

CP 的收益明显依赖 reward：MRR 均分基本不变，但本次 seed 波动变小；binary nDCG 平均改善，但不是每个 seed 都改善。不能把结论写成“CP 普遍提高检索分数或降低训练 seed 方差”。

固定状态的 MRR 探针给出很强的局部降噪证据：六组全参数梯度方差均下降约 110–419 倍，forward/backward 时间增加约 18%–22%。但诊断没有覆盖 graded reward，也没有覆盖实际 RL 训练轨迹，尚不能直接证明 graded CP 的检索收益由同等幅度的降噪造成。

## 1. 数据完整性与证据边界

- 两份 CSV 包含 27 个训练 run（9 方法 × 3 seed）和 1 条 E0，每个 run 有 12 个唯一 subset，共 336 条领域分数；未见缺项、重复或非零 errors。
- run 分数与 12 个已舍入领域分数的宏平均最大相差 0.0067，符合两处分别保留两位小数的精度范围。本文主表使用 run_summary，领域分析使用 subset_summary；末位可能有舍入差。SD 为三个 run 分数的样本标准差（ddof=1）。
- **E0 暂不纳入新协议增益结论**：此处记录名为 `G1-E0`，路径后缀为 `iclr2027__G1-E0/no_revision_available`，而不是计划的 `G1-R2-E0` 和新协议标识。其 15.07 总分及全部 12 个领域分数与历史 CSV 完全相同。这是来源核对缺口，不能据此直接断言旧缓存被复用，也不能认为已完成新协议重评。
- 27 个训练 run 的结果路径均带 `tokens-v2__pool-fp32`。但本地提供的文件没有逐 run contract/state、原始评测 JSON 和训练日志，因此无法独立确认三机训练数据哈希、最终 step 113、完整训练成本与运行硬件一致性。分数比较已完成，合同审计尚未完成。
- 两份 probe 的训练数据哈希一致：`59a9b2bc4f2ce7a0a890bc471b8ef784e3f92e01294523b2affd39f285b39a38`。它不同于计划记录的开发机快照，但计划明确允许使用另一份与 manifest 匹配的运行数据；不能仅凭差异判错。仍须用训练合同确认 27 个 run 也使用同一哈希。
- 两份 probe 记录的源码哈希与当前本地对应源码一致，诊断脚本哈希也一致。运行记录 `git_dirty=true`，因此应保留这些实际文件哈希；不能仅凭 commit 判断工作区内容。

## 2. 三 seed 主表

分数为 CSV 中 BRIGHT 宏平均分；seed 顺序固定为 42 / 3407 / 2026。E0 来源待核，不计算相对 E0 增益。

| 方法 | 42 | 3407 | 2026 | 均值 ± 样本 SD | 最差 seed |
|---|---:|---:|---:|---:|---:|
| RL-GradedNDCG64-CP | 20.27 | 21.15 | 20.75 | 20.72 ± 0.44 | 20.27 |
| RL-MRR64-SF | 18.54 | 19.78 | 19.87 | 19.40 ± 0.74 | 18.54 |
| RL-MRR64-CP | 19.33 | 19.36 | 19.44 | 19.38 ± 0.06 | 19.33 |
| LL-Graded | 19.13 | 19.30 | 19.67 | 19.37 ± 0.28 | 19.13 |
| RL-BinaryNDCG64-CP | 19.44 | 18.64 | 19.65 | 19.24 ± 0.53 | 18.64 |
| RL-BinaryNDCG64-SF | 18.33 | 18.88 | 19.07 | 18.76 ± 0.38 | 18.33 |
| RL-GradedNDCG64-SF | 18.23 | 18.85 | 18.53 | 18.54 ± 0.31 | 18.23 |
| CL | 17.87 | 17.04 | 18.25 | 17.72 ± 0.62 | 17.04 |
| LL-Binary | 15.87 | 16.05 | 16.63 | 16.18 ± 0.40 | 15.87 |

graded CP 最差 seed 的 20.27 也高于其他方法所有单次结果（最高 19.87）。这使它成为清晰的当前候选，但 n=3 和既有 benchmark 开发历史仍不足以支持广泛泛化或充分调优后的最优性结论。

## 3. 估计器、reward 与标签的影响

| 配对比较（CP−SF） | 42 | 3407 | 2026 | 平均差 | 差值 SD |
|---|---:|---:|---:|---:|---:|
| MRR64 | +0.79 | -0.42 | -0.43 | -0.02 | 0.70 |
| BinaryNDCG64 | +1.11 | -0.24 | +0.58 | +0.48 | 0.68 |
| GradedNDCG64 | +2.04 | +2.30 | +2.22 | +2.19 | 0.13 |

**Graded nDCG 是最明确的 CP 检索成功案例。** CP 对 SF 的 +2.04/+2.30/+2.22 很一致；相对 LL-Graded 为 +1.14/+1.85/+1.08。Graded SF 本身均值 18.54，低于 LL-Graded 19.37，因此当前证据支持特定 reward 与估计器的组合，而非“使用 RL 就优于匹配的排序监督”。

**MRR 的主要变化是本批稳定性和领域分配。** 均分差 −0.02，CP 的 SD 从 0.74 降到 0.057，最差 seed 从 18.54 提高到 19.33；但两个 seed 分数下降。只有三个重复，且 data seed 与 training seed 同时变化，不能把最终分数 SD 的下降直接归因于 rollout 噪声单一因素。MRR-CP 与 LL-Graded 的均分也几乎相同（19.38 vs 19.37）。

**Binary nDCG 有平均收益，稳定性没有改善。** CP 平均 +0.48，2/3 seed 为正，SD 反而从 0.38 增至 0.53。它不足以支撑所有 reward 都有稳定收益的论断。

**Graded 标签的收益与优化方法有交互。** LL-Graded 比 LL-Binary 高 3.18 分；graded CP 比 binary CP 高 1.48 分，三个 seed 都为正；但 graded SF 比 binary SF 低 0.22 分，三个 seed 都为负。因此“graded 一定更好”不成立，观察到的是更丰富监督与估计器/优化过程共同作用。两种 nDCG 下 CP 增益的差为约 1.70 分，但这只是本配方下的描述性差异。

## 4. 领域贡献：优势广泛，但有明确短板

以下是各领域先跨 seed 平均、再相减；领域等权，不能当作独立训练重复。

| subset | MRR CP−SF | Binary CP−SF | Graded CP−SF | Graded CP−LL-Graded |
|---|---:|---:|---:|---:|
| biology | +5.62 | +6.50 | +7.43 | +2.41 |
| earth_science | -5.76 | -2.49 | +2.72 | +3.40 |
| economics | -0.87 | +1.04 | +2.26 | +1.83 |
| psychology | -0.86 | +1.26 | +4.59 | +2.88 |
| robotics | +0.72 | -0.04 | +3.82 | +0.84 |
| stackoverflow | +4.11 | +2.45 | +3.02 | +3.23 |
| sustainable_living | -0.47 | +1.13 | +5.33 | +2.61 |
| pony | +0.15 | +0.71 | +0.39 | +0.33 |
| leetcode | -3.29 | -0.84 | -1.18 | -1.39 |
| aops | +0.12 | -1.11 | -0.68 | -0.39 |
| theoremqa_theorems | +0.66 | -2.42 | -1.12 | +0.73 |
| theoremqa_questions | -0.40 | -0.41 | -0.36 | -0.17 |

Graded CP 对 SF 在 8/12 个领域平均提升，主要来自 biology、sustainable_living、psychology、robotics、stackoverflow 和 earth_science；对 LL-Graded 在 9/12 个领域提升。因此收益不是单个异常领域独自支撑。相对 CL 为 11/12 个领域提升，平均 +3.00 分。

但 graded CP 对 SF 在 leetcode、aops 和两个 theoremqa subset 上均下降；对 LL-Graded 的短板是 leetcode、aops、theoremqa_questions。不能写成“全面改善推理检索”，更适合表述为本轮 BRIGHT 宏平均及多个知识领域的改善。

MRR 平均差接近零掩盖了明显的结构变化：biology +5.62、stackoverflow +4.11，同时 earth_science −5.76、leetcode −3.29。Binary CP 也有类似方向。因此“降噪没有作用”不是准确解释；可以确认优化后领域表现不同，但为何出现这种分配仍需训练过程和候选池诊断。

## 5. 梯度探针：降噪强，均值一致性仍是有限样本诊断

两个模型状态分别是固定 revision 的 E0 和新 LL-Binary 的 checkpoint-25。后者 `step=25`、权重文件哈希已记录；`checkpoint_state=null` 是未找到 RL 的 exploration_state，并不代表没有加载 LL checkpoint。两状态使用相同三个 batch（leetcode、stackoverflow、math-qa），每批 16 个独立 rollout seed，共 96 对、192 次 forward/backward。

全部配对的 rollout seed、实际动作 SHA256、完整 reward table SHA256 和 reward mean 一致；记录的梯度范数均有限。探针比较的是全部可训练参数的原始梯度，禁用 dropout，不执行 optimizer 更新。

| 状态 | batch / source | 方差 CP/SF | F/B 时间 SF→CP（秒） | (方差×时间) CP/SF | 均值差范数 / MC RMS |
|---|---|---:|---:|---:|---:|
| E0 | 0 / leetcode | 0.00912 | 5.58 → 6.61 | 0.01079 | 1.010 |
| E0 | 100 / stackoverflow | 0.00603 | 4.60 → 5.62 | 0.00737 | 0.963 |
| E0 | 200 / math-qa | 0.00686 | 4.88 → 5.90 | 0.00830 | 1.010 |
| LL step25 | 0 / leetcode | 0.00309 | 5.58 → 6.57 | 0.00364 | 1.016 |
| LL step25 | 100 / stackoverflow | 0.00522 | 4.59 → 5.60 | 0.00637 | 1.034 |
| LL step25 | 200 / math-qa | 0.00239 | 4.86 → 5.88 | 0.00289 | 1.031 |

CP 保留约 0.24%–0.91% 的 SF 方差。把额外计算计入，方差×平均 F/B 时间约为 SF 的 0.29%–1.08%，对应这一局部指标约 93–346 倍改善。该乘积是固定模型、固定 batch、独立抽样下的近似成本指标，**不是训练加速倍数，也不能推算 G 可直接缩小相同比例**。

六组去噪修正后的信号平方估计都为正；CP 的噪声 RMS / 修正信号范数约 0.37–0.67，SF 约 4.99–10.36。CP 的较小梯度范数与去除大量随机成分一致，不能仅据范数变小认定学习信号变弱。

配对均值差范数约为其 MC RMS 尺度的 0.963–1.034 倍，差异与抽样噪声处于同一量级；当前没有仅凭此指标可识别的超额均值差。但这不是显著性检验或无偏性证明，16 draws 也不能精确界定小偏差。SF/CP 样本均值余弦仅约 0.39–0.62，应结合 SF 的大均值估计误差解释，不能孤立判断两者优化目标不同。若要写入强均值一致性主张，按原计划累计独立 draws 至 64，必要时 256。

CUDA allocated 峰值两种估计器基本相同，随 batch 约为 33.08–39.16 GB（十进制）。CPU RSS 是进程累计高水位，不用于分配 SF/CP 各自额外内存。时间含初次调用影响，且是单卡诊断，不能替代 8 卡完整训练成本。

## 6. 为何强降噪没有变成所有 reward 的强检索收益？

现有数据能确认这两者不等价，但尚未识别具体原因。合理的待验证解释是：降噪提高了随机小候选目标的梯度估计精度，而最终 BRIGHT 使用确定性检索；reward 信息、候选覆盖和跨领域迁移仍决定目标是否有效。此外 AdamW 的二阶状态会随噪声改变，所以相同 LR 下的最终更新并非只改变原始梯度方差。

这些是机制假设。当前没有 sampled reward 学习曲线、同池 deterministic 指标、大池检索指标或 optimizer 更新统计，不能声称已定位为目标失配、候选不足或 AdamW 尺度问题。尤其当前探针只测 MRR，不能用它直接给 graded 的 +2.19 分做因果归因。

## 7. 建议的下一步优先级

1. **先补齐证据归档。** 核对/重新生成新协议 E0；同步 27 run 的 contract/state、原始评测 JSON、事件与训练日志，确认同数据、step 113，并补 train wall time/GPU-hours。这样才能回答 RL 是否值得额外成本。无需因此先推倒已有 27 次训练。
2. **主线聚焦 graded CP，保留 graded SF 与 LL-Graded 对照。** 在相同固定 batch 和相同模型权重上追加 graded reward 配对探针；优先覆盖 E0 及一个预先指定的 graded RL checkpoint。若强化均值一致性论证，累计至 64 个互不重复的 draws，必要时 256；不同模型状态不能合并计算一个方差。
3. **用已有 checkpoint 做三层机制分析。** 固定 query/候选/标签，对比 sampled reward、未经 RL 均值校准缩放的同池 deterministic 排序、同 source 大池检索。优先解释 graded CP 对 LL 的优势及 leetcode/aops 的损失；MRR CP/SF 可作为强降噪但均分不升的机制对照。这些训练 query 的结果只用于机制。
4. **需要算力主张时，再补 graded SF/CP × G32。** 复用当前 G64，以相同预算和三 seed 配对比较质量与实测训练成本，不从探针的百倍降噪直接推断训练采样成本。
5. **确认外部泛化后再扩 G2。** 冻结配方，按计划核对真正未参与选择的任务/查询并预先保存确认集合；若不存在，限定为已知 benchmark 的受控比较。BRIGHT 已参与开发，增加 training seed 不会使它变成未见测试。

当前可写入论文的主张是：在这套固定 G64/LR/步数配方和三个 seed 下，graded CP 提高 BRIGHT 宏平均并超过匹配的 LL-Graded；MRR 固定状态探针显示强全参数梯度降噪。暂不支持普遍提高所有 reward、普遍降低 seed 波动、无偏性已被实验证明、同算力优于监督方法或未见任务泛化已验证。

## 输入文件

- [run_summary.csv](_summary/g1_r2_bright/run_summary.csv)
- [subset_summary.csv](_summary/g1_r2_bright/subset_summary.csv)
- [E0 probe](../outputs/g1_r2_gradient_probe/gradient_probe/attempt-1.json)
- [LL step25 probe](../outputs/g1_r2_gradient_probe/gradient_probe_ll25/attempt-1.json)

本文仅分析本轮 R2，不把历史训练结果拼入三 seed 均值。所有数据核验均在 CPU 上读取现有 CSV/JSON 完成，未重新训练或评测模型。
