# G1 原始梯度诊断重分析

日期：2026-09-15。输入为 `outputs/rollout_gradients/` 下 27 份 JSON、81 个固定状态 probe，每个 probe 有 16 次 rollout 梯度估计。本记录修正[首次结果报告](rollout_variance_results.md)中的统计解释，未重跑模型或修改原始结果。

## 1. 修订结论

- **增大 G 的降噪符合约 1/√G 的量级。** 同一目标、同一模型和输入下，G 翻倍后噪声方差变为原来的 0.412–0.531，噪声 RMS 变为 0.642–0.729。原报告从 `noise/mean` 只变化约 8–10% 推断“强烈次于 √G”，应撤回。
- **当前单位方向逐文档 baseline 的负结果成立。** Final 上，MRR 方差增至 2.17–12.06 倍，graded nDCG 增至 2.36–3.16 倍；E0 上也没有一致的绝对方差收益。
- **换 reward 有条件性的正向信号。** 修正分母后，graded nDCG 在 final 的三个单 microbatch probe 上均优于 MRR，E0 八 microbatch 的三个 probe 也均优于 MRR；E0 单 microbatch 不一致。修正比值仍是小样本 plug-in 估计，尚不是确定的总体 SNR。
- **不能认定“vMF 固有噪声难以降低”。** 单次梯度方向一致性较低仍是事实，但 G 的直接方差证据表明采样预算能有效降低噪声。是否划算、能否改善完整训练结果，是另外的问题。

## 2. 数据核查与复现

运行 [analyze_rollout_gradients.py](../scripts/analyze_rollout_gradients.py)，仅需 Python 标准库：

```bash
python scripts/analyze_rollout_gradients.py
```

输出位于 `outputs/rollout_gradient_reanalysis/`：`summary.json` 保存输入 SHA-256、逐 probe 统计及 16 组条件配对核查；`per_probe.md` 展示全部 81 个 probe 的原始/修正指标和方差比。这些数值产物沿用 `outputs/` ignore 规则。

已核查每个 probe 的 16 个 seed 无重复且顺序一致；根据逐 draw 梯度范数和平均梯度范数重构的样本方差与原始统计一致。配对条件的模型标识、step、数据 hash、样本身份、精度和参数空间一致，配置差异限于设计因子及 run/output 名称。相同标签的条件还核对了完整 batch tensor hash。

MRR/graded 各自的 shared 与 counterfactual 配对，全部逐 draw `reward_mean` 完全一致。未保存动作 hash，不能将 reward 相同当作动作逐元素相同的独立证明。Graded 条件的完整 batch hash 按预期变化；只有样本身份和整体 hash，不能独立验证差异仅位于标签 tensor。

主单 microbatch 配对均来自 `c918bd8`；E0 八 microbatch 的 MRR 控制来自 `418bd5c`，其他条件来自 `c918bd8`。所有文件记录 `git_dirty=true`，未保存完整 dirty diff，因此不是对执行源码逐字相同的证明。

## 3. 原比值为何压缩差异

设独立梯度样本的真实均值为 μ，协方差迹为 V。脚本记录样本均值 ḡ 与无偏样本方差迹 V̂。原比值 √V̂/‖ḡ‖ 的分母也含有噪声：

\[
\mathbb E\|\bar g\|^2=\|\mu\|^2+\frac VN.
\]

真实信号较弱时，N=16 会使原比值经常落在 √N=4 附近。**4 不是数学上限**；本次 final C0 probe 0 的原比值实际为 **4.08485**，跨 probe 平均后才约为 3.896。

若真实噪声比为 ρ=√V/‖μ‖，按矩的量级近似，原比值约为 ρ/√(1+ρ²/N)。真实比值从 8 降到 8/√2 时，该近似只从 3.578 降到 3.266。因此不能用原比值的相对变化直接检验 1/√G 缩放。这是近似说明，不是随机比值期望的精确公式。

逐 probe 重算：

\[
\widehat{S^2}=\|\bar g\|^2-\frac{\widehat V}{N},\qquad
\hat\rho=\sqrt{\widehat V/\widehat{S^2}}\quad(\widehat{S^2}>0).
\]

S² 的估计无偏，开方及相除后的 ρ̂ **不无偏**。正值也不代表信号统计显著，只表示比值代数上可计算。负值保留并标注未分辨，不截断后求比值。不能先跨 probe 平均原比值再作此变换。

81 个 probe 中，final C0 probe 0 的 S² 估计为 **−226.274**，其修正比值记为未分辨。JSON 没有各次完整梯度或两两内积矩阵，不能从这些聚合量构造可信的梯度方向 bootstrap 区间。以下修正比值仅作描述性估计。

## 4. G：同一目标下直接比较噪声方差

G 改变采样数量，模型、目标和估计器规则一致。V̂ 的比较不依赖有噪声的平均梯度分母。表中为较大 G 的 V̂ / 较小 G 的 V̂：

| 状态 | probe | G16 → G32 | G32 → G64 |
|---|---:|---:|---:|
| E0 | 0 | 0.451 | 0.466 |
| E0 | 1 | 0.495 | 0.480 |
| E0 | 2 | 0.430 | 0.476 |
| Final | 0 | 0.464 | 0.476 |
| Final | 1 | 0.412 | 0.531 |
| Final | 2 | 0.466 | 0.455 |

例如 final probe 0，noise RMS 随 G16/32/64 为 **223.24 / 152.11 / 104.91**。G32→64 的 RMS 降至 0.690，接近 1/√2≈0.707。点估计与方差约按 1/G 缩放相符；没有误差区间，不声称已精确拟合缩放指数。

Product grid 组合数量随 G² 增加。缺少完整耗时/显存对照，不能据此断言 G64 是最佳预算配置，也不能外推达到某个总体 SNR 必需的 G。

## 5. 逐文档 baseline：直接方差负结果更明确

固定 reward 后，新 baseline 理论上保留当前文档 score-function 项的期望，query 项不变。因此可直接比较完整 encoder 梯度的方差，包括两侧协方差。

| 状态 / reward | CF/shared 方差比：probe 0 | probe 1 | probe 2 |
|---|---:|---:|---:|
| E0 / MRR | 1.877 | 2.117 | 1.812 |
| Final / MRR | 3.806 | 12.060 | 2.171 |
| E0 / graded nDCG | 0.942 | 1.111 | 1.266 |
| Final / graded nDCG | 2.893 | 3.158 | 2.364 |

E0 八 microbatch 下，MRR 方差比为 1.537/2.245/1.882，graded 为 1.048/1.722/1.182，也没有一致收益。

**暂停当前单位方向参考版本的完整训练是合理的。** 某些 probe 的修正比值变好，不能抵消同一目标下绝对方差变差的证据；有限采样会使两种估计器的平均梯度范数不同。结果不否定所有 factorized baseline，也不能证明信用分配与噪声无关。

## 6. Reward：存在条件性改善

下表为逐 probe 的修正噪声比 ρ̂。不同 reward 改变了目标/标签信息，不能仅凭绝对方差更小认定同一个梯度估计任务得到改善，也不能仅凭平均梯度范数缩小认定变化只是统一尺度缩小。

| 状态 / reward | probe 0 | probe 1 | probe 2 |
|---|---:|---:|---:|
| E0 / MRR | 6.87 | 15.81 | 13.10 |
| E0 / binary nDCG | 6.83 | 12.08 | 8.76 |
| E0 / graded nDCG | 8.62 | 14.20 | 9.21 |
| Final / MRR | 14.80 | 9.00 | 6.72 |
| Final / binary nDCG | 9.65 | 8.43 | 6.53 |
| Final / graded nDCG | 8.52 | 6.05 | 6.04 |
| E0，8-mb / MRR | 10.06 | 9.39 | 14.04 |
| E0，8-mb / binary nDCG | 8.00 | 7.33 | 6.60 |
| E0，8-mb / graded nDCG | 7.38 | 7.48 | 5.77 |

Final 单 microbatch 和 E0 八 microbatch 中，graded 相对 MRR 的三个点估计均更好；E0 单 microbatch 不一致。Binary nDCG 也有正向信号。不能据此宣称显著性或下游质量收益，但也不应将 reward 路线判定为完全无效。

## 7. 八 microbatch 与探索尺度的解释边界

八个不同 microbatch 平均后的真实信号是各自 μ 的平均，噪声协方差也相应变化。它不是“同一 batch 重复八次”的实验，不能对相对噪声比直接要求 √8 改善。聚合结果不足以独立识别“真实方向相互抵消”，也不足以推出 final 的八 microbatch 结果。

Alignment 改变 κ，也改变采样分布及对应的期望 reward 目标。保留“尚未找到跨状态一致的 alignment 改善”，撤回“与信噪比无关”和“已证实固有噪声”的表述。

## 8. 下一步

1. 若增加诊断预算，预先选定状态与 probe，将独立梯度估计次数 N 从 16 提高到 64/128，优先确认 shared baseline 下 MRR 与 binary/graded nDCG 的比较。N 用于估计统计量；增大 N 不会自动降低训练中每次更新的噪声。
2. 采样预算方向保留 G32/64，不重复已有同规格对照。如比较 G64 与多份独立 G32 梯度平均，应固定模型/输入、明确预算并记录耗时；目前没有“G 几乎无效”的依据。
3. 当前 CF 暂停；若以后更换参考或条件均值 baseline，仍先验证同一目标下的绝对方差。
4. 原独立 rollout 三 seed BRIGHT 结果不因本次重分析改变，它仍不足以单独分解原始 seed 差距。正式训练的质量、稳定性与成本需要各自证据。
