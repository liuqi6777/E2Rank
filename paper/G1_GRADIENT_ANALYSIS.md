# G1 梯度与更新规则诊断：W&B API 日志分析

分析日期：2026-09-14。项目：[defaultGroup/E2Rank-RL-v2](https://wandb.ai/defaultGroup/E2Rank-RL-v2)。本文补充 [G1 综合结果](G1_RESULTS.md)，只分析已有训练，不修改配方、不启动新训练。

## 1. 结论

**现有日志确认了更新规则带来的巨大梯度尺度差异，但不支持“DocMean 因梯度太小而没有学动”的简单解释。** Norm、DocMean、NormDocMean 的训练 reward 都持续改善，后 20 步平均 reward 均略高于完整配方，BRIGHT 却分别降低 3.43、6.77、8.20 分。当前更准确的表述是：这些改动显著改变了优化信号，而更高的候选池训练 reward 没有转化为更好的外部检索表现。

还发现一项配置口径问题：虽然 W&B 记录 `max_grad_norm=1`，但六个相关 Git 版本的 `scripts/zero3.json` 都没有 `gradient_clipping`。用远端相同版本的 Transformers/Accelerate 重放配置处理，该字段仍缺失；DeepSpeed 0.18.8 的缺省阈值为 0，即关闭裁剪。**按记录的代码和标准运行路径推断，这批训练应没有启用梯度裁剪。** W&B 的 `output.log` 未保留 engine 最终配置，因此这不是直接读取远端 engine 状态的确认；不能据此计算“实际 clipping 比例”，也不能用 `max_grad_norm=1` 宣称梯度被统一裁剪到 1。

对实验安排的影响：现有结果仍支持最终 MRR 配方内的消融结论；目前无需仅因总梯度量级不同，就追加大规模学习率搜索。已配置的 seed 3407/2026 应保持原配方，包括当前 DeepSpeed 设置。若另行启用 clipping，那会构成新配方，不能当作只改变 seed 的重复。

## 2. 数据取得与可追溯性

通过本机已有凭证调用 W&B Public API，使用 **`scan_history` 完整读取，而非默认抽样的 `history()`**。共读取 10 个 finished run：最终 MRR 配方、七项 MRR 0.90 消融、graded/binary LL-Scaled。每个 run 返回 114 条 history 行，其中 113 条包含梯度，分别对应 optimizer step 1–113；另有一条不含梯度的收尾行。总计 **1,130 个训练 step 记录**，没有缺失或重复 step。

以 Run ID 固定身份，不按可能重名的 display name 自动取最新运行。每个运行的 W&B `output_dir` 都与已导入 BRIGHT CSV 的结果目录前缀匹配。W&B 不包含本次 BRIGHT 逐领域评测，文中 BRIGHT 分数仍来自 [run_summary.csv](_summary/g1_bright/run_summary.csv)。

同时读取每个 run 的 `output.log`、`requirements.txt` 与 `wandb-metadata.json`。本地仅保留数值 history、白名单配置、所需环境信息、文件哈希及相关日志摘录；下载的完整临时文件在处理后删除。十个运行均记录 8 张 NVIDIA L20Y，以及 `torch 2.6.0 / transformers 4.52.3 / accelerate 1.13.0 / deepspeed 0.18.8 / wandb 0.25.1`。

W&B config 包含训练 seed、LR、batch、优化器和 DeepSpeed 文件路径，但没有完整记录 `advantage_norm`、`document_log_prob_reduction` 等 RL 专有参数。因此组件定义仍依据 Run ID、对应项目配置和代码解释，不把 W&B config 说成完整的历史运行配置快照。完整配方与三项更新规则消融的记录 Git SHA 不同，但这两个版本的 `src/grpo.py`、`src/grpo_trainer.py`、`src/policy_math.py` 和 `scripts/zero3.json` 内容一致。

### 指标口径

| 本文指标 | W&B 原始字段 | 含义 |
|---|---|---|
| 总梯度范数 | `train/train/grad_norm` | Trainer 从 DeepSpeed 取得的全局梯度范数；不是参数更新范数 |
| 训练 reward | `train/reward/mean` | 当前训练 batch 的采样动作在训练候选池上的 MRR@10 |
| Advantage std | `train/advantages/std` | 实际使用的 advantage 的聚合标准差，标准化版本与未标准化版本口径有意不同 |
| Query/document group std | `train/reward/mrr_in_batch/{query,documents}/group_std` | 对另一个动作轴平均后，本动作轴 reward 的组内波动；不是梯度范数 |
| 退化比例 | `train/reward/mrr_in_batch/{query,documents}/degenerate_frac` | 组内 reward 标准差低于实现阈值的样本比例 |
| Distinct reward levels | `train/reward/mrr_in_batch/n_distinct` | 每个 query 的 rollout grid 中实际出现的不同奖励值个数，随后平均 |

`train/train/grad_norm` 的双重 `train/` 前缀来自项目先重命名、W&B callback 再加前缀，不是两份不同指标。标量诊断由训练器跨 rank 聚合。本文全程均值对 113 个 step 等权，后期均值固定使用 step 94–113；这些不是独立样本的统计检验。

## 3. 核心 2×2：尺度确实变了，但训练 reward 没有变差

| 配方 | 平均总梯度范数 | 相对完整配方 | Advantage std 均值 | 后 20 步训练 MRR | BRIGHT nDCG@10 |
|---|---:|---:|---:|---:|---:|
| [完整配方：sum、不标准化](https://wandb.ai/defaultGroup/E2Rank-RL-v2/runs/staphpap) | 62.46 | 1.00× | 0.0647 | 0.7286 | **22.01** |
| [Norm：sum、标准化](https://wandb.ai/defaultGroup/E2Rank-RL-v2/runs/vs7gdwj0) | 751.87 | 12.04× | 0.8819 | 0.7349 | 18.58 |
| [DocMean：mean、不标准化](https://wandb.ai/defaultGroup/E2Rank-RL-v2/runs/lyfda2ic) | 14.43 | 0.23× | 0.0671 | 0.7339 | 15.24 |
| [NormDocMean：mean、标准化](https://wandb.ai/defaultGroup/E2Rank-RL-v2/runs/ntm7v4o1) | 209.22 | 3.35× | 0.8809 | 0.7479 | 13.81 |

![G1 MRR 更新规则的梯度、reward 与退化诊断](g1_gradient_analysis/gradient_diagnostics.png)

图中浅线为原始每步日志，深线为尾随 7 步平均；梯度和 advantage 使用对数纵轴。曲线不跨 run 聚合，不表示误差区间。退化比例图使用 query 轴；这四个 run 的 document 轴退化比例与 query 轴逐步相同，但两轴的 group std 不同。

### 3.1 第一步已能观察到纯更新规则的尺度差异

四个配置在第一步记录了完全相同的 reward 均值、query/document group std、distinct reward levels 和退化比例。其中 reward 为 0.533329，query group std 为 0.040320，document group std 为 0.053292；此时尚未经过不同配方的长期训练，梯度却已经明显不同：

| 配方 | 第一步总梯度范数 |
|---|---:|
| 完整配方 | 74.30 |
| Norm | 837.53 |
| DocMean | 13.91 |
| NormDocMean | 208.36 |

这支持“改变更新规则本身就改变了总梯度尺度”，而非将全部差异归因于训练后模型落入不同区域。不过，同一 reward 统计量不能单独证明每个输入和随机张量完全相同；该解释还依赖相同 E0、seed、已准备数据和声明的消融设置。

DocMean 的总梯度并非缩小为候选数的倒数，因为它只对 document 项除以有效文档数，query 项不受同样缩放；不同 query 的候选数又不同，最后还要叠加共享 encoder 中的向量梯度。Norm 则将每个 query/组件的 advantage 按各自的组内标准差缩放，既改变总体量级，也改变不同样本与组件的相对权重。

### 3.2 DocMean 没有表现出“没有训练起来”

完整配方的前/后 20 步平均 reward 为 **0.5613 → 0.7286**；DocMean 为 **0.5748 → 0.7339**。NormDocMean 的后期 reward 最高（0.7479），BRIGHT 却最低（13.81）。四个配置的梯度曲线均没有持续增大至发散的表现，日志中的全部 113 步梯度均为有限值；Norm 的大梯度主要体现为稳定的高量级，而不是少数异常尖峰。

因此，当前数据不支持“mean 把梯度压小，所以训练目标没有被改善”的解释。它也不证明已排除欠拟合：训练 reward 来自变化的训练 batch、采样动作和固定小候选池，不能替代完整训练集上的固定模型评测。可以确定的是，**这组训练 reward 的改善幅度不能解释 BRIGHT 的排序**；不能仅依据 reward 高低选配方。

### 3.3 不能把总梯度倍率当成有效学习率倍率

AdamW 使用一阶和二阶矩估计。在忽略 epsilon 等影响的近似下，若所有步骤的全部梯度统一乘一个常数，一阶矩也乘这个常数、二阶矩乘其平方，最终更新中的尺度会大幅抵消。因此，Norm 的总梯度约为 12 倍，不等于参数每步更新约为 12 倍，也不能直接把其学习率除以 12 当作公平性修正。

实际 Norm 和 DocMean 更复杂：它们分别按样本、动作组件或候选数改变权重，并非对整个梯度统一乘固定常数。这会改变梯度方向和 Adam 的历史状态。现有日志没有实际参数更新范数、query/document 梯度向量或二者余弦相似度，无法从总范数分解出各自贡献。

## 4. Reward 退化没有解释这三个消融的性能下降

| 配方 | Query group std | Document group std | Query/document 退化比例 | Distinct reward levels |
|---|---:|---:|---:|---:|
| 完整配方 | 0.0347 | 0.0494 | 27.24% | 3.550 |
| Norm | 0.0346 | 0.0519 | 26.14% | 3.551 |
| DocMean | 0.0372 | 0.0521 | 24.95% | 3.658 |
| NormDocMean | 0.0369 | 0.0515 | 26.29% | 3.524 |

四个配置的 reward 波动和实际奖励层级接近；DocMean 的退化比例甚至略低于完整配方。Document 轴的 group std 没有接近零，且在四个配置中都高于 query 轴。这些日志不支持“DocMean/Norm 的性能差是因为文档侧没有可用 reward 信号”。但它们测量的是 reward，而非 document 梯度，不能据此断言 document 梯度一定主导整体更新。

补充消融提供了不同方向的观察：

- **QPolicy / DPolicy：** 各自活动轴的退化比例为 47.37% / 40.02%，高于完整 product 配方的 27.24%；后期训练 reward 也更低。仅单侧动作的训练信号更容易退化，与其较低 BRIGHT 一致，但策略空间和 reward 分布也随动作组成改变，不能单独归因为梯度大小。
- **Paired：** 总梯度均值 106.32，高于 product 的 62.46；advantage std 为 0.1376，高于 0.0647，BRIGHT 反而较低。两个轴的 group std 都是 0.0946，因为 paired 使用同一条对角奖励向量，不能把它与 product 的轴边际波动当成完全相同的估计对象。
- **关闭校准：** 总梯度均值 61.77，与完整配方的 62.46 接近；退化比例 26.64%，没有比完整配方更高。其 BRIGHT 只低 0.29。这与“校准不是最终弱探索配方的主要增益来源”一致。
- **LL 对照：** Graded/binary LL 的总梯度均值分别为 2.50/5.71，但损失定义与 RL 不同，不能据绝对大小判定哪个训练更充分。其梯度总体随训练下降；完整统计保留于[附表](g1_gradient_analysis/statistics.md)。

## 5. Clipping 核查：声明的阈值与实际配置不是一回事

不能看到梯度大于 1 就认定梯度爆炸，也不能因为 `max_grad_norm=1` 就认定这些运行每一步都发生了裁剪。此次核查分为四层：

1. **远端配置事实：** 十个 run 的 W&B config 都记录 `max_grad_norm=1`、`deepspeed=./scripts/zero3.json`，但没有上传最终展开的 DeepSpeed engine 配置。取得的 `output.log` 中也没有 `gradient_clipping` 或 engine 启动配置行。
2. **代码版本事实：** W&B metadata 指向的六个 Git commit 在本地都可读取，其 `scripts/zero3.json` 均未声明 `gradient_clipping`。
3. **相同版本的配置处理复现：** 本地与远端的 Transformers、Accelerate、DeepSpeed、PyTorch 版本一致。调用 `HfTrainerDeepSpeedConfig.trainer_config_process/finalize` 和 `DeepSpeedPlugin.deepspeed_config_process`，原配置的裁剪字段最终仍缺失；只在内存中补入 `gradient_clipping: auto` 后，该字段才解析为 1.0。没有改写仓库中的训练配置。
4. **实现语义：** DeepSpeed 0.18.8 的 `GRADIENT_CLIPPING_DEFAULT=0.0`；ZeRO-3 的 `unscale_and_clip_grads` 仅在 `clip_grad>0` 时执行裁剪。Accelerate 的 DeepSpeed 分支把裁剪交给 engine，不额外按照 Trainer 参数裁一次。ZeRO-3 在内部裁剪之前保存 loss-scale 还原后的 `_global_grad_norm`。

由此可推断，**在记录的代码内容没有远端未提交修改、且走标准配置处理路径的前提下，这批运行未启用 clipping**。这是比“所有 step 的范数都大于 1，所以都被裁剪”更符合现有证据的解释；仍保留远端最终 engine 配置未保存这一限制。

复现结果见 [clipping_check.json](g1_gradient_analysis/clipping_check.json)，核查代码见 [check_clipping.py](g1_gradient_analysis/check_clipping.py)。它只处理配置和模型维度占位值，不实例化模型、不需要 GPU。

这一发现不自动使既有结果失效，也不应悄悄改变即将执行的 seed 重复。当前综合文档中的 `max_grad_norm=1` 应当理解为 Trainer 声明参数，不能继续表述为已核实的实际裁剪阈值。如果之后希望研究 clipping，应另建清楚命名的新实验。

## 6. 对论文解释与下一步的建议

可以把更新规则消融写得更具体：

> 在最终 MRR 配方中，advantage 标准化和文档平均分别将日志总梯度范数变为完整配方的约 12.04 倍和 0.23 倍，说明它们显著改变了优化信号尺度。然而，这些消融的后期训练 reward 和 reward 退化诊断并未恶化，而 BRIGHT 分数明显下降。因此，性能差异不能简单归因于训练信号消失或未能改善训练 reward；现有观察更支持更新规则影响了训练目标向外部检索任务的迁移。由于没有分组件梯度和实际参数更新统计，尚不能确认具体的梯度方向或优化机制。

现阶段优先执行已配置的 seed 重复与既定 G2，保持配方不变。只有论文要进一步主张某个精确机制时，再考虑补充：同一 batch 上 query/document 参数梯度的范数和夹角、实际 optimizer 更新范数，以及明确导出的 engine clipping 设置。单侧 policy run 的总梯度不能替代完整双侧 run 的分支梯度测量，因为采样分布和奖励计算也已变化。

本次没有追加训练、没有改动 clipping 或学习率，也没有把间接推断写成已经验证的机制。

## 7. 材料与复现

- [完整统计表](g1_gradient_analysis/statistics.md)与[统计 CSV](g1_gradient_analysis/statistics.csv)：十个 run 的全程、前 20 步、后 20 步、首步和极值。
- [核验与比较记录](g1_gradient_analysis/analysis.json)：数据完整性、首步匹配、各项倍率和原始文件哈希。
- [日志曲线 PNG](g1_gradient_analysis/gradient_diagnostics.png) / [SVG](g1_gradient_analysis/gradient_diagnostics.svg)。
- [API 读取脚本](g1_gradient_analysis/fetch_wandb.py)：凭证由本地 W&B 配置提供，不写入脚本；读取缓存存在时复用。原始标量与筛选后的环境证据位于 `g1_gradient_analysis/raw/`。
- [离线分析脚本](g1_gradient_analysis/analyze.py)：不访问网络，直接关联已导入的 BRIGHT CSV。

```bash
# Read-only API retrieval; reuses existing cached run snapshots.
python paper/g1_gradient_analysis/fetch_wandb.py

# Recompute statistics and plots without contacting W&B.
uv run --no-project --with matplotlib python paper/g1_gradient_analysis/analyze.py

# Reproduce config handling with the same installed versions as the remote runs.
python paper/g1_gradient_analysis/check_clipping.py
```
