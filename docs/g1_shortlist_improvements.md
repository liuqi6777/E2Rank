# G1-R2 shortlist 后续提点实验

最新结果：前三组 15 次与组合组 9 次训练均已完成；组合结果和当前结论见[第 5 节](#5-组合实验结果与当前结论)。

各组均从相同 E0 开始，复用现有 prepared 数据；113 optimizer steps、8 卡 × microbatch 16、
LR 5e-6、CP/G64、alignment 0.70、固定最终 checkpoint。training/data/rollout seeds 为
42、3407、2026。保持 joint full FT、无 KL、无辅助 InfoNCE，不追加 CL 预训练。
对照为已有 `G1-R2-RL-GradedNDCG64-CP-Align070-ShortlistUniform-K15-T1` 三 seed。
相同步数不表示相同墙钟耗时；新配方需记录实际训练耗时。

## 1. 更小的 shortlist：K7

仅把额外跨 query 候选由 K15 改为 K7。仍从跨卡全部候选中均匀抽样、T1、自有候选全部保留，
沿用 graded nDCG@10 和身份过滤。小池不足时按既有 mask 处理，不补重复候选。
配置并入 [sweep suite](../configs/experiments/iclr2027/suite_g1_shortlist_sweep.yaml)。

```bash
python scripts/run_g1_shortlist_sweep_r2.py check --recipes uniform-k7-t1
python -u scripts/run_g1_shortlist_sweep_r2.py train --recipes uniform-k7-t1
python -u scripts/run_g1_shortlist_sweep_r2.py train --recipes uniform-k7-t1 --seeds 3407 2026
python scripts/run_g1_shortlist_sweep_r2.py eval --recipes uniform-k7-t1 --seeds 42
```

所有脚本默认 `check`，接受 `--config` 指定机器上的 settings。`train` 要求恰好八张支持 BF16
的可见 CUDA GPU；每条训练后自动执行最终 BRIGHT，失败即停，非空输出目录拒绝覆盖。
重试使用 `--seeds` 只选择未完成的运行；已有模型使用 `eval`。
本地配置检查不启动 GPU 训练，也不代表已有新结果。

## 2. Graded / binary nDCG 混合奖励

保持 Uniform K15/T1，通过 `reward_shortlist_binary_weight: alpha` 设置：

\[
R=(1-\alpha)R_{\mathrm{graded\ nDCG@10}}+\alpha R_{\mathrm{binary\ nDCG@10}}.
\]

配置提供 alpha=0.25、0.50，各三个 seed，共六条；alpha=0 的既有基线不重跑。
Graded 项仍读取 `relevance_labels`（suite 指定 `relevance_scheme: graded`）；binary 项独立读取
原始 `positive_mask`，不按 teacher grade 阈值推断。两项有各自的 IDCG，使用完全相同的
actions、候选与跨 query 分数校准。先固定权重求和，再做线性 LOO/CP，与对两项分别做
同一 CP 再加权等价；不额外标准化 advantage，也不增加 encoder 前向或 action draws。

只支持 shortlist 路径的单一 unit-weight `ndcg_in_batch` 主项，alpha 必须有限且处于 [0,1]。
默认 0 保持旧路径与旧 checkpoint 合同；启用后缺少有效 boolean `positive_mask` 会报错。
日志同时记录 `reward/ndcg_in_batch/*`（原 grades）、`reward/binary_ndcg/*` 和混合 reward。
Checkpoint 记录 binary 标签来源与权重，恢复时拒绝切换目标。
配置：[mixed rewards suite](../configs/experiments/iclr2027/suite_g1_shortlist_mixed_rewards.yaml)。

```bash
python scripts/run_g1_shortlist_mixed_rewards_r2.py check
python -u scripts/run_g1_shortlist_mixed_rewards_r2.py train
python -u scripts/run_g1_shortlist_mixed_rewards_r2.py train --binary-weights 0.25 --seeds 3407
python scripts/run_g1_shortlist_mixed_rewards_r2.py eval --binary-weights 0.50 --seeds 42
```

## 3. Graded nDCG + 逐 pair 奖励与 CP

同样固定 Uniform K15/T1。主项为原 graded nDCG，辅助项为原始正例对负例的二值比较：

\[
J=E[R_{\mathrm{graded}}]+\lambda E\left[
\frac{1}{|\mathcal P_q|}\sum_{(p,n)\in\mathcal P_q}
\big(1[s_p>s_n]+\tfrac12 1[s_p=s_n]\big)\right].
\]

`reward_shortlist_pairwise_coef` 提供 lambda=0.25、0.50，各三个 seed，共六条。
这里 lambda 是加法系数，不像上一组 alpha 那样缩小主项；两个单独 suite 不自动组合；第 4 节使用独立 suite 显式组合。
所有原始正例（`positive_mask`）与所有有效自有负例、当前抽中的跨 query 负例配对。
正例之间不比较，也不做 negative-negative 比较；原始负例身份不由 teacher grades 推断。
每条 query 内所有 pairs 等权平均，然后对 query 平均；没有 pair 的 query 贡献零，仍保留在
batch 分母中。候选去重、已知正例过滤、route mask 和 padding 与 shortlist 主项一致。

每个 pair 使用独立的 query/document LOO：分别排除本 query draw 和本 document bundle。
Query 投影到 `span(query_mean, sampled_positive - negative)`，最多二维；固定跨 query
negative 的方向包含既有 frozen-document rescaling。Document 投影到
`span(document_mean, sampled_query)`，只更新该 pair 的自有采样端点，再累加各 pair 系数。
跨 query 文档仍为 detach 的固定方向，不新增其策略梯度。投影、actions、advantage 均 detach，
通过 live means 合成一个 surrogate，与主项一起进行一次 encoder backward。

**不能先把所有 pair 的 reward 合并后做一次 CP。** 本实现先逐 pair 投影再相加，
保留每份 reward 的局部条件空间。条件固定候选下，各项估计对应上述期望 reward；这不表示
它等价于全池 nDCG，也不预先保证完整参数梯度方差或 BRIGHT 分数改善。

实现按 query 和每 32 个 pairs 分块，不创建 `[B,Gq,Gd,pairs,D]` 张量；复用已有动作与
own/cross 分数，不新增模型调用或采样。默认系数 0 保持旧行为，只在 shortlist CP 路径启用。
日志的原 `reward_mean`/advantages 仍描述 listwise 主项；新增 `reward/pairwise/*` 记录 pair
均值、数量、有效波动比例和无 pair 比例，`reward/combined_mean` 记录完整目标均值。
`train/loss_pairwise{,_weighted}` 是策略梯度 surrogate，不等于负 reward。
Checkpoint 合同保存系数、标签、配对范围、归约与估计器版本，拒绝跨目标恢复。

配置：[pairwise suite](../configs/experiments/iclr2027/suite_g1_shortlist_pairwise.yaml)。

```bash
python scripts/run_g1_shortlist_pairwise_r2.py check
python -u scripts/run_g1_shortlist_pairwise_r2.py train
python -u scripts/run_g1_shortlist_pairwise_r2.py train --pairwise-coefs 0.25 --seeds 3407
python scripts/run_g1_shortlist_pairwise_r2.py eval --pairwise-coefs 0.50 --seeds 42
```

前三组共 **15 条运行**（3 + 6 + 6），已在提交 `cfd25d9` 中补齐结果；彼此和原基线的输出目录独立。
旧 51-run 结果分析脚本维持历史快照范围，不自动纳入这些结果或第 4 节的组合结果。

## 4. K7 + binary / pairwise 组合实验

2026-09-19：前三组的三 seed BRIGHT 均值 ± 样本 SD 如下，分数来自
[总分表](../paper/_summary/g1_r2_bright/run_summary.csv)。

| 配方 | 均值 ± SD | 相对 K15 原基线 |
|---|---:|---:|
| K15/T1 原基线 | 22.25 ± 0.27 | — |
| K7/T1 | 22.48 ± 0.10 | +0.23 |
| K15 + Binary 0.25 | 22.90 ± 0.30 | +0.65 |
| K15 + Binary 0.50 | 21.88 ± 0.32 | −0.37 |
| K15 + Pairwise 0.25 | 22.51 ± 0.23 | +0.27 |
| K15 + Pairwise 0.50 | 22.83 ± 0.30 | +0.59 |
| CL-Strong | 22.64 ± 0.20 | +0.40 |

Binary 0.25 和 Pairwise 0.50 相对原基线三个 seed 全部改善，均分超过 CL-Strong，
但尚不足以认定稳定优于 CL-Strong。前者总分最高，后者相对基线的领域均值在 9/12 项改善，
比前者的 7/12 更广；均分领先 CL 仍依赖 biology 优势。收益不能直接相加预测组合结果。

新增三组，每组 seeds 42、3407、2026，共 **9 条运行，结果已在提交 `29e7172` 中补齐**；从相同 E0 开始，
沿用本文开头的 113-step 协议，统一跨卡全部候选、Uniform K7/T1、alignment 0.70。
所有自有候选仍保留；K7 仅限制额外跨 query 候选。

| CLI recipe | binary alpha | pairwise lambda | 比较目的 |
|---|---:|---:|---|
| `k7-binary025` | 0.25 | 0 | 与已有 K15/Binary025 比较 K7 的增量 |
| `k7-pairwise050` | 0 | 0.50 | 与已有 K15/Pairwise050 比较 K7 的增量 |
| `k7-binary025-pairwise050` | 0.25 | 0.50 | 与上两组比较同时启用两个辅助目标的收益 |

统一目标为：

\[
J=E[(1-\alpha)R_{\mathrm{graded}}+\alpha R_{\mathrm{binary}}
+\lambda R_{\mathrm{pairwise}}].
\]

三者组合即 `0.75 * graded + 0.25 * binary + 0.50 * pairwise`，不除以总权重。
先对 graded/binary 混合项执行原 listwise LOO/CP；pairwise 沿用第 3 节的独立逐 pair
LOO/CP，再按 lambda 累加 surrogate，一次 encoder backward。禁止将 pairwise reward
并入 listwise reward 后只做一次 CP。现有实现已支持两个参数同时非零，不新增模型调用或 draws。

Binary 与 pairwise 共用原始 `positive_mask`，可能互补也可能重复强调二值标签。
K7 还会改变 pairwise 平均中自有/跨 query 负例的组成，相同 lambda 不保证相同梯度规模。
结合已有纯 K7 结果，可比较固定 K7 下两个辅助项的交互；没有 K15 双辅助项组合，
不据此声称完整识别 K 与两个奖励项的三因素交互。

日志中 `reward_mean` 和 advantages 对应 listwise 项：alpha 非零时是 graded/binary 混合项。
`reward/ndcg_in_batch/*`、`reward/binary_ndcg/*` 分别保留分项诊断；
`reward/pairwise/*` 对应 pair 项，`reward/combined_mean` 对应完整目标。
记录实际训练耗时、有效 pair 波动比例及 loss 分项；surrogate loss 不等于负 reward，
其数值也不能代替梯度范数。评估按配对 seed 差值、三 seed 均值和逐领域变化报告。

配置：[combinations suite](../configs/experiments/iclr2027/suite_g1_shortlist_combinations.yaml)。
独立入口默认只检查这 9 条；run ID 包含 K7 和启用的奖励项，输出目录与历史运行隔离。

```bash
# 检查全部 9 条；默认 action 也是 check。
python scripts/run_g1_shortlist_combinations_r2.py check
# 全部训练，每条完成后自动评估最终 BRIGHT。
python -u scripts/run_g1_shortlist_combinations_r2.py train
# 只试三者组合的 seed 42。
python -u scripts/run_g1_shortlist_combinations_r2.py train --recipes k7-binary025-pairwise050 --seeds 42
# 仅选择尚未完成的 seed 继续训练；已有模型则单独重试评估。
python -u scripts/run_g1_shortlist_combinations_r2.py train --recipes k7-binary025-pairwise050 --seeds 3407 2026
python scripts/run_g1_shortlist_combinations_r2.py eval --recipes k7-binary025-pairwise050 --seeds 42
```

上述全部训练与选择性训练命令是不同执行方式，不要对已完成目录重复启动 train。
支持 `--config`，八卡 BF16 要求、失败即停和拒绝覆盖非空目录的行为与前三组一致。
以上命令保留作复现与评估入口；这 9 条训练已完成，结果见第 5 节。

## 5. 组合实验结果与当前结论

结果来源：提交 `29e7172` 新增的 **9 次训练、108 条领域分数**；与前三组的 15 次运行合计
24 次后续训练。总分取 [run_summary.csv](../paper/_summary/g1_r2_bright/run_summary.csv)
的 `mean_task_score`，领域取 [subset_summary.csv](../paper/_summary/g1_r2_bright/subset_summary.csv)。
以下三 seed 顺序为 42/3407/2026，± 为样本 SD（ddof=1），领域先跨 seed 平均，差值单位为分。
本节所列 8 组三 seed 均无任务缺项、errors=0，每次有 12 个唯一领域；
总分与领域宏平均的最大舍入差为 0.00583。差值从 CSV 数值计算后舍入。

**当前已测最高均分为 K7 + Pairwise 0.50：23.04 ± 0.21；K7 + Binary 0.25 为
23.01 ± 0.06，两者仅差 0.04 分，尚不能判定优劣。三者组合为 22.78 ± 0.10，
在三个 seed 上均低于上述两个单辅助项配方，不采用当前权重的三者组合。**

### 5.1 总分和配对比较

| 配方 | 42 | 3407 | 2026 | 均值 ± 样本 SD | 相对 CL-Strong |
|---|---:|---:|---:|---:|---:|
| CL-Strong | 22.50 | 22.56 | 22.87 | 22.64 ± 0.20 | +0.00 |
| K15 原基线 | 22.41 | 21.94 | 22.39 | 22.25 ± 0.27 | -0.40 |
| K7 原基线 | 22.54 | 22.37 | 22.53 | 22.48 ± 0.10 | -0.16 |
| K15 + Binary 0.25 | 23.11 | 22.56 | 23.03 | 22.90 ± 0.30 | +0.26 |
| K15 + Pairwise 0.50 | 23.14 | 22.54 | 22.82 | 22.83 ± 0.30 | +0.19 |
| K7 + Binary 0.25 | 23.05 | 22.94 | 23.03 | 23.01 ± 0.06 | +0.36 |
| K7 + Pairwise 0.50 | 23.26 | 23.02 | 22.85 | 23.04 ± 0.21 | +0.40 |
| K7 + Binary 0.25 + Pairwise 0.50 | 22.87 | 22.68 | 22.78 | 22.78 ± 0.10 | +0.13 |

| 配对比较（前者减后者） | Δ42 | Δ3407 | Δ2026 | 平均差 |
|---|---:|---:|---:|---:|
| K7 + Binary 0.25 − K15 + Binary 0.25 | -0.06 | +0.38 | +0.00 | +0.11 |
| K7 + Pairwise 0.50 − K15 + Pairwise 0.50 | +0.12 | +0.48 | +0.03 | +0.21 |
| K7 + Binary 0.25 − K7 原基线 | +0.51 | +0.57 | +0.50 | +0.53 |
| K7 + Pairwise 0.50 − K7 原基线 | +0.72 | +0.65 | +0.32 | +0.56 |
| K7 + Binary 0.25 − CL-Strong | +0.55 | +0.38 | +0.16 | +0.36 |
| K7 + Pairwise 0.50 − CL-Strong | +0.76 | +0.46 | -0.02 | +0.40 |
| K7 + Binary 0.25 + Pairwise 0.50 − K7 + Binary 0.25 | -0.18 | -0.26 | -0.25 | -0.23 |
| K7 + Binary 0.25 + Pairwise 0.50 − K7 + Pairwise 0.50 | -0.39 | -0.34 | -0.07 | -0.27 |

K15→K7 对 Pairwise 的收益三个 seed 同向；对 Binary 的均分提升主要来自 seed 3407，
其余为微负或持平。K7/Binary 在三个 seed 上都高于 CL-Strong；K7/Pairwise 两胜一微负。
Binary 的跨 seed SD 较小，但只有三个 seed，不能据此宣称其训练稳定性已得到充分验证。

### 5.2 固定 K7 下的负交互

以最终 BRIGHT 总分 S 定义描述性的交互差：

\[
I=S_{\mathrm{binary+pairwise}}-S_{\mathrm{binary}}
-S_{\mathrm{pairwise}}+S_{\mathrm{K7}}.
\]

三个 seed 的 I 分别为 **−0.90、−0.91、−0.57**，均值 **−0.79**。
三者组合相对单 Binary 平均下降 0.23，相对单 Pairwise 下降 0.27，且逐 seed 全部下降。
因此现有权重下的两个辅助项未表现出可叠加收益；这不意味着所有 alpha/lambda 组合都无效。

I 衡量最终分数的非加性，不直接证明梯度冲突。Binary 与 pairwise 使用同一原始正例标签，
监督重叠、梯度方向或更新尺度变化均是待检验解释；当前 CSV 不提供区分这些解释的证据。

### 5.3 领域收益与剩余缺口

下表为三 seed 的领域均值；Binary 固定 alpha=0.25，Pairwise 固定 lambda=0.50。

| 领域 | K7 原基线 | K7 + Binary | K7 + Pairwise | K7 + 两项 | CL-Strong |
|---|---:|---:|---:|---:|---:|
| biology | 38.93 | 44.95 | 42.61 | 43.88 | 34.29 |
| earth_science | 36.10 | 36.37 | 36.65 | 35.50 | 34.09 |
| economics | 26.98 | 27.06 | 27.10 | 26.10 | 26.24 |
| psychology | 33.86 | 33.79 | 34.47 | 34.00 | 33.43 |
| robotics | 17.80 | 17.32 | 17.97 | 17.74 | 21.02 |
| stackoverflow | 26.42 | 28.68 | 28.19 | 29.24 | 28.78 |
| sustainable_living | 23.59 | 24.04 | 24.42 | 23.99 | 27.39 |
| pony | 2.30 | 2.21 | 2.25 | 2.02 | 1.16 |
| leetcode | 10.28 | 9.42 | 10.12 | 9.49 | 10.19 |
| aops | 2.79 | 2.63 | 2.58 | 2.19 | 3.11 |
| theoremqa_theorems | 30.86 | 29.82 | 30.50 | 29.68 | 32.31 |
| theoremqa_questions | 19.82 | 19.75 | 19.64 | 19.51 | 19.66 |

| 相对 K7 原基线的指标 | + Binary | + Pairwise | + 两项 |
|---|---:|---:|---:|
| 领域均值提升数量 | 5/12 | 7/12 | 4/12 |
| biology 增益 | +6.02 | +3.68 | +4.95 |
| 去掉 biology 后其余 11 领域平均增益 | +0.03 | +0.28 | −0.12 |

Binary 的净收益主要集中在 biology；Pairwise 在 biology 之外仍有更广的改善。
三者组合在 biology 外平均退步。移除 biology 仅用于敏感性分析，不替换官方全领域宏平均。

相对 CL-Strong，K7/Binary 和 K7/Pairwise 的 biology 优势分别为 +10.66、+8.32；
但 robotics 仍为 −3.70、−3.05，sustainable_living 为 −3.35、−2.97，
theoremqa_theorems 为 −2.49、−1.81。去掉 biology 后，其余 11 领域平均仍分别落后
CL-Strong 0.57、0.32 分。因此宏平均超过 CL 不等于普遍的跨领域优势。

### 5.4 配方选择和结论边界

保留 K7/Binary025 与 K7/Pairwise050 为两个工作候选：前者本轮三 seed 总分更一致，
后者均分略高且相对 K7 的领域收益更广；当前权重的三者组合不作为推荐配方。
若进一步确认超过 CL-Strong，应优先新增配对 seed 或独立评测。

可以报告：**在固定 113-step 条件下，K7 shortlist 上加入 pairwise 辅助奖励，使三 seed
BRIGHT 均分由 22.48 提高到 23.04（+0.56），三个 seed 均提升，均分高于 CL-Strong 0.40 分。**
该配方优化 `E[graded nDCG@10] + 0.5 E[pairwise]`，使用第 3 节的逐 pair LOO/CP。

这些实验支持完整辅助配方有效，但没有相同 pairwise 目标下的其他估计器对照，
不能单独证明收益来自逐 pair CP 的方差降低。辅助项同时改变目标结构和标签使用方式，
也不能将这两者的贡献完全分开。三个 seed 的 SD 不是置信区间，不作统计显著性主张。
反复使用同一 BRIGHT 选参的结果仍属探索性结果；实际训练耗时与梯度日志未在本节审计。
本节更新后续结果记录，不自动改变论文主配方或重写旧 51-run 历史快照。

## 本地验证

```bash
python -m pytest -q tests/test_shortlists.py tests/test_shortlist_objectives.py \
  tests/test_pairwise_projection.py tests/test_conditional_projection.py tests/test_rl_large_pool.py
```

验证包括互相冲突的原始正例/teacher 标签、混合权重端点、独立逐 cell SVD 梯度参考、
多正例/padding/固定池/同分/无 pair、分块与缓存分数、BF16 autocast、两个进程的不齐尾批
及无跨 query 候选、实际 Trainer 更新/模型调用数/保存恢复。真实八卡 CUDA 的训练成本与
BRIGHT 收益以实际实验为准；已回传的 BRIGHT 结果见第 4、5 节，耗时未在此审计。

2026-09-19：上述五个测试文件加 `tests/test_embedding_protocol.py` 共 **152 passed**。
两进程 CPU/Gloo 检查已通过；沙箱内 OpenMP 共享内存受限的测试在沙箱外复核。

2026-09-19 组合实验准备：9 条配置通过入口 `check` 和解析后的 `RLArguments` 校验，
确认 K/权重/seed/预算及独立输出目录。`test_shortlist_objectives.py` 与
`test_pairwise_projection.py` 共 **50 passed**（含 alpha=0.25、lambda=0.50 的独立梯度参考）；
其中 3 条双进程测试因沙箱 OpenMP 共享内存限制，在沙箱外复核通过。该准备阶段未启动 GPU 训练；
后续用户完成的 9 条训练结果已回传，见第 5 节。
