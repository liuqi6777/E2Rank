# G1-R2 shortlist 后续提点实验

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
这里 lambda 是加法系数，不像上一组 alpha 那样缩小主项；两个单独 suite 不自动组合。
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

三组共 **15 条新运行**（3 + 6 + 6），彼此和原基线的输出目录独立。旧 51-run 结果分析脚本
维持历史快照范围，不把这些待跑配置当成缺失历史结果，也不自动并入新的实验结论。

## 本地验证

```bash
python -m pytest -q tests/test_shortlists.py tests/test_shortlist_objectives.py \
  tests/test_pairwise_projection.py tests/test_conditional_projection.py tests/test_rl_large_pool.py
```

验证包括互相冲突的原始正例/teacher 标签、混合权重端点、独立逐 cell SVD 梯度参考、
多正例/padding/固定池/同分/无 pair、分块与缓存分数、BF16 autocast、两个进程的不齐尾批
及无跨 query 候选、实际 Trainer 更新/模型调用数/保存恢复。真实八卡 CUDA 的训练成本与
BRIGHT 收益需由上述实验测量；本地不启动正式训练。

2026-09-19：上述五个测试文件加 `tests/test_embedding_protocol.py` 共 **152 passed**。
两进程 CPU/Gloo 检查已通过；沙箱内 OpenMP 共享内存受限的测试在沙箱外复核。
