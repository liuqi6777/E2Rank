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
