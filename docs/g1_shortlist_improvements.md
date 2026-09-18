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
