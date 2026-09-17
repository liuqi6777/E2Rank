# G1-R2：纯 RL 的跨卡大负例池

本文描述固定跨 query 文档的原大池模式。新增的可采样、可接收所有 query reward 梯度的模式
见[跨 query 联合策略](cross_query_document_policy.md)，同时支持普通池和大池，以独立开关启用。

本轮只改变 G1-R2 alignment 0.90、CP、graded nDCG@10 配方的候选池：每条 query 可以使用八张卡当前
microbatch 的全部候选文档作为跨 query 负例。复用 Strong CL 的候选收集、去重和
已知正例过滤实现；训练文件、样本顺序和 batch 协议保持一致，不挖掘或新增数据。
跨 query 标签沿用零相关度约定，自有候选仍使用原 teacher grades。

默认依次运行 **CP、alignment 0.90、seeds 42 / 3407 / 2026**，本批不包含 SF。
配置以默认 G1-R2 的 G64 CP 对照为基准。其余为原固定 Qwen revision、
joint full FT、LR 5e-6、113 steps、8 卡 × microbatch 16、gradient accumulation 1、
无 dev、无梯度裁剪、最终 BRIGHT 12 subset 评测。`aux_infonce_coef: 0`，纯 RL。

## 服务器启动

同步代码后，在项目根目录、已激活的训练环境中执行；`configs/experiments_r2.yaml`
的 `G1.data` 指向原 prepared 数据目录，`output_dir` 指向本机实验输出根目录。
入口使用与 Strong CL 相同的直接启动方式，不接管旧队列或旧结果。

```bash
# 配置与数据文件预检，不加载模型或启动 GPU
python scripts/run_g1_rl_large_pool_r2.py check

# 依次跑 CP 三个种子，每个种子训练成功后自动评测最终模型
nohup python -u scripts/run_g1_rl_large_pool_r2.py train > g1_rl_large_pool_cp_align090.log 2>&1 &
tail -f g1_rl_large_pool_cp_align090.log
```

训练需恰好八张可见、支持 BF16 的 CUDA GPU；按需在命令前设置 `CUDA_VISIBLE_DEVICES`。
W&B 默认 offline；已显式设置的 `WANDB_MODE` 会保留。

```bash
# 分三台八卡机器运行时，各机器分别选择一个 seed
python -u scripts/run_g1_rl_large_pool_r2.py train --seeds 42
python -u scripts/run_g1_rl_large_pool_r2.py train --seeds 3407
python -u scripts/run_g1_rl_large_pool_r2.py train --seeds 2026

# 训练已完成，仅补最终评测
python -u scripts/run_g1_rl_large_pool_r2.py eval --seeds 42
```

输出：

```text
checkpoints/iclr2027-r2/
  .cl_strong_configs/G1-R2-RL-GradedNDCG64-CP-Align090-LargePool-s42.json
  G1-R2-RL-GradedNDCG64-CP-Align090-LargePool-s42/
    checkpoint-25/ ...
    exploration_state.json
    mteb_eval/bright/
  G1-R2-RL-GradedNDCG64-CP-Align090-LargePool-Seed3407-s3407/
  G1-R2-RL-GradedNDCG64-CP-Align090-LargePool-Seed2026-s2026/
```

展开配置复用启动器现有的 `.cl_strong_configs/` 目录，由独立 run ID 区分，不覆盖 CL 配置。
其他 seed 具有相同的模型/checkpoint/评测目录结构。输出名包含 `Align090`，标明默认 alignment 0.90。
训练输出非空时拒绝覆盖；训练中断后使用新的 settings `output_dir`，不自动恢复。
若同一命令中前一项已完成，后续重试请只选择未完成的 seed；已有模型用 `eval`。

## 实现与验证口径

- `reward_cross_device_negatives: true` 与 `ndcg_in_batch_include_negatives: true` 同时开启。
  目前支持静态 joint query/document product rollout、固定跨 query 文档和 shared
  document baseline，reward 为有明确正 cutoff 的 in-batch nDCG/MRR。
- 候选池与 Strong CL 相同，但梯度路径保持原 RL 设计：跨 query 文档使用 detach 的
  未扰动表示；自有 query/documents 继续采样并更新。保留 frozen-document rescaling。
  大池覆盖相同不代表与 Strong CL 的跨 query 文档直接梯度等价。
- 沿 query 动作轴分块计算完整 reward，保留候选顺序及同分时的原 `topk` 行为。
  没有截断候选、筛选 hard negatives 或改变 reward。避免实体化整个 `[B,G,G,pool]` 张量。
- CP 纳入全部有效固定候选方向。固定矩阵每条 query 只分解一次，采用保持
  `A A^T` 的 SVD 因子压缩并保留原矩阵宽度的数值秩阈值；分块计算后续投影。
  固定方向已保证满数值秩时，query 投影直接为恒等映射，document 投影保持原实现。
- 跨 rank 不齐的 batch/slate 会 padding 并屏蔽；按全局 query 数校正 rank-local loss。
  新候选池开关写入 checkpoint 状态，不允许与旧候选池混用恢复。

新增日志 `reward_pool/cross_candidates_mean`、`reward_pool/cross_candidates_max`、
`reward_pool/zero_reward_frac`。结合原 `reward/*/n_distinct`、各动作轴的
`group_std` / `degenerate_frac`、`projection/query_span_rank_mean` / `max` 和最终
BRIGHT 判断扩池效果。大池可能让 query 投影接近满秩，因此不能预设仍有原来的降噪幅度。

CPU 验证覆盖完整/分块 reward（含同分、padding、无可用跨 query 候选）、压缩/完整
SVD 投影、SF/CP 双进程跨卡梯度与单进程全局参考一致性（不齐 batch/slate）、
配置解析和原 Strong CL 的梯度回归；不代替真实八卡 CUDA 的显存和耗时测量。

```bash
python -m pytest -q tests/test_rl_large_pool.py tests/test_aux_infonce.py tests/test_conditional_projection.py
```
