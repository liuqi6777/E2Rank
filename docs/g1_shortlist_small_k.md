# G1-R2：T1 的小 K 与零额外候选对照

补充 K=0/1/3，各 seeds 42/3407/2026，共 9 次新训练。复用已有纯 graded
K7/T1（22.48 ± 0.10）和 K15/T1（22.25 ± 0.27），组成 K=0/1/3/7/15 曲线。
现有本地结果中的 K30/K60 使用 T4/T8，不能直接并入这条固定 T1 曲线。

所有运行固定 alignment 0.70、T1、Uniform、跨卡全部候选池、CP/G64、graded nDCG@10、
113 steps、8 卡 × microbatch 16、LR 5e-6、相同 E0 和 prepared 数据，最终 BRIGHT 评测。
不加 binary、pairwise 或 InfoNCE。与对应 seed 的 K7 展开配置相比，只改变 K 和运行/输出名称。

K 仅计算额外跨 query 候选，全部自有正负例一直保留。K0 使用
`reward_shortlist_count: 1`、`reward_shortlist_size: 0`、`reward_shortlist_hard_count: 0`，
继续执行 shortlist reward/LOO/CP 路径。`reward_shortlist_count: 0` 会关闭该路径，不能代替 K0。
K0 仍保留原候选池收集与全局 query 权重计算，以保持分布式执行协议；不把它当作吞吐优化实验。

## 运行

独立入口默认只选新实验，不重跑已有 K7/K15：

```bash
python scripts/run_g1_shortlist_small_k_r2.py check
python -u scripts/run_g1_shortlist_small_k_r2.py train

# 先看 seed42 的三个新点
python -u scripts/run_g1_shortlist_small_k_r2.py train --seeds 42
# 再补另外两个 seed
python -u scripts/run_g1_shortlist_small_k_r2.py train --seeds 3407 2026

# 单独运行或重试评测
python -u scripts/run_g1_shortlist_small_k_r2.py train --recipes uniform-k0-t1 --seeds 42
python scripts/run_g1_shortlist_small_k_r2.py eval --recipes uniform-k0-t1 --seeds 42
```

以上是不同选择方式，不要重复训练已有输出。支持 `--config /path/to/settings.yaml`；
每条训练成功后自动评测，非空输出目录拒绝覆盖。配置位于
[small-K suite](../configs/experiments/iclr2027/suite_g1_shortlist_small_k.yaml)。

## 如何判断

- K0 与 K1/K3/K7：检验额外跨 query 负例是否有收益，是否存在非零最佳 K。
- K1/K3 与 K7/K15：检验缩小 K 的收益是否延续或出现转折。
- 报告三 seed 均值、配对差值及领域变化；先看 seed42 仅作初筛。
- 同时比较 `reward/ndcg_in_batch/query|documents/group_std|degenerate_frac`、
  `projection/query_span_rank_mean` 与 `reward_pool/cross_candidates_mean`。
  reward 尺度随候选集改变，不能将训练 reward 更高直接解释为检索更好。
- K0 仍弱不能单独证明负例来源机制；K0 更好也不能证明其他预算下负例无用。

## 准备状态

2026-09-24：9 条入口预检通过，展开配置通过 `RLArguments` 校验，逐 seed 核对仅 K
与运行/输出名称区别于已有 K7。K0 的采样边界、reward、共享 encoder 梯度通过逐 cell
参考核验（SF/CP、rescaling 开关及 CPU BF16 autocast）。尚未启动 GPU 训练。
