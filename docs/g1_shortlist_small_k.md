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

## 原纯 graded 对照运行

以下 9 条已完成，保留命令供复现。当前入口默认改为下文新增的 6 条 Pairwise050；
复现纯 graded 对照需显式指定 recipes：

```bash
python scripts/run_g1_shortlist_small_k_r2.py check --recipes uniform-k0-t1 uniform-k1-t1 uniform-k3-t1
python -u scripts/run_g1_shortlist_small_k_r2.py train --recipes uniform-k0-t1 uniform-k1-t1 uniform-k3-t1

# 先看 seed42 的三个新点
python -u scripts/run_g1_shortlist_small_k_r2.py train --recipes uniform-k0-t1 uniform-k1-t1 uniform-k3-t1 --seeds 42
# 再补另外两个 seed
python -u scripts/run_g1_shortlist_small_k_r2.py train --recipes uniform-k0-t1 uniform-k1-t1 uniform-k3-t1 --seeds 3407 2026

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
参考核验（SF/CP、rescaling 开关及 CPU BF16 autocast）。当时尚未启动 GPU 训练；现已完成，结果见下节。

## 已完成结果与结论

9 条新增运行已全部回传。以下固定 T1、alignment 0.70、纯 graded reward，
三 seed 顺序为 42/3407/2026，± 为样本标准差（ddof=1）。分数来自
[run_summary.csv](../paper/_summary/g1_r2_bright/run_summary.csv)；领域来自
[subset_summary.csv](../paper/_summary/g1_r2_bright/subset_summary.csv)。
共核验 15 条运行、180 条领域分数，每条 12 个唯一领域，tasks_missing=0、errors=0；
总分与领域宏平均的最大舍入差为 0.00583。此核验不涉及训练日志和实际运行合同。

| K | seed42 | seed3407 | seed2026 | 均值 ± SD | 相对 K7 |
|---|---:|---:|---:|---:|---:|
| 0 | 22.87 | 22.53 | 22.77 | 22.72 ± 0.17 | +0.24 |
| 1 | 22.73 | 22.56 | 22.77 | 22.69 ± 0.11 | +0.21 |
| 3 | 22.55 | 22.59 | 22.68 | 22.61 ± 0.07 | +0.13 |
| 7 | 22.54 | 22.37 | 22.53 | 22.48 ± 0.10 | +0.00 |
| 15 | 22.41 | 21.94 | 22.39 | 22.25 ± 0.27 | -0.23 |

**均值随 K 从 15 降到 0 持续上升，没有观察到零额外候选端点的性能下降。**
K0−K7 的配对差为 +0.33/+0.16/+0.24，均值 +0.24；
K0−K15 为 +0.46/+0.59/+0.38，均值 +0.48。K1、K3 也在三个 seed 上均高于 K7。
但 K0−K1 只有 +0.14/−0.03/0.00（均值 +0.04），K0−K3 为
+0.32/−0.06/+0.09（均值 +0.12）；不能声称每个 seed 都随 K 单调变化或 K0 显著优于 K1。

CL-Strong 的三 seed 分数为 22.50/22.56/22.87（22.64 ± 0.20）；
K0 相对它为 +0.37/−0.03/−0.10，均值仅 +0.08，两负一正，尚不足以判定优于 CL。
纯 graded K0 的 22.72 仍低于已有 K7+Pairwise050 的 23.04 和 K7+Binary025 的 23.01。

### 领域变化

| 领域 | K0 | K1 | K3 | K7 | K15 | K0−K7 |
|---|---:|---:|---:|---:|---:|---:|
| biology | 40.55 | 40.25 | 39.98 | 38.93 | 37.56 | +1.62 |
| earth_science | 38.02 | 37.61 | 36.75 | 36.10 | 34.88 | +1.92 |
| economics | 28.02 | 28.15 | 27.38 | 26.98 | 26.51 | +1.04 |
| psychology | 34.73 | 34.72 | 34.65 | 33.86 | 33.11 | +0.87 |
| robotics | 18.22 | 17.78 | 17.81 | 17.80 | 17.71 | +0.42 |
| stackoverflow | 25.36 | 25.93 | 26.30 | 26.42 | 27.19 | -1.06 |
| sustainable_living | 23.67 | 23.75 | 23.65 | 23.59 | 24.10 | +0.08 |
| pony | 2.29 | 2.33 | 2.30 | 2.30 | 2.31 | -0.02 |
| leetcode | 10.12 | 9.69 | 10.04 | 10.28 | 10.40 | -0.15 |
| aops | 2.70 | 2.61 | 2.72 | 2.79 | 2.95 | -0.10 |
| theoremqa_theorems | 29.61 | 29.92 | 30.26 | 30.86 | 30.28 | -1.24 |
| theoremqa_questions | 19.40 | 19.48 | 19.46 | 19.82 | 19.92 | -0.42 |

K0 相对 K7 改善 6/12 个领域，主要收益为 earth_science +1.92、biology +1.62、
economics +1.04、psychology +0.87；主要下降为 theoremqa_theorems −1.24、stackoverflow −1.06。
去掉 biology 后，其他 11 领域平均仍提高 +0.12。小 K 的收益包含领域间取舍。

### 对原问题的修正

当前数据支持：在这套纯 graded、CP/G64、113-step 配方下，自有候选已经能提供有效监督；
额外跨 query 零相关度候选没有显示宏平均收益。此前“需要保留少量额外负例”的解释未被这轮结果支持。
K0 仍保留全部自有正负例，因此不能表述成“RL 不需要负例”。

原普通 CP 的“无 shortlist”会恢复同卡代表正例作为额外负例，与此处的 K0 不同。
新结果与原 CP 较差并不矛盾。此前固定 K15 的来源对照仍然成立，但不能据此推导额外候选必不可少。

奖励信号退化、CP 投影维度、零相关度标签的偏差均是待验证机制；仅凭最终评测无法确定各自贡献。
若下一步目标是提升最佳配方，优先比较 K0/K1 上的单辅助项与已有 K7 对照，
例如固定 Pairwise050；缩小 K 会同时改变 pairwise 中自有/跨 query pair 的组成，需一并解释。
目前不据此修改辅助项最佳配方，也不继续外推未测结果。

## 新增 K0/K1 + Pairwise0.50

新增两组，各 seeds 42/3407/2026，共 6 条，已接入同一 small-K suite 和脚本。
默认 `check/train/eval` 只选择这 6 条；原纯 graded 配方仍可通过 `--recipes` 显式选择。

| recipe | K | pairwise lambda | binary alpha |
|---|---:|---:|---:|
| `k0-pairwise050` | 0 | 0.50 | 0 |
| `k1-pairwise050` | 1 | 0.50 | 0 |

固定 T1、alignment 0.70、CP/G64、113 steps、LR 5e-6、8×16 batch、相同 E0，
目标为 `E[graded nDCG@10] + 0.5 E[pairwise]`。复用已有逐 pair LOO/CP 实现。
K0 仍比较自有正例与自有负例；K1 还加入抽中的额外负例。改变 K 同时改变
pairwise 平均中的候选组成，不应把差异仅归因于 listwise 难度。
输出名加入 `-Pairwise050`，与已完成纯 graded 实验分开。

```bash
# 默认检查/训练全部 6 条新增运行；每条训练完成后自动评测 BRIGHT
python scripts/run_g1_shortlist_small_k_r2.py check
python -u scripts/run_g1_shortlist_small_k_r2.py train

# 或先运行 seed42，再补其余两个 seed
python -u scripts/run_g1_shortlist_small_k_r2.py train --seeds 42
python -u scripts/run_g1_shortlist_small_k_r2.py train --seeds 3407 2026

# 单配方与重试评测
python -u scripts/run_g1_shortlist_small_k_r2.py train --recipes k0-pairwise050 --seeds 42
python scripts/run_g1_shortlist_small_k_r2.py eval --recipes k1-pairwise050 --seeds 42
```

以已有 K7+Pairwise050（23.04 ± 0.21）为固定辅助目标下的主要对照，
再分别与纯 graded K0/K1 比较辅助奖励的增量。报告三 seed 配对差值与领域变化。
准备验证包括 6 条入口预检、展开配置核对和 K0/K1 联合目标的独立 listwise/pairwise
梯度参考测试；尚未启动这 6 条 GPU 训练。
