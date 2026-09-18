# 当前跨卡 batch 池上的多 shortlist listwise RL

已完成结果见 [G1 Shortlist 结果总结](../paper/G1_SHORTLIST_RESULTS.md)：17 个配方、51 次训练，含三 seed 完整总分、领域分析和复算脚本。当前最佳已测均值为 Uniform K15/T1 / alignment 0.70 的 22.25，仍低于 CL-Strong 22.64。

保留每条 query 的全部自有候选（所有正例和自有 negatives），再从当前跨卡 batch 的其他文档中
抽取多组额外 negatives。每组分别计算 listwise reward、LOO 和 CP，最后平均 surrogate loss。
纯 RL 示例使用 graded nDCG@10，不加入 pairwise 或 InfoNCE 辅助项。

同卡其他 query 的候选已包含在跨卡池中；不再额外追加一次同卡代表正例。候选统一经过既有
身份过滤、去重、已知正例排除和 route mask。跨 query 文档按既有零相关度约定评分，维持
detached 均值及 frozen-document rescaling；直接策略梯度覆盖 query 和自有采样文档。
不涉及历史 queue、其他 accumulation microbatch 或全库检索。

## 固定预算与采样

```yaml
reward_cross_device_negatives: true
ndcg_in_batch_include_negatives: true
reward_shortlist_count: 8          # T；对照为 16；未配置时默认 0，原训练行为不变
reward_shortlist_size: 15          # K；不含自有候选
reward_shortlist_hard_count: 8     # 每组优先从高分区取的名额；0 为全池均匀采样
reward_shortlist_hard_pool_size: 128
rollout_seed: 42
```

每步使用 detached、未扰动的 query/document 均值点积选择候选，不使用 sampled actions 或
本次 rollout reward。每条 query 先把有效池按分数分成前 H 个高分候选和其余候选，分别随机
排列，按每组 8/7 的优先比例交错分组。高分区或其余区耗尽时，从另一部分尚未使用的文档
补齐；**整个有效池用完后才循环复用**。覆盖优先于维持耗尽后的 8/7 比例。

因此，有 P 个有效额外候选时，每组含 min(K,P) 个不同文档，T 组共覆盖 min(P,TK) 个不同文档。
P<K 时每组保留全部 P 篇并 padding；P=0 时退化为自有候选排序，仍保持分布式调用一致。
循环边界也不会导致单组重复。高分区由稳定分数排序定义，同分时按池内原顺序处理。

例如 P=2000、K=15 时，T=8 接触 120 篇（6%），T=16 接触 240 篇（12%），
不为穷尽 2000 篇自动增加 T。两个 Mixed 对照都固定 H=128，候选充足时可维持
每组 8 个高分区文档、7 个其余文档，避免仅因 H=64 在第 8 组耗尽而改变后续组的配比。
新 batch/模型状态下重新选择候选；不维护跨步“已见文档”集合，也不保证全训练期完整覆盖。
这些默认数值是初始实现设置，不代表当前推荐配方；已完成结果更支持 Uniform K15/T1。
高分不等于在当前扰动尺度下可学习，需同时
观察每组奖励变化和 LOO 退化。若几乎总为零，应调整难度配比或候选规模。

采样使用现有 rank/step/train-eval/microbatch 随机数框架中的独立 `shortlist` 流。
不会改变模型/dropout/data RNG，也不会改变 query/document action 流。
相同状态、其他采样参数与 seed 下，T=8 与 T=16 的前 8 组一致；增加 T 不改变采样流消费次数。
采样规则、参数和版本写入 checkpoint contract，恢复时禁止切换；旧 checkpoint 缺少该字段
视为未启用。与现有 RNG 一样，恢复保证针对 optimizer step 边界，world size 必须一致。

## 目标与估计器

对当步选定的候选集 C₁,…,C_T，优化 (1/T)∑ₜ E[R(Cₜ)]。候选选择停止梯度，CP/SF 估计的是
固定这些集合与固定跨 query 文档后的条件策略梯度。它不是全池 nDCG 的无偏分解。
后续组在某个分层耗尽后可能具有不同难度分布，不要求各组独立同分布。

每组在全部 Gq×Gd cells 中保持不变，使用自己的 query/document LOO。
Query projector 只包含本组全部实际评分文档；不对所有组的候选并集投影，也不先平均奖励
再投影。组间复用同一套 actions，自有文档仍参与每组 reward 和梯度；平均而非求和避免
仅因 T 增加就把更新尺度乘 T。

每组最多 20 篇自有候选、15 篇额外文档时，query span rank≤36，与 T 无关。
Document 投影沿用原实现。SF 也支持同样的 shortlist 目标，便于同动作配对比较。
初版要求静态 joint query/full-document vMF、固定 κ 或预定 schedule、product rollout、
LOO、无 advantage normalization、shared document baseline、sum reductions 和显式 rollout seed。
不与 sampled joint cross-query policy 组合。

Encoder 前向、跨卡池收集和动作采样次数均不随 T 增加，统一执行一次模型 backward。
大池只做一次均值评分；各组只收集 K 篇文档，逐组计算 reward/CP，不保留完整
`[B,T,Gq,Gd,pool]` 张量。计算增量主要是 T 组小榜单 reward 和投影；复用 actions 意味着
梯度相关，不能宣称总方差降低为 1/T。

## 运行与诊断

独立 [suite](../configs/experiments/iclr2027/suite_g1_shortlists.yaml) 提供 Mixed/Uniform × T8/T16 ×
seeds 42/3407/2026，共 12 条，均为 CP/G64/alignment 0.80，继承主结果选择及原 113 steps、8×16 batch、LR 5e-6、
prepared 数据和最终 BRIGHT 协议。未加入旧队列。配置保持 aux_infonce_coef=0。

| 配置 | 组数 | 每组额外负例 | 高分候选区 | 每步最多不同额外负例 | Seeds |
|---|---:|---|---|---:|---|
| Mixed-T8 | 8 | 8 个高分区 + 7 个其余池 | 前 128 个 | 120 | 42 / 3407 / 2026 |
| Mixed-T16 | 16 | 8 个高分区 + 7 个其余池 | 前 128 个 | 240 | 42 / 3407 / 2026 |
| Uniform-T8 | 8 | 全有效池均匀取 15 个 | 不使用分层 | 120 | 42 / 3407 / 2026 |
| Uniform-T16 | 16 | 全有效池均匀取 15 个 | 不使用分层 | 240 | 42 / 3407 / 2026 |

每组均保留全部自有候选。表中配比与覆盖数按有效候选充足计算，不足时按前述规则补齐或复用。
每条运行的 training/data/rollout seed 使用同一所选值，shortlist 流随 rollout seed 改变；
prepared 数据与固定预处理种子保持不变。seed 42 的原 run ID/输出目录保留，新增 seeds 使用
独立的 `-Seed3407` / `-Seed2026` run ID。

```bash
# 默认检查全部 12 条；不加载模型、不启动训练
python scripts/run_g1_shortlists_r2.py check

# 默认依次训练全部 12 条，每条成功后评测最终 BRIGHT
python scripts/run_g1_shortlists_r2.py train

# 只跑 Mixed/T8 的三个 seed
python scripts/run_g1_shortlists_r2.py train --sampling mixed --groups 8

# seed 42 已完成时，只补另外两个 seed，共 8 条
python scripts/run_g1_shortlists_r2.py train --seeds 3407 2026

# 仅运行一条
python scripts/run_g1_shortlists_r2.py train --sampling mixed --groups 8 --seeds 42
```

默认顺序为 Mixed-T8、Mixed-T16、Uniform-T8、Uniform-T16，每种配置内按 42、3407、2026
运行。任一训练或评测失败即停止。`--seeds` 可按种子拆分到不同八卡机器；重试时使用
sampling/groups/seeds 显式选择尚未完成的组合，脚本不会自动跳过已有输出。

`--config` 指向服务器 settings。check 沿用直接启动器，只校验展开配置/输入路径；训练要求
八张 BF16 CUDA 卡，输出非空拒绝覆盖。可用 eval 重试已训练模型的评测。其他 T/K/H 可通过
独立 suite overrides 或正常训练 JSON 配置指定，函数实现不限制为这两个组数。

### K/T 与轻量 hard 扩展实验

独立 [sweep suite](../configs/experiments/iclr2027/suite_g1_shortlist_sweep.yaml)
新增以下 7 个配置，各使用 seeds 42/3407/2026，共 21 条 run。
均沿用 CP/G64、alignment 0.80、113 steps 和原训练/评测协议。
K 仅指额外负例数，自有候选全部保留；T 组复用同一套 actions。

| `--recipes` 名称 | K | T | 每组高分区名额 | 最多不同额外负例 |
|---|---:|---:|---:|---:|
| `uniform-k15-t1` | 15 | 1 | 0 | 15 |
| `uniform-k15-t4` | 15 | 4 | 0 | 60 |
| `uniform-k30-t4` | 30 | 4 | 0 | 120 |
| `uniform-k30-t8` | 30 | 8 | 0 | 240 |
| `uniform-k60-t4` | 60 | 4 | 0 | 240 |
| `mixed-k15-t8-hard2` | 15 | 8 | 2 | 120 |
| `mixed-k15-t8-hard4` | 15 | 8 | 4 | 120 |

Mixed 的高分候选区仍为前 128 个，其余名额从其余池抽取；池不足时沿用上述补齐规则。
同覆盖对照复用已有 Uniform K15/T8（120）和 K15/T16（240），不重跑它们。
K×T 相同仅表示候选充足时覆盖数相同，不代表计算量相同。

```bash
# 检查全部 21 条，不启动 GPU 训练
python scripts/run_g1_shortlist_sweep_r2.py check

# 按表格顺序，每配置依次运行三个 seed；每条训练后评测 BRIGHT
python scripts/run_g1_shortlist_sweep_r2.py train

# 按 seed 拆到不同八卡机器
python scripts/run_g1_shortlist_sweep_r2.py train --seeds 3407

# 只跑指定配置，也可以同时选择多个配置和 seed
python scripts/run_g1_shortlist_sweep_r2.py train \
  --recipes uniform-k30-t4 mixed-k15-t8-hard2 --seeds 42 2026

# 仅重试指定模型的评测
python scripts/run_g1_shortlist_sweep_r2.py eval \
  --recipes uniform-k30-t4 --seeds 42
```

使用 `--config /path/to/settings.yaml` 指定训练机 settings。
新 run ID 显式包含 K/T，Mixed 额外包含 Hard2/Hard4，与原 suite 不重名。
启动器沿用八卡 BF16 检查、非空输出目录保护及失败即停止行为，不自动跳过已完成 run。

### 固定 K15/T1 的负例分布对照

已有三 seed 结果中，原普通 CP/Align080 为 20.76，跨卡全部候选 Uniform K15/T1 为 21.98。
原普通 CP 在每卡 batch=16 时，最多使用同卡其他 query 的 15 个代表正例作为额外负例；
因此两者数量上限接近，但候选来源和组成同时改变。新增
[distribution suite](../configs/experiments/iclr2027/suite_g1_shortlist_distribution.yaml)
补齐以下两个对照，各跑 seeds 42/3407/2026，共 6 条，不重跑已有配置。

| 来源范围 | 仅代表正例 | 全部候选 |
|---|---|---|
| 同卡当前 microbatch | 已有普通 CP/Align080；过滤后最多 15 个 | 新增 `local-all`，均匀抽 K15/T1 |
| 跨卡当前 microbatch | 新增 `cross-device-representatives`，均匀抽 K15/T1 | 已有 Uniform K15/T1 |

`reward_shortlist_pool_source` 默认 `cross_device_all`，保持旧行为与 checkpoint contract。
`local_all` 必须搭配 `reward_cross_device_negatives: false`；
`cross_device_representatives` 必须搭配 `reward_cross_device_negatives: true`。
两种对照都维持 `ndcg_in_batch_include_negatives: true`，使抽中的文档进入固定池 reward 路径；
**候选组成由 pool source 决定**。代表正例严格指每条 query 候选位置 0 的那个文档，
不包括其余自有正例，也不是从所有正例中重新选取。

所有自有候选、graded nDCG@10、CP/G64、alignment 0.80、113 steps、8×16 batch、LR 5e-6、
独立 shortlist RNG 和最终 BRIGHT 协议保持不变。跨 query 文档继续使用 detached 均值；
新来源复用现有身份去重、已知正例排除和 route mask。池不足 15 时保留全部有效候选并 padding，
不重复补足；应结合 `reward_pool/cross_candidates_mean` 检查实际数量。
同卡来源不收集跨卡文档，仅在分布式训练时归约 query 数以校正不齐尾 batch 的 loss 权重。
新来源写入 checkpoint contract，恢复时禁止切换来源；默认来源兼容已有 shortlist checkpoint。

```bash
python scripts/run_g1_shortlist_distribution_r2.py check
python scripts/run_g1_shortlist_distribution_r2.py train

# 可按 seed 分配到不同八卡机器，或仅选择一个来源
python scripts/run_g1_shortlist_distribution_r2.py train --seeds 3407
python scripts/run_g1_shortlist_distribution_r2.py train --recipes local-all --seeds 42
python scripts/run_g1_shortlist_distribution_r2.py eval --recipes cross-device-representatives
```

用 `--config /path/to/settings.yaml` 指定训练机 settings。每条训练成功后自动评测 BRIGHT，
非空输出目录拒绝覆盖，失败即停止。新 run ID 分别含 `LocalAll` 和 `CrossDeviceRepresentatives`。
比较同列可判断来源范围的影响，比较同行可判断候选组成的影响；原普通 CP 与新 shortlist 路径
并非逐动作严格等价，且过滤后实际数量可能不同，不能把均分差直接当作唯一因素的因果估计。

### 固定训练预算的 alignment sweep

[alignment suite](../configs/experiments/iclr2027/suite_g1_shortlist_alignment.yaml) 覆盖
alignment **0.65 / 0.70 / 0.80 / 0.90 / 0.95** × seeds **42 / 3407 / 2026**。
固定跨卡全部候选、Uniform K15/T1、CP/G64、LR 5e-6、8×16 batch 和 **113 steps**，
维持与 CL 相同的训练步数。每条运行的 training/data/rollout seed 一致。
较小 alignment 对应更强探索；所有配置保持固定 alignment，不启用退火或辅助 InfoNCE。

0.80 的三个 run ID 和输出目录与已有 K/T sweep 完全相同，用于复用已有结果。
启动脚本默认选择其余四个 alignment 和全部三个 seed，共 **12 条新增运行**，
按 alignment、seed 顺序执行，每条训练完成后自动评测最终 BRIGHT。

```bash
# 默认预检 12 条新增配置
python scripts/run_g1_shortlist_alignment_r2.py check

# 先用 seed42 筛选四个新增 alignment
python scripts/run_g1_shortlist_alignment_r2.py train --seeds 42

# 或运行全部 12 条新增实验
python scripts/run_g1_shortlist_alignment_r2.py train

# 指定 alignment 和 seed；可按 seed 分配不同八卡机器
python scripts/run_g1_shortlist_alignment_r2.py train --alignments 0.65 0.90 --seeds 3407 2026

# 包含已有 0.80 的完整配置预检，或只重新评测已有模型
python scripts/run_g1_shortlist_alignment_r2.py check --alignments 0.65 0.70 0.80 0.90 0.95
python scripts/run_g1_shortlist_alignment_r2.py eval --alignments 0.80 --seeds 42
```

`--config /path/to/settings.yaml` 指定训练机 settings。训练仍要求八张 BF16 CUDA 卡，
非空输出目录拒绝覆盖、失败即停止，不自动跳过已完成运行；显式选择 0.80 训练也遵守此规则。

### 日志

- `shortlist/pool_candidates_mean/max`：过滤后的来源池，不含自有候选。
- `shortlist/unique_candidates_mean`、`coverage_mean`、`repeat_fraction`：本步独立负例覆盖。
- `shortlist/hard_candidates_mean`、`selected_score_mean`：实际高分区名额和均值评分。
- `reward_pool/cross_candidates_mean/max`：每组额外候选数。
- `reward/<term>/query|documents/group_std|degenerate_frac`：分别统计各组的边际 LOO 信号。
- `shortlist/reward_std_across_lists`：每组平均 reward 的组间差异，不作为 LOO baseline。
- `projection/query_span_rank_mean/max`：每组投影的 rank，而非候选并集 rank。

不同采样方式改变榜单难度，训练 reward 不能直接横比检索质量；最终使用相同评测协议。
吞吐、显存和真实八卡效果需在训练机测量，本地验证不等于 BRIGHT 收益。

针对性验证：

```bash
python -m pytest -q tests/test_shortlists.py tests/test_rl_large_pool.py tests/test_conditional_projection.py
```

覆盖组间最大覆盖、分层耗尽补齐、循环复用、padding/空池、独立 RNG、逐 cell 参考梯度、
BF16 autocast、query/document 联合梯度、双进程不齐 batch/slate 及某 rank 无额外候选、
实际 Trainer 更新与 checkpoint 恢复；也检查多组不增加 encoder、action draw 或 gather 次数。
