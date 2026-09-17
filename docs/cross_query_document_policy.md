# 跨 query 文档的联合策略梯度

通过独立开关 `cross_query_document_gradients: true`，让实际参与其他 query 排序的文档动作，
接收那些 query 的 reward 所提供的策略梯度。默认关闭，已有固定跨 query 候选配方继续沿用原路径。
训练数据、候选身份、去重和已知正例过滤保持原口径。跨 query 标签继续沿用原零相关度约定。

## 候选范围与开关

| 范围 | `reward_cross_device_negatives` | `ndcg_in_batch_include_negatives` |
|---|---|---|
| 普通池：同卡其他 query 的代表正例 | false | false |
| 同卡其他 query 的全部候选 | false | true |
| 大池：跨卡其他 query 的全部候选 | true | true |

三种范围均支持 `cross_query_document_gradients: true`。自有文档始终保留全部有效槽位。
不因开启新策略自动扩大候选范围。普通池使用 collator 的代表正例/全部候选独立过滤 mask；
跨卡池复用 Strong CL 的身份过滤及不齐 batch/slate padding。重复文档按每条 query 的既有
顺序选择代表槽位，保留的槽位对应自己的策略随机变量。

`in_batch_use_sampled_documents` 必须保持 **false**。它是原来的采样消融开关，未实现完整的
跨 query 文档信用汇总；新开关单独定义新的联合策略，避免复用旧开关改变历史实验语义。

## 目标与梯度

每篇有效文档仍只采样 Gd 个动作。同一个 draw j 将池中所有文档的第 j 个动作组成全局 bundle，
供各 query 使用。Query 独立采样 Gq 个动作，计算 `R[b,i,j]`。没有为每条 query 再采样一遍文档。

训练目标为当前池内各 query 期望 reward 的平均：

\[
J(\theta)=\frac1B\sum_b E[R_b(e_b^q,\{e_m^d:m\in C_b\})].
\]

Query 的逐 cell LOO 排除当前 i；文档 LOO 排除当前整个 bundle j：

\[
A^q_{bij}=R_{bij}-\frac1{G_q-1}\sum_{i'\ne i}R_{bi'j},\qquad
A^d_{bij}=R_{bij}-\frac1{G_d-1}\sum_{j'\ne j}R_{bij'}.
\]

文档的 score-function 项对所有使用它的 query 求和，而不只使用该文档所属样本的 reward。
文档 bundle 相互独立，因此文档 baseline 不依赖当前 bundle 内任一文档动作；共享 encoder
不破坏这一条件。Query 仅使用自己的 reward，避免加入不依赖该动作的其他 query 项。

CP 保留每个 reward cell 的条件投影：query span 包含其均值和本 bundle 中该 query 的全部有效候选；
每个文档项使用 `span(document_mean, sampled_query_i)`，投影后才跨 query 汇总。
不把其他 query 的方向合并成一个文档投影空间。每项条件均值保持原 score 项的期望；完整共享
encoder 的总方差与训练效果仍需测量。

跨 query 文档现在也被采样，**不再对其分数应用 frozen-document rescaling**。在本模式支持的
joint 路径中所有文档均被采样，因此 `frozen_doc_rescale` 对 RL reward 无作用。
相比原固定跨 query 文档配方，这改变了随机目标和梯度覆盖；不能描述为对旧目标仅增加梯度项。

## 支持范围与资源

- 静态 joint query + bundled documents，exact vMF，固定 κ 或预定 alignment schedule。
- Product rollout、LOO、无 advantage normalization、shared document baseline、document sum。
- 带正 cutoff 的 in-batch binary/graded nDCG、MRR，及它们的固定加权和。
- CP 和 SF 都有完整梯度实现；示例实验采用 CP。
- 跨卡模式要求显式 `rollout_seed`（示例为 42），使用包含 rank 的独立 RNG，避免各卡同 seed
  的全局随机流产生相关动作。普通池也建议显式设置；既有示例均已设置。
- 不支持 dynamic retrieval、query-only/frozen encoder、learnable κ、旧 sampled-in-batch 消融、
  counterfactual baseline 或其他更新归约方式；配置阶段明确报错。

Encoder forward 不增加。跨卡分别收集 live means 和 detached actions；只有 means 使用可微
all-gather，其 backward 把文档系数汇总到所属 rank。按全局 query 数校正不齐 batch 的 loss，
与 DDP/ZeRO 的参数梯度平均配合。即使本 rank 没有有效跨 query 候选，也保留零系数池项以执行集体 backward。

Reward 沿 query draw 与 document draw 分块，文档 CP 沿候选分块，不构造完整
`[B,Gq,Gd,pool]` 分数张量或 `[B,Gq,Gd,pool,D]` 投影张量。仍需存储采样池 `[Gd,pool,D]`，
通信、点积和投影成本高于固定池；这不是同耗时替换。

大池中，对所有本地 query 都有效的公共候选，每个 document draw 只检查一次数值满秩。
使用最小奇异值下界和完整 span 的 Frobenius 上界保守认证；认证成功时 query 投影为恒等，
否则回到每条 query 的完整 SVD。该优化不删减候选或改变 rank 阈值。

## 运行

独立 suite 提供普通池和大池各 seeds 42 / 3407 / 2026，共六条配置：
`configs/experiments/iclr2027/suite_g1_cross_query_policy.yaml`。两种池保持 G1-R2 的 alignment 0.90、
CP/G64、LR 5e-6、113 steps、8×16 global batch、原数据、无辅助损失和最终 BRIGHT 评测，
分别配对原普通池 CP-0.90 和原大池 CP-0.90；未加入旧队列，也未启动 GPU 训练。

专用脚本为 `scripts/run_g1_cross_query_policy_r2.py`，复用已有大池/Strong CL 的直接启动器。
默认运行全部六条，先普通池三个 seed、后大池三个 seed，每条训练成功后自动评测最终 BRIGHT；任一步失败即停止。
每条的 training seed、data seed 和 rollout seed 使用同一个所选值，与原 G1 对照逐 seed 配对；
prepared 数据及其固定预处理种子不变。`--seeds` 可以只选择尚未完成的种子，也可以按 seed 分机器运行。
训练要求恰好八张可见且支持 BF16 的 CUDA GPU，W&B 默认 offline（保留显式设置的 `WANDB_MODE`）。
`check` 只检查配置展开和输入路径，不加载模型、不启动 GPU，也不做 manifest/hash 审计。

```bash
python scripts/run_g1_cross_query_policy_r2.py check

# 顺序训练普通池和大池各三个 seed，每条完成后自动评测
nohup python -u scripts/run_g1_cross_query_policy_r2.py train \
  > g1_cross_query_policy.log 2>&1 &

# 单独选择一种池的三个 seed（可在两台八卡机器分别执行）
python -u scripts/run_g1_cross_query_policy_r2.py train --pool local
python -u scripts/run_g1_cross_query_policy_r2.py train --pool large

# 单独选择一个 seed（默认普通池、大池都运行）
python -u scripts/run_g1_cross_query_policy_r2.py train --seeds 3407

# 如果 seed 42 已完成，只补剩余种子
python -u scripts/run_g1_cross_query_policy_r2.py train --seeds 3407 2026

# 训练完成后，只补评测
python scripts/run_g1_cross_query_policy_r2.py eval --pool large --seeds 42
```

`--config` 可指定机器上的 settings 文件。输出目录非空时拒绝覆盖；一条完成后重试，使用
`--pool` 与 `--seeds` 只选择剩余项，已有模型使用 `eval`。不同 seed 使用独立的 run ID 和输出目录；
seed 42 保留原有目录名。展开配置沿用启动器的 `.cl_strong_configs/` 目录。

通用入口仍可逐条运行，预检会执行该入口原有的数据合同检查：

```bash
# 普通池预检；数据目录使用现有 experiments_r2.yaml 中的 G1.data
python scripts/experiment.py check G1-R2-RL-GradedNDCG64-CP-CrossQuery \
  --suite configs/experiments/iclr2027/suite_g1_cross_query_policy.yaml \
  --config configs/experiments_r2.yaml --gpus 8

# 大池预检
python scripts/experiment.py check G1-R2-RL-GradedNDCG64-CP-CrossQuery-LargePool \
  --suite configs/experiments/iclr2027/suite_g1_cross_query_policy.yaml \
  --config configs/experiments_r2.yaml --gpus 8
```

正式训练将 `check` 改为 `train`，每条使用独立输出目录；正常入口完成最终评测。
建议训练机先用独立临时 run 做少量 step 的显存/耗时检查，再运行完整预算。

Checkpoint 的 `exploration_state.json` 保存新开关，缺少字段解释为 false。
开启/关闭新策略不能直接恢复同一个 optimizer/checkpoint 状态；可从相同初始化权重建立新 run。

日志保留 reward、各动作轴的 `group_std`/`degenerate_frac`、各 reward 的 `n_distinct`、
池大小/零奖励比例及 `projection/query_span_rank_mean`/`max`。
Document 轴统计描述每条 query 对全局 document bundle 的边际 reward，
不等于每篇文档最终跨 query 汇总后的梯度或 advantage。

## 验证

```bash
python -m pytest -q tests/test_cross_query_policy.py tests/test_rl_large_pool.py \
  tests/test_conditional_projection.py tests/test_aux_infonce.py
```

覆盖分块/完整 reward 一致性（含同分、padding、代表正例及全部候选）、逐 cell 显式梯度参考、
解析 vMF 双线性期望、公共 span 满秩/退化分支、跨 query 文档获得直接梯度、无额外 rescaling、
双进程梯度与全局参考一致（不齐 batch/slate、rank 无可用外部候选）、实际 Trainer 更新和恢复合同。
本机只有 CPU，真实八卡 CUDA/NCCL/ZeRO 的速度、显存及 BRIGHT 收益尚未验证。

本次完整 `tests/` 回归：235 passed、1 skipped；跳过项为 CUDA 不可用的精度测试。
双进程 CPU/Gloo 测试已运行并通过；小型 Qwen3 的 FP32/BF16 全参数配对诊断也已通过。
