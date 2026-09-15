# RL 的直接 InfoNCE 辅助损失

在现有策略损失上增加 `aux_infonce_coef * L_InfoNCE`。默认系数为 0，沿用原训练路径。
辅助项使用本次 encoder forward 的未扰动、归一化 embedding，直接反向传播；不进入
reward、advantage、采样分布或 frozen-candidate reward rescaling。

## 配置

```yaml
aux_infonce_coef: 0.1
aux_infonce_temperature: 0.03
aux_infonce_use_in_batch_negatives: true
```

系数必须为有限非负数，温度必须为有限正数。0.1 是初始试验值，未经过调优；它不能与
InfoNCE reward 的系数直接比较。温度独立于 `contrastive_temperature`。

## 正例与梯度口径

复用 joint CL 的损失函数。设每条 query 的已知正例集合为 P、有效负例集合为 N：

\[
L_{\mathrm{InfoNCE}}=\frac1B\sum_b\frac1{|P_b|}\sum_{p\in P_b}
\log\left(1+\sum_{n\in N_b}\exp((s_{bn}-s_{bp})/\tau)\right).
\]

- 必须提供 collator 生成的二值 `positive_mask`。所有已知正例均参与，正例之间不互相竞争；
  不用 teacher grade 的最高值或阈值推断正例。无有效负例的 query 贡献 0。
- padding 通过 `candidate_mask` 排除。in-batch 使用其他样本的代表正例，并沿用
  `in_batch_positive_mask` 排除已知正例和重复候选；跨 query 使用的文档表示停止梯度。
- 相似度和辅助损失用 fp32 计算，复用已有 forward，不再调用一次 encoder。
- 直接梯度范围由 `action_components` 决定：未采样的 query/文档槽位在辅助项中也停止梯度。
  legacy `positive` 指第一个代表正例槽位，`negative` 指其余槽位，后者可能包含额外的已知正例。
  哪些文档是正例由 `positive_mask` 决定，哪些分支可更新由 action 配置决定。
- 在共享 encoder 下，某个槽位没有直接梯度不等于它的表示在后续训练中永久冻结。

支持 `GRPOModel`、冻结静态候选的 `FixedCorpusGRPOModel` 和 G1 的
`DynamicRetrievalGRPOModel`。冻结索引模式只有 query 梯度；DR 的辅助项使用 batch 提供的
离线标注候选，RL reward 仍使用逐 rollout 全库检索结果。冻结索引模式的跨 query 辅助候选还
限定为同一 index route。未改变 G3 的独立 RAG 训练接口。

现有 `reward_type: infonce` 保持原行为：它在扰动表示上选最高标签的一个文档，并返回
温度缩放的负 InfoNCE。它与这里的多正例直接损失不是仅有梯度估计器不同；比较时还需对齐
候选、正例定义、温度、可训练分支和权重尺度。

## 日志与恢复

开启后记录：

- `train/loss_rl`：策略损失，不含 KL 和辅助项。
- `train/loss_infonce`：未乘系数的辅助损失。
- `train/loss_infonce_weighted`：辅助项对总损失的贡献。
- 原 `train/loss`：总损失；原 `kl` 仍记录未乘 `kl_coef` 的 KL。

以上标量沿用 trainer 的跨 microbatch/rank 聚合。`exploration_state.json` 保存辅助目标、
系数、温度和 in-batch 设置；改变已启用的辅助配置、在 RL-only 与辅助训练之间切换时，
不能直接恢复 optimizer/checkpoint 状态。应从所需初始化权重建立新 run。旧 checkpoint 在
辅助项关闭时继续兼容。

## G1 对照

新增可选行 `G1-A-MRR090-AuxInfoNCE`，继承 `G1-S-MRR32-Seed42` 的配置，只增加上述
三个辅助参数。二者保持训练/data/rollout seed 42、binary MRR@10、G32、alignment .90、
LR 5e-6、113 steps、global batch 128。`G1-J-CL` 可作为纯 CL 参照。

```bash
python scripts/experiment.py show G1-A-MRR090-AuxInfoNCE --gpus 8
python scripts/experiment.py check G1-A-MRR090-AuxInfoNCE --gpus 8
python scripts/experiment.py train G1-A-MRR090-AuxInfoNCE --gpus 8
```

本次增加实现与配置，未启动完整训练。这个可选行不会加入已有 G1 stability 批处理队列。

CPU 验证：

```bash
python -m pytest -q tests/test_aux_infonce.py
```

验证多正例和 padding 梯度、跨 query detach/排除 mask、reward 退化时的确定性辅助梯度、
各 action 配置的梯度范围、三种 wrapper 的 forward 复用和冻结索引、关闭后 reward 不变及恢复契约。
