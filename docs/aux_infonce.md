# RL 的直接 InfoNCE 辅助损失

在现有策略损失上增加 `aux_infonce_coef * L_InfoNCE`。默认系数为 0，沿用原训练路径。
辅助项使用本次 encoder forward 的未扰动、归一化 embedding，直接反向传播；不进入
reward、advantage、采样分布或 frozen-candidate reward rescaling。

## 配置

```yaml
aux_infonce_coef: 0.1
aux_infonce_temperature: 0.03
aux_infonce_use_in_batch_negatives: true
aux_infonce_strong_negatives: true
```

系数必须为有限非负数，温度必须为有限正数。0.1 是初始试验值，未经过调优；它不能与
InfoNCE reward 的系数直接比较。温度独立于 `contrastive_temperature`。

`aux_infonce_strong_negatives: true` 复用强化版 CL 的负样本实现：所有卡当前 microbatch
中的全部候选文档参与身份过滤后的跨 query 负样本池，保留 action 范围内的文档梯度。
它优先于旧的 `aux_infonce_use_in_batch_negatives` 开关；关闭 strong 时仍可使用原来的
同卡、代表正例、跨 query 文档 detach 版本。两个开关默认关闭，辅助系数为 0 时均不执行。
当前 G1-R2/G2-R2 纯 RL 行仍保持辅助系数 0，不自动变为 RL+CL。

## 正例与梯度口径

复用 joint CL 的损失函数。设每条 query 的已知正例集合为 P、有效负例集合为 N：

\[
L_{\mathrm{InfoNCE}}=\frac1B\sum_b\frac1{|P_b|}\sum_{p\in P_b}
\log\left(1+\sum_{n\in N_b}\exp((s_{bn}-s_{bp})/\tau)\right).
\]

- 必须提供 collator 生成的二值 `positive_mask`。所有已知正例均参与，正例之间不互相竞争；
  不用 teacher grade 的最高值或阈值推断正例。无有效负例的 query 贡献 0。
- padding 通过 `candidate_mask` 排除。旧版 in-batch 使用其他样本的代表正例，并沿用
  `in_batch_positive_mask` 排除已知正例和重复候选；跨 query 使用的文档表示停止梯度。
- 强化版收集跨卡候选及身份元数据，按相同规则过滤重复候选与已知正例；所有文档槽位
  均可供其他 query 使用。跨卡聚合可微，文档接收所有 query 的梯度；未启用的 action 分支
  仍由调用方 detach，冻结索引仍只有 query 梯度。同一索引 route 的限制同时用于跨卡候选。
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
系数、温度和 in-batch/strong 设置；改变已启用的辅助配置、在 RL-only 与辅助训练之间切换时，
不能直接恢复 optimizer/checkpoint 状态。应从所需初始化权重建立新 run。旧 checkpoint 在
辅助项关闭时继续兼容。

## G1-R2 配置草案

独立配置为 `configs/experiments/iclr2027/g1_r2_aux_infonce_strong.yaml`，继承
G1-R2 的 graded nDCG@10、CP、G64、alignment 0.90、joint full FT 配方，只增加
系数 0.1、温度 0.03 的强化版 InfoNCE 辅助项。辅助正例仍来自 binary positive_mask。
沿用 seed/data/rollout seed 42、LR 5e-6、113 steps、global batch 128、无 dev 和关闭裁剪。

目前仅准备配置，不注册到任何 suite、不修改启动队列、不启动训练。默认输出为
`checkpoints/iclr2027-r2/G1-R2-RL-GradedNDCG64-CP-AuxInfoNCE-Strong-s42/`。
0.1 为初始系数，未做额外超参数选择。

CPU 验证：

```bash
python -m pytest -q tests/test_aux_infonce.py
```

验证多正例和 padding 梯度、跨 query detach/排除 mask、reward 退化时的确定性辅助梯度、
各 action 配置的梯度范围、三种 wrapper 的 forward 复用和冻结索引、关闭后 reward 不变及恢复契约。
强化版还验证双进程全局 loss/encoder 梯度一致性（包含不齐的 batch/slate）、跨 route
过滤、joint 与冻结 wrapper 的 forward 复用，以及旧辅助设置的恢复兼容性。
