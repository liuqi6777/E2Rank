# 梯度估计器与配对诊断

## 已实现的变化

CL、RankNet、LambdaLoss 和静态候选 RL 统一使用 FP32 的归一化与评分。
Transformer 仍可使用 BF16；评分、投影的矩阵运算显式关闭 autocast 和 CUDA TF32。
转换到 FP32 无法恢复 backbone 已量化的 hidden state，但能避免点积、归一化再次损失精度。
冻结文档的监督训练评分和直接辅助 InfoNCE 也使用同一规则。

`gradient_estimator` 有两个值：

- `score_function`：原始 vMF score-function 梯度，默认值。
- `conditional_projection`：保持相同 vMF 动作、奖励和 LOO baseline，投影梯度的随机系数。

新的训练配置可设置：

```yaml
action_components: [[query], [positive, negative]]
gradient_estimator: conditional_projection
sampling_law: vmf
rollout: product
advantage_baseline: leave_one_out
advantage_norm: none
document_advantage_baseline: shared
document_log_prob_reduction: sum
sigma_learnable: false
in_batch_use_sampled_documents: false
```

初版只支持静态候选、on-policy 的 query + 全文档组联合采样，且 `reward_combine: sum`。
不支持的组合会报错。可以使用预先确定的 exploration schedule；不能学习 κ。
KL 和辅助 InfoNCE 仍是独立的直接损失，但隔离 rollout 噪声的配对诊断要求二者系数为 0。
checkpoint 会保存估计器名称；恢复训练时不能切换。诊断加载同一组权重作比较不属于恢复训练。

## 投影计算

对于每个文档 bundle j，query 的子空间包含其均值、bundle j 内全部有效文档动作，
以及实际参与任一 reward term 的跨样本固定候选。跨样本 mask、padding 与重复向量均处理。
先按 query LOO 权重求和，再对每个 bundle 单独投影；不会把全部 bundle 合成一个子空间。

每个文档动作的子空间为 `span(document_mean, sampled_query_i)`，逐 query draw 计算。
所有样本、优势、投影基和梯度系数都 detach，只有当前策略均值保留梯度。
实现避免生成 `[B, Gq, Gd, M, D]` 张量。Query 的 SVD 使用维度缩放的 FP32 数值秩阈值；
文档近共线方向用 FP64 构造切向量，再回到 FP32 做主要收缩计算。

两种 surrogate loss 的数值不要求相同，它们用来产生梯度。应比较奖励、梯度均值和方差。
条件投影的推导见 [方法重设计](../paper/METHOD_REDESIGN.md)；共享 Transformer 上的总梯度方差
是否下降需要实测，不能从单项条件方差直接推断。

## 固定状态配对诊断

新协议 G64 夜跑由 [run_g1_r2.py](../scripts/run_g1_r2.py) 自动安排 E0 与新 LL-Binary step 25 的 MRR 配对诊断；
配置与日志见[夜跑说明](g1_r2_overnight.md)。单独诊断新矩阵时同时传 `--suite configs/experiments/iclr2027/suite_r2.yaml` 和 `--config configs/experiments_r2.yaml`。

在有真实 prepared data 与足够内存的 GPU 机器上运行：

```bash
python scripts/diagnose_rollout_gradients.py \
  --run G1-A-MRRAlign090 \
  --compare-gradient-estimators \
  --batch-indices 0 1 2 \
  --rollout-seeds 42 3407 2026 0 1 2 3 4 5 6 7 8 9 10 11 12 \
  --precision bf16 --device cuda:0 \
  --output artifacts/diagnostics/g1_mrr090_projection_e0.json
```

默认使用 E0；使用 `--checkpoint /absolute/path/to/checkpoint` 比较某个已训练状态。
脚本没有 optimizer，不更新权重；禁用 dropout，固定 batch、候选和训练步数。
每个 seed 交替运行两种估计器，并校验实际动作与完整奖励表的 SHA256 完全一致。
`--microbatches-per-probe 8` 可以平均 8 个独立 microbatch loss，跨样本候选池仍各自独立。

输出包括：

- 全部可训练参数的梯度均值范数、噪声方差、投影/原始方差比、配对均值差及其 Monte Carlo 误差尺度。
- `signal_squared_unbiased = ||sample_mean||² − noise_variance/N`；非正时无法得到正的信号范数估计，修正的噪声/信号比为 null；正值也不代表统计显著。
- forward/backward 时间、含统计开销的时间、CUDA allocated/reserved 峰值；CPU RSS 为进程累计峰值。
- 源码哈希、模型 revision、本地 checkpoint 权重哈希、数据和 batch 哈希、embedding 协议。

统计使用完整参数向量。0.6B 配对统计约需 7.2 GB CPU 存储，另需模型、梯度和激活内存。
时间包含首次调用开销，CUDA reserved 峰值受缓存分配器影响；不应当作独立性能 benchmark。
16 个 seed 只是初筛；均值差较大或信号未分辨时增加独立 seed，不能用同一 seed 重复凑样本数。

## 验证与当前边界

```bash
.venv/bin/python -m pytest tests -q
```

覆盖逐组合显式投影参考、解析 vMF 双线性期望、共享编码器 Jacobian、重复/共线方向、
padding、跨样本候选并集、BF16 近分数排序、估计器恢复契约和实际 Trainer 的一步更新/保存。
小型随机初始化 Qwen3 的 FP32/BF16 全参数配对测试验证动作/奖励一致、统计正确且权重不变。

本地机器无 CUDA，尚未运行预训练 Qwen3-Embedding-0.6B 的全参数诊断或 BRIGHT 训练评测。
这些数值与小模型测试验证实现，不能证明真实模型方差降低或检索效果改善。

归一化精度也写入 checkpoint、索引和评测缓存 identity。缺少 FP32 pooling 标识的旧项目
checkpoint/固定索引不接受；使用原始 E0 重新训练或重建索引。Tokenization 规则仍为 v2，
MTEB 名称后缀现在为 `__tokens-v2__pool-fp32`。
