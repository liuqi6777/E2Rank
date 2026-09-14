# 逐文档反事实 baseline：实现与使用

状态：已实现，可做固定状态诊断；尚未运行正式 G1 对比训练，尚无降低梯度方差的实验结论。

## 1. 配置与计算

新增 `RLArguments.document_advantage_baseline`：

| 值 | 行为 |
|---|---|
| `shared`（默认） | 沿用现有组件级 advantage，文档组共享一个 advantage |
| `counterfactual` | 每个文档单独使用反事实 reward 差值 |

对第 i 个 query 动作和第 j 组文档动作，记 reward 为 Rᵢⱼ。将文档 m 的动作替换为该文档的单位均值方向 μₘ，保持其他动作不变，得到 Bᵢⱼₘ。文档侧 advantage 为：

\[
A^D_{jm}=\frac1{G_q}\sum_i(R_{ij}-B_{ijm}).
\]

用 Aⱼₘ 加权该文档自己的 vMF log-prob，先对文档求和，再对 batch 和文档 rollout 取平均。Query 侧仍采用现有边际化 LOO。新的文档 advantage 不再减一次 group/LOO baseline，也不做标准化。若使用原有 `document_log_prob_reduction: mean`，仍按每条样本的有效文档数缩放；G1 主配方继续用 `sum`。

参考动作是 **单位方向 μₘ**，不是期望动作 A_d(κ)μₘ。因此替换分数为 qᵢᵀμₘ，不额外乘 frozen-document scale。它替换的是一个 sampled 动作；其他候选的原有分数、标签、mask、frozen rescaling 均保持不变。该选择可能改善或恶化方差，需用诊断验证。

反事实 reward 和 advantage 全部停止梯度。仅替换自己的动作使 baseline 条件于其余动作时不依赖自身采样；文档独立采样下，减去该 baseline 不改变当前文档 score-function 项的期望。共享 encoder 参数不影响该条件性质。

## 2. 支持范围

首版要求以下配置；不兼容组合会在初始化时显式报错：

```yaml
action_components: [[query], [positive, negative]]
sampling_law: vmf
sigma_learnable: false
rollout: product
advantage_baseline: leave_one_out
advantage_norm: none
reward_combine: sum
in_batch_use_sampled_documents: false
document_advantage_baseline: counterfactual
```

也支持只采样联合文档组 `[[positive, negative]]`。固定 κ 或外部预定 alignment schedule 均可；不支持 learnable κ，因为新 advantage 不保证组内和为零，现有省略 vMF normalizer 的写法不足以计算其梯度。

沿用现有 reward 函数，支持 binary MRR、binary/graded nDCG，以及按固定权重直接求和的 reward terms。Dynamic retrieval、冻结文档的 query-only 策略、拆开的正负文档 action groups 不在首版范围内。

## 3. 固定状态对比命令

在训练机器的原环境中运行，默认使用 G1 MRR 配方的 E0，不更新参数：

```bash
python scripts/diagnose_rollout_gradients.py \
  --document-advantage-baseline shared \
  --output outputs/rollout_gradients/e0_mrr_shared.json

python scripts/diagnose_rollout_gradients.py \
  --document-advantage-baseline counterfactual \
  --output outputs/rollout_gradients/e0_mrr_counterfactual.json
```

两条命令保持相同的 checkpoint、step、batch 和 rollout seeds。新 baseline 不额外抽样；在相同输入、环境和采样配置下，不改变 query/document 动作或 RNG 调用次数。需要 global-batch 结构的诊断时，两条命令均加 `--microbatches-per-probe 8 --batch-indices 0 8 16`。

Graded nDCG 使用同一命令并加 `--run G1-A-NDCGAlign090`；binary nDCG 对应 `--run G1-A-BinaryNDCGAlign090`。比较不同 reward 时仍应使用同一份模型权重，不能分别加载各自训练后的模型再归因于 reward。完整协议见[后续实验计划](rollout_variance_experiment_plan.md)。

训练时可在新的训练 YAML 中设置 `document_advantage_baseline: counterfactual`。本次未增加完整训练的 suite 行；既有配置不设置该字段，继续使用 `shared`。

## 4. 日志、成本与恢复

新增日志前缀 `baseline/documents/counterfactual/`，包含 `mean`、`std`、`min`、`max`、`zero_frac`。这些统计针对边际化后的逐文档 advantage，只计有效文档；`zero_frac` 是严格等于零的比例，不使用 reward 退化阈值。

原有 `reward/.../documents/group_std` 和 `degenerate_frac` 继续描述原始 reward 在文档动作轴上的边际分布，便于与控制比较；它们不再描述实际逐文档 advantage 的分布。通用 `advantages_*` 统计会拼接 query advantage 和展平后的文档 advantage，文档条目数量更大，不宜与旧聚合量直接比较。

诊断 JSON 的每个 `draws` 条目新增 `metrics`，保留 reward 与 baseline 的日志标量；多个 microbatch 时逐项平均。主要判断仍使用完整参数梯度的噪声和方向一致性，不能只根据 `zero_frac` 增大判断改进有效。

实现复用 encoder 输出、采样动作和跨样本候选分数，按文档依次替换分数并重算 reward。没有额外 encoder forward；每个文档槽位多一次完整 reward-grid 计算。额外只保留一个可复用的替换分数表和逐文档 advantage，不同时堆叠所有反事实候选池。大量候选时 reward 重算仍可能很贵，GPU 耗时和显存需在真实诊断中测量。

训练 checkpoint 的 `exploration_state.json` 记录启用的文档 baseline。旧 checkpoint 缺少字段时解释为 `shared`，恢复训练时拒绝两种 baseline 之间的切换；固定状态诊断只加载 backbone，可以有意比较两个估计器。

实现位置：[GRPO](../src/grpo.py)、[配置验证](../src/config.py)、[诊断脚本](../scripts/diagnose_rollout_gradients.py)。本地行为测试遵循项目约定放在被忽略的 `scripts/tests/test_document_counterfactual.py`，没有强制加入版本控制。
