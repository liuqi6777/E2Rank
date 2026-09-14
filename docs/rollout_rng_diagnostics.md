# G1 rollout 随机性隔离与梯度诊断

目的：固定模型与训练候选池，只改变动作采样，直接测量梯度估计的波动；再通过完整训练判断这种波动是否足以解释 BRIGHT 的 seed 差距。

## 1. 固定状态的梯度诊断

在训练机器的项目根目录、原训练 Python 环境中运行。默认读取 `configs/experiments.yaml` 中的 G1 设置，并解析 `G1-A-MRRAlign090` 的完整配方。

```bash
# E0：三个固定的 16-query microbatch，每个做 16 次独立 rollout。
python scripts/diagnose_rollout_gradients.py \
  --output outputs/rollout_gradients/e0.json

# 同样的配方和 batch，在中间 checkpoint 上检查。
python scripts/diagnose_rollout_gradients.py \
  --checkpoint checkpoints/iclr2027/G1-A-MRRAlign090-s42/checkpoint-100 \
  --output outputs/rollout_gradients/step100.json
```

默认采样 seed 为 `42, 3407, 2026, 0, ..., 12`。可用 `--rollout-seeds` 显式指定至少两个不同 seed。这是固定模型上的 Monte Carlo 重复，**不是 16 次完整训练**。默认三个 probe 分别取 epoch-0 sampler 中第 0、1、2 个 microbatch；可通过 `--batch-indices` 改位置，不按结果挑 batch。

脚本复用真实 `EmbeddingDataset`、`EmbeddingDataCollator` 和 `GRPOModel`：query/document 长度沿用训练配置（当前 512/1024），保留真实多正例、候选 mask 和独立的 in-batch 候选池。每个 probe 的输入 tensor、样本 ID、来源和 SHA-256 都写入 JSON。仅支持 prepared v2、joint static-candidate GRPO、固定 exploration scale、KL=0 和非 EMA baseline。

默认诊断的是 **microbatch 梯度**。若波动大，应再看多个 microbatch 平均后的波动：

```bash
python scripts/diagnose_rollout_gradients.py \
  --microbatches-per-probe 8 --batch-indices 0 8 16 \
  --output outputs/rollout_gradients/e0_batch128.json
```

上面在单 GPU 上依次计算八个 size-16 的 loss，再平均梯度。候选池保持 size-16，不会错误地合并成一个 size-128 的 in-batch 池。这对应当前 global batch 128 的平均结构，但不是逐位重放八个分布式 rank 的原始采样。默认 microbatch 结果不能直接当作 global-batch 梯度方差。

脚本没有 optimizer，不更新参数，不发送 W&B 日志。关闭模型 dropout；每次重新前向/反向，仅 rollout RNG 改变。默认 fp32 参数 + bf16 autocast、启用配置中的 gradient checkpointing；这是单进程原始 loss 梯度诊断，不是 DeepSpeed/AdamW 更新量复现。可用 `--precision fp32` 做数值对照。不要用 `torchrun` 启动。

主机内存中维护两个完整 fp32 梯度累计向量；0.6B 模型仅此部分约需 4.8 GB，另外还需要模型、梯度和激活显存。不会保存所有采样的完整梯度。八 microbatch 模式的前后向成本约为默认模式的八倍。

`--checkpoint` 只替换 backbone；reward 等设置仍来自 `--run`。若有 `exploration_state.json`，自动读取其 step；否则从 0 开始，可用 `--step` 指定。当前 fixed alignment 不随 step 改变，step 仍参与 RNG 键。JSON 记录配置、模型权重 hash（本地 safetensors）、模型 revision、训练数据 hash 和代码版本。不同 checkpoint 间比较时，应确认 batch tensor hash 相同。

### 指标解释

设一个 probe 有 N 次梯度估计 g₁…gₙ，均值为 ḡ。每个梯度包含全部可训练参数；未连接参数按零坐标处理。

| JSON 字段 | 定义与用途 |
|---|---|
| `gradient_norm_mean/min/max` | 每次梯度范数的均值/范围，观察尺度波动 |
| `mean_gradient_norm` | ‖ḡ‖，不同于梯度范数的平均值 |
| `noise_rms` | √[Σ‖gᵢ−ḡ‖²/(N−1)]，采样噪声的样本 RMS |
| `noise_to_mean_ratio` | `noise_rms / ‖ḡ‖`；均值为零时记为 null |
| `mean_pairwise_cosine` | 非零梯度两两余弦的平均值，观察方向一致性；不足两个非零梯度记为 null |
| `zero_gradient_draws` | 零梯度采样数；这些采样仍参与均值和噪声统计 |
| `draws` | 每个 seed 的梯度范数、训练 reward、surrogate loss |

余弦通过 `‖Σ gᵢ/‖gᵢ‖‖²` 的恒等式计算，使用完整参数空间，不使用随机投影。累计使用 fp32，平方范数分块使用 fp64。

**判断重点**：reward 接近而梯度方向一致性低，支持“reward 标量掩盖了 rollout 更新方向噪声”。若八 microbatch 平均后噪声明显减弱，需要据此收紧结论；即使固定状态下噪声很大，也不能直接证明它造成了最终 BRIGHT 差距。16 次估计只是初步诊断，均值接近零时比值尤其不稳定，不设人为的因果判定阈值。

## 2. 固定数据、仅改变 rollout seed 的训练对照

```bash
bash scripts/run_g1_rollout_seed_repeats.sh check 8
bash scripts/run_g1_rollout_seed_repeats.sh train 8
```

| Run | 训练 seed | data_seed | rollout_seed |
|---|---:|---:|---:|
| `G1-A-MRR090-Rollout42` | 42 | 42 | 42 |
| `G1-A-MRR090-Rollout3407` | 42 | 42 | 3407 |
| `G1-A-MRR090-Rollout2026` | 42 | 42 | 2026 |

三行都从 E0 独立训练，沿用同一 prepared 数据、113 steps、LR 5e-6、global batch 128 / microbatch 16 和当前完整 MRR 配方。保持相同 GPU 数量、配置和运行环境。脚本先预检三行，再依次训练并执行既有 BRIGHT 最终评测；失败即停。输出在 `checkpoints/iclr2027/<Run>-s42/`，其中 `s42` 表示训练 seed，rollout seed 在 Run 名字里。

**三行都必须新跑。** 历史 `G1-A-MRRAlign090-s42` 使用全局 RNG，不等同于新的 `Rollout42`。历史三个 seed 的结果仍是“全部随机性一起变化”的实验；新三行构成独立的 rollout 对照。不要从历史结果复用 22.01 作为新控制组。

只运行某一行：

```bash
python scripts/experiment.py train G1-A-MRR090-Rollout3407 --gpus 8
```

## 3. 随机数与恢复约定

- `RLArguments.rollout_seed` 默认 `None`：严格沿用原来的全局 PyTorch RNG 路径。
- 设置后，只在 vMF / Gaussian 动作采样期间临时切换 CPU 和当前 CUDA 设备的 RNG，退出后恢复外部状态。vMF 的 Beta rejection sampling 也包含在隔离范围内。
- 每次 draw 的键包括 RNG 版本、rollout seed、global rank、optimizer step、train/eval 模式和该 step 内的调用序号。不同 rank、不同 accumulation microbatch、query/document 组件使用不同随机数；eval 调用不推进 train 计数。
- Trainer 在 optimizer 边界保存 checkpoint。恢复时下一步从 call 0 开始，与不中断训练一致；不依赖重新播放已完成步骤的动作采样。`exploration_state.json` 记录 rollout seed、RNG 版本和 world size；不兼容的恢复会报错。精确恢复仍要求相同 batch、accumulation、数据和代码配置。
- 当前不支持 dynamic-retrieval 的独立 rollout seed，会显式报错。

这次不改变数据 loader 的 seed 语义：三个新配置统一固定训练 seed 42，即可固定现有组批和 sampler。不能仅设置不同 `data_seed` 就假定已实现反向的数据随机性消融，现有 sampler 仍使用训练 `seed`。
