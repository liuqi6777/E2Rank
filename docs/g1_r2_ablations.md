# G1-R2 消融：三个独立任务入口

本轮基于 [R2 结果](../paper/G1_R2_RESULTS.md)，固定 **graded nDCG@10**，
从 E0 独立训练。读取 `configs/experiments_r2.yaml` 的数据与输出根目录；
变体定义在 [g1_r2_ablations.yaml](../configs/experiments/iclr2027/g1_r2_ablations.yaml)。
复用原 R2 的 SF/CP G64、alignment 0.90 结果作为对照，不重跑它们。

## 直接运行

只保留三个入口。每条命令在各自的 **8 卡任务环境** 中执行；每个入口内部串行跑所选实验。
同一机器同时提交多个任务时，需要各自分配互不重叠的 8 张 GPU。

| 入口 | 默认实验 | 三 seed 新训练数 |
|---|---|---:|
| `scripts/run_g1_r2_ablations.py` | SF-G32、CP-G32、SF-paired-G64 | 9 |
| `scripts/run_g1_r2_mechanisms.py` | 单侧 policy、Norm/DocMean/组合、关闭 rescaling | 21 |
| `scripts/run_g1_r2_exploration.py` | 宽 alignment 范围、退火、Gaussian 失配及其对照 | 48 |

默认三个 seed 为 42/3407/2026。全部选择时共 **26 个变体 × 3 seed = 78 次训练**，
每次结束后评测最终 BRIGHT。没有新的 E0 评测或梯度探针。
核心入口默认只跑 9 次，不会启动后两组。

```bash
# 任务 1：核心消融
nohup python -u scripts/run_g1_r2_ablations.py > g1_r2_ablations.log 2>&1 &

# 任务 2：其他机制
nohup python -u scripts/run_g1_r2_mechanisms.py > g1_r2_mechanisms.log 2>&1 &

# 任务 3：探索实验
nohup python -u scripts/run_g1_r2_exploration.py > g1_r2_exploration.log 2>&1 &
```

先做 CPU 预检，不加载模型、不占 GPU：

```bash
python scripts/run_g1_r2_ablations.py check
python scripts/run_g1_r2_mechanisms.py check
python scripts/run_g1_r2_exploration.py check
```

`--config /path/to/experiments_r2.yaml` 指定机器上的数据设置。
每次运行检查实际训练文件和同目录 manifest 的 SHA256、无 dev、预处理正例 seed 42。
各机器需要使用同一份数据及同一版本代码。

## 拆成更多任务，不增加入口文件

`--seeds` 筛选 seed，`--variants` 替换入口的默认实验列表；二者可以一起用。
以下两条使用相同入口，但运行不同实验，可分别提交：

```bash
python -u scripts/run_g1_r2_ablations.py --variants g32_sf --seeds 42
python -u scripts/run_g1_r2_ablations.py --variants g32_cp --seeds 42
```

探索也可按点或估计器分任务，例如：

```bash
python -u scripts/run_g1_r2_exploration.py --variants align040_sf align040_cp
python -u scripts/run_g1_r2_exploration.py --variants align053_sf align053_cp
python -u scripts/run_g1_r2_exploration.py --variants anneal_sf anneal_cp --seeds 3407
```

任务锁按 **run ID（变体 × seed）** 隔离。不同变体、不同 seed 可以共享输出根目录并行；
重复选择同一个 run 的任务会拒绝启动。锁保护共享文件系统中的重复任务，不调度 GPU；
独立磁盘上的任务仍需在提交时避免重复分配同一实验。

## 实验矩阵与解释

全部变体固定 LR 5e-6、113 steps、8 卡 × microbatch 16、G64（G32 变体除外）、
alignment 0.90（探索变体除外）、无 clipping、无 KL/辅助损失，使用最终 checkpoint。
同 seed 的训练数据顺序与 R2 一致，每个 epoch 在 source 内重新组合 microbatch。

| 组 | `--variants` 名称 | 相对对应 G64 对照的改动 |
|---|---|---|
| 核心 | `g32_sf`, `g32_cp` | G=32；估计器分别为 SF/CP |
| 核心 | `paired` | SF 的 rollout 改为 `diagonal` |
| 单侧 | `query_policy`, `document_policy` | SF 仅 query 或仅联合 document 动作 |
| 更新规则 | `norm` | SF 的 advantage 改为 per-component 标准化 |
| 更新规则 | `doc_mean` | SF 的 document log-prob 改为 mean |
| 更新规则 | `norm_doc_mean` | 同时标准化和 document mean |
| 校准 | `no_rescale_sf`, `no_rescale_cp` | 关闭 frozen-candidate rescaling |
| Alignment | `align040_sf/cp`, `align053_sf/cp`, `align065_sf/cp`, `align080_sf/cp`, `align095_sf/cp`, `align098_sf/cp` | SF/CP 分别固定 0.40/0.53/0.65/0.80/0.95/0.98 |
| 退火 | `anneal_sf`, `anneal_cp` | alignment 0.80→0.90，按既有实现线性调度 |
| Gaussian 对照 | `vmf_k755` | SF、固定 κ=755、`target_alignment=null`、vMF 采样 |
| Gaussian 失配 | `gaussian_k755` | 与 `vmf_k755` 相同 κ，仅改为 Gaussian 采样 |

表中 `align040_sf/cp` 是名称简写，实际 CLI 必须写 `align040_sf align040_cp`。
完整 alignment 曲线包含复用的 0.90，共七个点；本轮 0.53 是精确的配置值 0.53，
不复用历史 0.530237 的训练结果。

单侧、paired、Norm 和 DocMean 只用 SF；当前 CP 不支持这些组合，未解除其校验。
单侧 policy 仍更新共享 encoder，并非冻结另一套 encoder。
Norm/DocMean 改变样本与分支权重，固定 LR 比较不证明它们各自充分调优。
Product/paired 的 reward cell 数不同，同步数不等于同算力。

退火同时对照固定 0.80（本组 alignment 实验）和固定 0.90（原 R2）。
可以独立执行退火，不自动启动固定端点实验；汇总时相关结果齐全才计算差值。

**Gaussian 是已有实现的采样分布失配诊断**：实际采样是 projected Gaussian，
surrogate 仍使用 vMF log-density，不是正确 Gaussian policy-gradient 方法的比较。
保留与历史诊断一致的 κ=755，并新跑同 κ 的 vMF 对照；同 κ 不等于同平均 alignment，
因此结果不能单独归因为分布形状，也不能称为 alignment 0.90 的 Gaussian 替代。

## 输出、耗时与恢复

默认输出根目录沿用配置中的 `checkpoints/iclr2027-r2`，所有新 run 使用独立名称：

```text
checkpoints/iclr2027-r2/
  G1-R2-RL-GradedNDCG32-SF-s42/
    checkpoint-25/ ... checkpoint-113/
    mteb_eval/bright/
  ...
  .r2_ablations/
    <run ID>/
      contract.json
      state.json
      config.json
      train.log
      eval.log
      runtime.json
      timings.jsonl
      events.jsonl
    locks/
    queues/<variants>/seeds-<seeds>/summary.json
    queues/<variants>/seeds-<seeds>/summary.md
    summary.json
    summary.md
```

`runtime.json` 在训练启动时记录 GPU 型号、显存、Torch/CUDA/DeepSpeed 版本与 nproc。
`timings.jsonl` 分别记录每次训练/评测进程的 wall time、exit code；训练时间包括启动、
加载和保存，不是纯 optimizer 时间。单次成功训练的 8 卡分配成本可按 `seconds × 8 / 3600`
换算 GPU-hours。原 R2 对照没有这些 timing 文件时保留缺失值，不填入 probe 耗时。
跨硬件的时间不能直接解释为算法速度差异。

同一命令重启时：完整且合同一致的 run 跳过；训练成功但评测失败只补评测；
失败训练保留现场，不从部分 checkpoint 自动续训，也不覆盖已有权重。
一个 run 失败会记录并继续其他独立 run，任务最终返回非零。

需重训失败的单个 run 时，使用新的输出根目录，并只选择该变体/seed：

```bash
python scripts/run_g1_r2_ablations.py --variants g32_sf --seeds 42 \
  --output-root checkpoints/iclr2027-r2-ablation-retry
```

`--output-root` 只改变新实验输出；既有 G64 对照仍默认从 settings 的输出根目录读取。
对照位于其他位置时，用 `--reference-root` 指定。未修改原夜跑脚本、训练源码或原 R2 suite，
不会仅因新增这些入口而改变原夜跑的源码指纹。

## 汇集结果

运行中的自动汇总按变体集合和 seed 集合隔离，避免独立任务互相覆盖。
全部结束后可统一汇总所有已配置实验：

```bash
python scripts/run_g1_r2_ablations.py summary --variants all
```

只运行核心组时，省略 `--variants all`；也可以用 `--variants`、`--seeds` 精确选择已运行集合。
最终总表写入新输出根目录的 `.r2_ablations/summary.{json,md}`。
仅完整的所选 seed 组报告均值、样本 SD、最差值；单 seed 不报告 SD。
包含逐 seed 配对差、逐领域分数、训练耗时和每个 run 的数据哈希。
缺结果/合同或数据/训练源码不一致时返回非零，不用历史 CSV 代替。

各机独立磁盘时，按原目录结构汇集新 run 的 `mteb_eval/bright/` 与
`.r2_ablations/<run ID>/`；还需原 G64 SF/CP 六个 run 的 `mteb_eval/bright/` 和
`.r2_batch/<run ID>/{contract,state}.json`。只汇总不需要复制模型权重。
数据及源码哈希不一致时禁止聚合；路径可迁移，合同文件保留原内容。

补跑分散在不同输出根目录时，汇总前也按 run ID 汇集到同一个根目录。
新结果的合同/状态在训练时校验当前源码与配置；汇总时校验已保存合同及运行配置，
并核对各运行保存的 `src/` 源码哈希一致性。

## 本地验证

26 个变体均通过小型真实 Qwen3 的 graded reward 前向/反向测试，使用各自配置的 G32/G64、
policy、估计器和探索设置；另覆盖独立任务锁、评测失败重试、合同与数据/源码一致性、
缺失 seed 不计算均值。本地未启动真实 GPU 训练。
