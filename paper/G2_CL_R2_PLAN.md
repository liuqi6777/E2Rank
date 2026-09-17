# G2-CL 新一轮实验设计（2026-09-16）

> 2026-09-18 状态更新：E2Rank 的 D/E/W 普通 CL、Strong CL 及对应 RL 结果均已同步，详见 [G2-R2 结果](G2_R2_RESULTS.md)。脚本和配置恢复为原 D/E/W 版本，W 和 Strong CL 分支保留；本次仅更新文档，不重新训练。下文保存原设计、命令和当时验证记录，BGE-M3 完成状态不能由这份 E2Rank 汇总推断。

沿用原 G2 的 D/E/W 初始化、数据、训练预算、优化器和评测设置，应用昨晚的统一输入/数值协议与跨 epoch 重分组修复。先准备 E2Rank 三条全新 CL，seed 42；BGE-M3 三条作为后续同配方复现，在候选采样可复现问题解决后启动。本次完成设计与独立配置，不启动 GPU 训练。

配置：[日常设置](../configs/experiments_g2_cl_r2.yaml)、[新矩阵](../configs/experiments/iclr2027/suite_g2_cl_r2.yaml)、[训练预设](../configs/experiments/iclr2027/g2_cl_r2.yaml)。使用现有 `scripts/experiment.py`，显式传入新 suite/settings；旧 `run_g2_cl.sh` / `run_g2_bge_cl.sh` 仍指向历史目录。

## 新增强化版 CL（2026-09-17）

新增 `G2-R2-D-CL-Strong` / `G2-R2-E-CL-Strong` / `G2-R2-W-CL-Strong`，在原配方上启用全部候选的跨卡负样本池和完整文档梯度，保留去重/已知正例过滤。其余设置、seed 42、1200 步、temperature 0.03、callback 与最终 MTEB 均沿用对应原 CL；不把这次实现强化描述为经过 dev 调优的最优 CL。

W-CL-Strong 从原 `G2-R2-D-CL` 的 W0 初始化，与 W-CL/W-RL 保持同一起点；不从 D-CL-Strong 初始化。默认跑 D/E/W，W0 尚未就绪时可先选 D/E。

```bash
python scripts/run_g2_cl_strong_r2.py check
python -u scripts/run_g2_cl_strong_r2.py train
python -u scripts/run_g2_cl_strong_r2.py train --branches D E
python -u scripts/run_g2_cl_strong_r2.py eval --branches E
```

该独立入口只检查实际数据文件、W0 权重和输出目录，不调用 manifest/hash/receipt 校验或旧队列恢复逻辑。配置展开到输出根目录的 `.cl_strong_configs/`；最终模型和评测按 Strong run ID 保存。`eval` 只补评现有权重。原 `run_g2_cl_r2.sh` 保持原三条 CL 的启动行为。

## 1. 矩阵与依赖

| 顺序 | E2Rank run ID | 初始化 | 预算与用途 |
|---:|---|---|---|
| 1 | `G2-R2-D-CL` | `Qwen/Qwen3-0.6B`（B0） | 1200 步；建立新的 CL warm-up W0 |
| 2 | `G2-R2-E-CL` | 原始 `Qwen/Qwen3-Embedding-0.6B`（E0） | 1200 步；embedding 初始化监督基线 |
| 3 | `G2-R2-W-CL` | 本轮 D-CL 的最终权重（新 W0） | 再训练 1200 步；继续 CL 的监督基线 |

W-CL 只加载新 D-CL 模型，重新创建 optimizer/scheduler/step counter，并从本阶段 epoch 0 开始迭代数据；沿用原设计，不将两阶段改成一次连续 2400 步训练。
D/E 无权重依赖，可独立执行；W 必须等 D 的最终权重保存成功。E0 不加载任何 G1 微调权重。

BGE-M3 使用对应的 `G2-R2-BGE-D-CL` / `-E-CL` / `-W-CL`，每阶段 1 epoch、保存间隔 1000 步；W0 仅来自本数据集的新 D-CL。这里 BGE-M3 指训练数据，encoder 仍为 Qwen。
当前 `src/embedding_data.py` 使用未绑定 seed 的候选 RNG；新 suite 将这三条标为有实现前置条件，避免把设置了 data seed 误当作候选已经固定。先修复并验证其跨 rank/worker、方法的候选一致性，或冻结实际候选文件，再解除该条件。该修复不属于本次设计的已完成项。

## 2. 沿用的固定设置

| 项目 | 设置 |
|---|---|
| 数据 | E2Rank 现有 `train.jsonl`；BGE-M3 现有 source 目录；不重新划分、不设内部 dev |
| 标签/候选 | binary；E2Rank 沿用 `pos_index` 的已知正例；保留原 slate 16 和 masked device-local in-batch 代表正例、去重/已知正例过滤、跨 query 文档 detach |
| 目标 | joint full FT、InfoNCE、temperature 0.03、in-batch negatives；不引入 RL 或其他监督损失 |
| 训练长度 | query 512、document 1024；不启用长度桶；不限制每 source 样本数 |
| Batch/seed | 8 GPU × microbatch 16，accumulation 1，global batch 128；training/data seed 42 |
| 优化器 | AdamW，LR 5e-6，weight decay 0.01，linear schedule，warmup ratio 0.03 |
| Backbone | BF16、gradient checkpointing、原 `scripts/zero3.json`；沿用 G2 的裁剪配置，记录实际 engine 值 |
| 表示 | last-token pooling、left padding、append pad、原 query/document prompt；评测上限 8192 |
| 曲线评测 | `checkpoint-0` 与每个保存点运行 `MTEB(eng, v1, subset)`；E2Rank 间隔 200 步，BGE-M3 间隔 1000 步 |
| 最终评测/选模 | 固定预算最终模型，FP16 backbone，完整 `MTEB(eng, v2)`；不按中间分数选 checkpoint |

本轮仍是原来的单 seed 设计，不把 G1 的 113 步、三 seed、关闭裁剪等额外配方调整直接搬到 G2。若后续增加重复，D/E/W 必须整组配对 seed，W 使用对应 seed 的 D。

## 3. 应用昨晚的更改

1. `64d2e7d`：统一 tokenization v2。显式末尾 token 在截断后添加，预留一个位置；训练 collator、callback 与外部评测共用协议。pooling/归一化及 cosine 评分使用 FP32，评分区间关闭 autocast/TF32；backbone 精度沿用原设置。
2. `33c55ff`：保留原选样和按 source 丢尾规则，用 data seed 固定保留集合；每个 epoch 在 source 内重新组合完整 microbatch，再打乱整批顺序。同 seed 的 D/E/W 使用同一数据分组与阶段内顺序。
3. 昨晚条件投影及配对梯度诊断针对 RL，不改变本轮 InfoNCE。G1-R2 的 portable manifest 验证也不自动成为 G2 的数据合同。

旧 CL、W0、向量索引和结果缓存不能代替本轮训练产物。输出根目录为 `checkpoints/iclr2027-g2-cl-r2`，索引缓存也单独隔离；历史结果仅作附表。新 E0 固定 revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`。
B0 沿用原模型标识，目前尚未固定 immutable revision；正式启动前解析并记录其模型/config/tokenizer revision，必要时在 D 行 overrides 中固定 SHA。后续 D-RL 应使用同一 SHA。

## 4. 预算与结果判读

E2Rank 共 3 × 1200 = **3600 optimizer steps、460,800 次 query 呈现**。D 和 E 各 153,600 次；W 本阶段也是 153,600 次，但 B0→W-CL 整条路线含 D 前缀，共 2400 步、307,200 次呈现。呈现次数不等于独立 query 数。
BGE-M3 共三个 1-epoch 阶段；实际保留样本、丢尾、steps 在 GPU 机器核验后填写，不预估不存在的本地数据。保存点的 callback 与最终评测耗时单列，不把同 steps 解释为同算力。

主表报告每个初始化的最终完整 MTEB 聚合及逐任务分数；曲线作为固定预算轨迹，不挑峰值。D/E 初始化差异与 W 的额外 warm-up 成本分别说明；单 seed 不报告训练 seed SD。
后续 W-RL 从同一新 D-CL W0 出发，不能从 W-CL 最终权重出发；E-RL/D-RL 分别匹配 E0/B0。此阶段先建立新协议下的 CL 对照，不据此选择新的 RL 搜索矩阵。

## 5. 预检与执行方式

专用入口为 [run_g2_cl_r2.sh](../scripts/run_g2_cl_r2.sh)。在已激活训练环境的 8 卡 GPU 机器根目录，设置好数据路径后：

```bash
bash scripts/run_g2_cl_r2.sh check
nohup bash scripts/run_g2_cl_r2.sh train > g2_cl_r2.log 2>&1 &
tail -f g2_cl_r2.log
```

脚本固定 8 卡、按 D→E→W 串行，每条自动完成最终 MTEB；任一步失败即停止。默认 W&B offline，保留已显式设置的模式。它只运行 E2Rank 三条 CL。

在 GPU 机器根目录，先修改新 settings 中的数据路径，使其与旧实验实际使用的数据一致。下面的函数始终传入新矩阵和设置：

```bash
g2cl() {
  python scripts/experiment.py "$@" --gpus 8 \
    --suite configs/experiments/iclr2027/suite_g2_cl_r2.yaml \
    --config configs/experiments_g2_cl_r2.yaml
}

g2cl check G2-R2-D-CL
g2cl check G2-R2-E-CL
g2cl show G2-R2-W-CL

# 正式执行时，前一步成功才进入下一步；train 会自动跑最终完整 MTEB。
g2cl train G2-R2-D-CL &&
g2cl train G2-R2-E-CL &&
g2cl check G2-R2-W-CL &&
g2cl train G2-R2-W-CL
```

第一轮 W 预检缺少新 D-CL 是预期的依赖未完成；D 完成后再运行 W 的正式 check。BGE-M3 先保留为 blocked，不直接续用历史 W0。
运行前记录训练数据 SHA256（目录数据记录文件清单和逐文件 SHA）、代码 commit、依赖/解析后的 MTEB 任务与数据版本、B0 revision。现有入口保存展开配置和命令到 `.launches/<run>-s42/`，但不自动校验 G2 数据 manifest/hash，需把这些证据另存到运行根目录。
输出已存在或有 launch receipt 时入口会拒绝覆盖；若训练成功但最终评测失败，保留模型并只重跑 receipt 中的 `post_train_commands`，不重新调用 train。故障重训使用新根目录。

本地缺少 E2Rank `data/train.jsonl`、BGE-M3 目录及新 W0；CPU 验证可检查完整展开配置与依赖，但不代表真实 GPU/数据预检已通过。
已完成六条新旧配置的结构化比较：目标、temperature、batch、seed、长度、优化器/DeepSpeed、保存间隔和最终评测均沿用历史设置；W 依赖只指向本轮同数据集 D-CL。实际入口预检仅报告上述缺失输入/依赖和 BGE 候选 RNG 前置条件，未发现其他未决配置。
