# G1 新协议整批运行

这批复用旧实验中较稳定的 G64、alignment 0.90 和 LR 5e-6，重新比较三种 reward 与两种梯度估计器。
实验 seed 为 42/3407/2026，可通过 `--seeds` 选择本机执行哪些；113 optimizer steps、每台机器 8 卡 × microbatch 16；不设 dev，沿用现有丢尾规则，每个 epoch 在 source 内重新组合 microbatch。

采样协议为 `per_source_epoch_shuffle_retained_tail_v1`：先按原规则确定本次运行保留的样本，再用 `data_seed + epoch` 的独立 RNG 在 source 内重排样本并重新分批，最后打乱整批顺序。启用长度桶时，重分组也限制在同一长度桶内；只够一个完整 microbatch 的组，其同伴集合自然不变。同 data seed、同 epoch 的各方法和各 rank 生成相同的全局顺序，由 Accelerate 分配完整 microbatch；dataset 索引不变，支持持久化 DataLoader workers。

三台机器应同步到包含此修复的同一个提交后启动，命令不变。运行合同已包含采样协议和源码指纹；若已用固定同伴版本启动过 R2，保留旧结果并修改 `configs/experiments_r2.yaml` 的 `output_dir` 后重跑，脚本会拒绝在原合同下混用两种采样行为。

## 一条命令启动

在 GPU 机器的项目根目录、已激活训练环境后运行：

```bash
nohup python -u scripts/run_g1_r2.py > g1_r2_night.log 2>&1 &
```

脚本默认执行全部任务，无需写 `run`。使用 8 张可见 CUDA GPU；可在启动前用 `CUDA_VISIBLE_DEVICES` 指定。
没有时长上限，直到队列结束。G2/G3、额外 G/alignment 搜索和整套 MTEB 不在本批内。
W&B 默认 offline，避免夜间登录提示；已显式设置的 `WANDB_MODE` 会保留。

只检查配置和实际训练文件哈希，不下载模型、不启动 GPU：

```bash
python scripts/run_g1_r2.py check
```

查看进度、重建汇总：

```bash
tail -f g1_r2_night.log
python scripts/run_g1_r2.py summary
```

## 三台机器按 seed 并行

三台机器使用同一版本代码、相同训练文件内容，每台分别执行对应的一条命令：

```bash
# 机器 1：9 次训练，加 E0 评测与两次公共梯度诊断
nohup python -u scripts/run_g1_r2.py --seeds 42 > g1_r2_s42.log 2>&1 &
# 机器 2：9 次训练
nohup python -u scripts/run_g1_r2.py --seeds 3407 > g1_r2_s3407.log 2>&1 &
# 机器 3：9 次训练
nohup python -u scripts/run_g1_r2.py --seeds 2026 > g1_r2_s2026.log 2>&1 &
```

`--seeds` 也接受多个值，例如 `--seeds 3407 2026`；不传则执行全部三个。`--seed` 是相同参数的别名。
E0 评测和两次公共诊断只属于 seed 42 的队列；其他机器不等待 seed 42 的结果，也不重复这些任务。
`check` 和 `summary` 支持相同参数，例如 `python scripts/run_g1_r2.py check --seeds 3407`。

每个单 seed 队列的事件和汇总写到 `.r2_batch/queues/seeds-<seed>/`，训练状态和日志仍按 run ID 保存。
单 seed 汇总显示该 seed 的分数和 CP−SF 差，不计算样本 SD。不同 seed 可以共用输出根目录，互不覆盖事件、汇总或运行状态。
文件锁按 seed 分配，包含相同 seed 的两份队列不能同时运行；全量队列会占用三个 seed 的锁。

三台使用共享输出根目录时，全部完成后直接运行 `python scripts/run_g1_r2.py summary`，生成 `.r2_batch/summary.md` 的三 seed 汇总。
若使用独立磁盘，先把每个 run 的 `mteb_eval/bright/`、对应 `.r2_batch/<run ID>/` 状态/合同，以及 `queues/` 和公共诊断结果按原目录结构汇集到同一个输出根目录，再执行该命令；仅汇总分数不需要复制模型权重。

## 27 次训练与执行顺序

| 方法 | 标签/目标 | G | 估计器 |
|---|---|---:|---|
| CL | 全部已知正例，multi-positive InfoNCE | — | 直接监督梯度 |
| LL-Binary | binary LambdaLoss@10 | — | 直接监督梯度 |
| LL-Graded | teacher graded LambdaLoss@10 | — | 直接监督梯度 |
| RL-MRR64-SF / CP | binary MRR@10 | 64 | 原估计器 / 条件投影 |
| RL-GradedNDCG64-SF / CP | graded nDCG@10 | 64 | 原估计器 / 条件投影 |
| RL-BinaryNDCG64-SF / CP | binary nDCG@10 | 64 | 原估计器 / 条件投影 |

默认单机全量执行时，先做 E0 固定状态配对梯度诊断和 E0 BRIGHT 评测，再按表中顺序完成 seed 42、3407、2026。
指定 `--seeds` 后，只执行所选 seed 的九方法比较，方法内部顺序不变。
seed 42 的 LL-Binary 训练后，追加其 step 25 权重上的同状态配对诊断。两次诊断都使用 MRR64，固定 epoch-0 microbatch 位置 0/100/200、16 个独立 rollout seed。
这些位置在看结果之前确定；它们不是 dev，也不保证代表全部 source。每次诊断比较同一份权重、同一批实际动作和奖励。

按整批运行安排，脚本不等待人判断降噪幅度后再排三 seed；诊断失败会记录，后续独立任务继续执行。
不根据 BRIGHT 分数动态挑方法、修改配方或提前停止。`--skip-probe` 可显式跳过两次独立梯度诊断。

每次训练完成后只评测最终模型的 BRIGHT，必须包含全部 12 个 subset 才记为成功。
保留 step 25/50/75/100/113 的模型 checkpoint，便于后续机制分析；`save_only_model: true` 不保存 optimizer 状态，不支持从这些 checkpoint 续训。
训练样本的 sampled / deterministic 同池 / 大池三层检索分析尚未自动接入本脚本，不属于本批已完成的验证。

## 配置与输出

- 日常配置：[experiments_r2.yaml](../configs/experiments_r2.yaml)，设置数据路径和独立输出根目录。
- 矩阵：[suite_r2.yaml](../configs/experiments/iclr2027/suite_r2.yaml)，含 27 条训练和 1 条 E0 评测。
- 训练设置：[g1_r2.yaml](../configs/experiments/iclr2027/g1_r2.yaml)，固定模型 revision、无 dev、无裁剪和模型 checkpoint 保存规则。
- 夜跑入口：[run_g1_r2.py](../scripts/run_g1_r2.py)。旧配置与旧批量脚本保持历史用途。

完整配方的关键值由预检核对；修改实验目标或超参时，应修改对应矩阵和预检合同，并使用新的输出根目录。
数据路径可迁移，文件内容须匹配计划记录的 SHA256；复用文本，不重新划分或生成训练数据。
同一个 seed 的 SF/CP 运行配置只允许在估计器、名称、输出目录上不同。

默认输出：

```text
checkpoints/iclr2027-r2/
  G1-R2-E0-s42/mteb_eval/bright/
  G1-R2-CL-s42/
  G1-R2-RL-MRR64-CP-s42/
  ...
  .r2_batch/
    events.jsonl
    summary.json
    summary.md
    queues/seeds-<seed>/
      events.jsonl
      summary.json
      summary.md
    locks/seed-<seed>.lock
    gradient_probe/
    gradient_probe_ll25/
    <run ID>/
      contract.json
      config.json
      state.json
      train.log
      eval.log
```

`contract.json` 记录实际展开配置、数据/源码 SHA256、DeepSpeed 配置与评测命令。
实际 DeepSpeed engine 参数在 `train.log` 中输出；两处梯度裁剪均显式设为 0。
原始 E0 权重和 tokenizer 均固定 revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`；输入为 token protocol v2，pooling/评分 FP32。
训练 backbone BF16，外部评测统一 FP16 backbone，与历史评测精度一致；模型输出和结果缓存使用独立 R2 路径。

## 重启与失败处理

重新执行同一条启动命令即可续跑队列：

- 已完成：核对配置/源码/数据身份、最终模型和完整评测结果后跳过。
- 训练成功、评测失败：只重跑评测，不重新训练。
- 训练失败或中断：保留现场并标记失败，继续其他独立任务；不自动覆盖或从部分 checkpoint 续训。需要重训时使用新的输出目录。
- 已存在但不是本脚本创建的输出：拒绝接管。
- 梯度诊断失败：保留已写出的部分统计；下次使用新 attempt 文件重试。

同一输出根目录按 seed 加锁，不同 seed 可并行。任何失败都会写入各自的事件/汇总，脚本最终返回非零。
汇总仅对完整的所选 seed 计算均值和最差分数；单 seed 的样本 SD 为 null。三 seed 总表要求三个结果齐全；同时报告每个 reward 内的 CP−SF 配对差，以及逐领域结果。
训练时长记录在事件日志中；n=3 是稳定性复现，不能当作充分调优或严格显著性的证明。
