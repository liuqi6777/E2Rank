# G1 过夜稳定性实验：7 个配方 × 3 个 seed

日期：2026-09-15。目标是 **BRIGHT 结果好，同时完整训练不随 seed 大幅变化**。执行环境为单机 8 卡；约 10 小时仅用于估算，按用户要求**不设时间停止条件，一直执行完队列**。

> **完成状态：**21 次训练及完整 BRIGHT 评测均已完成。结果、配对比较和结论见 [G1 过夜稳定性实验结果](g1_stability_overnight_results.md)。下文保留执行前的实验设计与决策口径。

以下保留执行前状态：配置与运行脚本已就绪，尚未启动正式训练。共 **21 次全新训练**，包含新控制组；完整新批次便于各配方在同一轮条件下比较。跨机器文件 hash 不同不能单独说明训练样本变化，也不意味着旧控制结果无效。

## 1. 比较矩阵与执行顺序

每个配方依次运行 seed 42、3407、2026，先完成一整组的三次重复，再进入下一配方。每次训练后立即评测最终模型的完整 BRIGHT。

| 顺序 | Recipe / run 名中间部分 | Reward / 标签 | G | LR | 次数 | 要回答的问题 |
|---|---|---|---:|---:|---:|---|
| 01–03 | `MRR32` | MRR@10 / binary | 32 | 5e-6 | 3 | 当前配方在本轮环境下的质量和整体 seed 波动 |
| 04–06 | `GradedNDCG64` | nDCG@10 / graded | 64 | 5e-6 | 3 | 丰富反馈与增加采样预算的组合能否改善最终训练 |
| 07–09 | `MRR64` | MRR@10 / binary | 64 | 5e-6 | 3 | 对照 MRR32，增加采样预算是否有效 |
| 10–12 | `GradedNDCG32` | nDCG@10 / graded | 32 | 5e-6 | 3 | 对照 MRR32 与 GradedNDCG64，区分 reward 和 G 的作用 |
| 13–15 | `BinaryNDCG32` | nDCG@10 / binary | 32 | 5e-6 | 3 | 区分 nDCG 指标变化与 graded 标签信息变化 |
| 16–18 | `GradedNDCG64LRHalf` | nDCG@10 / graded | 64 | 2.5e-6 | 3 | 较小学习率能否进一步改善稳定性且保留质量 |
| 19–21 | `MRR64LRHalf` | MRR@10 / binary | 64 | 2.5e-6 | 3 | 在 MRR 下检验相同的学习率折中 |

Run ID 为 `G1-S-<Recipe>-Seed<seed>`，例如 `G1-S-GradedNDCG64-Seed3407`；输出为 `checkpoints/iclr2027/<Run>-s<seed>/`。完整配置在 [suite.yaml](../configs/experiments/iclr2027/suite.yaml)，运行脚本会核对每行的核心实验契约，避免错配。

这是一组训练改进实验，不以梯度诊断是否达到某个阈值作为启动条件。当前单位方向逐文档 baseline 不进入本批。

## 2. Seed 与固定条件

本轮每条重复同时设置：

| 重复 | training seed | data_seed | 独立 rollout_seed |
|---|---:|---:|---:|
| 1 | 42 | 42 | 42 |
| 2 | 3407 | 3407 | 3407 |
| 3 | 2026 | 2026 | 2026 |

这直接检验整体 seed 稳定性，不是只变 rollout seed 的因果隔离；不能将新控制与旧的三个固定 data run 混合计算标准差。各配方采用同样的 seed 配对，利于比较；不同训练轨迹不意味着实际采样动作一直相同。

所有运行均从 E0 `Qwen/Qwen3-Embedding-0.6B` 开始 full FT，固定 113 optimizer steps、global batch 128 / microbatch 16、8 卡、alignment 0.90、双侧 vMF product、LOO、无 advantage normalization、document log-prob sum、frozen-candidate rescaling、shared document baseline、KL=0。只改变表中列出的 reward/标签、G、LR 和重复 seed，不恢复其他 run 的优化器状态。

使用同一份 prepared 数据，不重新抽取代表正例；既有预处理 seed 42 保持不变。学习率在 suite 的 protocol 和实际 overrides 中均显式设置，避免公共 settings 或显示摘要掩盖低 LR 对照。

## 3. 运行命令

在训练机器上进入项目根目录，激活原训练环境，先预检：

```bash
python scripts/run_g1_stability.py check
```

直接在现有 tmux/screen 会话运行：

```bash
python scripts/run_g1_stability.py train
```

若希望退出终端后继续运行，可用：

```bash
mkdir -p outputs/g1_stability_night
nohup python -u scripts/run_g1_stability.py train \
  > "outputs/g1_stability_night/launcher-$(date +%Y%m%d-%H%M%S).log" 2>&1 < /dev/null &
```

脚本只支持本次约定的 8 卡，不设置小时数或截止时间，也不强制中断训练/评测。全队列预检通过后开始；训练模式还会先检查至少 8 个可见 CUDA 设备。现有 runner 完成每次训练和 BRIGHT；单次返回失败则保存失败状态、继续下一行，不自动降 G、改 LR、覆盖输出或重试。

已完成的输出会被既有预检保护，不会因重启队列而重跑。若手动中断后需要从尚未开始的第 10 行继续：

```bash
python scripts/run_g1_stability.py check --start-at 10
python scripts/run_g1_stability.py train --start-at 10
```

`--start-at` 表示队列位置，不是 checkpoint resume。失败行保留其输出与 launch receipt，需另行检查恢复；后续行照常运行。

## 4. 日志与自动汇总

每次启动建立独立目录 `outputs/g1_stability_night/<UTC时间戳>/`：

- `plan.json`：完整解析配置、数据 hash、代码 commit/dirty 标识。
- `01_<Run>.log` 等：逐次训练及 BRIGHT 的完整输出。
- `status.jsonl`：成功/失败、返回码和每次训练加评测的耗时。

每行结束后刷新 `outputs/g1_stability_night/summary/results.md` 和 `results.json`。手动重新汇总：

```bash
python scripts/run_g1_stability.py summary
```

汇总只读取各 run 的最终 `mteb_eval/bright/**/BrightRetrieval.json`，要求完整 12 个 subset；部分评测不计算宏平均。只有三个 seed 都有完整成绩时，才报告该配方的均值、样本标准差、最差与最好 seed。失败或缺失项明确保留，不用剩下两个结果冒充三 seed。

逐 run 表头固定为 biology、earth_science、economics、psychology、robotics、stackoverflow、sustainable_living、pony、leetcode、aops、theoremqa_theorems、theoremqa_questions，保持每个 subset 按顺序列出。

## 5. 明早如何作决定

主要看每个配方的 **BRIGHT 均值、样本标准差、最差 seed 成绩**，并检查 12 个领域是否发生集中退化。先与本批 MRR32 控制比较：均值不降、标准差不升、最差 seed 不降且至少一项改善的方案优先。出现质量/稳定性折中时同时呈现，不用事后设定的单一分数隐藏折中，也不挑最好 seed 替代均值。

具体对比：MRR32→MRR64；GradedNDCG32→GradedNDCG64；MRR32→BinaryNDCG32→GradedNDCG32；两个 G64 配方分别比较原 LR 与半 LR。MRR/graded × G32/G64 的四格还能判断两种改动的组合是否互补。

三个 seed 是配置开发证据，不能可靠估计尾部失败概率。BRIGHT 本轮用于配置选择，后续写论文应如实标注这一点；优胜方案再用未参与选择的新 seed 验证，不把本轮最佳点估计直接当最终定论。

## 6. 运行时间

已有 W&B 缓存中，MRR/G32、MRR/G64、graded nDCG/G32 的训练时间分别约 705、731、693 秒。按该硬件与数据条件，21 次训练本身约 4.1–4.3 小时，模型加载、保存及 BRIGHT 评测另计。历史训练时间不是本轮总耗时保证；缺少完整 BRIGHT wall-time 对照，无法保证全部在 10 小时内结束，超过也会继续执行至队列结束。
