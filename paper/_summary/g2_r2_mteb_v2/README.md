# G2-R2 / MTEB English v2

2026-11-10：原 `iclr2027_rebuttal/table12_seeds/` 与本目录的
`cl_to_reler_*` 已合并。三个重复 CL run 的 123 个任务分数及 27 个聚合指标
完全一致，合并时去重；所有原有分数不变。本次没有修改论文或 rebuttal 正文。

## 统一入口

| 文件 | 内容 |
| --- | --- |
| [run_summary.csv](run_summary.csv) | 全部 12 个 run 的总体与 Retrieval 指标 |
| [type_summary.csv](type_summary.csv) | 全部 run 的七类指标 |
| [task_summary.csv](task_summary.csv) | 全部 run 的 492 个任务分数 |
| [retrieval_summary.csv](retrieval_summary.csv) | 每个 run 的 10 个 Retrieval 任务 |
| [seeds/per_run.csv](seeds/per_run.csv) | 三方法 × 三 seed 的完整精度指标、初始化和结果路径 |
| [seeds/per_task.csv](seeds/per_task.csv) | 九个主要 run 的 369 个完整精度任务分数 |
| [seeds/grouped.csv](seeds/grouped.csv) | 三方法的均值和样本 SD |
| [seeds/paired.csv](seeds/paired.csv) | 三种两两比较：逐 seed 差值及均值和样本 SD |
| [seeds/results.json](seeds/results.json) | 完整精度数据、任务类型和聚合结果 |

根目录四张表沿用原格式，显示两位小数。统计以 `seeds/` 的完整精度数据为准。
全部 run 均有 41 个任务；主要比较使用 seed 42、3407、2026。
另外三条 seed42 历史 run（普通 D-CL、旧 D-RL、K7 RELER）保留在根目录表中，
不混入三 seed 均值。

`infonce` = CL-Strong；`reler` = 从 Qwen3-0.6B 直接训练的 K0 RELER；
`cl_reler` = 从各自 seed 的 CL-Strong 最终 checkpoint 继续训练的 K0 RELER。
前两者各训练 1200 steps；第三者为 1200 CL + 1200 RELER steps，沿用同一份
E2Rank 数据，第二阶段重新初始化 optimizer/scheduler。当前尚无 CL → CL 结果；
[续训对照代码](../../../docs/g2_cl_to_cl.md)已准备好，完成后以 `cl_cl` 合并进同一处。

## 结果与 rebuttal 判断

下表均为 `main_score × 100`；± 为三个 seed 的样本 SD。最后一列先按 seed
计算 `(CL → RELER) − CL-Strong`，再求差值的均值和样本 SD。

| 指标 | CL-Strong | Direct RELER | CL → RELER | 配对变化 |
| --- | ---: | ---: | ---: | ---: |
| Classification | 72.59 ± 0.22 | 72.98 ± 0.46 | 73.46 ± 0.16 | +0.869 ± 0.207 |
| Clustering | 44.37 ± 0.16 | 45.56 ± 0.24 | 45.72 ± 0.21 | +1.346 ± 0.286 |
| PairClassification | 82.09 ± 0.03 | 79.94 ± 0.24 | 80.75 ± 0.05 | −1.341 ± 0.079 |
| Reranking | 44.96 ± 0.30 | 45.41 ± 0.27 | 45.18 ± 0.24 | +0.224 ± 0.135 |
| Retrieval | 51.63 ± 0.10 | 50.39 ± 0.31 | 50.92 ± 0.13 | −0.710 ± 0.043 |
| STS | 75.89 ± 0.15 | 76.98 ± 0.22 | 77.06 ± 0.05 | +1.171 ± 0.156 |
| Summarization | 28.99 ± 0.45 | 25.94 ± 0.59 | 26.32 ± 0.16 | −2.667 ± 0.519 |
| 41-task average | 60.98 ± 0.03 | 61.01 ± 0.15 | 61.34 ± 0.06 | +0.364 ± 0.030 |
| 7-type average | 57.22 ± 0.06 | 56.74 ± 0.13 | 57.06 ± 0.03 | −0.158 ± 0.073 |

**可以直接回应 nBHM-Q1：已经完成同数据、逐 seed 对应 checkpoint 的显式
CL → RELER 实验。** 41 任务均值在三个 seed 分别提升 +0.363、+0.395、+0.334。
但结论应同时报告 Retrieval 和类型均值下降，不能写成“CL 后继续 RELER 普遍有益”
或“检索效果提升”。四个类型提升、三个下降，三 seed 的方向一致。
任务均值与类型均值的差别来自权重：Classification/Clustering/STS 分别有
8/8/9 个任务，Summarization 只有 1 个任务。

**对 3QUB-W4，这是补充证据，原有负结果仍成立。** 与 direct RELER 比，
CL → RELER 的 Retrieval 高 +0.532 ± 0.203，41 任务均值高 +0.329 ± 0.203；
两者在三个 seed 上均提高。但 CL → RELER 仍低于 CL-Strong 的 Retrieval，
且总训练预算更长，不能将差值单独归因于 CL 初始化或 RL 目标。

**对 vQkB-W2，只能作为 BRIGHT 之外的补充评测。** 本次是 MTEB English v2，
并非完整标准 BEIR，也没有评测 BRIGHT checkpoint 的零样本迁移。
导出没有新增训练数据规模对照；1200 × 128 = 153,600 是每阶段的 query exposures，
不能代替唯一 query 数。因此不能据此声称解决了 BEIR transfer 或 data-abundant
场景的优势问题。没有 CL → CL，也不能排除额外训练预算的影响。

Retrieval 的下降分布在多个任务：ArguAna、CQADupstackGaming、CQADupstackUnix、
ClimateFEVERHardNegatives、FEVERHardNegatives、FiQA2018、Touche2020.v3
在三个 seed 上都下降；HotpotQAHardNegatives、SCIDOCS、TRECCOVID 在三个 seed
上都提高。前者共 7/10 个任务，不能用总体 embedding 均值掩盖检索代价。
现有数据也不能判定退化的机制是遗忘、数据分布差异或训练过久。

本次核对的是已导出的完整精度逐任务分数与 suite/checkpoint 对应信息；
没有重新运行训练或 MTEB 评测。正式 rebuttal 中原有“尚未做 CL → RL”的文字
需在作者决定采用本组结果后再更新。

## 后续更新

三个入口的 `summary` 统一更新本目录，保留其他组已经归档的结果：

```bash
python scripts/run_g2_table12_seeds.py summary
python scripts/run_g2_cl_to_reler.py summary
python scripts/run_g2_cl_to_cl.py summary
```

可用 `--config` 指定训练机路径，或 `--results-root <root>` 读取按 run 目录放置的
最终评测 JSON。缺少原始文件时会保留并提示使用完整归档；没有完整记录时，
三 seed 均值和 SD 留空。损坏或不一致的数据会报错。
配置与运行命令见 [直接训练说明](../../../docs/g2_table12_seeds.md)、
[RELER 续训说明](../../../docs/g2_cl_to_reler.md) 和
[CL 续训说明](../../../docs/g2_cl_to_cl.md)。CL → CL 汇总会新增
`cl_cl_minus_infonce` 与 `cl_reler_minus_cl_cl` 两组比较；结果齐全前均值和 SD 留空。

论文构表入口已改为读取 `seeds/`，仍仅选取 `infonce` / `reler` 两方法。
2026-11-09 的历史 evidence/patch 保持原样，其中的旧 `table12_seeds` 路径
对应这里的这六条记录。
