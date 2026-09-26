# 论文补充分析结果

本目录与 [分析计划](../ANALYSIS_PLAN.md) 同处 `paper/` 下，用于保存后续分析输出。现已收录 F1 匹配主配方梯度探针、F2 训练 reward 曲线，F3 的 12 子集三 seed 配对差值图，以及 seed 3407 的 F3/F5 四领域检索案例。

Git 忽略本目录下的 `.npy`、`.npz` 数组及对应的 `.partial` 中间文件，embedding 大文件仅保存在本地；报告、CSV/JSON 统计、embedding manifest 和图表仍可纳入版本控制。

[代表 checkpoint 配置](../analysis/bright_representatives.json) 已转为拟定的 `K=0` 主配方；其 `output_dir` 为 `paper/analysis_results/iclr2027_representatives_k0_s3407/`，checkpoint 从新 suite 的 `checkpoints/iclr2027-final-k0/` 读取。`K=0 + pairwise` 主配方三 seed 与纯 graded 三 seed 的最终权重及中间 checkpoint 已在新根目录就位，seed 3407 的 F3/F5 四领域案例已在最终表格分数锁定后运行。从仓库根目录执行计划中的命令，脚本会自动创建以下目录：

```text
iclr2027_representatives_k0_s3407/
├── <subset>/
│   ├── data/                       # 固定文本、qrels、manifest
│   ├── <model>/
│   │   ├── embeddings/             # query/document embedding 缓存
│   │   ├── retrieval/              # 查询指标、排名、正例排名
│   │   └── perturb/                # 查询扰动分析
│   └── geometry/                  # 固定文档对、谱、邻域、候选并集
└── reports/<content_id>/           # report.md 和汇总 CSV
```

后续批次在本目录下另设子目录，并在对应配置中指定 `output_dir`。回传时保留配置、已完成的 `retrieval/`、`geometry/`、`perturb/`、`reports/` 以及运行时间和峰值显存记录；不需要回传完整 embedding 数组或模型权重。

训练 reward 曲线使用独立的 `reward_curves_k0/` 子目录，六条 `K=0` 运行见[运行清单](../analysis/reward_curve_runs.csv)。逐步 `history.csv`、运行与字段来源 `run_manifest.json`、聚合后的 `curves.csv` / `curves_per_seed.csv`、最终 BRIGHT 对照 `bright_final.csv`、图注 `README.md` 及论文图 PDF/PNG 已由 `scripts/export_reward_curves_k0.py` 导出（2026-09-24）。匹配主配方的梯度图、逐 probe 统计和协议见 [`gradient_probe_k0_pairwise/`](gradient_probe_k0_pairwise/)，与旧 local-pool 探针分开。四领域报告见 [`iclr2027_representatives_k0_s3407/FINDINGS.md`](iclr2027_representatives_k0_s3407/FINDINGS.md)，附录案例图由 `scripts/plot_embedding_case.py` 从报告 CSV 绘制。

F3 的 12 子集三 seed 配对领域图与逐点 CSV 见 [`bright_domain_deltas_k0/`](bright_domain_deltas_k0/)，由 `scripts/plot_f3_domain_deltas.py` 从 `paper/_summary/k0_bright/subset_summary.csv` 绘制。
