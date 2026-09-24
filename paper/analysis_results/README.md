# 论文补充分析结果

本目录与 [分析计划](../ANALYSIS_PLAN.md) 同处 `paper/` 下，用于保存后续分析输出。目前尚未运行分析，没有新增实验结果。

Git 忽略本目录下的 `.npy`、`.npz` 数组及对应的 `.partial` 中间文件，embedding 大文件仅保存在本地；报告、CSV/JSON 统计、embedding manifest 和图表仍可纳入版本控制。

[代表 checkpoint 配置](../analysis/bright_representatives.json) 已转为拟定的 `K=0` 主配方；其 `output_dir` 为 `paper/analysis_results/iclr2027_representatives_k0_s3407/`，checkpoint 从新 suite 的 `checkpoints/iclr2027-final-k0/` 读取。新 RELER checkpoint 尚待训练，已有基线须先从旧输出导入。从仓库根目录执行计划中的命令，脚本会自动创建以下目录：

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

训练 reward 曲线使用独立的 `reward_curves_k0/` 子目录，六条 `K=0` 运行见[运行清单](../analysis/reward_curve_runs.csv)。待取得对应 history 后，保存逐步 `history.csv`、运行与字段来源 `run_manifest.json`、聚合后的 `curves.csv`、图注 `README.md` 及论文图 PDF/PNG。目前尚未导出这组日志或生成曲线。梯度图的原始逐 probe 统计、配置与图也应放在独立子目录，区分新组合 reward 探针与旧 local-pool 探针。
