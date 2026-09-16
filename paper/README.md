# Paper workspace

2026-09-17 的 [G2-RL-R2 计划](G2_RL_R2_PLAN.md)已按最新 G1-R2 结果落地：E2Rank graded nDCG、ReasonEmbed binary nDCG，均使用 CP/G64/alignment 0.80；各 D/E/W 三条 RL、仅 seed 42，与现有新协议 CL 配对。两个运行脚本与独立配置已提供，未启动 GPU 训练。

论文正文已于 2026-09-16 同步新协议与 G1-R2 结果：方法部分加入条件投影推导；实验部分报告 27 次训练、三 seed 主表和全参数配对梯度探针；附录包含全部九方法的逐领域结果。G2 与 G3 保持计划状态。当前 PDF：[`iclr2027/build/main.pdf`](iclr2027/build/main.pdf)。结果来源见 [G1_R2_RESULTS.md](G1_R2_RESULTS.md)，后续计划见 [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) 和 [G2_CL_R2_PLAN.md](G2_CL_R2_PLAN.md)。

最新结果文档已于 2026-09-17 扩展到 **117 次训练 + 2 条 E0**：包含全部 G1-R2 消融、动态检索和 DIVER 初始化实验，见 [G1_R2_RESULTS.md](G1_R2_RESULTS.md) 与 [完整附表](g1_r2_results/all_runs.md)。CP 的优势跨 alignment/G 保持；DIVER 上 RL 优于 LL/CL，但仍低于自身 E0。论文 LaTeX/PDF 尚未纳入这批新增结果。用 `python scripts/analyze_g1_r2_results.py` 可重新生成结果支撑表与输入审计。

结果呈现决策（2026-09-17）：**主结果采用 graded CP/G64/alignment 0.80，配对 SF-0.80**，分别为 20.76 ± 0.20 与 18.81 ± 0.30；对 SF 的增益 +1.94，对 LL-Graded +1.39。0.80 根据本轮探索消融选择；机制消融和梯度探针保留 0.90 的匹配对照，原 run 名称与配置不变。当前 LaTeX/PDF 仍为原 0.90 版本，尚未同步这项呈现决策。

论文纳入决策（2026-09-17）：**DIVER 实验仅作内部记录，不纳入论文正文、附录、实验计数或主张。** 数据匹配问题是内部优先假设，尚未验证；上述 117 次训练与两条 E0 是归档总数。

G1 已完成实验的综合分析见 [G1_RESULTS.md](G1_RESULTS.md)，包括全部 12 个 BRIGHT subset 的主表、探索与机制消融、领域分析和复现材料。[完整附表](g1_results/all_runs.md) 收录 45 次训练和 E0 的结果；这些材料独立于论文 LaTeX 正文。

W&B 完整日志的后续诊断见 [G1_GRADIENT_ANALYSIS.md](G1_GRADIENT_ANALYSIS.md)，涵盖更新规则的梯度尺度、训练 reward 与 BRIGHT 差异，以及实际 clipping 配置的核查。

Seed 重复后的最新机制讨论见 [rollout 方差复盘](../docs/rollout_variance_discussion.md)，区分已验证的 reward 分辨率、现有边际化实现、待检验的梯度信噪比假设和后续对照。

当前 rollout seed 对照之后的工作见[后续实验计划](../docs/rollout_variance_experiment_plan.md)：先固定状态比较 reward 和逐文档反事实 baseline，再按诊断证据选择完整训练重复。

```text
paper/
├── README.md                 # this guide
├── EXPERIMENT_PLAN.md        # experiment source of truth
├── archive/                  # local historical PDFs; ignored by Git
├── scripts/
│   ├── draw_pipeline_figure.py
│   └── draw_rag_query_only_figure.py
└── iclr2027/
    ├── main.tex              # paper entry point
    ├── references.bib
    ├── sections/             # manuscript sections and appendix
    ├── figs/                 # figures referenced by main.tex
    ├── build/                # generated PDF and LaTeX intermediates
    ├── Makefile
    └── *.sty, *.bst          # self-contained ICLR style dependencies
```

## Common commands

Run these commands from `paper/iclr2027`:

```bash
# Full bibliography and cross-reference build.
make all

# One LaTeX pass for text-only iteration.
make quick

# Open the current compiled paper.
make view

# Remove generated intermediates but keep build/main.pdf.
make clean
```
