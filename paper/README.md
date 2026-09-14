# Paper workspace

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

Regenerate the main method figure from the repository root:

```bash
python paper/scripts/draw_pipeline_figure.py
python paper/scripts/draw_rag_query_only_figure.py
```

The figure scripts require Matplotlib. The manuscript consumes the PDFs under
`iclr2027/figs/`; the PNGs are local previews and need not be kept.
