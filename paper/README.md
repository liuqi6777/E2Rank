# Paper workspace

G1 已完成实验的综合分析见 [G1_RESULTS.md](G1_RESULTS.md)，包括全部 12 个 BRIGHT subset 的主表、探索与机制消融、领域分析和复现材料。[完整附表](g1_results/all_runs.md) 收录 45 次训练和 E0 的结果；这些材料独立于论文 LaTeX 正文。

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
