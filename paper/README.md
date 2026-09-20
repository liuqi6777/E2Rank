# Paper workspace

2026-09-20：补充分析的[计划](ANALYSIS_PLAN.md)、[代表 checkpoint 配置](analysis/bright_representatives.json)和[结果目录](analysis_results/README.md)统一放在 `paper/` 下，避开 `iclr2027/` 的 Git ignore 规则。配置的输出路径为 `paper/analysis_results/iclr2027_representatives_s3407/`；分析尚未运行，RAG 仍是后续必须完成的实验。

补充分析现分为 embedding 几何与检索行为、梯度诊断、RL 训练 reward 曲线三组。Reward 曲线选取 RELER、Listwise 各三个 seed，见[运行清单](analysis/reward_curve_runs.csv)；优先复用已有训练日志，输出到 `paper/analysis_results/reward_curves/`，目前尚未生成曲线。

2026-09-20 论文呈现精简：监督对照保留 InfoNCE（历史 CL-Strong 配方）与 LambdaLoss（graded 标签），未微调基线使用模型名称 Qwen3-Embedding-0.6B。正文、附录及后续分析统一清理原 local-pool InfoNCE、binary LambdaLoss 和 binary mixture 的展示；主表、奖励消融和初始化表由原始结果重新生成。表格内容统一为 `\scriptsize`，当前 [PDF](iclr2027/build/main.pdf) 为 24 页，已完成编译及版面检查。下面的旧轮次说明保留作历史记录。

第 3 节复核：统一联合分布、求导条件、梯度方向和符号；3.3 从单个 query action 的几何直觉出发解释 conditional projection，再推导逐单元格 LOO 更新及其保证。18 类代数/梯度数值核对通过，详见 [写作复核记录](iclr2027/WRITING_REVIEW.md)；数值实现细节集中在附录。

2026-09-19 写作修订：`iclr2027/` 已完成 Introduction / Method 重写，补入 CP 正式命题与证明，并同步主配方 alignment 0.80、原 0.90 奖励对照、78 次核心消融、51 次 shortlist、CL-Strong、动态检索及 G2 九条结果。正文明确区分匹配估计器收益与较强监督配方比较；方法名称统一为 RLOO / RLOO+CP（历史实验 ID 不变），3.2 直接说明与 GRPO 的梯度对应条件及奖励标准化对无偏性的影响，G3 仍无结果。新增附录表直接从原始 CSV 复算，未启动训练或改写结果。当前 [PDF](iclr2027/build/main.pdf) 与 [写作复核记录](iclr2027/WRITING_REVIEW.md) 对应本轮修订。该目录沿用现有 Git ignore 设置，论文源码与 PDF 更新在本地。

2026-09-19：shortlist 四组实验已整理为 [G1 Shortlist 结果总结](G1_SHORTLIST_RESULTS.md)，覆盖 **17 个配方、51 次训练**。当前最佳已测 Uniform K15/T1 / alignment 0.70 为 **22.25 ± 0.27**，CL-Strong 为 **22.64 ± 0.20**；双方均为 113 steps。文档包含 K/T、hard 比例、候选来源、alignment 和领域分析。用 `python scripts/analyze_g1_shortlist_results.py` 复算；shortlist 结果现已纳入候选对照及附录，原匹配主配方保持不变。

2026-09-18 后续：新增 [ReasonEmbed 三方法实验](../docs/g2_reasonembed_cl.md)，D/E/W 各 CL、CL-Strong、binary CP，共九条 seed 42 训练，**每阶段 1 epoch**，最终 BRIGHT；新输出根目录隔离旧 1200-step 产物。配置和入口已准备，尚未启动训练。下文 E2Rank 的结果与呈现建议保持不变。

2026-09-18：E2Rank G2-R2 的 **D/E/W × CL / CL-Strong / RL 九条结果已同步**，见 [结果与呈现建议](G2_R2_RESULTS.md) 和 [原始总分表](_summary/g2_r2_mteb_v2/run_summary.csv)。三个分支的 RL Retrieval 均值均低于两种 CL；建议正文若讨论直接训练则聚焦 D，附录保留完整结果。脚本和配置已恢复原 D/E/W 版本，本次只更新文档，没有重跑或新增训练。G2 完整结果现已纳入附录，正文简述其检索负结果。

论文正文已于 2026-09-16 同步新协议与 G1-R2 结果：方法部分加入条件投影推导；实验部分报告 27 次训练、三 seed 主表和全参数配对梯度探针；附录包含全部九方法的逐领域结果。当时将 G2/G3 列为计划；本轮已同步 G2，仅 G3 继续作为未测扩展。当前 PDF：[`iclr2027/build/main.pdf`](iclr2027/build/main.pdf)。结果来源见 [G1_R2_RESULTS.md](G1_R2_RESULTS.md)，后续计划见 [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) 和 [G2_CL_R2_PLAN.md](G2_CL_R2_PLAN.md)。

最新结果文档已于 2026-09-17 扩展到 **117 次训练 + 2 条 E0**：包含全部 G1-R2 消融、动态检索和 DIVER 初始化实验，见 [G1_R2_RESULTS.md](G1_R2_RESULTS.md) 与 [完整附表](g1_r2_results/all_runs.md)。CP 的优势跨 alignment/G 保持；DIVER 上 RL 优于 LL/CL，但仍低于自身 E0。Qwen 消融与动态结果现已纳入论文；DIVER 继续遵循下述不纳入决定。用 `python scripts/analyze_g1_r2_results.py` 可重新生成结果支撑表与输入审计。

结果呈现决策（2026-09-17）：**主结果采用 graded CP/G64/alignment 0.80，配对 SF-0.80**，分别为 20.76 ± 0.20 与 18.81 ± 0.30；对 SF 的增益 +1.94，对 LL-Graded +1.39。0.80 根据本轮探索消融选择；机制消融和梯度探针保留 0.90 的匹配对照，原 run 名称与配置不变。本轮 LaTeX/PDF 已同步该决定，原 0.90 结果保留为奖励与机制对照。

论文纳入决策（2026-09-17）：**DIVER 实验仅作内部记录，不纳入论文正文、附录、实验计数或主张。** 数据匹配问题是内部优先假设，尚未验证；上述 117 次训练与两条 E0 是归档总数。

G1 已完成实验的综合分析见 [G1_RESULTS.md](G1_RESULTS.md)，包括全部 12 个 BRIGHT subset 的主表、探索与机制消融、领域分析和复现材料。[完整附表](g1_results/all_runs.md) 收录 45 次训练和 E0 的结果；这些材料独立于论文 LaTeX 正文。

W&B 完整日志的后续诊断见 [G1_GRADIENT_ANALYSIS.md](G1_GRADIENT_ANALYSIS.md)，涵盖更新规则的梯度尺度、训练 reward 与 BRIGHT 差异，以及实际 clipping 配置的核查。

Seed 重复后的最新机制讨论见 [rollout 方差复盘](../docs/rollout_variance_discussion.md)，区分已验证的 reward 分辨率、现有边际化实现、待检验的梯度信噪比假设和后续对照。

当前 rollout seed 对照之后的工作见[后续实验计划](../docs/rollout_variance_experiment_plan.md)：先固定状态比较 reward 和逐文档反事实 baseline，再按诊断证据选择完整训练重复。

```text
paper/
├── README.md                 # this guide
├── EXPERIMENT_PLAN.md        # experiment source of truth
├── ANALYSIS_PLAN.md          # supplementary analyses and pending experiments
├── analysis/                # checkpoint analysis configurations
├── analysis_results/        # analysis outputs, grouped by batch
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
