# Paper workspace

2026-09-26 GPT-4 query 重评已纳入论文：主表与逐 seed 表使用 `_summary/bright_gpt4_reasoning/` 的 30 个完整结果、360 条子集分数，原 query 数值保持不变。GPT-4 下 RELER 相对 InfoNCE 在 Q3E-0.6B、Q3E-4B、BGE-M3 上分别为 **+0.63 / +0.78 / +3.22**，九个配对 seed 均为正；三个模型在两种 query 设置下均领先。正文和附录明确直接读取官方 `gpt4_reason.query`，附录 C.1.5 补充逐 seed 主结果；旧拼接人工标注 reasoning 的分数不再用于论文。4B 各方法学习率按作者最新确认统一写为 `2e-4`。第 8 页分析段落和图注恢复原文，仅在第 7 页压缩结果解读以保持正文及结论 9 页。完整 PDF 27 页，主表、逐 seed 表和相关正文已核对，编译无未定义引用或 overfull。当前稿件：[PDF](iclr2027/build/main.pdf)；[结果与核验记录](BRIGHT_GPT4_REEVAL_REVIEW_2026-09-26.md)。本条覆盖下方历史 GPT reasoning 分数和输入方式说明。

2026-09-26 附录结构重组（以本条为准）：按 A 理论推导、B 方法实现、C BRIGHT 实验、D 固定索引 RAG、E 通用 LLM 初始化实验、F 扩展相关工作组织。BRIGHT 的设置、文献参考、完整消融、参数扫描和梯度测量集中在 C；RAG 的语料、训练、query anchor、生成协议及逐数据集结果集中在 D。Exploration 配置并入 C.1 超参数，梯度协议与定量结果统一为 C.5；旧的通用 Experimental Details / Supplementary Experimental Results 分拆结构不再使用。源码由 sections/appendix.tex 统一引入六个主题文件，保留所有原有 label，并同步正文和复现声明的引用。数学、实验数值及表格数据不变；编译无未定义引用或 overfull，已检查附录版面。正文仍为 9 页，完整 PDF 为 26 页。

2026-09-26 实验语言润色：重写正文实验的结果叙述，并同步精简附录 QA、初始化及完整梯度分析中的数字串和重复总结。正文保留支持比较的关键差值，将其余分数留给表格；按作者偏好，各结果段以简短的结论句开头，再接数字与解释，Main Results 分为跨模型的 baseline 总体比较与不同模型的收益差异两段，减少模板化的推断句。表格、图、实验设置和主要结论不变；核对正文差值并完成编译与视觉检查，正文及结论仍为 9 页，完整 PDF 为 25 页。

2026-09-26 按作者取舍补充附录：摘要将 InfoNCE 比较限定为 original queries；paired/单侧采样按相同 graded nDCG 协议描述。附录 D.5 补全 RAG 的 3/2/1/0 passage grades、InfoNCE answer masking、训练及生成器配置；C.3 补全 CMP 的数值投影、精度和梯度累积实现；D.8 新增九组完整梯度 probe 定量表、SNR 定义、配对诊断和梯度云选轴规则。正文修正 probe 为三个状态 × 三个 microbatch，并引用附录定量表。新表由 `iclr2027/build_gradient_results.py` 从已有 `probes.csv` 生成，已接入常规表格构建。保留整体配方比较、现有调参说明、训练耗时表述与 Avg. 主线，不新增实验或修改结果。当前 PDF 25 页，正文与结论仍在前 9 页。

2026-09-25 RAG 训练数据与篇幅：正文明确 NQ 和 HotpotQA 是域内训练集，分别使用 DPR 人工正例 passage 和 HotpotQA 标注的 supporting facts，并映射到冻结语料作为二元相关标签；附录说明证据映射不完整的 query 会被排除。Conclusion 改为概括方法、主要趋势和适用边界的总结，不再逐项复述数值；完整编译后正文与结论在第 9 页结束，PDF 共 24 页，无 overfull 或未定义引用。

2026-09-25 RAG 正文表样式：Table 3 的九个数值列均居中，每列按显示的两位小数加粗最高值（并列同样加粗）；表格生成器和图注已同步。完整编译、页 9 视觉检查通过，正文和结论仍在前 9 页，PDF 共 24 页。

2026-09-25 RAG 协议说明：正文不再使用 `answer-masked` 展示名，表格统一标作 InfoNCE。评测每个问题只对冻结索引检索一次 top-10，并交给冻结生成器作答；InfoNCE 训练使用一次性挖掘的固定候选池。正文说明 `wiki18_100w` 语料来自 FlashRAG 的 2018 Wikipedia passage 发布集，附录补充 21,015,324 条、100-word passage 和数据集链接；索引由本文用 Q3E-0.6B 建立并冻结。实验数值不变，PDF 24 页，正文和结论在前 9 页。

2026-09-25 RAG 表格与分析：正文 Table 3 按 In-domain、Out-of-domain、Avg. 三组展示 EM/F1/Hit@10（9 个指标列），实验设置逐一列出 2 个训练数据集和 5 个非训练数据集；附录 E.4 只保留逐数据集及 query 数，去掉重复的分组汇总。结果分析说明检索与答案质量总体同向但不完全一致、RLOO 配方在各分组均优于 InfoNCE、混合 answer reward 的小幅互补作用；逐数据集差异放在附录。结果值不变，完整编译后 PDF 为 24 页，正文和结论仍在前 9 页。

2026-09-25 最终范围调整：全文移除候选数量 K 的实验、符号与旧控制说明。删除 Additional Candidate Count 小节、候选构造控制及 frozen-score 校准公式；理论投影空间去掉额外固定候选扩展，设置直接描述 own-query candidates。补充实验现为 E.1 完整消融、E.2 λ/alignment、E.3 BRIGHT 参考 baseline、E.4 RAG。已清理正文引用、图注、超参数表与限制中的对应内容，历史数据和生成文件仅作归档。PDF 23 页，正文与结论仍在前 9 页；8pt 表格字号保持统一，编译与视觉检查通过。本条覆盖下方 candidate K 展示安排。

2026-09-25 表格字号统一：正文与附录所有表格内容统一 8pt / 9.5pt 行距，标准差沿用统一的 6.5pt 辅助字号。移除新 subset 表和参考 baseline 表的整体缩放，后者用 Q3E 简写与单个长方法名换行适配宽度；表格不再因填满版心而出现不同字号。编译及修改页视觉检查通过，无 overfull 或未定义引用。

2026-09-25 BRIGHT 附录重整（以本条为准）：补充实验依次为 Table 2 全部消融行的 12-subset 表、Figure 2(c,d) 的 λ/alignment 完整结果、candidate K 完整结果、文献参考 baseline、RAG 完整结果。K 复用已有组合 reward 的 0/1/7/15 与 graded-only 的 0/1/3/7/15，每项三 seed，不新增训练。参考 baseline 已从 Extended Related Work 移到补充实验；旧 local-pool/4B K7、额外 policy-role、初始化和固定索引 BRIGHT 扩展，以及重复主表 seed 表、梯度汇总表均不再收入论文，对应旧设置和引用同步清理。梯度测量协议保留。新表由 `iclr2027/build_k0_results.py` 从已审计 subset CSV 生成，Avg. 保留 grouped mean/SD；原始记录和未引用的历史表保留用于追溯。当前 PDF 26 页，正文与结论仍为前 9 页。

2026-09-25 Table 2 补充 paired rollout：下半块改为 Rollout and CMP controls，依次展示 RLOO / paired（18.71±0.53）、RLOO / product（20.85±0.68）、RLOO+CMP / product（22.72±0.18）。数据直接复用现有三 seed 汇总；图注不再把所有行写为 product，正文保留 paired reward 路径等价性未验证的限制。

2026-09-25 alignment 入正文：Figure 2 采用三列四面板，左侧梯度、中间收窄的 reward，最右侧上下排列 λ / alignment。右侧两图统一分数刻度，实心点标主设置。alignment 复用当前 K0/Pairwise050 的四组汇总；正文说明 ρ=0.80 比主设置仅高 0.08，配对 seed 差异正负混合。附录删除重复的当前配方 alignment 表，历史协议控制保留。正文及结论仍为 9 页，完整 PDF 为 29 页。

2026-09-25 分析图补充：正文 Figure 2 现为梯度云、共享 graded reward、pairwise 权重敏感性三图并排。新增面板直接读取已有三 seed 汇总，展示 λ=0/0.5/1 的均值与样本 SD，突出 λ=0.5；附录删除重复的 λ 表及解读，改为引用正文图。正文和结论仍在前 9 页，整篇为 29 页。

2026-09-25 消融与分析精简：正文消融表按 reward design / CMP control 分块，共享 graded-only 参考行在两块中重复，保留领域分数；正文补充简短的参数敏感性总结。Training Analysis 用左右双面板展示梯度云与共享 graded reward（三 seed 均值与 SD）。附录移除梯度/reward 图、difference 图与 case analysis 图及案例解读，保留梯度协议、数值表和参数扫描。新图由 `scripts/plot_training_analysis.py` 从既有 CSV 重画，不新增训练或改动分数。当前 PDF 为 30 页，正文及结论在前 9 页；本条覆盖下方历史展示安排。

2026-09-25 主表样式：Q3E 的全名和缩写定义移至实验部分首次出现处，引言仅保留 Qwen3-Embedding 全名。Table 1 按 Original query、GPT-reasoning query 分为两大块，各块内按 BGE-M3、Q3E-0.6B、Q3E-4B 排列；模型组不再单列标题，以原模型基线行及细横线区分。各模型组顺序为原模型、LambdaLoss、InfoNCE、RELER；RELER 行使用图稿绿色 `#1BAF7A` 的 10% 浅底色，最优加粗、第二高的不同显示值加下划线（均包含并列），区块及标题下增加横线，Avg. 左对齐。表格生成器已同步，数据审计、编译与实验页视觉检查通过，正文和结论仍在前 9 页。

2026-09-25 模型命名：论文正文、表格、图注和分析图不再使用 `E0`。基线行改用 `Q3E-0.6B`、`Q3E-4B`、`BGE-M3`；引言定义 Qwen3-Embedding（Q3E），梯度初始状态写作 Before fine-tuning。表格生成器和两张分析图的绘图脚本已同步，内部运行 ID 保留用于追溯。数据审计、编译及改动图表检查通过；当前完整 PDF 为 33 页，正文和结论仍在前 9 页。

2026-09-25 实验叙述调整：实验章分为 §4.1 **Reasoning-Intensive Retrieval** 和 §4.2 **Retrieval-Augmented Generation**。Retrieval 内含 Experimental Setup（Baselines、Training setup、Evaluation 三段）、Main Results、Ablation Studies、Gradient Analysis；RAG 内含 Experimental Setup 和 Results and Analysis。设置精简去重，数据清洗明细由附录承接。表序仍为 BRIGHT、消融、RAG。编译、交叉引用及实验页视觉检查通过；正文和结论仍在前 9 页，复现声明位于第 9 页，伦理声明位于第 10 页，完整 PDF 为 32 页。

2026-09-25 实验展示重排：按用户决定，**BRIGHT 联合检索训练与固定索引 RAG 均进入正文**。正文实验采用一张完整 BRIGHT 大表、两张紧凑表（RAG 总览、reward/CMP 关键消融）和一张梯度图；逐 seed/QA 数据、领域差值图、训练曲线和参数扫描进入按问题组织的附录，历史 local-pool/K7 对照另组。当前分配与展示理由见[结果整理文档](ICLR2027_RESULTS_REORGANIZATION.md#当前展示结构2026-09-25-重排)。按用户反馈，表 1 现将 Qwen3-0.6B/Qwen3-4B/BGE-M3 × original/GPT reasoning 分为六区块，完整展示 12 领域及 Avg.，共 24 行；附录不再重复领域表。BGE-M3 数据来自[新汇总](_summary/bge_m3_main/)。本轮只复用已有结果重建展示，未新增训练或改变分数。新增文献参考表后 PDF 共 32 页，正文及结论在前 9 页，复现/伦理声明在第 10 页；数据审计、完整编译与改动页的视觉检查已完成。下方按日期保留的记录描述当时状态，以本条和当前结果文档为准。

2026-09-24：当前实验表以[结果整理文档](ICLR2027_RESULTS_REORGANIZATION.md)为准；梯度、训练 reward 与检索行为的论文图以[分析图计划](ANALYSIS_PLAN.md)为准。[独立 suite](../docs/iclr2027_final_k0_suite.md)把 36 次新训练与 28 个已完成结果导入项汇入新输出根目录。代表 checkpoint 配置及 reward 曲线清单已改为拟定的 0.6B `K=0 + pairwise` 主配方，相关新训练和日志导出仍待完成。下方较早日期的 `K=7` 决策均为历史记录。

2026-09-20 写作与排版优化：摘要和引言从 embedding model 在语义检索中的地位切入，聚焦针对 BRIGHT 的优化及相同训练数据下的比较；ReasonRank 细节留在实验设置。方法正文精简，辅助推导移至附录，保留 CP 的直觉、核心公式与保证。表格保持 `\scriptsize`，恢复默认行列间距，仅三张超宽领域表局部缩小列间距；主表的子集列居中、Avg. 左对齐，两层表头加粗。Algorithm 1 改为展示奖励矩阵、逐 pair LOO/CP 和参数更新的数学伪代码。当前 [PDF](iclr2027/build/main.pdf) 为 24 页，正文、复现与伦理说明在前 8 页内；已完成编译及版面检查。

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
