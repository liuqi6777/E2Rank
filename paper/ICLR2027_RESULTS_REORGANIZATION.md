# ICLR 2027：BRIGHT 论文表格与结果解读

2026-09-26 GPT-4 query 重评已纳入论文：主表与逐 seed 表使用 `_summary/bright_gpt4_reasoning/` 的 30 个完整结果、360 条子集分数，原 query 数值保持不变。GPT-4 下 RELER 相对 InfoNCE 在 Q3E-0.6B、Q3E-4B、BGE-M3 上分别为 **+0.63 / +0.78 / +3.22**，九个配对 seed 均为正；三个模型在两种 query 设置下均领先。正文和附录明确直接读取官方 `gpt4_reason.query`，附录 C.1.5 补充逐 seed 主结果；旧拼接人工标注 reasoning 的分数不再用于论文。4B 各方法学习率按作者最新确认统一写为 `2e-4`。第 8 页分析段落和图注恢复原文，仅在第 7 页压缩结果解读以保持正文及结论 9 页。完整 PDF 27 页，主表、逐 seed 表和相关正文已核对，编译无未定义引用或 overfull。当前稿件：[PDF](iclr2027/build/main.pdf)；[结果与核验记录](BRIGHT_GPT4_REEVAL_REVIEW_2026-09-26.md)。本条覆盖下方历史 GPT reasoning 分数和输入方式说明。

2026-09-25 RAG 正文表样式：Table 3 九个结果列居中；按两位小数的显示值对每列最高分加粗，并列最高一起加粗。图注明示标记含义，附录逐数据集表未改。数据不变；PDF 24 页，正文和结论仍在前 9 页。

2026-09-25 RAG 协议表述：显示名由 `InfoNCE, answer-masked` 简化为 `InfoNCE`，内部 run ID 不变。正文交代评测每题一次 top-10 检索及冻结生成器；训练的 InfoNCE 则使用一次性挖掘的固定候选池。`wiki18_100w` 来自 FlashRAG 发布的 2018 Wikipedia passage 语料，本文用 Q3E-0.6B 重新建立并冻结索引；附录给出精确规模和数据集链接。结果值不变，完整 PDF 24 页，正文与结论仍在前 9 页。

2026-09-25 RAG 表格与分析更新：正文 Table 3 按 In-domain、Out-of-domain、Avg. 三组展示 EM、F1、Hit@10，共 9 个指标列；正文实验设置列明训练域 NQ/HotpotQA 与非训练域 PopQA/TriviaQA/2WikiMultihopQA/MuSiQue/Bamboogle，各组和总平均均按数据集等权。正文分析聚焦检索与答案质量的总体正向关系及不一致处、RLOO 配方相对 InfoNCE 的分组优势、混合 answer reward 的小幅互补作用；逐数据集解释放在附录 E.4，正文不以单个未展示的数据集解释汇总。附录保留逐数据集表与 query 数，删除与正文重复的分组汇总表。由 `build_rag_results.py` 生成正文和附录表；数据值不变。完整编译后 PDF 为 24 页，正文与结论仍在前 9 页。

2026-09-25 最新决定：candidate K 不再属于论文内容，全文删除该符号、实验表、引用与旧候选控制实现说明。附录投影空间采用实际使用的自有候选，移除额外固定候选 F 扩展及对应校准公式。当前补充结果只保留 E.1 完整消融、E.2 λ/alignment、E.3 BRIGHT 参考 baseline、E.4 RAG；历史 K 数据仍存档，不纳入稿件。当前 PDF 23 页，正文与结论仍为前 9 页。本条覆盖以下旧范围表。

表格排版统一：所有正文及附录表格的主体字号为 8pt、行距 9.5pt，SD 为一致的 6.5pt；新 subset 表与 reference baseline 表均不使用 `resizebox`，通过列距、简写和长行名换行控制宽度。

2026-09-25 附录范围收敛（覆盖下方历史安排）：

| 小节 | 当前内容 |
| --- | --- |
| E.1 Complete BRIGHT Ablation Results | Table 2 的 4 条 reward + 5 条 rollout/CMP/policy 行，完整 12 subset + Avg.±SD；保留路径限制 |
| E.2 Pairwise Weight and Alignment | Figure 2(c,d) 的 λ=0/0.5/1、ρ=0.6/0.7/0.8/0.9 完整 subset 结果 |
| E.3 Additional Candidate Count | 已有组合 reward K=0/1/7/15 与 graded-only K=0/1/3/7/15；均三 seed，不新增训练 |
| E.4 BRIGHT Reference Baselines | 从 Extended Related Work 移来的文献参考表；来源、协议差异说明保留 |
| E.5 Complete Fixed-Index QA Results | 保留现有 RAG 分组和逐数据集结果 |

旧 local-pool、旧 alignment/negative sampling、旧 4B 配方、额外 policy-role、初始化与 fixed-index BRIGHT 扩展不再进入论文；主表逐 seed 表和梯度汇总表也移出 PDF。同步删去旧训练设置、探索配置和已失效引用，保留主配方梯度测量协议。新增三张 subset 表直接由 `build_k0_results.py` 使用原始审计数据生成；历史文件仍保留作记录，不属于当前附录。完整 PDF 为 26 页，正文及结论仍在前 9 页。下方按历史记录保留，不作为当前展示要求。

2026-09-25 消融表更新：Table 2 为 4 行 reward design + 3 行 rollout/CMP controls。后一块固定 graded nDCG，列出 RLOO / paired、RLOO / product、RLOO+CMP / product，方便分别比较 rollout 和 CMP。Paired 数据来自已有 OwnCandidates-Paired 三 seed；正文明确单独 reward 路径的等价性尚未验证，避免把该比较写成严格隔离的 rollout 效应。

2026-09-25 最终紧凑布局：Figure 2 的左、中两列为梯度云和收窄的 reward 曲线，右列上下叠放 λ 与 alignment 敏感性，共四面板。两张敏感性图共用相同分数范围，填充点标明主设置；数据均直接读取已有三 seed 汇总。当前配方 alignment 表及对应解读移出附录，正文说明均值范围及 ρ=0.80 相对 0.70 的微小差异。历史 local-pool alignment 控制继续单独保留。正文和结论在前 9 页，整篇 29 页；本条覆盖下方旧布局说明。

2026-09-25 三面板更新：Training Analysis 的 Figure 2 改为梯度云 / graded training reward / pairwise 权重敏感性并排。权重图复用 grouped CSV 中 λ=0/0.5/1 的三 seed 均值和样本 SD，突出 λ=0.5；附录不再重复权重表及对应解读，仅保留正文图引用。完整 PDF 为 29 页，正文与结论仍在前 9 页。本条覆盖下方旧双面板安排。

2026-09-25 最新展示调整（覆盖下方历史图稿安排）：消融表分 reward design（4 行）和 CMP control（2 行），在两块中重复 graded-only 参考，以便就近比较；正文保留领域权衡，并补充简短参数敏感性总结。Training Analysis 采用梯度云与共享 graded reward 并排的单张双面板图。附录仅保留梯度协议和数值诊断，不再展示梯度、reward、difference 或 case analysis 图；删除原 Training Dynamics and Retrieval Behavior 小节，后续小节顺延。正文及结论在前 9 页，声明在第 10 页，完整 PDF 为 30 页。主结果分数和 Table 1 不变。

更新：2026-09-25。0.6B 的论文 suite 已覆盖全部 **64 个 original-query 结果（36 个新增训练、28 个导入项）**，以及主表 **10 个 GPT-reasoning-query 评测**。三个 CSV 中共有 **888 个子集分数**，每次评测均含 12 个 subset，训练方法均覆盖 seed 42/3407/2026，E0 评测一次。按当前表格计划，0.6B 尚缺训练 **0 次**，尚缺主表评测 **0 次**。4B 另有 **22 个 original-query 结果（21 个训练、E0 一次）**、**10 个 GPT-reasoning-query 评测**和 **384 个子集分数**，与 4B suite 完全匹配。BGE-M3 已补齐 **10 个 original-query 结果（9 次训练、E0 一次）**、**10 个 GPT-reasoning-query 评测**及 **240 个子集分数**；三个 backbone 的主表均已齐全。

主配方维持 `K=0`、graded nDCG@10 + pairwise `λ=0.5`、RLOO+CMP、`G=64`、`ρ=0.70`。4B 与 BGE-M3 主表的两种 query 均已补齐。按用户本轮决定，**联合检索训练与固定索引 RAG 均为正文贡献**；RAG 复用已有结果，不新增训练。

数据源：[逐运行汇总](_summary/k0_bright/k0_bright_per_run.csv)、[方法汇总](_summary/k0_bright/k0_bright_grouped.csv)、[子集明细](_summary/k0_bright/subset_summary.csv)。[生成器](iclr2027/build_k0_results.py)已接入论文 `build_results.py`；[审计](iclr2027/k0_results_audit.json)保存各表使用的运行、seed、分数和聚合结果。无需手填新表。4B 的[逐运行汇总](_summary/q4b_lora_bright/per_run_summary.csv)、[方法汇总](_summary/q4b_lora_bright/grouped_summary.csv)、[子集明细](_summary/q4b_lora_bright/subset_summary.csv)已接入同一生成器；见[4B 审计](iclr2027/q4b_results_audit.json)。BGE-M3 的[逐运行汇总](_summary/bge_m3_main/bge_m3_per_run.csv)、[方法汇总](_summary/bge_m3_main/bge_m3_grouped.csv)、[子集明细](_summary/bge_m3_main/subset_summary.csv)也已接入；见[BGE-M3 审计](iclr2027/bge_m3_results_audit.json)。

所有分数为 nDCG@10 × 100。Avg. 和样本 SD 保留 grouped CSV 的报告值，避免对已舍入的 per-run 分数再次聚合造成尾数漂移；12 个领域及 7/2/3 三类均值由四位小数的 subset 分数计算。12-subset Avg. 不等于三类均值的简单平均。各层 CSV 已交叉核验，舍入可能使最后一位相差 0.01，最终如有全精度数据可统一重导出。旧表中的 `22.72 ± 0.17` 等已按新导出更新。

## 当前展示结构（2026-09-25 重排）

实验按“推理密集型检索是否有效 → reward/CMP 各自提供什么 → 降噪机制是否成立 → 固定索引下是否改善问答”展开。实验章分为 §4.1 Reasoning-Intensive Retrieval 和 §4.2 Retrieval-Augmented Generation 两大块，各自包含设置与结果。§4.1 的 Experimental Setup 按 Baselines、Training setup、Evaluation 三段展开，后接 Main Results、Ablation Studies 和 Gradient Analysis；§4.2 包含 Experimental Setup 与 Results and Analysis。正文保留 **3 张表 + 1 张实验图**（方法示意图另计）。此前文档中的“正文四表”安排已被下表替代。

| 位置 | 证据与展示形式 | 负责回答的问题 |
| --- | --- | --- |
| 正文 §4.1，表 1 | 完整 BRIGHT 大表：先 Original query、后 GPT-reasoning query 两大块；每块内按 BGE-M3、Q3E-0.6B、Q3E-4B 分三组，每组原模型/LambdaLoss/InfoNCE/RELER 四行；省去模型标题行，以细横线分组；横向为 12 subset + Avg. | 总体效果、query 形式与模型条件的差异，以及收益和退步分别出现在哪些领域 |
| 正文 §4.1，表 2 | 合并 reward 与 CMP：4 条 CMP reward 行 + 1 条纯 graded RLOO 行；保留三类别及 Avg. | Pairwise 的 +0.66 和纯 graded 下 CMP 的 +1.87，并显示领域权衡 |
| 正文 §4.1，图 2（分析编号 F1） | 单面板梯度云，图与图注左右排列；图注给出九个 probe 的完整梯度方差下降范围 | 主配方组合 reward 下的完整梯度噪声是否下降 |
| 正文 §4.2，表 3 | 固定索引 QA：5 方法；In-domain、Out-of-domain、Avg. 各报 EM/F1/Hit@10，共 9 个指标列 | 下游问答是否改善、能否迁移、检索命中是否等于回答质量 |
| 附录 E.1 | 三个 backbone/两种 query 的逐 seed 表、Qwen3-0.6B 的 12 领域配对差值图；不重复正文领域表 | 正文主表的完整性、seed 差异和领域异质性 |
| 附录 E.4 | RAG 的 7 数据集逐项 EM/F1/Hit@10 表，标出 query 数；正文已有分组汇总 | 正文 RAG 聚合的核查和数据集差异 |
| 附录 E.3 | 完整梯度云/SNR 双面板、V、V×t、均值相容性和协议 | 梯度图的定量支撑 |
| 附录 E.4 | 组合 reward 的 K、ρ、λ sweep；纯 graded K sweep 单独标明 | 主配方敏感性，不混合变量或协议 |
| 附录 E.5 | 训练 reward 聚合/逐 seed 曲线、四领域单 seed 查询行为案例 | 训练诊断与解释性案例，不代替泛化证据 |
| 附录 E.6 | rollout/policy-role 对照，保留实现路径差异说明 | 额外组件证据及其解释限制 |
| 附录 E.7 | earlier local-pool、旧 alignment、负采样及 4B K7 控制 | 历史协议内的补充证据，与 K0 主配方明确区分 |
| 附录 E.8 | 初始化实验、固定索引 BRIGHT 扩展 | 主线以外的适用范围和负结果 |
| 生成产物/归档 | 紧凑 BRIGHT 总览、拆分的 Qwen3/BGE-M3 领域表、历史 local-pool 领域表及旧逐 seed 表只保留源表和生成器 | 避免在附录重复堆放细粒度历史结果，保证仍可追溯 |

2026-09-25 主表扩展：按用户反馈，将 12 领域结果移回正文表 1，六区块共 24 行，保留 8pt 字号；附录删去重复的两张领域表。正文仍为 9 页，声明位于第 10 页，整份 PDF 为 30 页。

2026-09-25 图稿精简：按用户反馈，正文删除 SNR 面板，采用紧凑梯度云和侧置短图注，降低版面占用；完整双面板和 SNR 定义移至附录 E.3。

### 展示与解读规则

- **模型命名**：论文基线行使用 Q3E-0.6B、Q3E-4B、BGE-M3；实验部分首次出现时定义 Qwen3-Embedding（Q3E）。论文不再展示 E0，梯度初始状态写作 Before fine-tuning。本文档下方的 E0 及源数据运行 ID 保留作内部对照。
- **正文用表做配方比较，用图解释机制。** 三个 backbone、两种 query 和全部 12 领域均留正文；逐 seed 明细与配对差值图置于附录。训练 reward 曲线不再占正文，因为它不能证明固定查询上的泛化或收敛提速。
- BRIGHT 主表在每个 backbone/query 区块内按显示精度标记各列排名：最高值加粗、第二高的不同值加下划线，并列同样标记，不表示显著性。主表按 query 形式分为两大块，各块内模型顺序为 BGE-M3、Q3E-0.6B、Q3E-4B；模型名由基线行标识，不再单列标题。RELER 行用图稿绿色的 10% 浅底色；方法顺序为原模型、LambdaLoss、InfoNCE、RELER；Avg. 左对齐，query 标题左对齐、斜体、不加粗；标题下和模型组间以横线分隔。RAG 单 seed 表不加标准差、不以极小均值差渲染稳定胜出。正文保留 E0，而不仅与下降的 InfoNCE 比较。
- **CMP 的两种证据分开解读**：纯 graded 完整训练的 +1.87；组合 reward 固定状态 probe 的降噪。不能把表中最后一行叫作“完整 RELER w/o CMP”，也不能用 probe 替代该训练对照。
- 三 seed、九个 probe、64 draws、12 领域和查询数有各自含义；不相互充当独立重复。图注写清楚误差带为 seed SD，还是固定状态 probe 点。
- 当前主要限制在正文紧邻结果交代：0.6B reasoning 的 −0.15、4B seed 差异正负混合、RAG 域外均值弱/下降、直接 answer reward 未体现明确额外收益。
- `build_results.py` 一次重建所有当前正文和附录表。BRIGHT 继续复用已有审计；新增 `build_rag_results.py` 核对 5 配方 × 7 数据集、query 数及分组聚合。所有本轮改动均是展示与写作，不改变训练或评测值。

**落实与验收**：LaTeX 和表格生成器已同步；当前 PDF 共 33 页，正文及结论在前 9 页，复现声明在第 9 页、伦理声明在第 10 页。已核对 BRIGHT 与 RAG 数值来源，完整编译后无未定义引用、重复 label 或 overfull；已逐页检查本轮修改的实验及结果附录。

### 正文 RAG 表及可支持的结论

| Method | In-domain EM | In-domain F1 | In-domain Hit@10 | Out-of-domain EM | Out-of-domain F1 | Out-of-domain Hit@10 | Avg. EM | Avg. F1 | Avg. Hit@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q3E-0.6B | 33.79 | 44.17 | 65.67 | 31.71 | 39.40 | 51.94 | 32.30 | 40.76 | 55.86 |
| InfoNCE | 33.33 | 43.58 | 63.29 | 30.17 | 37.70 | 46.52 | 31.08 | 39.38 | 51.31 |
| RLOO, graded nDCG | 34.86 | 45.29 | 67.24 | 31.51 | 39.22 | 51.97 | 32.47 | 40.95 | 56.33 |
| RLOO, answer F1 | 34.57 | 45.02 | 66.52 | 31.24 | 39.02 | 52.43 | 32.19 | 40.73 | 56.46 |
| RLOO, nDCG + F1 | 34.72 | 45.04 | 66.72 | 31.95 | 39.42 | 51.93 | 32.74 | 41.03 | 56.15 |

数据源：[RAG 逐数据集汇总](_summary/g3_r2_qa/per_dataset_results.csv)；[正文表体](iclr2027/tables/rag_overview.tex)。所有训练行都是 anchor 0.5、seed 42。Graded reward 的域内 F1 相对 E0 为 +1.12，全部七域为 +0.19，其余五域为 −0.18；混合 reward 相对 graded 的全部七域差仅 +0.08，去掉 Bamboogle 后不保持。将它作为“固定索引下可利用任务反馈适配 query encoder”的正文应用证据，同时明确泛化和 answer reward 的限制。

## BRIGHT 数据明细：正文表 1 完整展示 12 个领域

正文使用 [bright_main_full.tex](iclr2027/tables/bright_main_full.tex) 的完整大表，按三个 backbone × 两种 query 分六个区块，每块只包含 E0、InfoNCE、LambdaLoss 和 RELER。全部 12 个 subset 和 Avg. 在正文同表展示；附录 E.1 保留逐 seed 结果与领域差值图。原[紧凑总览](iclr2027/tables/bright_overview.tex)继续生成供核查，不再进入 PDF。下方先列 Qwen3 明细，随后汇总 BGE-M3。两种 query 复用同一 checkpoint；GPT-reasoning 在有 reasoning 时追加到 query，否则回退原 query。

| Method | Bio. | Earth. | Econ. | Psy. | Rob. | Stack. | Sus. | Leet. | Pony | AoPS | TheoQ. | TheoT. | Avg. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Qwen3-Embedding-0.6B; Original query** | | | | | | | | | | | | | |
| E0 | 13.4 | 27.8 | 18.0 | 16.6 | 12.0 | 12.8 | 13.2 | 14.2 | 0.7 | 3.3 | 17.2 | 32.1 | 15.10 |
| InfoNCE | 34.3 | 34.1 | 26.2 | 33.4 | 21.0 | 28.8 | 27.4 | 10.2 | 1.2 | 3.1 | 19.7 | 32.3 | 22.64 ± 0.20 |
| LambdaLoss | 24.3 | 25.5 | 23.9 | 29.0 | 17.6 | 22.9 | 21.9 | 12.4 | 1.4 | 3.3 | 21.0 | 29.0 | 19.36 ± 0.28 |
| RELER | 45.9 | 37.3 | 28.6 | 34.0 | 18.4 | 29.3 | 25.2 | 9.4 | 2.2 | 2.3 | 19.0 | 28.9 | 23.38 ± 0.17 |
| **Qwen3-Embedding-0.6B; GPT-reasoning query** | | | | | | | | | | | | | |
| E0 | 26.5 | 42.0 | 33.7 | 38.7 | 19.5 | 15.1 | 28.6 | 14.2 | 0.7 | 7.0 | 25.9 | 72.1 | 27.00 |
| InfoNCE | 59.8 | 49.4 | 49.1 | 56.1 | 36.1 | 34.0 | 49.0 | 10.2 | 1.2 | 8.8 | 27.4 | 67.4 | 37.38 ± 0.08 |
| LambdaLoss | 42.0 | 39.4 | 40.0 | 49.5 | 28.0 | 26.8 | 43.3 | 12.4 | 1.4 | 7.0 | 27.4 | 62.1 | 31.61 ± 0.43 |
| RELER | 64.2 | 53.6 | 47.0 | 54.9 | 33.6 | 33.8 | 46.0 | 9.4 | 2.2 | 6.7 | 26.0 | 69.3 | 37.23 ± 0.38 |
| **Qwen3-Embedding-4B; Original query** | | | | | | | | | | | | | |
| E0 | 17.2 | 34.6 | 16.4 | 23.0 | 13.0 | 15.9 | 17.1 | 21.2 | 2.0 | 3.7 | 19.6 | 40.9 | 18.70 |
| InfoNCE | 48.5 | 47.0 | 32.0 | 42.3 | 26.7 | 34.7 | 35.9 | 15.2 | 6.6 | 3.8 | 23.5 | 44.7 | 30.08 ± 0.16 |
| LambdaLoss | 32.3 | 35.5 | 27.5 | 36.0 | 19.9 | 27.0 | 30.8 | 18.5 | 3.6 | 3.8 | 23.3 | 42.0 | 25.01 ± 0.37 |
| RELER | 48.9 | 50.2 | 32.9 | 44.3 | 25.8 | 33.4 | 33.6 | 13.8 | 11.3 | 3.6 | 20.7 | 46.9 | 30.46 ± 0.55 |
| **Qwen3-Embedding-4B; GPT-reasoning query** | | | | | | | | | | | | | |
| E0 | 33.0 | 47.9 | 35.2 | 46.2 | 21.0 | 18.9 | 31.3 | 21.2 | 2.0 | 6.4 | 27.0 | 79.2 | 30.78 |
| InfoNCE | 64.3 | 56.2 | 49.4 | 54.6 | 37.0 | 40.3 | 50.6 | 15.2 | 6.6 | 9.7 | 29.3 | 65.8 | 39.91 ± 0.56 |
| LambdaLoss | 45.3 | 46.1 | 43.6 | 50.1 | 29.8 | 31.3 | 49.3 | 18.5 | 3.6 | 6.8 | 27.7 | 68.4 | 35.03 ± 0.75 |
| RELER | 63.6 | 58.7 | 49.8 | 57.4 | 37.4 | 38.0 | 50.8 | 13.8 | 11.3 | 7.3 | 25.7 | 66.3 | 40.00 ± 0.37 |

[论文表体](iclr2027/tables/bright_main_k0.tex)。0.6B original query 下，RELER 相对 InfoNCE 为 **+0.74**，三个配对 seed 为 **+0.77/+1.01/+0.42**（42/3407/2026）。相对 LambdaLoss 为 **+4.02**。GPT-reasoning 下 RELER 比 InfoNCE 低 **0.15**，配对差为 **−0.15/+0.30/−0.61**；不能写成两种 query 下都优于 InfoNCE，也不据此声称显著差异。4B 的 K0 主配方沿用 LR `2e-4`，其他训练与评测约定见现有 4B 文档。

4B original query 下，RELER 相对 InfoNCE 为 **+0.38**，配对 seed 差为 **−0.19/+0.89/+0.44**；GPT-reasoning 下为 **+0.09**，配对差为 **+0.22/−0.18/+0.24**。两种 query 的三 seed 均非一致获益，不声称显著或稳定优势。相对 LambdaLoss 的均值差分别为 **+5.45/+4.97**。4B RELER 使用 LR `2e-4`，InfoNCE/LambdaLoss 使用 `1e-4`，因此是完整训练配方比较；0.6B 全参数微调与 4B LoRA 也不构成受控的 scaling 实验。

### BGE-M3：新增的正文主表条件

| Method | Original query | GPT-reasoning query |
| --- | ---: | ---: |
| E0 | 10.66 | 21.41 |
| InfoNCE | 12.63 ± 0.11 | 22.33 ± 0.32 |
| LambdaLoss | 11.40 ± 0.08 | 18.74 ± 0.20 |
| RELER | **14.49 ± 0.04** | **26.07 ± 0.05** |
| RELER − InfoNCE | **+1.86** | **+3.74** |

RELER 相对 InfoNCE 的配对 seed 差（42/3407/2026）分别为 original **+1.81/+1.95/+1.83**、GPT-reasoning **+3.72/+4.06/+3.45**。这为另一个 encoder family 提供了三 seed 同向的总体收益证据；不意味着所有领域均受益。两种 query 的 Coding 类均比 InfoNCE 低 **3.49**，完整 12 subset 结果见[正文表体](iclr2027/tables/bright_main_full.tex)。

BGE-M3 与 0.6B 使用相同 prepared ReasonRank、113 steps、global batch 128、学习率 `5e-6` 及全参数微调，但沿用各模型的编码协议：BGE-M3 为 CLS pooling、右侧 padding、无末尾附加 token、无 query/document 前缀。协议与模型固定版本已写入论文附录；详见[实验说明](../docs/iclr2027_bge_m3_main.md)。正文保留四方法的相同结构，BGE-M3 的两种 query 各占一个区块，与两个 Qwen3 backbone 的完整领域结果合为表 1。

## 正文表 2 的 reward 区块

固定 K=0、CMP、G=64、ρ=0.70、113 steps。只有第一行加入 pairwise；binary 两行使用原始正例标注。

| Reward | StackExchange | Coding | Theorem-based | Avg. |
| --- | ---: | ---: | ---: | ---: |
| Graded nDCG@10 + pairwise (λ=0.5) | 31.25 | 5.79 | 16.73 | 23.38 ± 0.17 |
| Graded nDCG@10 | 29.80 | 6.21 | 17.24 | 22.72 ± 0.18 |
| Binary nDCG@10 | 21.74 | 4.49 | 16.07 | 17.45 ± 0.65 |
| Binary MRR@10 | 23.42 | 4.69 | 17.83 | 18.90 ± 0.43 |

[论文表体](iclr2027/tables/reward_ablation_k0.tex)。Pairwise 带来 **+0.66** 的 Avg.，逐 seed 均为正；收益集中于 StackExchange，Coding 和 Theorem-based 下降。Graded nDCG 明显高于两个 binary reward 的观测均值，但不把该比较解释成所有任务上的普遍结论。

## 附录 E.4：纯 graded reward 下的额外候选数

固定 reward 与 CMP，只改变加入的跨 query 负例数量 K，自有候选始终保留。

| Additional negatives K | StackExchange | Coding | Theorem-based | Avg. |
| --- | ---: | ---: | ---: | ---: |
| 0 | 29.80 | 6.21 | 17.24 | 22.72 ± 0.18 |
| 1 | 29.74 | 6.01 | 17.34 | 22.69 ± 0.11 |
| 3 | 29.50 | 6.17 | 17.48 | 22.61 ± 0.07 |
| 7 | 29.10 | 6.29 | 17.82 | 22.48 ± 0.09 |
| 15 | 28.72 | 6.35 | 17.72 | 22.24 ± 0.27 |

[论文表体](iclr2027/tables/candidate_count_graded.tex)。K=0 的总平均最高，与 K=1 差距很小；组合 reward 的对应检验单独放附录 E.4，不混合两个变量。

## CMP、rollout 与 policy role：五行进入正文表 2，12 subset 明细置于附录 E.1

所有行均为纯 graded reward、无 pairwise、仅自有候选、ρ=0.70、G=64。后三行均无 CMP，以 RLOO product 双侧 policy 为参照；单侧 policy 仍训练共享 encoder。

| Configuration | StackExchange | Coding | Theorem-based | Avg. |
| --- | ---: | ---: | ---: | ---: |
| RLOO+CMP, product, query + document | 29.80 | 6.21 | 17.24 | 22.72 ± 0.18 |
| RLOO (w/o CMP), product, query + document | 26.22 | 5.79 | 18.36 | 20.85 ± 0.68 |
| paired rollout | 22.55 | 6.68 | 17.78 | 18.71 ± 0.53 |
| query policy only | 17.92 | 8.64 | 15.85 | 15.85 ± 0.26 |
| document policy only | 22.10 | 5.98 | 18.25 | 18.45 ± 1.56 |

[论文表体](iclr2027/tables/rl_components_k0.tex)。CMP 相对 RLOO product 提升 **1.87**，三个 seed 均为正。RLOO product 的均值高于 paired/query-only/doc-only；doc-only 的 seed SD 达 **1.56**。Product 使用 G² 个 reward cell，paired 使用 G 个，不能称为等计算量比较。

前两行共用 K0 shortlist 路径；后三行使用 own-candidate 非 shortlist 路径。结果 CSV 不能证明两路径的 reward/梯度等价。已有固定分数 reward 核对，仍需保留同批次 RLOO product 梯度核对的记录后，才能将后三行解释为纯 rollout/policy-role 效应。该表不测量组合 reward 下的 CMP 收益。

## 附录 E.4：组合 reward 下的 K sweep

固定 graded nDCG + pairwise λ=0.5，其余设置与主配方一致。

| Additional negatives K | StackExchange | Coding | Theorem-based | Avg. |
| --- | ---: | ---: | ---: | ---: |
| 0 | 31.25 | 5.79 | 16.73 | 23.38 ± 0.17 |
| 1 | 30.95 | 5.82 | 16.90 | 23.25 ± 0.05 |
| 7 | 30.20 | 6.18 | 17.57 | 23.04 ± 0.20 |
| 15 | 29.78 | 6.12 | 17.75 | 22.83 ± 0.30 |

[论文表体](iclr2027/tables/pairwise_k_sweep.tex)。K=0 在每个 seed 均高于 K=1/7/15，支持当前 own-candidate 配方。K 增加也改变 pairwise 中可用的负例对，属于本候选数干预的一部分。

## 附录 E.4：alignment sweep

固定 K=0、λ=0.5，只改变 ρ。

| Alignment ρ | StackExchange | Coding | Theorem-based | Avg. |
| --- | ---: | ---: | ---: | ---: |
| 0.60 | 31.47 | 5.43 | 16.39 | 23.36 ± 0.25 |
| 0.70 | 31.25 | 5.79 | 16.73 | 23.38 ± 0.17 |
| 0.80 | 31.25 | 6.03 | 16.92 | 23.46 ± 0.25 |
| 0.90 | 30.48 | 6.16 | 17.45 | 23.17 ± 0.19 |

[论文表体](iclr2027/tables/alignment_k0_pairwise.tex)。均值范围为 23.17–23.46；ρ=0.80 比约定主配方高 **0.08**，配对提升并非三个 seed 一致。不改主配方，也不把这一小差距当作可靠的最优点。

## 附录 E.4：pairwise 权重 sweep

固定 K=0、ρ=0.70，只改变 λ。

| Pairwise weight λ | StackExchange | Coding | Theorem-based | Avg. |
| --- | ---: | ---: | ---: | ---: |
| 0 | 29.80 | 6.21 | 17.24 | 22.72 ± 0.18 |
| 0.5 | 31.25 | 5.79 | 16.73 | 23.38 ± 0.17 |
| 1.0 | 30.02 | 5.48 | 16.57 | 22.57 ± 0.18 |

[论文表体](iclr2027/tables/pairwise_lambda_k0.tex)。λ=0.5 最好；λ=1.0 比 λ=0.5 低 **0.81**，且低于纯 graded 的观测均值。因此不采用“权重越大越好”的表述。历史 K=15、λ=0.25 不放入这张固定 K 的表。

## 附录：4B 配方与历史组件消融

| Method | K | LR | StackExchange | Coding | Theorem-based | Avg. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E0 | -- | -- | 19.58 | 11.59 | 21.40 | 18.70 |
| InfoNCE | -- | 10^{-4} | 38.16 | 10.88 | 24.02 | 30.08 ± 0.16 |
| LambdaLoss | -- | 10^{-4} | 29.86 | 11.08 | 23.01 | 25.01 ± 0.37 |
| RELER (main) | 0 | 2×10^{-4} | 38.45 | 12.54 | 23.74 | 30.46 ± 0.55 |
| RELER (prior recipe) | 7 | 10^{-4} | 36.20 | 10.66 | 24.43 | 29.00 ± 0.15 |
| w/o pairwise, CMP | 7 | 10^{-4} | 34.60 | 10.55 | 24.18 | 27.99 ± 0.24 |
| w/o pairwise, RLOO | 7 | 10^{-4} | 28.45 | 10.49 | 23.20 | 24.14 ± 0.24 |
| w/o score calibration | 7 | 10^{-4} | 34.73 | 10.12 | 23.75 | 27.89 ± 0.44 |
| RELER (LR calibration) | 7 | 2×10^{-4} | 37.34 | 11.47 | 24.59 | 29.84 ± 0.07 |

[论文表体](iclr2027/tables/recipes_4b.tex)。上半部分是主表 original-query 结果；下半部分为旧 K=7 配方及其消融。K7、LR=1e-4 条件下，pairwise 为 **+1.01**，纯 graded 下 CMP 为 **+3.85**，移除 frozen-score calibration 为 **−1.11**。三项差异在所有配对 seed 中同向，但不能当作 K0 主配方的组件效应。

LR calibration 行保留先前独立导出的结果 29.84 ± 0.07；新 q4b 目录未重复包含它。相对该行，K0/LR=2e-4 的均值高 0.62；相对旧 K7/LR=1e-4 行高 1.46，后者同时改变 K 和学习率，不是单因素消融。

## 剩余工作与分析图

0.6B 本轮计划内训练及主表评测已齐。下列工作不等于新增完整训练：

- 核对并保存 own-candidate 与 K0 shortlist 路径的同批次 reward/梯度等价记录，以及 binary MRR 的 cutoff/正例身份记录。
- 组合 reward 下的匹配全参数梯度探针已完成并保留为正文 F1；尚缺的是相同组合 reward 下无 CMP 的完整训练对照，二者不能混淆。
- [reward 曲线目录](analysis_results/reward_curves_k0/)的六条 history、manifest、CSV、PDF/PNG 已核对并接入附录 E.5；不再列为待审核接入或正文必备。
- 12 领域三 seed 差值图已进入附录 E.1；四领域单 seed 查询案例已进入 E.5。仍未完成的是 12 领域跨 seed 的查询级扩展，不能从 subset 均分倒推。
- GPT-reasoning 的缺失/回退/截断率及训练成本仍需对应日志或数据统计，结果 CSV 本身不提供这些证据。

详细图稿方案见[分析计划](ANALYSIS_PLAN.md)。4B 的主表及当前 suite 结果已齐，BGE-M3 的九次训练与两套主表评测已齐；RAG 复用此前已完成的评测，不纳入上述 BRIGHT suite 计数。论文与 table drafts 继续按既有 Git ignore 规则管理。
