# ICLR 2027 最终复核（2026-09-26）

## 改稿后的收尾复核

后续修复：按作者授权，第 3.4 节 CMP 解释已改为“投影保留已检索文档的分数，但不能保证全索引 top-k 列表不变，因为其他文档可能越过检索阈值”。下方该项已解决；评测协议和复现配置两项状态不变。

本条以当前 27 页 PDF 和实际引用的源码为准，覆盖下方初审状态描述。本轮只读核对论文、配置及结果，在临时目录独立编译；未修改正文、训练/评测代码或实验数据。

- **新增表述问题：固定索引 CMP 说明。** `sections/method.tex:202–206` 的 “we can not only use the documents retrieved with the fixed index in training” 表达不准确；限制针对投影只覆盖已检索文档不足以保留全索引检索结果，不是说训练不能仅使用检索列表反馈。建议替换为：`We do not apply CMP in this setting: projecting onto the span of the policy mean and the retrieved document vectors preserves their scores, but does not in general preserve the full-index top-k list, since documents outside that span can cross the retrieval cutoff.` 本轮未直接修改。
- **既有协议差异仍存在。** BRIGHT 的 `excluded_ids` 未过滤，涉及此前核实的 AoPS、LeetCode、TheoremQA-Q，两套 query 均受该协议差异影响，分数影响未量化。附录 C.1.2 已披露，不等于实现已修复。重新对照 [BRIGHT 官方入口](https://github.com/xlang-ai/BRIGHT/blob/main/run.py) 与 [MTEB 1.38.32 加载器](https://github.com/embeddings-benchmark/mteb/blob/1.38.32/mteb/tasks/Retrieval/eng/BrightRetrieval.py) 确认；遵循作者暂不处理的决定，未启动重评。
- **复现配置待与作者确认对齐。** 论文按作者要求写 4B 全部 `2e-4`；`configs/experiments/iclr2027/suite_qwen3_embedding_4b_lora.yaml:20` 仍是默认 `1e-4`，InfoNCE/LambdaLoss 未覆盖，RELER-K0 单独覆盖为 `2e-4`。应以实际运行记录核定公开复现配置，不能仅靠旧 suite 推断集群实际学习率。本轮保留作者确认的正文，没有重写实验配置。

验证通过：

- 99 个 original-query 与 30 个 corrected GPT-4-query 结果的来源、完整性和聚合校验；新 GPT-4 共 360 条子集分数。
- 六个模型/query 组合的 RELER 平均值均高于 InfoNCE、LambdaLoss 和预训练模型；BGE-M3 对 InfoNCE 增益最大，而对预训练编码器的增益是 Q3E-4B 最大。Q3E-4B original 的 seed 42 配对差仍为 -0.19，因此跨设置平均胜出不应解释为每个 seed 胜出。
- RAG 五个配方 × 七个数据集的宏平均及附录差值重新计算吻合；九组完整梯度方差比为 291.1849–590.0307。
- 独立 `pdflatex → bibtex → pdflatex` 构建至引用稳定，输出文本与当前 PDF 逐页一致；共 27 页、结论在第 9 页、复现/AI 声明在第 10 页，无 Overfull、未定义引用或重复 label。核查第 7–9 页，表格与分析图完整；匿名标题正常、PDF Author 为空。

本轮对集群原始结果/训练日志无访问权限；数值结论来自已上传导出，未新做完整数学证明审阅或匿名代码全站审计。先前作者已取舍的整体配方比较、调参和耗时问题，不作为本轮新增错误。

---

审阅对象：当前 `iclr2027/main.tex`、其实际引用的正文/附录/表格、26 页 PDF、相关配置和结果导出。沿用此前作者对整体配方比较、调参、训练耗时和实验展示范围的取舍。初次复核未修改论文、实验配置或训练代码；随后按作者授权修正评测入口并完成 GPT-4 query 重评，现已将新结果同步论文。以下初审记录保留问题来源。

结论（用户追问 reasoning query 后修正）：发现一处阻断当前 GPT-reasoning 结果解释的评测协议错误，另有一处学习率事实错误及一处英文表达问题。此前将 reasoning 问题仅判为回退说明缺失的结论撤回。数学、数值聚合和编译检查通过，不能据此证明评测输入协议正确。

更新：GPT-4 query 重评已完成 30/30，新导出的 360 条子集分数及聚合统计已交叉核验通过，见[重评核验](BRIGHT_GPT4_REEVAL_REVIEW_2026-09-26.md)。论文主表、逐 seed 表和正文/附录说明现已同步。4B 学习率按作者最新确认统一写为 `2e-4`；结论英文表达与摘要拼写已修正。excluded_ids 独立问题仍按用户决定暂不处理，并已在附录说明实际协议。

## 1. Q3E-4B 学习率（作者已确认论文统一写 2e-4）

作者最新明确指示：4B 各训练方法的论文学习率均为 `2e-4`。正文、附录超参数表注和生成器中的主表方法标签已按该确认统一。0.6B、BGE-M3 仍为 `5e-6`。

初审曾依据仓库 suite 默认 `1e-4`、RELER-K0 覆盖 `2e-4`，建议区分 InfoNCE/LambdaLoss 与 RELER；此写作建议已由作者的最新确认覆盖。本次未修改 suite 或检查集群训练日志，不将论文文字更新记录为配置/运行层面的重新核验。

## 2. [P0] GPT-reasoning 路径读取了标注 reasoning，而非官方 GPT reasoning queries

- 位置：`iclr2027/sections/experiments.tex:43`；`iclr2027/sections/appendix_bright.tex:23` 的 evaluation details 尚无这部分定义。
- 修复前实现：`eval_mteb/run_mteb.py:285` 加载的是 `xlangai/BRIGHT` 的 **`examples` 配置**；第 266 行起使用其中的 `reasoning` 字段，构造 `query + "\n\n" + reasoning.strip()`；`None`、空串、`empty`、`n/a` 回退原 query。
- 固定版本：`src/fixed_corpus/router.py:18` 的 revision 为 `3066d29c9651a576c8aba4832d249807b181ecae`，与 `data/audit_reasonrank_bright/download_manifest.json` 一致。
- 独立读取该 revision 的 12 个本地 examples Parquet：共 1,384 个问题，其中 254 个回退，全部来自 LeetCode（142/142）和 Pony（112/112）。其他十个子集没有回退。主表这两列在两种 query setting 中相同，与实现吻合。

**官方协议不同。** [BRIGHT run.py](https://github.com/xlang-ai/BRIGHT/blob/main/run.py) 在 `--reasoning` 有值时加载 `{args.reasoning}_reason` 配置，直接把该配置中的 `query` 送入检索器；例如 GPT-4 使用 `gpt4_reason`。它不拼接 `examples.reasoning`。[数据说明](https://github.com/xlang-ai/BRIGHT/blob/main/Dataset_documentation.md) 将 reasoning 与相关文档标注联系起来，并要求标注者说明 query 与文档的相关性及其推理过程。

本地样例也证实问题：`theoremqa_theorems` 第一条 reasoning 明确讨论给定文档如何证明 Pigeonhole Principle，并解释题目答案。把这种相关性解释作为 query 输入可能引入评测标注信息，不能解释为模型仅基于原 query 生成的 GPT 推理。

影响：0.6B、4B、BGE-M3 的 reasoning 脚本共享该入口，因此当前主表整个 GPT-reasoning 区块都不能继续作为官方 GPT reasoning 协议结果使用。需要修正数据来源后复用 checkpoint 重新评测，或从投稿结果中移除该区块。该问题不涉及 original-query 分支，也不需要因此重新训练。该数据源问题现已修复为直接读取 `gpt4_reason.query`，新目录为 `query-gpt4-reasoning/`；旧分数保留；现已完成 30/30 重评并核验新导出，论文主表和逐 seed 表已替换旧分数，结果解读与输入说明已同步。

此前的 254 个回退计数仍然成立，但它是错误数据源路径的行为，不是官方 GPT reasoning 缺失范围。

## 新增 [P0] MTEB 未处理 BRIGHT excluded_ids（按用户要求暂不处理）

固定的 MTEB 1.38.32 加载器不读取 excluded_ids，检索器也未过滤。官方要求在 top-k 前排除这些文档。AoPS、LeetCode、TheoremQA-Q 的原 query 和 reasoning 评测均存在此协议偏差，实际分数变化尚未测量；其他九个子集为空或 N/A。此前关于 original-query 不受影响的判断仅适用于 reasoning 数据源问题。

用户决定本次不处理该独立问题：相关过滤代码和原 query 目录变更已撤回，启动脚本仅安排主表 30 个 reasoning setting。原 query 保持原状，不重跑，不自动复制或改写其旧分数。此问题仍然存在，不能记录为已解决。

## 3. [P3] 结论开头有不自然的英文表达

位置：`iclr2027/main.tex:138`。

已将 `by vMF exploring in the embedding space` 改为 `through vMF exploration in embedding space`，并修复摘要 `adreess` → `address`。不影响技术含义。

## 初审已完成的核验（更新后的编译结果见 README）

- 数学与实现：复核 product-rollout LOO、`1/G²` 缩放、CMP 条件化与逐项方差保证、pairwise CMP、vMF sampling，以及对应实现。未发现新的可复现错误；理论没有把逐项方差保证误写成总梯度方差保证。
- BRIGHT：以只读方式重新调用表格生成器的来源校验函数。0.6B 为 67 条运行、924 个子集分数；4B 为 22 条运行、384 个分数；BGE-M3 为 10 条运行、240 个分数。共 99 条运行、1,548 个子集分数；seed、query set、子集完整性及均值/SD 检查通过。最大聚合舍入差为 0.006 分。
- RAG：独立复算五个 recipe × 七个数据集的宏平均。混合 reward 的 EM/F1/Hit@10 为 32.9557/41.1000/56.3829，与主表 32.96/41.10/56.38 一致；正文和附录引用的当前差值匹配。
- 梯度 probe：九组完整梯度方差比复算得到 291.1849–590.0307 倍，支持文中 291–590 倍的表述。
- 编译：在临时目录完成独立 `pdflatex → bibtex → pdflatex` 多轮构建直至引用稳定。重编译 PDF 的版面文本与当前 PDF 一致，共 26 页，结论在第 9 页结束。无未定义 citation/reference、重复 label 或 Overfull；仅有既有非阻断 hyperref/Underfull 提示。
- 版面：检查全部 26 页缩略图，并放大检查正文主表、消融/分析图和 RAG/结论页；未发现裁切、重叠或漏图。
- PDF 匿名：第一页为 Anonymous authors，PDF Author metadata 为空，含 AI Use Statement。当前正文页数符合 [ICLR 2027 Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines) 的 9 页限制；引用、附录和 AI/reproducibility statements 不计入正文页限。
- 重点引用：抽查近邻工作对 action、retrieval 和 gradient estimator 的描述，未确认新的描述错误。GRIT 的 ICLR 2025 年份与[正式会议论文集](https://proceedings.iclr.cc/paper_files/paper/2025/hash/70cf215430492f7d34830a24e744b3f1-Abstract-Conference.html)一致，不能因 2024 workshop 版本而判为错误。DIVER 当前作者名单与 [arXiv v1](https://arxiv.org/abs/2508.07995v1)一致；[v5](https://arxiv.org/abs/2508.07995v5) 已调整名单。可选地给该条目补上 v1 链接以固定所引版本，或整体更新为 v5，未将其列为确定错署作者。
- 匿名代码链接：浏览器可打开 [RELER-F029 README](https://anonymous.4open.science/r/RELER-F029/README.md)，高层方法描述与正文一致。本轮只核查首页与 README，没有完成全部公开文件的逐项匿名审计。

静态公式/代码与导出数据复核不能替代重新运行训练或评测，本轮没有启动这些任务。
