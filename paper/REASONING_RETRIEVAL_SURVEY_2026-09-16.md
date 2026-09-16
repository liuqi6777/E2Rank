# Reasoning retrieval 调研（2026-09-16）

**后续决策更新**：下文为最初调研建议，执行范围已由用户收窄。G1 新增 DIVER-0.6B 与
ReasonEmbed-4B 作为 E0，各自仅比较 CL / graded LL / graded RL；不新增 benchmark，也不铺开
公开模型横向 baseline。ReasonEmbed-4B 权重已通过 Hugging Face API 确认并固定 revision，
此前“未确认 4B checkpoint”的记录已过时。当前执行合同见 [核心实验说明](../docs/g1_reasoning_e0.md)。

检索范围：2025–2026 年 reasoning-intensive text retrieval，补充早期评测工作。结论针对本项目的 embedding RL / InfoNCE / RankNet / LambdaLoss 对照，而非泛指所有 agentic RAG。来源优先为论文、作者仓库和数据卡。公开分数是作者报告或指定复评结果，本次没有运行模型评测，也没有完成新数据去重审计。

## 结论与建议

1. **不能只保留 ReasonIR 作为领域 baseline。** 最小新增外部评测集建议为 BM25、ReasonIR-8B、DIVER-Retriever-0.6B、ReasonEmbed-Qwen3-8B，以及 NVIDIA llama-nv-embed-reasoning-3b。DIVER-0.6B 尤其适合当前 Qwen3-Embedding-0.6B 的规模对比。
2. **最值得复现的训练目标是 ReasonEmbed 的 RI-InfoNCE 和 SyCL 的 Wasserstein list-wise loss。** 前者是 reasoning 数据上的难度自适应监督学习，后者直接处理多级 relevance。它们比再加一个大型生成式 reranker 更能检验本项目 RL 的增益来源。需要用相同 backbone、候选、数据暴露和最终 checkpoint 协议复现，不能用公开 checkpoint 分数代替目标函数对照。
3. **最适合扩数据的是 ReasonEmbed，其次是 ReasonIR 与 RaDeR；ReasonAug 适合低成本的数学/定理方向验证。** ReasonEmbed 的多正例和多级标注尤其贴合当前实现，但公开文件是否保留完整 graded scores 仍须核查。
4. **第二个 reasoning 评测优先 R2MED。** BRIGHT-Pro 适合补充多方面证据覆盖分析，但它是 BRIGHT 的扩展，不能当作完全独立的第二个泛化集。
5. **保持当前已冻结的 G1 批次。** 新 baseline 和数据另开后续实验，不把新数据的收益混入现有 4,963 条 ReasonRank 上的目标函数结论。数据增量解决 seed 不稳定只是待检验假设。

以上优先级为结合项目预算与研究问题作出的判断。

## 论文与 baseline 候选

| 工作 / 时间 | 核心变化 | 资源与对本项目的用途 |
|---|---|---|
| [ReasonIR](https://arxiv.org/abs/2504.20595)，2025-04，COLM 2025 | 用 reasoning query 与看似相关但无帮助的 hard negative 训练 dense retriever；推理可放在训练数据中，部署仍为向量检索 | [代码 / 模型 / 数据入口](https://github.com/facebookresearch/ReasonIR)。必须保留为领域参照；论文摘要的 29.9 与加 reranker 后的 36.9 不能混成一个单阶段成绩 |
| [RaDeR](https://aclanthology.org/2025.emnlp-main.1011/)，2025-05，EMNLP 2025 | 从数学解题的 retrieval-augmented MCTS 路径提取 query–theorem 监督；正确答案验证提供文档效用的代理信号 | [代码](https://github.com/Debrup-61/RaDeR)。建议作为第二批外部 baseline，尤其关注数学领域迁移与数据生成，而不是直接搬用完整混合检索分数 |
| [DIVER](https://arxiv.org/abs/2508.07995)，2025-08 | 文档处理、迭代 query expansion、synthetic-data dense retriever、helpfulness reranker 组成多阶段系统 | [仓库](https://github.com/AQ-MedAI/Diver) 已发布 0.6B / 1.7B / 4B / 4B-1020 retriever。0.6B 作者报告 BRIGHT 25.2，4B-1020 报告 31.9；推荐先跑 0.6B。45.8 等完整 pipeline 分数不属于同预算 dense 对照 |
| [BGE-Reasoner](https://github.com/FlagOpen/FlagEmbedding/blob/master/research/BGE_Reasoner/README.md)，2025-08 起 | rewriter / embedder / reranker 的端到端系统 | 发布了 embedder、部分训练数据和 top-2000 检索结果；部分组件仍为 TBA。适合作为外部强模型或固定候选来源。历史 0923 embedder 报告 original-query 37.1，完整系统 45.2；版本要固定 |
| [ReasonEmbed](https://aclanthology.org/2026.acl-long.54/)，2025-10 预印本，ACL 2026 | ReMixer：生成 query 后排除生成源文档，再挖候选并进行推理相关性标注；Redapter：用 reasoning intensity 加权 InfoNCE | 原始 query 下 Qwen3-4B / 8B 报告 37.1 / 38.1。建议加入领域强 baseline，并在相同小模型上复现 RI-InfoNCE；[代码入口](https://github.com/VectorSpaceLab/agentic-search)，[数据](https://huggingface.co/datasets/hanhainebula/reason-embed-data) |
| [llama-nv-embed-reasoning-3b](https://huggingface.co/nvidia/llama-nv-embed-reasoning-3b)，2026，模型发布 | reasoning-focused 合成语料与公开 reasoning 训练数据混合 | 官方卡报告 BRIGHT short-doc 12 域平均 38.3，并提供 MTEB BRIGHT(v1.1) 评测方法。建议补一个更新且规模较小的强外部模型；这是模型资源，不应杜撰为单独论文 |
| [SyCL / Beyond Contrastive Learning](https://aclanthology.org/2025.findings-emnlp.1245/)，2025-03，EMNLP 2025 Findings | 为真实 query 合成多级相关文档，使用 Wasserstein distance 做 list-wise dense retriever 训练 | [代码](https://github.com/BatsResearch/sycl) 与 [数据](https://huggingface.co/datasets/BatsResearch/sycl) 已公开。不是 BRIGHT 专用方法，但与 graded nDCG、RankNet、LambdaLoss 的研究问题直接相邻，推荐为训练目标补充对照 |
| [Revela](https://arxiv.org/abs/2506.16552)，2025-06，ICLR 2026 | 不依赖标注 query–doc，通过跨文档上下文和 retriever 加权 attention 的语言模型训练学习检索器 | [代码 / 3B 模型入口](https://github.com/TRUMANCFY/Revela)。适合“无标注也能学 reasoning retrieval”的相关工作；涉及额外 LM 训练机制，不是当前最小对照必选项 |
| [RTriever / BRIGHT-Pro](https://arxiv.org/abs/2605.04018)，2026-05，ACL 2026 | 用互补 positives 与 positive-conditioned hard negatives 学习多方面证据检索；扩展 BRIGHT gold 与评测 | [代码、数据和 RTriever-4B](https://github.com/yale-nlp/Bright-Pro)。适合后续多正例/证据覆盖方向。论文描述 RTriever-Synth，但本次确认的仓库入口主要是 benchmark 和模型；训练集公开位置与完整性待确认 |
| [RGLT](https://arxiv.org/abs/2608.14107)，2026-08 预印本 | silent tokens 形成非自回归 latent reasoning，CoT 重建和阶段检索效用监督连接中间状态与检索增益 | 非常新的 latent reasoning 相关工作。论文 BRIGHT 34.2 已在榜单出现，但本次未确认可下载代码/权重；先引用和跟踪，不列为已可执行 baseline |

ReasonEmbed 的公开权重已确认：[hanhainebula/reason-embed-qwen3-8b-0928](https://huggingface.co/hanhainebula/reason-embed-qwen3-8b-0928)，卡片明确采用 RI-InfoNCE。论文 4B 变体的 checkpoint 本次未确认，先不承诺可执行。

BGE-Reasoner 与 ReasonEmbed 作者及资源有明显关联；不要把不同日期的 BGE embedder 与 ReasonEmbed 当作两个完全独立训练配方，复现前需核对 model card、权重 revision 与训练说明。

## 显式 reasoning、RL 与 agentic 方向的边界

- [O1 Embedder](https://github.com/RuiranYan/o1embedder)：先生成 thoughts，再 embedding；通过 retrieval committee 筛选 thoughts，behavior cloning 与 contrastive learning 联合训练。公开 [Ruiran/msmarco_thought](https://huggingface.co/datasets/Ruiran/msmarco_thought)。适合研究“显式 CoT 的额外 query-time compute 是否必要”，当前优先列 related work。
- [LREM](https://arxiv.org/abs/2510.14321)：电商 query–CoT–item 的 SFT / InfoNCE 起步，再用 RL 改进 reasoning trajectories。需要引用以界定“retrieval utility 驱动 RL”的既有工作；它的 RL action 是 reasoning 文本，不能视为当前 vMF embedding action 的等价实现。本次未确认可直接用于公开 BRIGHT 复现的完整资源。
- [Embed-RL](https://arxiv.org/abs/2602.13823)：多模态生成式 embedding 中，embedder-guided RL 训练 reasoner 产生与检索目标相关的 T-CoT。与奖励设计有关，但不需要为当前纯文本 G1 加入多模态评测。
- [Rank-R1](https://arxiv.org/abs/2503.06034)：GRPO 训练 setwise reasoning reranker；[公开代码](https://github.com/ielab/llm-rankers/tree/main/Rank-R1)、[模型](https://huggingface.co/ielabgroup/Rank-R1-14B-v0.1)。当前 ReasonRank 也是 reranker 来源；这些方法适合独立的 reranking/system 表，不能回答 embedding objective 的公平比较。
- [AgentIR](https://arxiv.org/abs/2603.04384)，2026-03：把 agent 当前 reasoning trace 与 search query 联合编码，DR-Synth 由 QA 数据合成 Deep Research retriever 监督。[代码](https://github.com/texttron/AgentIR)、[项目页含数据入口](https://texttron.github.io/AgentIR/)。若恢复 G3，这是比只讨论 Search-R1 更直接的 retriever baseline；输入和评价是 agent 场景，不能把 BrowseComp-Plus agent accuracy 当作 BRIGHT nDCG。
- [Your Dense Retriever is Secretly an Expeditious Reasoner](https://arxiv.org/abs/2510.21727)：AdaQR 在 fast dense reasoning 与 deep LLM reasoning 间路由，适合效率和 embedding-space reasoning 的 related work；不是当前最小训练矩阵必选项。

调研未建立“此前没有人对 continuous embedding 做 RL”的证据。创新性需要比较 action space、reward、梯度估计器与部署计算，而不能仅写“首次 RL reasoning retrieval”。

## 可用训练数据

| 数据 | 已核实的公开规模 / 形态 | 推荐用途与需要核查的事项 |
|---|---|---|
| [ReasonEmbed v0928](https://huggingface.co/datasets/hanhainebula/reason-embed-data/blob/main/README.md) | 81,659 query，12 个领域 split；论文平均约 12 positives / 86 negatives，标注 1–5，3–5 为正、1–2 为负 | **首选 reasoning 扩数据。** 相对当前 4,963 query 约 16.5 倍。保留多正例；先核查 JSONL 是否有完整 scores / IDs。不能把 pos/neg 内部排列直接当 teacher ranking，亦不能把训练 difficulty score 当 relevance gain |
| [ReasonIR](https://huggingface.co/datasets/reasonir/reasonir-data) | 两个 configs：`vl`、`hq`；当前页面合计 345,491 rows，HQ 约 101k。query/pos/neg 含 instruction–content 包装 | VL 提供完整正负文本；HQ 的 positive 为 ID，按官方脚本与 BRIGHT `documents` join。适合扩大 binary reasoning 监督；多个数据来源和两个 configs 分开追踪。公开 synthetic 集规模不等于论文全部混合训练样本数 |
| [RaDeR datasets collection](https://huggingface.co/collections/Raderspace/rader-training-datasets) | `MATH_qCoT_LLMquery_lexicalquery` 约 35.9k；加入 question-as-query 的版本约 43.2k；MATH+NuminaMath allquerytypes 约 122k | 数学推理定向补充。多个版本有包含关系，不应简单相加；同一 math problem 的不同 query types 必须成组划分。公开 page rows 与独立问题数需分开统计 |
| [ReasonAug](https://huggingface.co/datasets/siyue/ReasonAug) | 约 10.9k train rows；[作者说明](https://github.com/siyue-zhang/DiffEmbed) 用 GPT-4o-mini 按数学、物理、金融、代码定理生成 question–solution pairs | 适合快速验证。相同定理下的 pairs 互为正例，不同定理作负例；转换需要 theorem/group 标识，避免 in-batch false negatives。这种概念标签是近似检索监督，不等于证明不同定理永远无帮助 |
| [BGE-Reasoner partial data](https://huggingface.co/datasets/hanhainebula/bge-reasoner-data/tree/main/bge-reasoner-data-0904) | 官方仓库公开的是部分训练集，未核实独立 query 数 | 可作历史版本/数据消融，不优先于更清楚的 ReasonEmbed v0928；先核查交集再混合 |
| [SyCL](https://huggingface.co/datasets/BatsResearch/sycl) | 按 MS MARCO query 合成 graduated relevance ranking contexts，官方训练脚本区分 binary / multilevel / synth+real | **首选 graded-label 机制验证。** 与 reasoning-specific 扩数据分开，可检验 RL 是否优于可微 list-wise 监督，不把它描述成天然需要数学推理的集 |
| RTriever-Synth / AgentIR DR-Synth | 论文或项目描述了互补证据 / agent query 合成方法；DR-Synth 有项目页数据入口 | 后续 G3 与 aspect coverage 有价值；RTriever-Synth 的完整可下载训练包仍待确认，不作为本次已就绪数据承诺 |

## 可用评测数据

| 数据 | 评测价值 | 优先级 |
|---|---|---|
| [R2MED](https://github.com/R2MED/R2MED) | 876 医学 reasoning queries，公开 corpus 与 qrels；与 BRIGHT 形成领域迁移检验 | **优先补充外部 reasoning test**。其 Biology 等来源仍应与现有训练集做 query/source 重叠检查，不自动宣称完全无交集 |
| [RAR-b](https://github.com/gowitheflow-1998/RAR-b) | 把数学、代码、常识等 reasoning tasks 转换为检索；full corpus 和 multiple-choice 两个设置，已接 MTEB | 第二优先；采用 full setting 并按任务报告，避免把 answer retrieval 与真实 evidence retrieval 解释成同一种能力 |
| [BRIGHT-Pro](https://github.com/yale-nlp/Bright-Pro) | 专家补充 multi-aspect gold evidence；提供 static / agentic-search 协议，数据为 `yale-nlp/Bright-Pro` | 适合 analysis / 后续 set-level reward。先采用无需付费 judge 的 static 协议；不是独立于 BRIGHT 的 held-out benchmark |
| [BrowseComp-Plus](https://arxiv.org/abs/2508.06600) | Deep Research 的固定语料检索 / agent 评测，AgentIR 与 RGLT 等新工作使用 | G3 再考虑，优先区分 retrieval recall 与完整 agent accuracy；本次没有核实接入成本和数据版本 |

多模态 MM-BRIGHT、MR²-Bench 等属于相邻发展方向；当前项目为纯文本 encoder，不建议此时扩展主实验范围。

## 最小可执行后续方案

1. **外部模型评测**：复用当前逐领域 BRIGHT corpus、excluded IDs 和 original-query 协议，跑 BM25、ReasonIR-8B、DIVER-0.6B、ReasonEmbed-8B、NVIDIA reasoning-3B；RaDeR 与额外 BGE 版本放第二批。保存各自原生 instruction / pooling / truncation 与 immutable revision，不能强行共用 Qwen 的末尾读出协议。
2. **训练目标对照**：在同一个 0.6B backbone 与冻结数据上保留当前 CL / RankNet / LambdaLoss / RL；若拿得到所需 RI 标注再加 RI-InfoNCE，graded-label 实验再加 SyCL Wasserstein。这两个扩展均应独立于当前夜跑队列。
3. **reasoning 数据规模实验**：优先 ReasonEmbed，但先按原始问题/生成源做重叠审计，核查 scores 和 IDs；冻结候选后，CL 与 RL 使用完全相同子集，再比较小数据与扩数据、多 seed 的均值和标准差。不要只有 RL 用更多/更难数据。
4. **泛化验证**：相同最终 checkpoint 补 R2MED，必要时 RAR-b；BRIGHT-Pro 作为 evidence coverage 分析。

训练从 BRIGHT corpus 合成的数据，应标明“target-corpus-aware，测试 query held out”。使用公开 corpus 本身不等于测试题泄漏；去重 query 也不能把这个设置改称跨语料泛化。当前 ReasonRank 的清理事实以项目审计和冻结 manifest 为准，不由本调研替代。

注意公开分数差异：ReasonIR 原论文摘要报告 29.9；NVIDIA 卡的 original-query baseline 表列 24.4。需要核对 query 扩展、corpus 版本、excluded IDs、长度与模型模板，并统一复评；不能选择最高分拼表。完整系统榜单亦混有 rewrite、hybrid、reranking 和 test-time scaling，适合另表注明配置。

本次产物为调研与实验建议；没有修改现有训练计划、配置、数据或启动 GPU/API 任务。
