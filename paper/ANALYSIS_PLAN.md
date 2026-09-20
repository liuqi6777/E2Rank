# 论文补充分析与待跑实验

更新：2026-09-20。本轮已改论文、重算表格并准备分析配置；没有加载训练模型、启动 GPU 任务或生成新的实验分数。RAG 仍是后续要完成的实验，本轮只暂缓其结果写作。

补充分析分为三组：**embedding 几何与检索行为、梯度诊断、RL 训练 reward 曲线**。Reward 曲线优先复用已有训练日志，可先于 GPU 分析整理；RAG 另作为后续必须完成的端到端实验。

## 1. 已固定的主方法与证据范围

论文主方法命名为 **RELER**：从 Qwen3-Embedding-0.6B 开始，113 steps，joint full fine-tuning，G=64，alignment=0.70，每步一次均匀采样最多 7 个额外跨 query 负例，graded nDCG@10 + 0.5 × binary pairwise reward，逐项 LOO/CP。主表使用 42、3407、2026 三个 seed，不单独挑 seed 3407 当主结果。

用户指定的运行是 `G1-R2-RL-GradedNDCG64-CP-Align070-ShortlistUniform-K7-T1-Pairwise050-Seed3407-s3407`。同配方三 seed 为 23.26 / 23.02 / 22.85，均值 23.04、样本 SD 0.21；InfoNCE 为 22.64、SD 0.20。新表由 [build_results.py](iclr2027/build_results.py) 生成，内部运行映射和复算值保存在 [results_audit.json](iclr2027/results_audit.json)，这些工程标识不进入论文正文。

论文中的 InfoNCE 对应历史 `G1-R2-CL-Strong` 运行，LambdaLoss 对应 `G1-R2-LL-Graded`；配置保留实际 checkpoint 路径。未微调基线使用模型名称 Qwen3-Embedding-0.6B。后续图表统一沿用这套名称。

主表的领域顺序与分组沿用 [BRIGHT 论文 Table 2](https://arxiv.org/pdf/2407.12883)：StackExchange、Coding、Theorem-based。领域列只给三 seed 平均，总平均列给 mean ± sample SD。旧梯度探针使用本地候选和单一 reward，不能作为新组合奖励已经验证的证据。

## 2. 第一批建议：五个最终 checkpoint，四个领域

建议先跑配置 [analysis/bright_representatives.json](analysis/bright_representatives.json)。模型别名与用途：

| 配置别名 | 论文名称 / checkpoint | 要回答的问题 |
|---|---|---|
| E0 | 原始 Qwen3-Embedding-0.6B | 适配前的排名与几何 |
| InfoNCE | InfoNCE，seed 3407，step 113 | 与跨卡候选上的对比学习配方比较 |
| LambdaLoss | LambdaLoss，seed 3407，step 113 | 与 metric-aware surrogate 比较 |
| RELER | 主方法，seed 3407，step 113 | 完整方法的变化 |
| Listwise | K7、alignment 0.70、无 pairwise，seed 3407，step 113 | pairwise 项改变了什么 |

统一 seed 3407 是固定的案例分析选择，符合用户指定 checkpoint，且不是主方法分数最高的 seed。分析图不能把它称作三 seed 结果。优先领域固定为 Biology、Earth Science、Robotics、TheoremQA theorems，同时覆盖主方法相对 InfoNCE 的优势与劣势；不得只展示 Biology。

配置中的相对 checkpoint 路径依据现有运行 ID 构造，需在训练机上核对实际保存位置。若 checkpoint 根目录不同，只修改 `models.*.path`；E0 使用固定模型 revision。最终模型通常位于 run 根目录，不假定一定存在 `checkpoint-113` 子目录。保持原始 query、8192 截断上限、FP16 backbone 和 FP32 pooling/scoring。显存不足可减小 encoding batch size，不改变长度或精度。

配置保存在 `paper/analysis/`，本批输出统一写入 `paper/analysis_results/iclr2027_representatives_s3407/`，与本计划同处 `paper/` 下；输出结构见 [结果目录说明](analysis_results/README.md)。这些分析文件不再位于被忽略的论文源码目录中。后续批次使用 `paper/analysis_results/` 下的独立子目录，避免覆盖本批结果。

从仓库根目录执行：

```bash
ANALYSIS_CONFIG=paper/analysis/bright_representatives.json

# 先跑一个模型、一个领域，核对无扰动检索是否重现原有评测。
python scripts/analyze_embeddings.py encode --config "$ANALYSIS_CONFIG" --models RELER --subsets biology
python scripts/analyze_embeddings.py retrieval --config "$ANALYSIS_CONFIG" --models RELER --subsets biology

# 核对后补全模型与四个领域；输入相同的已有缓存会复用。
python scripts/analyze_embeddings.py encode --config "$ANALYSIS_CONFIG"
python scripts/analyze_embeddings.py retrieval --config "$ANALYSIS_CONFIG"
python scripts/analyze_embeddings.py geometry --config "$ANALYSIS_CONFIG"
python scripts/analyze_embeddings.py report --config "$ANALYSIS_CONFIG"

# 扰动属于第二阶段，可在前面结果确认后再运行。
python scripts/analyze_embeddings.py perturb --config "$ANALYSIS_CONFIG"
python scripts/analyze_embeddings.py report --config "$ANALYSIS_CONFIG"
```

验证门槛：无扰动 nDCG@10 先与该 checkpoint 的现有逐领域结果对照。脚本分数为 0–1，主表为 0–100；先统一尺度。如果超出已有两位小数舍入能解释的差异，先查数据版本、instruction、模型路径、tokenization、precision 和 tie handling，再解释分析图。记录实际运行时间和峰值显存，以首个完整领域估计其余工作量。

该入口读取固定 checkpoint，不更新参数。现有离线分析实现见 [embedding_analysis.md](../docs/embedding_analysis.md)。本次仅验证配置格式、命令入口和模型映射，真实 checkpoint 的编码与评测仍由训练机运行。

## 3. 建议形成的图，以及需要回传的数据

| 图 | 横轴 / 纵轴或布局 | 数据与比较 | 支持的解释 |
|---|---|---|---|
| 领域收益图 | 12 个领域；nDCG@10 差值 | 现有三 seed 表，RELER−InfoNCE 与 RELER−Listwise；均值加三个配对 seed 点 | 收益分布和跨 seed 一致性；这张图不需新评测 |
| 正例排名迁移图 | E0 的最佳正例 rank 桶（1、2–10、11–100、101+）；适配后的 rank 桶或 nDCG 改变量 | 固定 query ID，对比 E0、InfoNCE、RELER、Listwise | 改善来自越过 top-10 边界还是已命中查询内的重排 |
| 固定文档对的间隔图 | 横轴 InfoNCE 的归一化边界距离；纵轴 RELER 的距离；按领域分面 | 所有模型 top-100 文档并集加全部正例，E0 固定同一正例/未标注文档对 | 相同语义对的排序边界如何改变；避免每个模型自行挑对造成选择偏差 |
| 扰动曲线 | alignment；采样后的 nDCG@10 及相对各自干净基线的下降 | 同一批 query ID，64 次 query-side vMF draws，五模型 | 检索对局部 query 扰动的敏感性，不等同训练时 joint-policy reward |
| 奖励分项诊断（第二批） | checkpoint step；reward 波动、有效 pair 比例、梯度夹角/范数 | 固定 batch 和动作，分别取得 listwise 与 pairwise 项 | 区分奖励退化、分项尺度和方向变化；不能仅凭 surrogate loss 推断梯度冲突 |
| RL 训练 reward 曲线 | optimizer step；共享 graded nDCG 与主方法奖励分项 | RELER、Listwise，各三个 seed；详见第 5 节 | 展示训练目标的优化过程、平台与跨 seed 波动，结合最终 BRIGHT 结果解读 |

第一批不以二维 t-SNE/UMAP 散点作为主要几何证据；检索排名、相同文档对的 margin 和扰动表现更直接。若画分布图，保留逐 query 点/分位数，按领域分面，避免大领域主导汇总。按 query bootstrap 只表示该 seed 下的查询不确定性，不表示训练 seed 不确定性。

默认扰动配置在共享候选中重排，图标题必须注明 `candidate reranking`，不能写成全库鲁棒性。如果这一结果有价值，再用独立输出目录、`perturb.scope="corpus"` 重搜全库；保持模型、query ID、alignment 和 draws 不变。评测不使用训练时 frozen-score calibration。先对齐 `alignment=1` 的干净基线。

请回传配置和四领域的 `retrieval/`、`geometry/`、`perturb/`、`reports/` 目录（若暂未跑 perturb，可先回传前三者中的已完成部分），以及时间/显存记录；无需传完整 embedding 数组或模型权重。最终可视化使用矢量 PDF，正文保留最能回答问题的 2–3 张图，其余放附录。

## 4. 第二批：少量训练过程 checkpoint 与梯度诊断

先看第一批结果，再考虑 InfoNCE 与 RELER 的 **step 25、50、113**，仍固定 seed 3407 和同一批四领域 query。现有最终模型复用，优先只补两个中间 checkpoint 的检索；如果中间权重未保存，先报告缺失，不自动重训。若需要正式训练轨迹，使用原有 callbacks 的同协议评测结果；不得将不同 tokenization 的历史结果拼成一条曲线。

现有 `scripts/diagnose_rollout_gradients.py` 支持固定权重的本地候选、单一奖励 SF/CP 探针，可在新的代表 checkpoint 上复核旧机制。**它目前不能直接给出主方法的完整 SF/CP 对照**：现有 reward 捕获路径没有覆盖 shortlist 的所有奖励表，而 pairwise 项始终走逐 pair CP；只切换 `gradient_estimator` 会得到混合估计器，不是纯 SF。

若要增加“主方法也有同样降噪”的论文结论，先补齐诊断实现，再运行以下预先指定的矩阵：

- 固定状态：E0、RELER step 50、RELER step 113；固定 seed 3407 的训练样本选择。
- 先复用原三个微批次位置 0/100/200 并记录实际 source/query ID，不将其称为覆盖所有训练域的抽样。
- 每批 64 次独立重复；每次将相同负例 ID、query/document actions、listwise reward 和逐 pair reward 提供给两估计器。先固定负例集合测条件方差，再另测重新采样负例的总波动。
- 无 optimizer step、关闭 dropout；计算完整参数梯度的方差、配对均值差/MC RMS、分项范数和夹角、时间及显存。逐 pair 的未投影对照需对相同 reward 使用正确的 score-function 项，不能比较不同奖励。
- 输出区分 listwise-only、pairwise-only 与完整和，显式保留两分项协方差。图画 CP/SF 方差比和 variance × time，不能将其解释为端到端训练加速。

这部分是实施计划，不是已经可直接运行的新探针。本轮保留旧探针的原始适用范围，论文中没有填入预期结果。

## 5. 第三组：代表性 RL 训练 reward 曲线

这组分析复用已完成训练的逐步日志，优先整理为论文中的一张双面板图。代表配方固定如下，均为 Qwen3-Embedding-0.6B、CP/G64、alignment 0.70、Uniform K7/T1、113 optimizer steps；每个配方保留 42、3407、2026 三个 seed，共六条运行。选择依据是主方法及直接的奖励消融，不根据曲线是否平滑、是否单调或最终分数挑 seed。

| 代表配方 | 训练目标 | 曲线用途 |
|---|---|---|
| RELER | graded nDCG + 0.50 × binary pairwise | 主方法的学习过程与奖励分项变化 |
| Listwise | graded nDCG | 比较加入 pairwise 项后的训练轨迹 |

具体运行名、seed 和字段映射见 [reward_curve_runs.csv](analysis/reward_curve_runs.csv)，六条运行均已与 [现有 BRIGHT 总分表](_summary/g1_r2_bright/run_summary.csv) 对上。本地 `paper/g1_gradient_analysis/raw/` 目前只有旧 G1 日志，尚未取得本组六条运行的逐步 history，不能用旧曲线替代。本清单中的 `run` 是项目运行名，取得日志时仍须核实唯一的 W&B run ID 或对应的本地日志路径。

### 图的布局与指标

- **面板 A：共同指标。** 横轴为 optimizer step 1–113，纵轴为训练候选池上采样动作的 graded nDCG@10（0–1），比较 Listwise 和 RELER。两者的该指标均来自 `train/reward/mean`，保持相同候选协议与探索设置。
- **面板 B：RELER 分项。** 同时画 graded 项、`0.50 × train/reward/pairwise/mean` 和完整的 `train/reward/combined_mean`，展示两部分如何构成训练目标。核对逐步 `combined = graded + 0.50 × pairwise`，容许日志舍入误差。两个配方自己的完整目标定义不同，不以总 reward 的绝对高低给它们排序。
- **跨 seed 与平滑。** 默认画原始逐 step 的三 seed 均值和样本 SD，附录保留单 seed 曲线。若可读性确需平滑，统一使用每个 seed 的 5-step trailing mean，再计算跨 seed 均值与 SD，并在图注声明；原始数据和无平滑图必须保留。`reward/std` 是训练 rollout 内波动，不能作为跨 seed 阴影带。

图注明确这是训练 batch/训练候选上的 reward，随 step 数据和模型均在变化；训练曲线与最终 BRIGHT 泛化结果配合解读，不据平滑程度推断参数梯度方差，也不代替第 4 节的固定状态探针。InfoNCE 与 LambdaLoss 的训练 loss 不纳入同一 reward 纵轴。

### 日志、输出与完成条件

优先读取已有 `trainer_state.json` 的 `log_history` 或完整 W&B history。W&B 导出可参考 [已有读取脚本](g1_gradient_analysis/fetch_wandb.py) 的 `scan_history` 用法，换成上述六条运行并核对 seed、配置和 checkpoint 路径；保留稀疏字段，避免把只在部分行出现的指标过滤掉。本地 Trainer 日志通常没有 W&B 添加的 `train/` 前缀，字段转换需记录。横轴使用实际 optimizer global step，不能直接使用 W&B `_step`。

导出后核查 step 范围、缺失和重复记录、字段定义及奖励权重。缺少 history 时先补导出，缺少某个 seed 时明确标注可用数量；不将缺失值补零，不将同一步重复日志计作额外 seed，不为补图自动重训。现有代码的日志含义与 [奖励实现说明](../docs/g1_shortlist_improvements.md) 一致，最终以对应训练版本及实际字段为准。

输出统一放在 `paper/analysis_results/reward_curves/`：保存逐步标量 `history.csv`、运行与字段来源 `run_manifest.json`、聚合数据 `curves.csv`、图注草稿 `README.md`，以及 `reward_curves.pdf` / `reward_curves.png`。PDF 用于论文，PNG 用于预览。完成条件是六条日志身份与步数核对通过、分项与完整目标一致、图中均值/SD 可从 CSV 复算，再将图和相应描述纳入论文；目前仅完成分析设计与运行清单。

## 6. 跨 seed 确认与 RAG 后续

第一批有明确结论后，再用 42/2026 的最终 InfoNCE、RELER，以及必要时 Listwise，复核相同四领域的 query-level 分析。新增 seed 不能消除 BRIGHT 已参与选参的事实；如果需要更强泛化主张，应另留未用于选择的任务或查询。

**RAG 保留为后续必须完成的实验。** 方法 3.4 和现有流程图保留冻结索引、query-only policy 与答案奖励接口；当前正文不写尚未完成的 RAG 结果。后续沿用项目的 [RAG 计划与开发记录](../docs/g3_rag_plan.md) 及 [实验配置](../configs/experiments/iclr2027/suite_g3_r2.yaml)，先在训练机核实已有阶段的 checkpoint/检索与生成评测状态，再确定还需运行的部分。

正式论文实验至少区分原始 encoder、contrastive query adaptation、ranking-reward adaptation 和 answer-reward adaptation；固定文档索引、generator、prompt、context budget、decoding 与答案评测规则。报告检索指标和实际生成 EM/F1，以及生成调用/时间成本，不能用 answer containment 检索指标替代端到端答案质量。数据与 checkpoint 清单确定、结果跑完后再补实验表和结论；本轮不据旧开发记录推断正式 RAG 实验已完成。
