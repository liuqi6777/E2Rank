# Experiment Plan: Reward-Based Optimization of Embedding Retrievers

## 2026-09-12：固定探索强度趋势实验（最新安排）

围绕 **binary MRR@10、teacher-graded nDCG@10 与 binary nDCG@10 三种 reward 配方**，
分别固定 reward 与其余训练设置，只改变 vMF 的
`target_alignment = E[cos(action, mean)]`。值越大，探索越弱。
采用 **0.40 / 0.5302373892742263 / 0.65 / 0.80 / 0.90 / 0.95** 六点：
0.40 检查比旧默认更强的探索，0.65 补齐中间区间，0.90/0.95 检查弱探索端是否回落。

| 期望余弦 | binary MRR@10 | graded nDCG@10 | binary nDCG@10 |
|---|---|---|---|
| 0.40 | G1-A-MRRAlign040，待训练 | G1-A-NDCGAlign040，待训练 | G1-A-BinaryNDCGAlign040，待训练 |
| 0.5302373892742263 | G1-A-MRR，复用 19.21 | G1-J-RL，复用 18.14 | G1-A-Binary，复用 18.42 |
| 0.65 | G1-A-MRRAlign065，待训练 | G1-A-NDCGAlign065，待训练 | G1-A-BinaryNDCGAlign065，待训练 |
| 0.80 | G1-J-RL-MRRSmall，待训练 | G1-A-FixedSmall，复用 18.98 | G1-A-BinaryNDCGAlign080，待训练 |
| 0.90 | G1-A-MRRAlign090，待训练 | G1-A-NDCGAlign090，待训练 | G1-A-BinaryNDCGAlign090，待训练 |
| 0.95 | G1-A-MRRAlign095，待训练 | G1-A-NDCGAlign095，待训练 | G1-A-BinaryNDCGAlign095，待训练 |

共 18 个配置点，复用 4 个已有结果，14 个待训练点均从 E0 独立启动；双侧 vMF product、G=32、
leave-one-out、无标准化、文档求和、分数校准、113 steps、LR 5e-6、
global batch 128 / microbatch 16、seed 42 保持一致。全程固定探索，不使用退火。
训练时按实际 embedding 维度反解 κ；target_alignment 优先于继承的 kappa=755。
不同 κ 也会改变 score-function 梯度尺度，因此这里检验的是固定优化器下探索配方的
端到端效果，不将趋势完全归因于采样半径。

每个点只评测最终 checkpoint 的 BRIGHT；横轴为期望余弦，纵轴为 12 领域宏平均
nDCG@10，同时保留领域分数。分别绘制三条六点曲线，FixedSmall 只属于 graded nDCG
曲线，不混入 Anneal，不将 E0 15.07 当作“零探索训练”点。全量报告三条曲线，
观察各自最优区间、同强度下的差异，以及 reward 配方是否改变最合适的探索强度。
binary MRR 与 binary nDCG 使用完全相同的已知正例标签，比较奖励指标；
graded nDCG 与 binary nDCG 保持指标不变，比较标签来源和粒度。
graded nDCG 与 binary MRR 的直接比较同时包含标签和指标变化。
各对照均在相同期望余弦处比较，并观察最合适的探索强度是否随 reward 配方变化。
其他配置在本轮固定，已有消融作为选择依据，不据此声称已找到所有组件的全局最优组合。
本轮属于 BRIGHT 配置开发，结果不是独立于调参过程的最终测试证据。

使用 `bash scripts/run_g1_mrr_exploration.sh check 8` 预检，
`bash scripts/run_g1_mrr_exploration.sh train 8` 顺序训练五点，每点训练后自动执行 BRIGHT。
任一步失败即停止，不覆盖既有运行；若 MRRSmall 已完成，复用其结果，单独启动其余四行。
新增 nDCG 入口 `bash scripts/run_g1_ndcg_exploration.sh check 8` / `train 8`，
只训练四个缺失点，不重复默认强度与 FixedSmall。
binary nDCG 入口为 `bash scripts/run_g1_binary_ndcg_exploration.sh check 8` / `train 8`，
只训练五个缺失点，复用 G1-A-Binary。相较两条曲线新增 5 行；
当前共 43 行（41 次训练、2 次评测），其中 37 个核心训练、4 个可选训练。
三组共待执行 14 次训练；LL-Scaled 独立，不计入探索曲线。
用户将在其他训练平台运行，目前无法登录；本次只准备配置与批量入口，没有启动训练。

## 2026-09-12：第二轮配置开发（当前生效，覆盖下文冲突安排）

用户补充 E0 的 BRIGHT nDCG@10 为 **15.07**；来源为用户提供的结果，
尚未加入本地逐领域 CSV。第一轮 `_summary/g1_bright` 已有 16 个训练配置，
每个覆盖 12 个领域。主 RL 为 18.14（较 E0 +3.07），FixedSmall 为 18.98
（+3.91），MRR 为 19.21（+4.14），RankNet 为 17.96，LL 为 10.53。

当前优先目标改为开发更好的 BRIGHT 配置。G1 不再要求 MTEB retention、
逐 query bootstrap 或成本测量；它们也不作为下一轮训练的前置条件。
相应不主张已经验证通用检索能力保持、统计显著性或计算效率优势。
G2 的外部检索评测与 G3 的答案评测属于各自研究问题，不受这一 G1 决定影响。

下一批增加 **2 次独立训练**，都从 E0 开始，沿用相同数据、seed 42、
full FT、LR 5e-6、113 steps、global batch 128、microbatch 16 和最终 checkpoint：

| ID | 配方 | 目的 / 对照 |
|---|---|---|
| G1-J-RL-MRRSmall | binary MRR@10，固定期望余弦 0.80；G=32，双侧 product，leave-one-out，无标准化，文档求和，保留校准 | 优先候选；相对 G1-A-MRR 只减小探索，另与 FixedSmall 比较 reward 设置 |
| G1-J-LL-Scaled | graded LambdaLoss，sigma=1/0.03，其他设置沿用 G1-J-LL | 将 pairwise logistic 输入尺度对齐现有 RankNet；不改变 teacher grades 或 nDCG cutoff |

MRRSmall 的假设是：强调首个已知正例的反馈，与较小的球面扰动可能互补。
两个单项收益不能相加预测组合分数；不引入退火、额外标准化或更长预算。
LL 的尺度修改有现有 RankNet temperature 作为依据，但效果尚未验证；
同 logistic 尺度不意味着梯度、优化动态完全相同。LL-Scaled 与原 graded RL /
FixedSmall 是同标签同 cutoff 的对照，不能称为 MRRSmall 的完全同目标监督控制。

本轮是查看第一轮 BRIGHT 结果后进行的配置开发，后续分数按这一背景报告，
不继续宣称 BRIGHT 从未用于配置选择。旧运行 ID、配方和结果保留；新配置使用独立
输出目录，尚无结果，不自动替换主表的实测行。当前共注册 30 行（28 次训练、2 次评测）；
其中核心训练 24 次、第一轮可选训练 4 次。下文 22/26 次预算为第一轮历史安排。
本次仅更新计划与可解析配置，没有启动训练。

## 2026-09-12：实验优先级与执行顺序（当前生效）

核心预算为 **22 次训练**。新增 Norm / DocMean / NormDocMean，和主方法组成统一
leave-one-out baseline 的 2×2 对照；Gaussian mismatch 与 MRR 降为可选。
探索收缩和固定终点强度必须成对安排，启用这两项后为 24 次；四项可选全部启用为 26 次。
两个 E0 评测不计入训练次数。主方法保持固定 κ=755，不按消融结果更换配方。

推荐顺序：G1 同标签 LL/RL → G1 CL/RN → 更新规则 2×2 → 其他核心机制 → G2 → G1-DR → G3。
详见第 5 节。顺序是资源安排，不是基于测试分数的模型选择或自动批量执行。
配置中的 `execution_stages` 与 `priority` 记录该顺序；`list` / `show` 可直接查看。
本次仅修订计划、运行行与论文，没有启动实验。

## 2026-09-12：第一轮方法修订

本节覆盖下文涉及旧估计器的历史说明。标签、候选、G1/G2 预算和评测协议不变；没有因本次修订启动训练。

- 主策略采用 `advantage_baseline=leave_one_out`、`advantage_norm=none`、`document_log_prob_reduction=sum`。
  Document log-density 始终按有效文档求和；`mean` 只在 loss 层施加旧版 `1/n_valid` 权重。
- 默认仍固定 `kappa=755`、`G=32`，不预先声称新更新规则能提升检索。
  旧版可由 `configs/grpo/legacy.yaml` 复现：group baseline、per-component normalization、document mean。
- `configs/grpo/annealed.yaml` 提供独立、未经效果验证的预定调度：期望余弦从 0.53 线性增加到 0.80，
  按实际向量维度反解 κ。第 t 个 optimizer update（从 0 计数）使用 t/max(T−1,1)；
  同一步的 accumulation microbatches 共用 κ，采样与评分使用同值。此时每步对应 J_{κ_t}，不是同一个固定目标。
- Joint、固定候选 query-only、G1-DR 和 RAG 共用 baseline 语义与调度。
  Checkpoint 新增 `exploration_state.json`，记录配方、估计器及进度；恢复时校验配方，
  下一步以 Trainer 恢复的 global_step/max_steps 为准。缺少新版元数据的旧 checkpoint 不自动续训新版策略。
- Leave-one-out 去掉 group baseline 的 (1−1/G) 因子。无偏性仅指冻结当前 cross-query 向量条件下的原始梯度；
  不包括 detached 向量随共享 encoder 刷新的依赖，也不包括 optimizer/gradient clipping 的变换。
- 探索方差修正为 `A/κ * ||δ_perp||² + A' * g²`；Gaussian 翻转概率使用完整方差。
  `g*` 仅作边界附近的近似尺度。κ=755、d=1024 的期望余弦约 0.5302，不属于小角度扰动。
- 关闭归一化后仍记录近退化组比例，但不因日志阈值清零非零的原始 advantage。
  `exploration/own_boundary_flip_rate` 仅覆盖 own-list 中涉及 top-K 的不同 grade、非平局相邻对；不代表全库变化。
- 第二轮的逐文档 counterfactual baseline、几何保持、奖励标签重定义、各向异性探索均未加入本轮主方法。

本轮验证：25 项 CPU 数学/接口检查通过，包含确定性球面积分、完整离散组枚举、维度反解、
两路径 baseline 一致性、padding 梯度、固定候选封装、动态检索、Trainer 累积/恢复和配置传递。
12 条已实现 RL 配方完成参数解析；未实现的 G3-RetRL 保留原 blocker。论文全量 LaTeX 编译通过，
无未定义引用或 overfull box；未启动模型训练、语料编码、生成器调用或检索基准实验。
旧测试套件存在两个独立失效点：`test_frozen_corpus.py` 导入当前及 HEAD 均不存在的
`validate_training_against_bright_documents`；`test_g1_dynamic_retrieval.py` 的 suite 测试把根目录
定位到 `scripts/`。未将这些旧测试计入通过数，也未扩展本轮范围修复；当前固定候选及配置接口
已由新增 `scripts/tests/test_policy_math.py` 覆盖。测试目录与论文源码沿用仓库现有的本地保留规则。




**修订：2026-09-11。本文档是实验设计的 source of truth。**

本版替代此前以 E2Rank → MTEB 为主、base-LLM 训练仅作可选边界实验的方案。
采用三组实验：①开放权重 embedding model 的 reasoning retrieval 适配；
②从未经 embedding 专项训练的通用 LLM 开始的较大规模 embedding 训练；③固定索引的 RAG 优化。
所有结果均待测；本文档不表示对应实现已就绪。主实验统一 **full fine-tuning、单 seed 42**。
本次只修订实验计划，论文正文、表格和 runner 尚须按本版同步。

日常配置入口已收敛到 `configs/experiments.yaml` 和 `scripts/experiment.py`。
G1 对外数据接口为 `id/query/positives/negatives/source`。预处理关联审计元数据并将
候选顺序、标签、去重标识写入 `train.ready.jsonl`；loader 直接读取，batch mask 动态构造。
当前目录为 `data/processed/reasonrank_multi/`（旧单正例产物保留），
仍为 4,963 条全量训练；底层 suite 保留作高级配置。

**G1 初始运行参数已定：** E0、full FT、seed 42、AdamW、LR 5e-6、global batch 128、
每卡 microbatch 16、113 optimizer steps（14,464 query exposures）、linear schedule、warmup 0.03、
weight decay 0.01、max grad norm 1.0；每 25 步保存，报告最终模型。按 GPU 数自动调整梯度累积。
RL 使用 group size 32、kappa 755；CL/RankNet temperature 0.03。此版未经 dev 调参。
G1 不再将模型 revision 和 manifest/hash 校验作为启动条件；保留必需的结构检查。

**G1 最新决定（2026-09-11）：** 保留全部去重后的已知正例与负例；固定 seed 仅选择
一个 in-batch 代表正例，不删除其余正例。
使用全部 4,963 条清理后的非 MSMARCO 数据训练，不划分 dev。提前固定配置/训练预算，
报告最终 checkpoint；G1 不执行下文通用的 dev 网格调参或最佳 checkpoint 选择。
BRIGHT 仅作最终评测。G2/G3 的开发集协议暂不改变。

**本轮评审决定（2026-09-11）：** LL 保留当前 sigma=1.0 与训练配方先跑，暂不追加
尺度调参；G1-A-MRR 的直接控制改为 G1-A-Binary。普通固定候选训练与 G1-DR
共享完整已知标注；DR 保留完整 corpus，其奖励有效性由实验确定。
G3 的无检索 generator 基线和答案/证据变化联合诊断后移，不作为当前 G1/G2 的前置条件。

## 0. 研究问题与贡献边界

| 组别 | 初始化与数据 | 核心问题 | 主要评测 |
|---|---|---|---|
| G1：Reasoning adaptation（主实验） | 现有开放权重 embedding model；清理后的 ReasonRank | 相同数据下，RL 是否优于监督适配？document policy/exploration 是否必要？ | BRIGHT；MTEB retrieval 作为能力保持检查 |
| G2：General-LLM embedding training | 未经 embedding 专项训练的通用 LLM；E2Rank listwise，约 156k，实际规模待审计 | RL 能否直接学习检索表示？共同 warm-up 后是否仍有收益？ | 固定的外部检索任务集与学习曲线 |
| G3：RAG optimization | 预先固定的 embedding checkpoint；独立 QA 数据 | 固定索引时，答案反馈能否改善检索和生成质量？ | Answer EM/F1；检索指标辅助 |

三组共同支持 embedding-space reward optimization，分别覆盖初始化、更新约束和反馈类型。
G1/G3 属于后训练和任务适配；G2 检验更早的表示学习阶段。不能预先宣称 RL 可以替代
对比学习建立初始空间，也不能因为共同对比学习 warm-up 使用了 156k 数据，就声称 RL
本身已在这个规模得到验证。G2 的 RL 阶段必须实际使用并记录较大规模的数据预算。

“约 156k”称为较大规模 listwise 训练，不等同于通用 embedding 预训练规模。
BRIGHT 提升不单独证明学会推理。G1 full-corpus 动态检索作为独立实验行 `G1-DR`，
不计入主结果或 RL 单因素消融。

## 1. 共同实验约束

### 1.1 初始化、参数与版本

- G1 主模型沿用计划中的 `Qwen/Qwen3-Embedding-0.6B`，记为 E0。
  正式运行前固定 checkpoint revision、tokenizer、pooling、prompt 和最大长度。
- G2 使用未经 embedding 专项训练的通用 LLM，记为 B0；当前为 `Qwen/Qwen3-0.6B`。
  允许通用语言模型后训练，不将 B0 描述为纯预训练 checkpoint。
  优先选择规模相近、便于控制架构差异的模型，不能直接把 E0 当作 B0。
  同一组内严格同初始化；不同模型组之间不作只归因于目标函数的比较。
- G3 默认从原始 E0 开始，不自动继承 G1 或 G2 的最佳模型，避免引入额外训练混杂。
- Joint 默认一个共享 encoder，通过 query/document 两种角色更新全部参数；
  最终评测必须用当前 document encoder 重新生成文档向量。
- Query-only 从同一个 E0 创建可训练 query 分支和冻结 document 分支；全部 query
  参数可训练。固定的是文档表示版本，不是声称真实部署中的 corpus 永远不变。
- 主实验不使用 LoRA；第二个 embedding 模型作为泛化扩展，不展开完整消融。

### 1.2 公平对照与预算

每个直接比较内固定初始化、数据与顺序、候选构造、标签来源、tokenization、
长度、microbatch、梯度累积、优化器族、精度和 checkpoint/evaluation schedule。
G1 主对照全部使用 joint encoder；RL action 组成只在预先声明的消融中改变。

主要结果按处理的 query/examples 与输入 token 预算比较；同时报告 steps、GPU-hours、
wall-clock、峰值显存、采样和 reward 成本。相同硬件才作时间比较。
产品组合的 G² 次 reward evaluation 不能隐藏在 encoder 复用的计算说明中。

各组、各分支内监督与 RL 使用相同数量的调参试验。默认每个 objective 使用
3 个 learning rates × 3 个目标专属设置，共 9 个短试验；LR、temperature、concentration
等数值网格、每次试验预算及总训练预算须在 Phase 0 写入 manifest。
预算不足时统一缩减直接比较各方的试验数，不单独减少强基线。
G2 直接训练和 warm-up 后训练可以有各自的网格，但每个对照内部相等。

所有训练 seed 为 42。数据划分使用单独固定的 split seed/manifest，不依赖训练 RNG。
不计划 multi-seed，也不因结果差距小临时增加 seeds。
可选 paired query bootstrap 仅描述固定模型下的测试样本不确定性，不代表训练方差。

### 1.3 评测与模型选择

G1/G2 提前固定配置与训练预算，报告最终 checkpoint，不使用 dev 选模。
G3 使用独立 QA dev 选模；并列时按更早 step、预先固定的 config 顺序处理。
外部最终评测不用于选配置或挑任务。

以确定性均值 embedding 的检索结果为主。采样 reward 上升不能替代确定性质量提升。
固定评测数据 revision、task list、候选、检索协议与聚合权重，输出逐任务分数。
MTEB 全部任务平均分不是 G1 主指标；不为追求可区分性而在看到方法结果后筛选任务。

## 2. G1：开放 embedding model 的 reasoning retrieval 适配

### 2.1 数据与清理

使用 [ReasonRank 审计](REASONRANK_BRIGHT_AUDIT.md) 中固定版本的数据。
默认主训练数据排除 MSMARCO，保留 reasoning 来源；含 MSMARCO 混合训练是可选扩展。

| 阶段 | 记录数 | 状态 |
|---|---:|---|
| 原始 train | 6,721 | 已实测 |
| 保守隔离 BRIGHT 匹配项后 | 6,588 | 隔离 133 条，含疑似误报，不是完整语义去污染认证 |
| 再保守排除标签冲突、全正例、内部重复 | 6,498 | 审计估算 |
| 再排除 MSMARCO | 4,963 | 主数据量估算 |
| 全量用于 train，不留 dev | 4,963 | 已生成；此前 4,463/500 划分另存保留 |

正式训练前完成疑似同题复核并冻结排除 manifest；完整保留 BRIGHT 测试集。
G1 不再划分 train/dev，清理后全部用于训练。预处理支持可选的分层分组 dev 划分供复现。
原始 val 50 条全为 math-theorem，不用于全局选模。
共享候选文档不自动等于同题，不据此合并整个数据集。
记录最终样本数、来源分布、排除原因与 hashes；表中估算不能替代最终 manifest。

重写转换流程，保留可变长候选、document IDs、source、teacher permutation 和
`relevant_docids`，实现 padding/mask。当前 first-16 converter 会丢掉 19.7% 的记录，
不能作为本组输入。重复文档文本去重，正负标签冲突记录默认隔离；移除不足两个有效
候选或没有正负区分的列表。Teacher 排序和原始 relevance 不得静默互换。

### 2.2 主监督与候选协议

主比较保留全部去重后的已知正例与负例；seed 42 按 query 独立固定抽取一个正例，
仅作为 in-batch 代表，不改变本条 query 的候选集合。
候选集合相同，但正例身份与排序监督分开保存。RL 主 reward 为 teacher-derived graded nDCG@10：
保留候选中 teacher 第 1 名为 3，第 2–5 名为 2，第 6–10 名为 1，其余为 0。
E2Rank pilot 支持该设置，ReasonRank 的收益由 binary 消融验证。
不把 teacher 第一名当正例；BRIGHT 评测保留完整官方标签。

| Objective | 主监督定义 |
|---|---|
| Multi-positive InfoNCE | 每个正例分别与全部有效负例计算 log-softmax loss，其他正例不进入分母；先对正例平均，再对 query 平均 |
| RankNet | 使用保留候选的完整 teacher 排序构造有序 pairs |
| LambdaLoss | LambdaRank variant：pairwise logistic × 当前排序的 `|ΔnDCG@10|`；使用与 RL 相同的 graded 标签与候选，`gain=2^rel-1`、`sigma=1.0` |
| Proposed RL | teacher-derived graded 标签与相同候选上的 exact nDCG@10 |

主协议沿用扩展候选池设计：加入其他记录固定抽取的正例作为 in-batch 候选，
所有 objective 相同。合并同文本/同 ID 项；跨 query 的已知正例（含去重别名）被 mask 掉，
其余跨 query 候选按负例近似，明确其不完整标注与 false-negative 风险。
跨 query document embeddings detach，文档只通过自身记录得到梯度。

记录 microbatch、world size、accumulation；当前 in-batch 池是 device-local，
梯度累积不扩大候选池。开发集固定候选，不随评测 batch 改变。
Own-candidates-only 消融检查扩展候选的作用。Mean-score calibration 只在确有冻结候选时评测。
CL 使用已知正例，排序方法使用 teacher 信号，监督信息并不完全一致；LL 与 RL 是同 graded 目标的主要对照，不能仅由 CL 对比归因于估计器。

### 2.3 主结果表

新 ID 是实验逻辑 ID，不代表已有 runnable config。

| ID | Objective | 更新范围 |
|---|---|---|
| G1-E0 | 无训练 | 原始模型，评测复用 |
| G1-J-CL | Multi-positive InfoNCE | Joint |
| G1-J-RN | RankNet | Joint |
| G1-J-LL | LambdaLoss | Joint |
| G1-J-RL | Proposed nDCG RL | Joint；query 与 document policy |

共 **4 个主训练配置**，加原始 E0 评测。四个 objective 使用相同候选和训练预算。
G1-J-RL 超过 E0 只证明额外训练有益；超过 CL 但不超过 LL 不能证明优于 metric-aware 监督。

`G1-DR` 是单独的动态检索实验：从 E0 初始化并冻结分 source 的完整 document index，
只更新 query encoder。每个 sampled query action 检索 top-20，以去重后的
完整已知候选 teacher order 按 3/2/1/0 构造 grades，计算 nDCG@10；保留所有已知正例身份，
不因 in-batch 代表抽样而从 corpus 或 qrels 删除其余正例，也不将 binary 正例身份
强行转换为 teacher 高 grade。未出现在完整已知 qrels 中的文档 gain 为 0。
新版 ready 已保留完整候选 IDs 与 grades，DR 通过同一 collator 加载完整已知 qrels，
IDCG 使用完整已知 qrels 而非仅本次检索结果计算。
其最近控制为 `G1-A-QPolicy`，但二者
同时改变 candidate access 和 document update scope，因此不解释为单因素因果对照。

**主结果：** BRIGHT nDCG@10，报告固定官方协议下逐领域与宏平均。
冻结短/长文档设置、query instruction、excluded-document handling 和全 corpus 检索协议；
不能用训练中的短候选列表重排结果代替 BRIGHT retrieval。
**能力保持：** 预先固定的 MTEB retrieval 子集及相对 E0 的变化，作为辅助结果。
**选模：** G1 不使用 dev，不执行目标专属网格调参；提前固定配置和预算，报告最终 checkpoint。
训练 reward/损失仅用于数值诊断，不用 BRIGHT 或 MTEB 挑选 checkpoint/超参数。

### 2.4 主要消融与理论诊断（集中在 G1）

所有消融从相同 E0 开始，使用同一数据顺序、候选池、113 步预算和最终 checkpoint 协议。
复用主方法的**结果作为对照**；不从已训练的 G1-J-RL checkpoint 继续微调消融，也不隐式重训主方法。

| ID | 相对直接控制的变化 | 优先级 |
|---|---|---|
| G1-A-Norm | 对 G1-J-RL 恢复 per-component advantage 标准化，文档仍求和 | 核心 |
| G1-A-DocMean | 对 G1-J-RL 恢复文档 `1/n_valid` 权重，不标准化 | 核心 |
| G1-A-NormDocMean | 同时恢复标准化和文档平均；结合前三格分析交互 | 核心 |
| G1-A-Paired | 对 G1-J-RL 改为同 G 的 paired/diagonal rollout | 核心 |
| G1-A-QPolicy | 对 G1-J-RL 只移除 document policy/exploration，仍训练共享 encoder | 核心 |
| G1-A-DPolicy | 对 G1-J-RL 只移除 query policy/exploration，仍训练共享 encoder | 核心 |
| G1-A-Cal | 对 G1-J-RL 禁用 frozen-candidate mean-score rescaling | 核心 |
| G1-A-Binary | 对 G1-J-RL 将 teacher grades 改为全部已知正例 binary 标签 | 核心 |
| G1-A-Anneal | 与主方法相同初始探索强度，期望余弦预定线性增加到 0.80 | 可选，与 FixedSmall 成对 |
| G1-A-FixedSmall | 全程固定期望余弦 0.80；控制 Anneal 的终点探索强度 | 可选，与 Anneal 成对 |
| G1-A-Gaussion | projected-Gaussian 采样，但沿用 vMF log-density；仅研究采样/评分失配 | 可选诊断 |
| G1-A-MRR | 以 G1-A-Binary 为直接控制，仅将 binary nDCG@10 改为 binary MRR@10 | 可选 |
| G1-A-Own / G1-A-G | 去除 in-batch candidates / group size 质量成本曲线 | 次要，尚未加入可启动行 |

更新规则的四格必须保持相同 leave-one-out baseline，不能把 baseline 也一起切回旧值：

| | 文档求和 | 文档平均 |
|---|---|---|
| 不标准化 | G1-J-RL（复用主结果） | G1-A-DocMean |
| per-component 标准化 | G1-A-Norm | G1-A-NormDocMean |

令四格最终评测结果分别为 S_none,sum、S_norm,sum、S_none,mean、S_norm,mean。
分别报告两个背景下的简单效应，以及交互差分
`(S_norm,mean − S_none,mean) − (S_norm,sum − S_none,sum)`。
单 seed 的交互差分只是描述性证据，不据此声称跨 seed 显著性。
NormDocMean **不是完整旧版复现**：旧版 baseline 为 group，当前四格均为 leave-one-out。
Baseline 常数因子的作用由数学/数值检查验证，不单独增加完整训练。

四格的 LR、梯度裁剪和 optimizer 保持一致；报告裁剪前梯度范数、角色夹角和裁剪比例，
避免把最终差异全部解释为梯度方向变化。标准化与文档平均也会改变尺度，而 Adam/裁剪未必尺度不变。

探索三方比较复用固定 κ=755 的主结果，另加 Anneal 和 FixedSmall 两次训练。
对当前 Qwen3-Embedding-0.6B（1024 维），Anneal 起点固定为 `A_1024(755)=0.5302373892742263`，
终点为 0.80；FixedSmall 全程为 0.80。由实际维度反解 κ，其他配置与主方法一致。
这与底层 annealed.yaml 的四舍五入示例 0.53 有区别；正式命名行使用精确起点。
切换模型、改变主方法的探索配置时，应先重新声明成套对照；当前两个命名行固定适用的初始化模型。
启用可选实验应在看到最终测试分数前按资源/论文主张决定，并将两项都报告；不以更优者替换主方法。

Paired 的 G 次 reward 与 product 的 G² 次存在成本差异；报告 example-matched 结果，
另做短 matched-time 比较或质量/成本曲线。G² 个组合相关，不是 G² 独立 joint samples。

`G1-A-DPolicy` 与 `G1-A-QPolicy` 分别隔离 document 和 query action 的贡献；二者都保留
joint encoder 更新，因此与 `G1-J-RL` 的差异只在被采样和施加 score-function 梯度的 policy role。
`G1-A-Gaussion` 使用 `sampling_law=gaussian` 的归一化高斯采样；当前实现仍以 vMF
log-density 计算 surrogate，故该行检验的是 vMF 采样一致性，而非完整重推导的 Gaussian
policy-gradient estimator。正式报告须明确这一点。

固定 checkpoint/minibatches 重复采样，测 paired/product 梯度估计的噪声、相对高采样
参考的偏差及 per-role norms，并记录是否启用 advantage normalization。
另做低成本小探索极限诊断：关闭 advantage normalization，固定 frozen candidates，
比较直接 InfoNCE 梯度与平均 score-function 梯度，分别检查角色再施加一致角色权重。
报告 cosine、relative error 和采样预算；有限样本噪声不等于理论极限失败。

诊断与完整训练分开计数。已经接入 raw marginal reward spread、近退化比例、mean alignment、
own-list 边界翻转率。还需补充的观察项如下，不能把计划中的日志写成已有结果：

| 观察项 | 用途与协议 | 当前状态 |
|---|---|---|
| 同候选池的 sampled / deterministic reward | 固定标签和候选；确定性版本使用原始 cosine，不作 sampled-document 衰减校准；不当成全库检索分数 | 待接入 |
| 裁剪前 query/document 梯度范数、夹角、裁剪比例 | 区分方向冲突、尺度与优化器影响；在固定诊断 batch 上采集，单列额外成本 | 待接入 |
| 按候选长度分组的确定性 reward 与边界翻转 | 观察 `1/n_valid` 的影响；预先固定分组 2–5、6–10、11–20，报告样本数；不据此重新选训练数据 | 待接入 |
| sampled/frozen 分数分布与冻结候选 top-K 占比 | 解释 calibration，不将均值匹配当成完整分布匹配 | 待接入 |

归一化后的 advantage std 不能独自说明信号质量。
旧版 per-component normalization 和 document-role `1/n_valid` 权重不等于未加权联合
目标的无偏梯度；新版默认已移除这两项。Mean-score calibration 仍只匹配一阶分数矩，不是期望 ranking reward。

## 3. G2：从通用 LLM 开始的较大规模 embedding 训练

G2 初始运行参数沿用旧 scratch 的 `Qwen/Qwen3-0.6B` 表示协议与 full FT 配方：
LR 5e-6、global batch 128、microbatch 16、linear schedule、warmup ratio 0.03。
每次独立运行训练 1200 步，每 200 步保存。第二阶段统一从 G2-D-CL 的最终模型初始化，
新建 optimizer、scheduler、step counter 和数据迭代，各训练 1200 步。

### 3.1 数据、表示与标签

使用原始 E2Rank listwise artifact（约 156k，须核实实际版本、规模和候选长度）。
直接读取原始 `data/train.jsonl`，全量用于训练，不另留内部 dev/test。
不默认使用混入 ReasonRank 的 `train_v2`；审计来源与外部评测 overlap，固定所有方法的训练文件。
不需要 ReasonRank 预处理或专门的 manifest loader。

固定 B0 的 pooling、retrieval prompts、归一化与 scoring；这些在所有目标间保持相同。
所有路线均 full FT、joint query/document。B0 直接评测仅作初始化诊断，不假定已有检索能力。

保留完整 teacher order，并为 metric-aware 目标定义 teacher-derived grades：
rank 1 / 2–5 / 6–10 / 其余对应 3 / 2 / 1 / 0；主 RL 优化 nDCG@10。
CL 使用 rank-1 正例和相同候选池；LL 使用与 RL 完全相同的 grades/cutoff。
跨 query 候选构造保持一致，明确其近似负例属性。
CL 与 RL 使用的监督信息粒度不同，因此 **LL 是必需的同标签强对照**。
可选 RankNet 补充 full-order 控制；不能只比较单正例 CL 就归因于 RL 估计器。
此处 grades 是 teacher 目标，不是人工相关性；MRR 若报告，明确 grade ≥2 的定义，
G1/G2 graded 均采用相同 teacher 排名分段；已知正例身份另行保存。

### 3.2 两个问题、两组对照

| ID | 路线 | 目的 |
|---|---|---|
| G2-D-CL | B0 → CL | 直接 embedding 训练基线 |
| G2-D-LL | B0 → LambdaLoss | 同 grades 的监督对照 |
| G2-D-RL | B0 → RL | 检验直接从未经 embedding 专项训练的通用 LLM 启动 |
| G2-W-CL | B0 → CL 最终模型 W0 → 重新训练 CL | 后续训练对照 |
| G2-W-LL | B0 → 同一个 W0 → LL | 排除切换到 metric-aware 目标本身的收益 |
| G2-W-RL | B0 → 同一个 W0 → RL | 检验已有初始空间后的 RL 收益 |

W0 为 G2-D-CL 完成 1200 步后的最终模型，只训练一次，三条第二阶段分支共享该权重。
G2-W-CL/LL/RL 均为独立运行：重置 optimizer、scheduler、step counter 和数据迭代，
各重新训练 1200 步。W-CL 不复用 D-CL 的后半段，入口不恢复 trainer 状态。

共 6 个训练执行：3 个直接训练和 3 个从 CL 最终权重初始化的第二阶段训练。
直接路线预算 1200 步；两阶段路线计入共同前缀后为 2400 步，不能称为等总预算对比。
第二阶段 CL/LL/RL 的起点、候选、优化设置及新增预算一致，可用于比较后训练目标。

### 3.3 评测、失败情形与结论

Phase 0 固定一组外部检索任务，可选自 MTEB retrieval；包含任务与 aggregation 必须
在看到目标间效果差异前冻结。报告逐任务结果、宏平均及随处理样本数/时间的学习曲线。
提前固定训练预算并评测最终 checkpoint，不做内部 dev/test 划分或选模，外部任务检验迁移。
训练集 teacher-target 指标仅作训练诊断，不作为 held-out 测试结果。BRIGHT 可补充，但不能
替代本组通用检索证据。不得只凭同分布 teacher-target 改善宣称通用能力提高。

与强 embedding 初始化相比，base 可能留出更大改善空间，但不能保证 MTEB 可区分。
若各方法外部效果接近，如实呈现，并根据预算/效率和学习曲线限定结论，不事后挑任务。

直接 RL 的稀疏信号、退化 group、低 reward variance 和检索空间形成速度是本组要测的
问题。使用预先统一的非有限值/数值失效规则中止并保存失败运行；不要隐藏不收敛结果。
若直接 RL 无效而 W-RL 有效，应将贡献定位为已有检索表示上的 post-training。
若直接 RL 同时在外部检索有效，才扩展对早期 embedding 训练适用性的主张。

## 4. G3：固定索引的 RAG 优化

默认初始化 E0，所有方法 full FT query encoder，冻结 document encoder、corpus vectors、
index 和 generator。独立 QA train/dev/test，不能用 ReasonRank listwise 标签替代答案监督。
现有 NQ/HotpotQA、wiki18、Qwen2.5 generator 配置只是候选方案；Phase 0 验证实际
数据、答案/证据映射和版本后固定。优先完整完成一个 QA 任务，第二个作为扩展。

| ID | Query 训练 | 角色 |
|---|---|---|
| G3-E0 | 无 | 原始检索器 |
| G3-CL | InfoNCE（多正例按实际标注处理） | 检索监督适配 |
| G3-RetRL | Retrieval nDCG reward | 与下游答案 reward 区分 |
| G3-AnsRL | Generated-answer F1 reward | 直接下游反馈的核心证据 |

主指标和选模指标为固定 QA dev/test 的 answer F1；EM、Recall@5/20、MRR 为辅助。
固定 generator、prompt、top-k、context budget、decoding、答案归一化和聚合方式。
所有行使用同一套答案评测，不能把 answer-containing passage MRR 当作生成答案 F1。
报告训练 GPU-hours、index searches、generator calls/tokens、cache reuse 与实际 group sizes。

监督训练使用离线候选；RL 不使用 CL 意义上的正负候选，action 直接动态检索冻结的完整
corpus，再由检索或答案 reward 评分。candidate manifest 在 RL 中只承载 query、答案和证据
元数据，不限制 action 的检索空间。因此主实验比较的是完整学习范式，不把方法差异仅归因于
目标函数，也不要求 shared-candidate 控制。若额外运行 fixed-slate RL，它只作为 reranking
诊断，不作为 G3 主实验的前置条件。各行仍需统一 E0、corpus/index、数据划分和最终评测，
并分别报告 query exposures、index searches、generator calls/tokens 与 GPU-hours。

若 AnsRL 未完成，只能声称改善 RAG 检索，不能声称已验证直接答案优化。
固定索引无需重编码的优势属于本组所有适配方法，不是 RL 独有。

## 5. 执行顺序与预算

### Phase 0 — 数据和实现前置条件

- [x] G1 保守隔离、多正例保留、去重和全量 train manifest（4,963 条）。
- [x] G1 空字符串补齐和有效候选 mask 的文本层 helper。
- [x] G1 候选 mask 接入 collator、有效文档编码、CL/RankNet 和 RL reward/log-prob/KL；CPU 数值检查通过。
- [ ] G1 最终同题复核。
- [x] G1 主对照与 DR 共享完整 teacher qrels 与全部已知正例；CPU 检查完整标注的 DR 奖励与 IDCG。
- [ ] G2 E2Rank 版本与规模、外部 overlap 审计、固定全量 train 文件。
- [ ] 固定 E0、B0、pooling/prompts 和各组评测协议；校验 full FT 与冻结分支。
- [x] G1 接入逐正例独立分母的 multi-positive CL、teacher-order RankNet 和可变长 RL；G2 保留 rank-1 CL。
- [x] 实现指定 LambdaLoss variant；G1/G2 使用 teacher grades 3/2/1/0，padding 不参与 loss。
- [ ] 明确两套标签的 MRR 阈值，确保 padding 不参与 reward。
- [ ] 接入最终 checkpoint 的固定候选排序评测与全 corpus 检索。
- [ ] 核对固定 LR/主训练预算和 G2 两阶段预算；G1/G2 不使用 dev 网格选模。
- [ ] G3 QA、index、generator manifests；验证 RL action 动态检索完整冻结 corpus，且训练
      不消费离线 candidate IDs。
- [ ] 所有命令 dry-run，检查 resolved configs，再提交 GPU；尚无已完成结果。

### Phase 1 — 数值检查、协议固定与短运行准备

先完成配置展开、padding/梯度/断点恢复检查；本轮已有 CPU 验证不必因每个新 YAML 行重复训练。
正式 GPU 前仅安排能够发现数值或接口故障的短 smoke，并将其成本与主训练分开记录。
若修复实现或修改配方，须在正式比较前冻结新版本，让全部相关对照采用一致版本。
不以 BRIGHT/MTEB 的早期分数调 LR、选择 κ、筛掉数据或替换主方法。
G3 的 generator 可复现性及成本检查在 G3 正式训练前完成，不阻塞 G1/G2。

### Phase 2 — G1 主效果和更新规则（第一批）

| 顺序 | 执行项 | 次数 | 目的 / 前置条件 |
|---|---|---:|---|
| 0 | G1-E0 | 0 次训练 | 固定 BRIGHT 评测协议，原始模型结果只评测一次并复用 |
| 1 | G1-J-LL → G1-J-RL | 2 | 最先得到同 grades/cutoff 下的监督与 RL 对照；两项独立，可按资源重叠 |
| 2 | G1-J-CL → G1-J-RN | 2 | 补齐常规表示学习与完整 teacher 顺序的主对照 |
| 3 | G1-A-Norm → G1-A-DocMean → G1-A-NormDocMean | 3 | 完成更新规则 2×2，复用主 RL 为第四格；均从 E0 开始 |

这批合计 7 次训练。LL/RL 的差异先帮助理解效用，四格进一步解释标准化与文档权重。
即使先得到的结果不利于 RL，也记录并完成预先声明的比较；测试分数不决定哪格成为新默认。
只有明确数值/实现故障才先修复再统一重跑受影响对照，不用无限补跑寻找正结果。

### Phase 3 — G1 其他核心机制（第二批）

依次建议 `G1-A-Paired → G1-A-QPolicy → G1-A-DPolicy → G1-A-Cal → G1-A-Binary`，共 5 次。
先检查 product rollout 和两侧探索的必要性，再检查分数校准与 teacher 标签来源。
Paired/product 继续报告相同 query exposure 的结果及单列的质量/时间诊断。
这些行无权重依赖，可以在资源允许时调度重叠，但都复用同一个固定主方法结果作控制。
两批 G1 共 12 次训练，尚不包含固定索引 G1-DR。

### Phase 4 — G2 初始化与共同 warm-up（第三批）

先执行 `G2-D-CL`，其最终模型就是 W0；随后安排 `G2-D-LL / G2-D-RL` 与
`G2-W-LL / G2-W-RL / G2-W-CL`，合计 6 次。
W 系列只有对 D-CL 最终权重的真实依赖，无需等待 D-LL 或 D-RL；各自新建 optimizer、scheduler 和数据迭代。
不把 D-CL 再训练一遍当作 warm-up，也不从 G1 最优消融继承模型或配方。
数据准备就绪、资源允许时，G2-D-CL 可与 G1 后续机制运行重叠；顺序不构成额外阻塞条件。
保留直接 RL 不收敛或无收益的结果，用于界定初始化条件。

### Phase 5 — 固定索引部署与下游反馈（第四批）

1. 准备并核验 G1 分领域 E0 索引后，执行 `G1-DR`（1 次）。它与 QPolicy 不同的 candidate access
   和文档更新范围必须同时注明，不按单因素消融解释。
2. G3 的 QA/index/generator 协议和 answer-F1 选模能力接入后，先得到 `G3-E0`，再安排
   `G3-CL → G3-AnsRL → G3-RetRL`（3 次）。优先完成答案反馈问题；RetRL 还需 nDCG 实现。
   现有 `rag_selection` / `rag_ndcg` blockers 保留，不用 source-aware MRR 顶替 nDCG。

G1-DR 的索引准备可提前进行，但不是 G1 joint 或 G2 的前置依赖。
G3 与前两组无 checkpoint 依赖；其准备完成后可与其他独立运行重叠。

### Phase 6 — 可选扩展（单独申明预算）

优先成对执行 `G1-A-Anneal` 与 `G1-A-FixedSmall`（2 次），同时复用主方法作为固定初始探索对照。
若保留分布失配或奖励接口的额外主张，再安排 `G1-A-Gaussion` 与 `G1-A-MRR`（各 1 次）；
MRR 的直接控制为已经完成的 Binary。无需为了可选行重训主方法或 Binary。
可选与核心区分是预先声明的研究范围，不按结果好坏决定是否报告。

### 核心预算与删减顺序

| 部分 | 主训练执行数，不含 smoke 和诊断 |
|---|---:|
| G1：四个 joint objective 主对照 | 4 |
| G1：Norm / DocMean / NormDocMean | 3 |
| G1：Paired / QPolicy / DPolicy / Cal / Binary | 5 |
| G1：full-corpus 动态检索 | 1 |
| G2：三个直接训练 + 三个共同 CL 初始化 | 6 |
| G3：CL / AnsRL / RetRL | 3 |
| **核心合计** | **22** |
| 可选：Anneal + FixedSmall（成对） | +2 |
| 可选：Gaussian mismatch + MRR | +2 |
| **核心 + 探索对照 / 全部已注册训练** | **24 / 26** |

Suite 共 28 行：22 个核心训练、4 个可选训练、2 个 E0 评测。Own/G 尚未注册，不计入上述预算。
训练执行数不等于 GPU-hour；原始 checkpoint 评测、完整检索评测、索引编码、generator 调用、
paired/product 时间诊断和 smoke 单列成本。共享 CL 前缀只计算一次。

预算不足时先取消可选扩展；探索对照应成对取消。保留 LL/RL、完整 2×2、核心 product/role 证据、
G2 初始化问题与 G3 答案反馈。若仍不足，显式缩小论文主张与整组范围，不只删掉不利结果或关键对照。

## 6. Runner 与论文同步清单

本版使用 G1/G2/G3 命名以避免复用旧 ID 混淆；旧 C1–C4 是 E2Rank/E0 方案，不能直接
改名当作 G1 结果；旧 A9/A6/R2 分别对应新的 Paired/Cal/MRR 概念，但需适配新标签与数据。
旧 A3 query-only exploration 对应现在的 `G1-A-QPolicy`：共享 encoder 继续更新，只移除
document policy action。G1 full-corpus 动态检索映射为独立的 `G1-DR`。
G3 监督对照为 CL，不运行 RAG LambdaLoss。

旧 `posttrain_*.sh` 已退役；当前运行入口为 `scripts/experiment.py`，以 `check RUN` 核实实现就绪状态。
新增 logical IDs 需要显式 config/runner mapping；不要使用旧 `all` 模式代替新核心集合。
保证 checkpoint 复用、训练数和实际预算可追踪，旧运行目录不覆盖。

下一次正文同步需修改：整体定位、三组实验结构、G1/G2 teacher-grade 监督及 CL 正例身份、
base 初始化与共同 warm-up、MTEB 辅助/外部评测角色、RAG 答案指标，以及对应表格。
未经测量的结果保持 TBD；full-FT 不能通过禁用 adapter 得到初始参考策略。


### 配置入口（2026-09-11 已建立）

三组 logical IDs 已映射到 `configs/experiments/iclr2027/suite.yaml`，使用
`scripts/experiments/iclr2027.py list/resolve/check/launch`。详见
[运行说明](../configs/experiments/iclr2027/README.md)。`resolve` 可无 GPU 展开，
`check` 明确列出数据、预算及实现缺项；正式运行是否就绪以其输出为准。


ReasonRank 当前产物为 `embedding_candidates_v2`，保存完整多正例。Loader 兼容旧 v1
供历史复现，但不能将旧产物重命名为 v2 恢复已删除的正例；须从原始 parquet 重新生成。
E2Rank 的 teacher-ranking 输入属于第二组实验，继续支持。
