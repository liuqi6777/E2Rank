# Experiment Plan: Reward-Based Optimization of Embedding Retrievers

**修订：2026-09-11。本文档是实验设计的 source of truth。**

本版替代此前以 E2Rank → MTEB 为主、base-LLM 训练仅作可选边界实验的方案。
采用三组实验：①开放权重 embedding model 的 reasoning retrieval 适配；
②从 base LLM 开始的较大规模 embedding 训练；③固定索引的 RAG 优化。
所有结果均待测；本文档不表示对应实现已就绪。主实验统一 **full fine-tuning、单 seed 42**。
本次只修订实验计划，论文正文、表格和 runner 尚须按本版同步。

日常配置入口已收敛到 `configs/experiments.yaml` 和 `scripts/experiment.py`。
G1 对外数据接口为 `id/query/positive/negatives/source`。预处理关联审计元数据并将
候选顺序、标签、去重标识写入 `train.ready.jsonl`；loader 直接读取，batch mask 动态构造。
当前目录为 `data/processed/reasonrank_simple/`，
仍为 4,963 条全量训练；底层 suite 保留作高级配置。

**G1 初始运行参数已定：** E0、full FT、seed 42、AdamW、LR 5e-6、global batch 32、
每卡 microbatch 8、450 optimizer steps、linear schedule、warmup 0.03、weight decay 0.01、
max grad norm 1.0；每 100 步保存，报告最终模型。按 GPU 数自动调整梯度累积。
RL 使用 group size 32、kappa 755；CL/RankNet temperature 0.03。此版未经 dev 调参。
G1 不再将模型 revision 和 manifest/hash 校验作为启动条件；保留必需的结构检查。

**G1 最新决定（2026-09-11）：** 固定 seed 随机选择一个正例、删除其余已知正例，
使用全部 4,963 条清理后的非 MSMARCO 数据训练，不划分 dev。提前固定配置/训练预算，
报告最终 checkpoint；G1 不执行下文通用的 dev 网格调参或最佳 checkpoint 选择。
BRIGHT 仅作最终评测。G2/G3 的开发集协议暂不改变。

## 0. 研究问题与贡献边界

| 组别 | 初始化与数据 | 核心问题 | 主要评测 |
|---|---|---|---|
| G1：Reasoning adaptation（主实验） | 现有开放权重 embedding model；清理后的 ReasonRank | 相同数据下，RL 是否优于监督适配？固定文档表示时是否仍有效？ | BRIGHT；MTEB retrieval 作为能力保持检查 |
| G2：Base-LLM embedding training | 非 embedding 专用的 base LLM；E2Rank listwise，约 156k，实际规模待审计 | RL 能否直接学习检索表示？共同 warm-up 后是否仍有收益？ | 固定的外部检索任务集、held-out E2Rank 与学习曲线 |
| G3：RAG optimization | 预先固定的 embedding checkpoint；独立 QA 数据 | 固定索引时，答案反馈能否改善检索和生成质量？ | Answer EM/F1；检索指标辅助 |

三组共同支持 embedding-space reward optimization，分别覆盖初始化、更新约束和反馈类型。
G1/G3 属于后训练和任务适配；G2 检验更早的表示学习阶段。不能预先宣称 RL 可以替代
对比学习建立初始空间，也不能因为共同对比学习 warm-up 使用了 156k 数据，就声称 RL
本身已在这个规模得到验证。G2 的 RL 阶段必须实际使用并记录较大规模的数据预算。

“约 156k”称为较大规模 listwise 训练，不等同于通用 embedding 预训练规模。
BRIGHT 提升不单独证明学会推理；固定索引是所有 query-only 对照共享的设置。
在线适配仍是潜在应用，不作本实验已验证的贡献。

## 1. 共同实验约束

### 1.1 初始化、参数与版本

- G1 主模型沿用计划中的 `Qwen/Qwen3-Embedding-0.6B`，记为 E0。
  正式运行前固定 checkpoint revision、tokenizer、pooling、prompt 和最大长度。
- G2 使用一个明确的 base LLM，记为 B0；具体 checkpoint 在 Phase 0 固定。
  优先选择规模相近、便于控制架构差异的模型，不能直接把 E0 当作 B0。
  同一组内严格同初始化；不同模型组之间不作只归因于目标函数的比较。
- G3 默认从原始 E0 开始，不自动继承 G1 或 G2 的最佳模型，避免引入额外训练混杂。
- Joint 默认一个共享 encoder，通过 query/document 两种角色更新全部参数；
  最终评测必须用当前 document encoder 重新生成文档向量。
- Query-only 从同一个 E0 创建可训练 query 分支和冻结 document 分支；全部 query
  参数可训练。固定的是文档表示版本，不是声称真实部署中的 corpus 永远不变。
- 主实验不使用 LoRA；第二个 embedding 模型作为泛化扩展，不展开完整消融。

### 1.2 公平对照与预算

每个直接比较内固定初始化、可训练分支、数据与顺序、候选构造、标签来源、tokenization、
长度、microbatch、梯度累积、优化器族、精度和 checkpoint/evaluation schedule。
Joint 与 query-only 的可训练范围不同，分别作 objective 内部对照，再比较更新约束。

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

只用独立开发集选择超参数/checkpoint；先按主 dev 指标，再按更早 step、预先固定的
config 顺序处理并列。各组指标在下文固定。外部最终评测不用于选配置或挑任务。

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

主比较用 seed 42 按 query 独立固定抽取一个原始正例，删除其他已知正例，保留负例。
各方法使用同一份 **single-positive binary relevance**，RL reward 为 nDCG@10。
不把 teacher 第一名当正例；BRIGHT 评测保留完整官方标签。

| Objective | 主监督定义 |
|---|---|
| Single-positive InfoNCE | 对固定抽取正例计算 log-softmax loss；其余有效候选为负例 |
| RankNet | 仅构造已标注正例高于负例的 pairs，不强制正例之间的 teacher 次序 |
| LambdaLoss | LambdaRank variant：pairwise logistic × 当前排序的 `|ΔnDCG@10|`；使用相同 binary gains 与候选，`gain=2^rel-1`、`sigma=1.0` |
| Proposed RL | 同标签与候选上的 exact nDCG@10 |

主协议沿用扩展候选池设计：加入其他记录固定抽取的正例作为 in-batch 候选，
所有 objective 相同。合并同文本/同 ID 项；若当前 query 被删除的已知正例重新出现则 mask 掉，
其余跨 query 候选按负例近似，明确其不完整标注与 false-negative 风险。
跨 query document embeddings detach，文档只通过自身记录得到梯度。

记录 microbatch、world size、accumulation；当前 in-batch 池是 device-local，
梯度累积不扩大候选池。开发集固定候选，不随评测 batch 改变。
Own-query-only 消融检查扩展候选的作用。Mean-score calibration 只在确有冻结候选时评测。
Teacher-order agreement 是可选的独立监督实验，不能与主 binary 比较混为一张受控结果表。

### 2.3 主结果表

新 ID 是实验逻辑 ID，不代表已有 runnable config。

| ID | Objective | 更新范围 |
|---|---|---|
| G1-E0 | 无训练 | 原始模型，评测复用 |
| G1-J-CL / G1-Q-CL | Single-positive InfoNCE | Joint / Query-only |
| G1-J-RN / G1-Q-RN | RankNet | Joint / Query-only |
| G1-J-LL / G1-Q-LL | LambdaLoss | Joint / Query-only |
| G1-J-RL / G1-Q-RL | Proposed nDCG RL | Joint / Query-only |

共 **8 个训练配置**，加原始 E0 评测。每个更新范围都包含监督对照。
G1-J-RL 超过 E0 只证明额外训练有益；超过 CL 但不超过 LL 不能证明优于 metric-aware 监督。

**主结果：** BRIGHT nDCG@10，报告固定官方协议下逐领域与宏平均。
冻结短/长文档设置、query instruction、excluded-document handling 和全 corpus 检索协议；
不能用训练中的短候选列表重排结果代替 BRIGHT retrieval。
**能力保持：** 预先固定的 MTEB retrieval 子集及相对 E0 的变化，作为辅助结果。
**选模：** G1 不使用 dev，不执行目标专属网格调参；提前固定配置和预算，报告最终 checkpoint。
训练 reward/损失仅用于数值诊断，不用 BRIGHT 或 MTEB 挑选 checkpoint/超参数。

### 2.4 主要消融与理论诊断（集中在 G1）

复用 G1-J-RL checkpoint/config，不在 ablation runner 中隐式重训主方法。

| ID | 改动 | 优先级 |
|---|---|---|
| G1-A-Paired | 同 G、相同 component samples，使用 paired/diagonal 而非 product rollout | 核心 |
| G1-A-Cal | 禁用 frozen-candidate mean-score rescaling | 核心 |
| G1-A-MRR | reward 改为 binary MRR@10，相关阈值为 label=1 | 核心 |
| G1-A-QExplore | joint encoder 仍训练，只移除 document exploration | 可选；不同于冻结 document encoder 的 G1-Q-RL |
| G1-A-Own | 不加 in-batch candidates | 次要 |
| G1-A-G | 改变 G 的质量/成本曲线 | 次要 |

Paired 的 G 次 reward 与 product 的 G² 次存在成本差异；报告 example-matched 结果，
另做短 matched-time 比较或质量/成本曲线。G² 个组合相关，不是 G² 独立 joint samples。

固定 checkpoint/minibatches 重复采样，测 paired/product 梯度估计的噪声、相对高采样
参考的偏差及 per-role norms，并记录是否启用 advantage normalization。
另做低成本小探索极限诊断：关闭 advantage normalization，固定 frozen candidates，
比较直接 InfoNCE 梯度与平均 score-function 梯度，分别检查角色再施加一致角色权重。
报告 cosine、relative error 和采样预算；有限样本噪声不等于理论极限失败。

持续记录 raw marginal reward spread、退化 group 比例、每角色梯度、mean alignment、
sampled/frozen score 分布及冻结候选 top-K 占比。归一化后的 advantage std 不能独自说明信号质量。
实践中的 per-component normalization 和 document-role `1/n_valid` 权重不等于未加权联合
目标的无偏梯度；mean-score calibration 匹配一阶分数矩，不是期望 ranking reward。

## 3. G2：从 base LLM 开始的较大规模 embedding 训练

### 3.1 数据、表示与标签

使用原始 E2Rank listwise artifact（约 156k，须核实实际版本、规模和候选长度）。
不默认使用混入 ReasonRank 的 `train_v2`。审计来源和外部评测 overlap 后，按同题簇冻结
train/dev/test manifest；开发/测试比例默认各 5%，以实际组数为准。
现有 loader 分别 shuffle 构造 train/dev 的方式不能保证互补，须改成共享 manifest。

固定 B0 的 pooling、retrieval prompts、归一化与 scoring；这些在所有目标间保持相同。
所有路线均 full FT、joint query/document。B0 直接评测仅作初始化诊断，不假定已有检索能力。

保留完整 teacher order，并为 metric-aware 目标定义 teacher-derived grades：
rank 1 / 2–5 / 6–10 / 其余对应 3 / 2 / 1 / 0；主 RL 优化 nDCG@10。
CL 使用 rank-1 正例和相同候选池；LL 使用与 RL 完全相同的 grades/cutoff。
跨 query 候选构造保持一致，明确其近似负例属性。
CL 与 RL 使用的监督信息粒度不同，因此 **LL 是必需的同标签强对照**。
可选 RankNet 补充 full-order 控制；不能只比较单正例 CL 就归因于 RL 估计器。
此处 grades 是 teacher 目标，不是人工相关性；MRR 若报告，明确 grade ≥2 的定义，
不要沿用 G1 binary relevance 的阈值解释。

### 3.2 两个问题、两组对照

| ID | 路线 | 目的 |
|---|---|---|
| G2-D-CL | B0 → CL | 直接 embedding 训练基线 |
| G2-D-LL | B0 → LambdaLoss | 同 grades 的监督对照 |
| G2-D-RL | B0 → RL | 检验直接从 base 启动 |
| G2-W-CL | B0 → 共同 CL warm-up W0 → 继续 CL | 后续训练对照 |
| G2-W-LL | B0 → 同一个 W0 → LL | 排除切换到 metric-aware 目标本身的收益 |
| G2-W-RL | B0 → 同一个 W0 → RL | 检验已有初始空间后的 RL 收益 |

W0 只训练一次。预先声明总 example/token budget T 与 warm-up budget W（0<W<T）；
所有 warm-up 分支使用相同 W0，后续各用 T-W。直接分支各用 T。
G2-D-CL 在 W 处的 checkpoint 可作为 W0，但必须预先确定配置/预算，不能根据 RL 测试
结果挑 warm-up。若共同 warm-up 后 CL 的配置不变且训练轨迹完全相同，G2-W-CL 直接
复用 G2-D-CL 的后半段；如改变 optimizer/schedule，必须额外运行并计入成本。
明确分支是否重置 optimizer、schedule 和 step counter，并保持 warm-up 后比较一致。

默认 **5 个独立训练执行**：D-CL、D-LL、D-RL，以及从已保存 W0 分叉的 W-LL/W-RL；
W-CL 为 D-CL 后半段复用。六个逻辑结果行不等于六次从头训练。
直接 vs warm-up 比较报告总预算；后续目标比较同时报告 continuation 预算和共同前缀成本。
不能让 warm-up 路线额外多读一轮数据后与直接路线称为等预算。

### 3.3 评测、失败情形与结论

Phase 0 固定一组外部检索任务，可选自 MTEB retrieval；包含任务与 aggregation 必须
在看到目标间效果差异前冻结。报告逐任务结果、宏平均及随处理样本数/时间的学习曲线。
选模采用 held-out E2Rank dev teacher-grade nDCG@10；E2Rank test 报告 teacher-target
nDCG、MRR、pairwise 与 rank-1 accuracy，外部任务检验迁移。BRIGHT 可补充，但不能
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
| G3-LL | LambdaLoss，同检索 relevance/candidate 协议 | 强监督检索对照 |
| G3-RetRL | Retrieval nDCG reward | 与下游答案 reward 区分 |
| G3-AnsRL | Generated-answer F1 reward | 直接下游反馈的核心证据 |

主指标和选模指标为固定 QA dev/test 的 answer F1；EM、Recall@5/20、MRR 为辅助。
固定 generator、prompt、top-k、context budget、decoding、答案归一化和聚合方式。
所有行使用同一套答案评测，不能把 answer-containing passage MRR 当作生成答案 F1。
报告训练 GPU-hours、index searches、generator calls/tokens、cache reuse 与实际 group sizes。

**候选访问公平性须先解决。** 当前监督训练使用离线候选、RL 动态检索，不能直接把差异
归因于目标函数。默认增加 shared-candidate 控制：同一冻结候选 manifest 下比较
CL/LL/RetRL；动态检索版本作为部署结果单列。如改用共同动态 mining，必须在训练前
冻结规则并统一候选刷新预算。额外控制运行单独计费，不隐藏在四个主训练配置中。

若 AnsRL 未完成，只能声称改善 RAG 检索，不能声称已验证直接答案优化。
固定索引无需重编码的优势属于本组所有适配方法，不是 RL 独有。

## 5. 执行顺序与预算

### Phase 0 — 数据和实现前置条件

- [x] G1 保守隔离、单正例抽取、去重和全量 train manifest（4,963 条）。
- [x] G1 空字符串补齐和有效候选 mask 的文本层 helper。
- [x] G1 候选 mask 接入 collator、有效文档编码、CL/RankNet 和 RL reward/log-prob/KL；CPU 数值检查通过。
- [ ] G1 最终同题复核。
- [ ] G2 E2Rank 版本与规模、外部 overlap 审计、唯一 train/dev/test manifest。
- [ ] 固定 E0、B0、pooling/prompts 和各组评测协议；校验 full FT 与冻结分支。
- [x] 接入 single-positive CL、binary RankNet 和可变长 RL。
- [x] 实现指定 LambdaLoss variant；G1 使用 binary relevance，G2 使用 teacher grades 3/2/1/0，padding 不参与 loss。
- [ ] 明确两套标签的 MRR 阈值，确保 padding 不参与 reward。
- [ ] 修复 split loader；实现固定候选 deterministic dev evaluation 与全 corpus 检索。
- [ ] 固定各组 LR/目标网格、trial 与主训练预算、tie-break 和 G2 的 T/W。
- [ ] G3 QA、index、generator manifests 和候选访问控制。
- [ ] 所有命令 dry-run，检查 resolved configs，再提交 GPU；尚无已完成结果。

### Phase 1 — 短运行与诊断

G1 各 objective 和两种更新范围跑 smoke；核对 frozen document embeddings/index hash。
G2 用 B0 短跑 CL/LL/RL，检查 reward/advantage 信号；只基于训练/dev 冻结预算。
G3 验证固定 generator 的可复现输出和 reward 成本。完成低成本梯度诊断。
这些短运行不取代正式预算，也不根据最终测试分数决定保留哪条路线。

### Phase 2 — G1 主实验与机制

先固定 G1 配置与预算，完成八个训练配置，以最终 checkpoint 做 BRIGHT/保持性评测。
主要消融 Paired、Cal、MRR 复用 G1-J-RL；额外 QExplore/Own/G 按具体问题安排。
约 5k 数据使用预先声明的统一更新预算，不机械沿用大数据一轮默认值；不进行 dev calibration。

### Phase 3 — G2 两组对照

在更大 E2Rank 数据上完成 direct 与 common-warm-up 两组结果。
保存 W0 并明确复用，不把 G2 再降为默认可删的附录边界实验。
保留从 base 直接 RL 不收敛或无收益的结果，用于决定论文贡献边界。

### Phase 4 — G3 下游反馈

完成四个训练配置、原始 E0 与候选访问控制，优先保证 AnsRL 与公平监督对照。
若 GPU/数据准备允许，组间可以调度重叠；各组内部配置选择和共享 checkpoint 仍有依赖。

### 核心预算与删减顺序

| 部分 | 主训练执行数，不含调参 |
|---|---:|
| G1：4 objectives × 2 update settings | 8 |
| G1：Paired、Cal、MRR | 3 |
| G2：按 D-CL/W0/W-CL 复用方案 | 5 |
| G3：CL、LL、RetRL、AnsRL | 4 |
| **合计** | **20** |

另计原始 checkpoint 评测、调参、paired/product 短 matched-time 比较、G3 候选访问
控制、共享 warm-up 的存储与生成器评测开销。20 是训练执行数量，不是完整 GPU-hour
报价，也不意味着每项训练耗时相同。若 G2-W-CL 无法轨迹复用，增加一次 continuation。

预算不足时先删第二模型/第二 QA 数据集、混合数据扩展、额外 G/分布/奖励混合消融。
然后缩减各对照一致的调参预算。保留三组核心问题、同标签强监督基线、paired/product
证据和 RAG 答案 reward。若仍不足，明确缩小论文贡献范围，不以省略关键对照保留主张。

## 6. Runner 与论文同步清单

本版使用 G1/G2/G3 命名以避免复用旧 ID 混淆；旧 C1–C4 是 E2Rank/E0 方案，不能直接
改名当作 G1 结果；旧 A9/A6/R2 分别对应新的 Paired/Cal/MRR 概念，但需适配新标签与数据。
旧 A3 query-only exploration 不等于 G1-Q 的 document encoder 冻结。
旧 RAG RankNet 对照可保留作额外结果，主强监督行计划为 LL。

旧 `posttrain_*.sh` 已退役；当前运行入口为 `scripts/experiment.py`，以 `check RUN` 核实实现就绪状态。
新增 logical IDs 需要显式 config/runner mapping；不要使用旧 `all` 模式代替新核心集合。
保证 checkpoint 复用、训练数和实际预算可追踪，旧运行目录不覆盖。

下一次正文同步需修改：整体定位、三组实验结构、G1 binary 与 G2 teacher-grade 监督、
base 初始化与共同 warm-up、MTEB 辅助/外部评测角色、RAG 答案指标，以及对应表格。
未经测量的结果保持 TBD；full-FT 不能通过禁用 adapter 得到初始参考策略。


### 配置入口（2026-09-11 已建立）

三组 logical IDs 已映射到 `configs/experiments/iclr2027/suite.yaml`，使用
`scripts/experiments/iclr2027.py list/resolve/check/launch`。详见
[运行说明](../configs/experiments/iclr2027/README.md)。`resolve` 可无 GPU 展开，
`check` 明确列出数据、预算及实现缺项；正式运行当前仍未就绪。建立映射不等于完成
Phase 0 的 trainer 改造，尤其不能把 query exploration 配置当成冻结 document branch。


ReasonRank 训练仅接受预处理后的 `embedding_candidates_v1` ready 记录；旧单正例 schema
和运行时格式转换已移除。E2Rank 的 teacher-ranking 输入属于第二组实验，继续支持。
