# 方法设计审查：先核清表示协议，再判断 RL 的价值

审查日期：2026-09-16；结果快照：2026-09-15；代码基线：`934ab1d`。核对了当前论文正文/附录、G1/G2/G3 配置、主训练与奖励实现、数据准备和加载、评测入口，以及本地 75 行 G1 结果和已有梯度诊断。本次新增本地 CPU 数值检查，没有启动训练，没有修改训练配置或历史结果。

**结论：基础 policy-gradient 构造有成立的数学依据，但“精确计算排序奖励，就更直接地优化最终检索”这个核心动机缺少充分支撑。当前同时存在一个已确认的表示协议偏差、若干削弱训练信号的数据/候选细节，以及高维 score-function 估计的效率问题。不能把所有负结果归结为 seed，也不能据此认定整条 RL 路线无效。**

证据与复现：[evidence.json](method_audit/evidence.json)、[reproduce.py](method_audit/reproduce.py)。所有本次数值均来自这里；历史日志结论另行注明。

## 1. 结果现在支持什么

从本地 CSV 重算，单位为 BRIGHT nDCG@10 × 100，SD 为三个 seed 的样本标准差。

| 方法 | 均值 ± SD | 证据范围 |
|---|---:|---|
| E0 | 15.07 | 原有评测协议 |
| CL | 17.35 ± 0.35 | 三 seed |
| MRR32 | 17.71 ± 0.76 | 新独立 rollout RNG 协议，三 seed |
| MRR64 | 18.89 ± 0.41 | 三 seed |
| Graded nDCG64 | 18.85 ± 0.52 | 三 seed |
| Graded LambdaLoss-Scaled | 19.50 | 仅 seed 42 |

因此，RL 的 G64 配方相对普通 CL 的增益有三 seed 支持；相对最强监督方法的优势没有建立。历史 22.01 是开发过程中真实出现的单次高点，不是稳定效果。G32→64 在两个 reward 下均提高约 1.2 分，支持估计噪声是一个实际瓶颈，但不是全部原因。[最新稳定性结果](../docs/g1_stability_overnight_results.md)

## 2. 已确认的实现细节：Embedding 路径的末尾 token 重复

### 2.1 当前 Embedding 路径确实产生两个末尾 token

[模型配置](../configs/model/qwen3_embedding_0.6b.yaml:7) 声明 `append_token: pad`；[collator](../src/embedding_data.py:733) 和 [评测 tokenize](../eval_mteb/qwen3_embedding_model.py:78) 手动追加后，仍以默认 `add_special_tokens=True` 调用 tokenizer。

本地 Qwen3-Embedding-0.6B tokenizer 的 `TemplateProcessing` 已自动追加 token `151643`：

```text
官方输入方式：test               → [1944, 151643]
当前项目方式：test<|endoftext|>  → [1944, 151643, 151643]
```

两个末尾 token 的 attention mask 都是 1；最后一个是实际参与计算的读出位置。官方示例直接 tokenize 文本，依赖 tokenizer 自动追加。[Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)

**后续补充核查：这个结论不适用于项目中的普通 `Qwen/Qwen3-0.6B` tokenizer。** 该 tokenizer 只有 `ByteLevel` 后处理，原生 `test → [1944]`，手动追加后为 `[1944, 151643]`，没有重复；但长文本截断会把手动末尾裁掉。Embedding 版有 `TemplateProcessing`，自动末尾会在截断后保留。因此不能把两个模型的 `append_token` 同时改成 `none`，也不能根据相同 YAML 字段断言表示协议相同。固定 revision 的复现见 [compare_qwen_tokenizers.py](method_audit/compare_qwen_tokenizers.py) 和 [tokenizer_comparison.json](method_audit/tokenizer_comparison.json)。

用本地原始模型、CPU fp32、官方示例的两 query/两 document 复核：

| 相似度 | 官方单末尾 token | 项目双末尾 token |
|---|---:|---:|
| China query / Beijing document | 0.764557 | 0.746212 |
| Gravity query / gravity document | 0.599955 | 0.558238 |

同一文本在两种协议下的 embedding cosine 为 0.978856–0.992846。单 token 路径复现官方示例分数，双 token 路径改变输出。这确认了**表示协议偏差及其数值影响**；四条示例不能证明修复后 BRIGHT 必然上涨。

### 2.2 为什么它值得优先处理

- 当前所谓 E0 是“使用项目双末尾协议的原始权重”，不能直接当作官方原生协议的 E0。
- 训练与评测都重复追加，避免了某一种简单的不一致，但不代表协议与预训练匹配。
- 截断会移除手动末尾 token，自动末尾 token 仍保留，因而长短输入可能使用不同数量的末尾 token。本地 G1 有 310/4,963 条 query 超过当前 512-token 训练限长，约 6.25%。
- 两份本地缓存的 tokenizer 内容相同；远端各历史 checkpoint 保存的 tokenizer 尚未逐一核对，不能把本地复现无条件推广到所有历史运行。

**处理顺序：先核对真实 checkpoint 的 token IDs，再独立评测原始 E0 的原生/历史协议；若采用修订协议，相关方法从共同初始化重新比较。** 对历史微调模型换 token 只能算协议敏感性评测。新评测使用独立结果目录，避免复用旧缓存。不要只改 RL 或只改评测后与旧表直接比较。

具体的新协议建议与基础模型/Embedding 的训练差异，见 [后续方法改进设计](METHOD_REDESIGN.md#1-先把两类-qwen-模型的差别说清楚)。本报告上述双末尾 E0 数值只针对 Embedding 分支。

## 3. 核心方法缺口：精确奖励不等于部署目标

### 3.1 训练与部署之间有三处变化

当前训练实际优化的是：

> 在一个固定小候选池上，对随机 query/document 向量的排序奖励取期望；额外负例是经均值缩放的确定性向量，并且停止梯度。

最终评价的是：

> 使用确定性 encoder，在一个领域的完整 corpus 上检索，以该测试任务的相关性标签计算指标。

这里分别改变了 **向量分布、候选分布和部分监督语义**。使用同名 nDCG 并没有消除这些差异。MRR 还增加了与测试 nDCG 的指标差异。[方法目标](iclr2027/sections/method.tex:16)

在当前 `d=1024, alignment=0.90` 下，κ≈4846.29，且严格有：

\[
E\|e-h\|^2=2(1-0.9)=0.2.
\]

RMS 距离约 0.447；双侧采样的期望分数为 `0.81 × deterministic cosine`。这不是可以自动忽略的扰动。统一的期望分数缩放本身不改变排序，真正的区别来自随机波动及其边界翻转；均值排序与随机排序的平均奖励没有一般等价关系。

### 3.2 frozen-candidate rescaling 只匹配均值

论文已正确承认 `R(E[s]) != E[R(s)]`。因此该校准不能证明完全采样奖励的无偏性。停止跨 query 梯度也使更新只对“本 batch 中固定其他向量”的条件目标成立，而非整个随 encoder 刷新的 batch 目标的完整梯度。这是可用的训练近似，但理论主张必须保留这个条件。[校准定义](iclr2027/sections/method.tex:174)

### 3.3 最缺的是一条目标对应关系的实测

在相同 query 和固定 checkpoint 下并列测量：

1. sampled 小候选池 reward；
2. 相同候选、相同标签下的 deterministic reward，使用未经校准缩放的 cosine；
3. 扩大候选池/全库后的 deterministic 排名、Recall@K 与 nDCG。

按 query 检查各级改善是否传递，并看进入全库 top-K 的错误文档是否在训练中出现过。已有训练 reward 与 BRIGHT 曲线不能代替这个检查，因为 batch、模型、候选和标签同时变化。

若 1 上涨、2 不涨，优先处理随机目标与部署的差异；若 2 上涨、3 不涨，优先处理候选覆盖和泛化。现在尚未把这两种失败分开。

## 4. 高维动作估计：数学正确，仍可能很低效

### 4.1 没有证据要求推翻基础公式

当前实现正确区分 query/document marginals；LOO 的 `G/(G-1)` 修正、detached action、log-density 对 live mean 求导以及有效文档 log-prob 求和，在所声明的条件目标下是合理的。

本次用现有 sampler、log-prob 和 LOO，在可解析线性 reward 上做 4,096 组 × G32 的数值检查：采样 alignment=0.900067；估计梯度与解析梯度 cosine=0.999945，相对误差约 1.07%。这是局部估计器检查，不是整个训练系统的正确性证明。

每份 rollout 只用一次更新，当前没有“缺 PPO clipping 所以算法错误”的结论。对完整文档组求和也符合联合密度，不能把它简单改成平均并继续称为同一目标的无偏梯度。

### 4.2 G² 奖励组合没有提供 G² 个独立方向

G64 的 4,096 个 reward 组合只来自 64 个 query action 和 64 个文档组 action。每个文档组约含 17 篇、每篇 1024 维，却共享组级 advantage。一个关键排序变化会同时加权许多与它无关的随机坐标；这些坐标只在期望中抵消。

已有固定状态重分析显示 G32→64 的噪声方差通常约减半；当前单位方向参考的 counterfactual baseline 则常增加方差。它们支持继续研究估计器结构，不支持把“无偏”视为“低方差”。[梯度重分析](../docs/rollout_gradient_reanalysis.md)

高维策略的 baseline 本来就是有条件的方差控制问题，既有研究也不保证任意 action-dependent baseline 在实证中都有收益。[Factorized baselines](https://arxiv.org/abs/1803.07246)、[The Mirage of Action-Dependent Baselines](https://research.google/pubs/the-mirage-of-action-dependent-baselines-in-reinforcement-learning/)

### 4.3 一个此前未利用的结构：去掉奖励看不到的方向

这是本次提出并做了 toy 验证的候选方法，不是已经验证的 G1 修复。

对固定的一组文档和 query mean `h`，奖励只观察 query action 与文档的内积。令 `S` 为 `h` 与这些文档（含 frozen candidates）张成的线性空间，`Π_S` 为正交投影。vMF 密度仅依赖 `hᵀe`，在 S 的正交补上有反射对称性，因此：

\[
E[e\mid\Pi_S e]=\Pi_S e,\qquad
E[R(e)e]=E[R(e)\Pi_S e].
\]

即可以在 score-function 梯度中解析去掉奖励无法观察的正交噪声，保留原采样分布与 reward。这里处理的是梯度估计，不是把部署 embedding 改成低维。

在本次 `d=1024 / G64 / 256 repeats` 的两文档 MRR toy 中，奖励只有一个有效切向方向。投影前后该方向的梯度逐组相同，梯度噪声方差由 1516.94 降到 4.24，比例 0.00280。这只说明结构性机会存在；不能把 toy 的比例当作真实 encoder 或 BRIGHT 收益。

对当前 product 实现，投影必须保留每个 `(query action, document action)` 的条件：query 侧每个 reward cell 只涉及最多 35 篇文档，相关空间至多 36 维（包含 h）；document 侧在固定 query action 下只需该 query 与本 document mean 张成的空间。若先把全部 G 组采样文档合成一个大空间，它可能覆盖 1024 维，失去意义。

**实现前必须推导逐 cell 的 LOO 形式，detach 投影算子，并验证完整共享 encoder 梯度均值、方差与两侧协方差。** 各 cell 的条件不同，不能仅凭逐项投影就宣称总梯度方差一定下降。该方向与已经得到负结果的“用单位均值替换文档作为反事实 reward baseline”不同。

后续已完成逐 cell 推导，以及共享归一化线性 encoder 的联合采样数值验证；真实 Transformer 梯度验证仍待完成。详见 [改进设计](METHOD_REDESIGN.md) 和 [action_design_evidence.json](method_audit/action_design_evidence.json)。

## 5. 监督和候选：当前信息远没有表面上丰富

### 5.1 多正例数据与 MRR 的目标不完全对应

当前 artifact 有 4,963 条记录，其中 **3,836 条（77.29%）有多个正例**，共有 15,830 次正例出现。MRR 只看第一个正例名次：第一个正例保持第 1 名时，把其他正例从第 100 名移到第 2 名，也不会改变 MRR。

这解释了它为什么可能丢掉与 multi-positive nDCG/Recall 相关的训练信号。但稳定性实验里 binary nDCG 没有稳定获胜，所以不能据此直接宣布“换 nDCG 就能解决”。需要在候选和估计器都对齐时判断。

### 5.2 Teacher grades 是另一种监督，不是 binary labels 的细化

本次直接统计当前 ready 文件：

| 现象 | 数量 / 比例 |
|---|---:|
| teacher 第 1 名属于 binary 负例的记录 | 1,359 / 4,963，27.38% |
| binary 正例被赋 teacher grade 0 | 2,063 / 15,830，13.03% |
| binary 负例被赋非零 teacher grade | 34,542 / 69,477，49.72% |
| teacher 把 binary 负例排到某正例之前的正负对 | 32,339 / 207,558，15.58% |

最后两项不自动代表错误标注：ordinal ranking 和是否相关不是同一问题。尤其每个足够长的 slate 固定给十篇文档非零 grade，本身会给很多 binary 负例非零收益。ReasonRank 原作也分别构造 pointwise binary 与 listwise teacher labels，不能把两者当作一个 gold label 的两种编码。[ReasonRank 原文](https://arxiv.org/html/2508.07050v1)

因此，“graded 比 binary reward 更密”与“获得了更准确监督”必须分开。Binary MRR RL 与 graded LambdaLoss 比较可回答实际方法优劣；若声称差异来自 RL 估计器，仍需同标签、相近目标的对照。

### 5.3 候选池小，而且 in-batch 同伴跨 epoch 固定

训练自有候选平均 17.19 篇，最多 20 篇；microbatch=16 只增加最多 15 个代表正例作为负例，并非 global batch=128 个候选。掩码后更少，跨设备梯度平均不会扩大检索候选池。

[dataset](../src/embedding_data.py:449) 在构建时固定 16 条一组；[sampler](../src/embedding_data.py:594) 每个 epoch 只打乱整组顺序。本次实际检查 epoch 0/1 的组集合完全相同。对同一个 query，跨 epoch 仍是相同的最多 15 个跨 query 负例；向量会随模型刷新，文档身份不刷新。

此外，尾部不足 16 的记录在 dataset 构建时永久丢弃，本地实际是 **4,896 条，丢了 67 条**；sustainable_living 从 54 条变成 48 条。改变 training seed 还会改变被丢弃的身份和固定分组，方差并不只来自顺序或 rollout。

这些是所有共享 loader 方法的限制，不足以单独解释 RL−CL，但削弱了“全量训练”和“充足 in-batch negatives”的描述。候选刷新比继续增加同一小池上的 reward 组合数更值得做对照。若加入模型挖掘负例，必须处理未标注文档中的假负例，并让监督方法使用相同候选。

### 5.4 真实文本长度需要纳入解释

当前全部 85,307 次训练 document 出现，在该 tokenizer/手动末尾协议下的长度中位数为 173，最大只有 516，没有一篇触及 `d_max_len=1024`。所以单纯提高 d_max_len 不会增加当前训练文档的信息。它们与实际完整 corpus 文本是否一致，仍需按 document ID 对照；当前不能凭长度直接断言原始数据被如何截断。

训练 query 有 310 条超过 512，最大 6,271；需要检查被截去的是否是问题条件或代码。G1 训练的九个 source 均落到通用 instruction，而 BRIGHT 使用领域 instruction，也是需要显式记录的迁移差异。

## 6. 其他会影响解释或复现的细节

### 6.1 监督基线与 RL 的排序精度没有完全对齐

[BaselineModel](../src/train_baseline.py:242) 直接对模型 dtype 的向量 matmul；RL 的 score table 显式转换并归一化为 fp32。CPU bf16 输入检查得到 baseline score 为 bf16，RL 在关闭 autocast 时为 fp32。near-tie 的排序或 LambdaLoss 权重因此可能受不同精度影响。

RL 的 `.float()` 也没有显式关闭外层 autocast；CPU autocast 会将其 einsum 降回 bf16。本次没有验证远端实际 DeepSpeed 的 autocast 状态，因此这是条件性问题，不能说历史 reward 已确认以 bf16 排序。后续应打印实际 score dtype，并统一排序、归一化与 loss 的 fp32 区域。

### 6.2 BGE-M3 的选样 RNG 未受全局 seed 控制

[EmbeddingDataset](../src/embedding_data.py:238) 创建未显式 seed 的独立 `random.Random()`，并把它传给 BGE-M3 的正负例选择。`set_seed(42)` 不会控制这个独立实例；不同 run 的同 seed、同数据，不能保证取到相同 slate，恢复训练也没有保存这个 RNG 的状态。

这是 G2 BGE-M3 配对实验的实质复现缺口；G1 ready 数据和 E2Rank 固定 listwise 输入不走该选样分支，不能用它解释 G1 低分。应按 run/data seed、record ID、epoch 等设计确定性选样或冻结实际候选 artifact。

### 6.3 其他已知偏差不能再被遗忘

- `max_grad_norm=1` 不等于 DeepSpeed 真正启用了 clipping；已有审计推断当前零裁剪配置，不能继续按“统一裁剪到 1”解释梯度。[已完成的裁剪核查](G1_GRADIENT_ANALYSIS.md)
- 附录把 InfoNCE reward 写成同一 multi-positive CL 的负值；当前 `reward_type=infonce` 实际只选一个最大 label 文档，并返回带温度尺度的值。新增直接辅助 InfoNCE 不会自动修复旧 reward 的定义。[现有说明](../docs/aux_infonce.md)
- Full FT 不保留独立 reference encoder，当前主训练也没有 representation retention 约束。这是设计选择，不能直接叫 bug；但需要外部检索保留度评测，才能区分定向适配和通用能力损失。

## 7. 实验论证：多 seed 解决不了测试集开发问题

当前 suite 明确记录了根据 BRIGHT 做配置开发；本地已有 75 行结果，包含多轮 reward、alignment、G 和更新规则探索。论文却仍写“external benchmarks do not select configurations”。这与实际过程不符。[正文](iclr2027/sections/experiments.tex:23)、[suite](../configs/experiments/iclr2027/suite.yaml)

**新 seed 可以验证固定配方的训练随机性，不能把已反复用于选配方的 BRIGHT 重新变成独立测试集。** 应把这些结果标为开发证据，冻结候选方法，再用未用于该轮开发的独立任务或 query 集确认泛化。若无法取得独立确认集，就收窄结论并如实报告选择过程。

其他论证缺口：

1. 为最强 LambdaLoss 补齐重复，不能继续只对 RL 搜索并和单次或未校准监督基线比较。
2. 同样 LR/steps 不代表对每种目标都合理，也不代表相同算力；先报告受控固定配方结果，再明确区分有限预算调优结果。
3. G1 的监督标签本来能用于强排序 surrogate，RL 要证明额外估计成本带来的收益。仅以 reward 不可微作为动机不足以建立 superiority。
4. G2 把在成熟 E0、小数据上开发的 G32 配方直接用于 B0/W0/更大数据，只能检验统一配方迁移，不能据 B0 不成功推出“RL 不能训练 embedding”。本地没有 G2 完整数值，不能将 G1 原因直接当作 G2 的诊断。
5. G3 对答案效用的优化更能体现独特价值，但尚无完整本地结果。CL 离线候选与 RL 动态全库检索的比较同时改变数据访问方式；若要归因于 reward，应补充共享候选的监督控制。还需记录真正进入 2048-token generator context 的文档，而不只是 retrieved top-10，以及不同 rollout 实际产生多少不同答案/奖励。

## 8. 论文必须同步的事实

当前 LaTeX 明显落后于代码和记录：摘要仍写实验全部 pending；主方法仍以 graded nDCG 为默认；附录仍以 κ=755 为主配方；实验仍写单 seed、没有 benchmark 调参；G2 仍写 teacher rank-1 正例、旧 LL 矩阵；G3 仍写 dev F1 选模，而当前配置使用完整 train、固定 final checkpoint。

这些不是低分的物理原因，但会让读者审查的“方法”与实际运行的对象不同。应先确定最终要报告的协议，再统一重写方法、实验和结果，避免继续在旧设计上追加零散解释。

## 9. 建议的下一步顺序

| 顺序 | 要回答的问题 | 最小工作 | 完成判据 |
|---|---|---|---|
| 1 | 目前比较的是正确的表示协议吗？ | 检查实际 tokenizer；E0 原生/历史协议独立评测；确认真实 score dtype | 输入 token、读出位置、score 精度与评测目录可追溯 |
| 2 | 训练 reward 的改善在哪一级失效？ | 现有 E0/CL/RL checkpoint 上做 sampled-slate → deterministic-slate → larger-pool/full-corpus 诊断 | 区分随机目标失配与候选迁移失败 |
| 3 | 高维噪声能否在不换 reward 下减少？ | 条件投影候选的逐项推导和固定真实 encoder 梯度验证 | 均值与原估计一致到采样误差范围，完整梯度方差/耗时确实改善 |
| 4 | 辅助确定性目标有无必要？ | 复用已实现的 RL+InfoNCE，测辅助/策略梯度的范数、夹角；必要时做少量配对训练 | 分清是辅助监督单独贡献，还是组合相对纯 CL/LL 提供额外收益 |
| 5 | 最终方法是否值得用？ | 冻结协议后，对 CL、强 LL 和候选 RL 做匹配重复与独立确认 | 同时报告效果、波动、成本和能力保留 |

第 4 步的系数 0.1 只是现有初值；loss 标量的比例不代表梯度贡献的比例。它是一个明确改变目标的混合方法，不能作为无偏原始 policy-gradient 的实现修复。

**我不建议现在就把整个项目推倒重来，也不建议继续围绕 22.01 扩大扫参。** 优先解决可核实的协议偏差，并把“目标有没有对齐”和“梯度有没有被有效估计”分开。若修订后 RL 仍不能超过匹配的 LambdaLoss，论文就应把贡献放在奖励接口、估计效率和真正的下游答案反馈，而不是承诺 RL 普遍优于监督排序。

## 10. 本次验证与边界

执行：

```bash
.venv/bin/python paper/method_audit/reproduce.py --model-probe
```

核对了全部本地 ready 记录、token 长度与 tokenizer 后处理、真实 sampler 分组、结果 CSV；用缓存的 0.6B 模型验证协议敏感性；用现有 vMF/LOO 实现检查解析梯度；用两文档 MRR toy 检查条件投影机会。

本次没有访问远端训练 checkpoint，没有运行完整 BRIGHT 或新训练。对历史运行的推断以已有日志为限；toy 不能替代真实 encoder 和多 seed 的训练证据。新文件只保存审查报告、复现代码和聚合数值，不包含新增训练样本文本副本。
