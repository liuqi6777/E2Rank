# 条件投影、大负例池与逐 pair 辅助 reward

日期：2026-09-17。状态：讨论记录与候选设计；逐 pair 辅助估计器尚未实现，未启动新的训练或梯度方差实验。

## 1. 当前结论与选择

当前问题是：普通池的 query 条件投影空间只有几十维；跨卡大池可能张满 Qwen3-Embedding-0.6B 的 1024 维空间，使 query 侧 CP 退化为 SF，而此前 SF 的随机梯度噪声正是需要解决的问题。

用户反馈：扩大负例池后没有观察到明显收益，日志中的 query span rank 非常接近 1024。本记录没有读取或审计服务器上的对应实验输出，因此这些是用户报告的观测，不是新增的实验结论。

目前倾向的候选设计是：**小池 listwise 主 reward + 大池逐 pair 辅助 reward，分别计算 LOO、分别投影，再合并梯度。** 主项仍优化排序指标，辅助项增加负例覆盖，两者都采用随机动作上的策略梯度，不引入直接监督辅助 loss。

已经作出的讨论选择：

- 暂不运行小池/大池 × SF/CP 的固定状态完整梯度方差对比。
- 分数空间扰动与条件积分留作后续工作。
- 小池 CP + 大池 InfoNCE 是已有实现的替代方案，但用户当前更倾向纯 RL 的方法定位。
- 不继续把 triplet margin 作为直接辅助 loss 的优先方向；若使用直接辅助 loss，优先复用现有 InfoNCE。
- 全池差值修正虽然有无偏推导，但当前认为实现和验证成本偏高，不作为本次优先方向。

这里的“倾向”不代表训练有效性已经确认，也不代表 pair 选择、文档梯度范围及超参数已经定案。

## 2. 现有实现与候选数量

### 2.1 Query 与 document 投影不同

固定某个 document bundle，设 query 单位均值为 h，所有参与 reward 的有效文档方向为 d₁,…,d_M，包含自有采样文档和固定跨 query 文档。当前 query 侧使用：

\[
S=\operatorname{span}(h,d_1,\ldots,d_M),\qquad
r=\operatorname{rank}[h,d_1,\ldots,d_M].
\]

因此：

\[
r\le\min(D,M+1).
\]

因为均值归一化会在反传中消除 h 的径向方向，保留的切向空间维数为 r−1，被删除的方向数为 D−r。r=D 时 query 投影为恒等映射；r 接近 D 时，可删除方向很少。

Document 侧每个 reward cell 使用 span(document_mean, sampled_query)，最多 2 维，反传后最多保留一个切向方向。扩池不会使每个 document 项的投影空间一起满秩。

所以，“大池导致 CP 失效”应精确表述为：**大池使 query 侧通过投影删除无关方向的作用消失或接近消失；document 侧投影仍然存在。**

实际代码统计的是 SVD 数值秩；重复、padding 和近线性相关方向可能降低 rank。语义相似不等于线性相关，独立连续 vMF 采样也可能使相似文档动作在数值上变得独立。增加候选不乘 rollout 数量：当前是每个 document bundle 单独投影，而非把全部 bundles 合成一个空间。

实现与说明：[conditional_projection.py](../src/conditional_projection.py)、[梯度估计器](gradient_estimators.md)。

### 2.2 当前 G1-R2 普通池与大池

配置为每卡 microbatch 16、八卡 global batch 128。普通池使用自有候选，以及同卡其他 query 的代表正例；大池使用跨卡其他 query 的全部候选。

| 范围 | Query 的 reward 候选 | 过滤前数量 |
|---|---|---|
| 普通池 | 自有候选 + 其他 15 个同卡 query 的代表正例 | 自有候选数 + 15 |
| 同卡全候选池 | 自有候选 + 其他 15 个同卡 query 的全部候选 | 当前数据最多 320 |
| 八卡大池 | 自有候选 + 其他 127 个 query 的全部候选 | 当前数据最多 2560 |

本地 `data/processed/reasonrank_multi/train.ready.jsonl` 在本次核对中包含 4963 条记录，每条候选数为 2–20，平均 17.1886。普通池过滤前平均约 32.19、最多 35 篇，query span rank 最多 36。这个统计只描述本地文件，不替代服务器实际 batch 的计数。

“大池 1024”不是候选数的固定设置。大池候选可能超过 1024，但 span rank 受 embedding 维度限制，最多为 1024。应分别检查：

- `projection/query_span_rank_mean/max`：子空间数值秩。
- `reward_pool/cross_candidates_mean/max`：过滤后的跨 query 候选数，不含自有候选。

`slate_size: 20` 是现有 G1 配置中的 legacy parser 占位，不能据此额外截断 prepared 候选；上述最多 20 是本地文件的实际统计。

来源：[普通池配置](../configs/experiments/iclr2027/suite_r2.yaml)、[大池配置](../configs/experiments/iclr2027/suite_g1_rl_large_pool.yaml)、[大池说明](g1_r2_large_pool.md)。

### 2.3 现有观测能支持什么

rank 接近 D，支持“扩池削弱 query 侧 CP 几何降噪空间”的解释。但不能仅凭 rank 推出完整模型梯度方差比例，也不能认定它是检索指标没有提升的唯一原因。

原因是剩余方向上的噪声能量、encoder Jacobian，以及 query/document、不同 cells 的协方差都会影响总方差。即使只删除少数方向，这些方向也可能承载较多噪声；相反，删除很多方向也不等于按维数比例降低完整梯度方差。

## 3. 为什么 reward 只计 top-K，投影仍不能直接只留 top-K

当前 nDCG/MRR 的 top-K 从拼接后的全候选分数中选出，而后才只给前 K 名计分。见 [rewards.py](../src/rewards.py)。

本节 d_m 表示实际用于内积评分的向量；若固定文档分数有 rescaling，其系数吸收进 d_m。投影 span 不因非零标量缩放改变，但 top-K 边界不等式必须使用真实评分向量。

### 3.1 原始梯度与全候选 CP

固定文档、标签和 κ，令 q∼vMF(h,κ)，J_h=∂h/∂θ。归一化使 J_hᵀh=0，因此 query 梯度为：

\[
g_q=\kappa J_h^T E[R(q)q].
\]

LOO baseline b 来自其他独立 query draws，因 E[q] 平行于 h，有 J_hᵀE[bq]=0。

全候选空间 S 包含 h 和全部 reward 分数方向。给定 x=Π_Sq，全部分数和 reward 已确定；正交补上的 vMF 密度具有反射对称性，所以：

\[
E[q\mid\Pi_Sq]=\Pi_Sq,\qquad E[Aq]=E[A\Pi_Sq].
\]

这些期望等式是精确线性代数下的论证；数值实现还使用数值秩阈值。

### 3.2 按当前 action 选择 top-K 空间会引入选择约束

令 I(q) 为本次 top-K 的有序文档身份，S_I=span(h,{d_m:m∈I})，P_I=Π_{S_I}。直接替换后的偏差是：

\[
E[\widehat g_K]-g_q
=-\kappa J_h^TE[A(I-P_{I(q)})q].
\]

这个期望通常不为零。P_I 依赖当前 q，因此即便 b 与 q 独立，J_hᵀE[bP_Iq] 也未必为零，LOO 不会自动修复偏差。

只看 top-1：R(q)=1[qᵀd₊>max_n qᵀd_n]。虽然只有第一名计分，“它是第一名”仍要求 qᵀ(d₊−d_n)>0 对所有 negatives 成立。

更一般地，固定 T=(I,x)，x=P_Iq，写 q=x+u、u∈S_I⊥、||u||=ρ=√(1−||x||²)。设第 K 名分数为 t_K，每个未入选文档都要求：

\[
((I-P_I)d_n)^Tu\le t_K-d_n^Tx.
\]

给定 T 后，剩余方向在满足这些不等式的球面区域 Ω 上均匀分布，而不是在整个剩余球面上均匀分布。因此正确的条件均值是：

\[
E[q\mid T]=x+E[u\mid u\in\Omega].
\]

Reward 是 T 的函数，所以 E[Rq]=E[R E[q|T]] 始终成立；困难在于 Ω 一般不对称，其剩余均值不为零。直接 top-K 投影遗漏的就是这一项。

### 3.3 与当前 LOO 对应的解析反例

取 D=3、h=e₃、d₊=e₁、d₋=e₂，κ>0，R(q)=1[q₁>q₂]。由前两个坐标的旋转对称性，E[R]=1/2，且存在 c>0，使 E[Rq₁]=c、E[Rq₂]=−c。

真实切向梯度为 κc(e₁−e₂)。按获胜文档选择 top-1 空间时：

\[
P_Iq=\begin{cases}(q_1,0,q_3),&R=1,\\(0,q_2,q_3),&R=0.\end{cases}
\]

LOO baseline 与当前 action 独立，且期望为 1/2。直接投影后的梯度期望为 κc(e₁−e₂)/2，即真实梯度的一半。

本次对话用确定性数值积分核对 κ=5：原始 LOO 两个切向坐标为 (0.579757,−0.579757)，top-1 投影后为 (0.289878,−0.289878)。这是解析反例的数值核对，不是真实 encoder 梯度方差实验。一般几何下偏差也可能改变方向，不能统一乘常数修复。

### 3.4 严格安全条件及其局限

若对所有 n∉I 都有：

\[
t_K-d_n^Tx\ge\rho\|(I-P_I)d_n\|,
\]

则剩余方向无论怎么变化，未入选文档都无法超过第 K 名。忽略同分退化情形，此时 Ω 是完整剩余球面，E[u|T]=0，可以安全使用 top-K 投影。

可以据此构造“通过时 top-K 投影，否则原始 action/全候选 CP”的无偏混合估计器，但不能保证始终低 rank。

对大池、小 K、靠近边界的 hard negatives，预期通过率偏低；这只是几何判断，没有测量通过率。在 L 维完整剩余球面上，文档剩余分数的标准差为 ρ||(I−P_I)d_n||/√L，安全条件抵御的最大变化是其 √L 倍。L≈1000 时约为 32 倍，检查很保守。

## 4. 优先候选：小池主 reward + 大池逐 pair 辅助 reward

### 4.1 优化目标

主项沿用小池 graded nDCG@K；辅助项比较偏好文档 a 与较差文档 b：

\[
R_{ab}=1[s_a>s_b],\qquad
J=E[R_{\mathrm{main}}]+\lambda\frac1{|\mathcal P|}\sum_{(a,b)\in\mathcal P}E[R_{ab}].
\]

同分可按固定规则给 1/2；必须与实现的 tie convention 对齐。该目标是 listwise 与 pairwise 的期望 reward 混合，不是全池 nDCG/MRR 的等价分解。

等权 pair 是初版候选。若使用权重，应由固定标签或其他动作无关信息定义，并按权重总和归一化。不能把依赖本次全池 predicted rank 的权重直接代入低维证明。

### 4.2 Pair 的低维结构

固定 document bundle j。若 pair 两侧都采样，定义 Δ_{j,ab}=e^d_{ja}−e^d_{jb}。则：

\[
R_{ij,ab}=1[(e_i^q)^T\Delta_{j,ab}>0],\qquad
S_{j,ab}=\operatorname{span}(h_q,\Delta_{j,ab}).
\]

每个 pair 的空间最多 2 维，归一化反传后最多一个切向方向。若一侧为固定跨 query 文档，并沿用 frozen-document rescaling，差向量必须使用实际评分系数，例如 Δ=e^d_{ja}−γf_b，而不是忽略 γ。

两个方向虽然相似，只有差向量才是该二值比较真正观察的方向。若改用同时依赖两者绝对分数的 reward，则应重新检查空间，而不能沿用这个维数结论。

### 4.3 逐项 LOO 与 query 梯度

对每个 pair 单独构造：

\[
A^q_{ij,ab}=R_{ij,ab}-\frac1{G_q-1}\sum_{i'\ne i}R_{i'j,ab},
\qquad
A^d_{ij,ab}=R_{ij,ab}-\frac1{G_d-1}\sum_{j'\ne j}R_{ij',ab}.
\]

主项有自己的 LOO。Query 的更新为：

\[
\widehat g_q=\widehat g_{q,\mathrm{main}}
+\frac{\lambda\kappa}{|\mathcal P|G_qG_d}
J_q^T\sum_{ab}\sum_{ij}
A^q_{ij,ab}\Pi_{S_{j,ab}}e_i^q.
\]

固定 j 和 pair 后，投影空间不依赖当前 query draw i，pair reward 只依赖保留的差方向，因此可沿用反射对称性证明每项保持期望。总目标的梯度再由线性性相加。

利用线性性，可先对 i 求加权向量，再为每个 j/pair 投影，不必实体化每个 cell 的 D×D projector。不能先把全部 pair 的差向量合成一个全局空间；该空间又可能满秩。

**不能先合成 total reward，再走现有一次 CP。** 当前实现用 reward 的全部输入方向建统一 query span，这样会重新引入大池满秩问题。新的设计需要按项构造随机系数与 projector，最后累加 detached 系数，形成一次 encoder backward 的 surrogate。

### 4.4 Document 梯度范围

对每个可训练文档 m，固定当前 query action 和其他文档动作后，其 pair reward 只通过 qᵀe_m 依赖该文档动作。Document 项仍可投影到 span(h_m,q_i)，使用该 pair 的 document LOO，再累加与 m 有关的 pairs。

需要完整处理 query 和 document 两侧，不能只增加 query 辅助系数，却把实现称为完整联合策略梯度。

跨 query 文档有两种不同口径，尚待选择：

| 口径 | 跨 query 文档 | 辅助策略梯度范围 |
|---|---|---|
| 沿用原 fixed pool | 未扰动、detach 的固定方向 | Query 与自有采样文档；不补跨 query 文档 encoder 梯度 |
| Sampled joint pool | 文档动作参与全局 bundle | 参与 pair 的跨 query 文档也接收 reward 梯度，汇总回所在 rank |

第二种可参考 [跨 query 联合策略](cross_query_document_policy.md)，但这是不同随机目标与梯度覆盖，不能与 fixed pool 混为一谈。Fixed pool 下的“无偏”是针对既定固定方向条件下的策略梯度；共享 encoder 的间接更新不等于这些固定方向接收了直接梯度。

### 4.5 负例覆盖与 pair 抽样

候选的最小版本是：保留小池主 reward；辅助项用自有已知 positives 与跨 query 大池 negatives 比较。正例身份由 `positive_mask` 定义，跨 query 候选沿用现有身份过滤与相对当前 query 的负例约定。未标注的跨 query 候选不等于已经证明不相关。

另外讨论过按 teacher grades 构造 y_a>y_b 的 pairs；这是可扩展选项，不是当前已决定的标签口径。Teacher grade 与二值正例身份可能不同，需明确采用哪一种，不能混用“positive”的含义。

可在动作采样前均匀抽取 pairs，固定后让全部 Gq/Gd draws 使用同一组 pairs。Pair 样本均值无偏估计全 pair 平均目标；非均匀抽样则需用明确的抽样概率加权。

若按未扰动均值选 hard pairs 且不校正，优化的是所选 pairs 的条件目标。若按当前 sampled action 只选排错 pairs，则 selection 依赖当前动作，不能直接套用上述无偏证明。

### 4.6 收益与风险边界

预期机会：更多负例提供更多局部比较，每份 reward 只更新其能解释的差方向；累加后的梯度可以覆盖整个空间，不等于每项都需要保留整个空间的随机噪声。

尚未证明或验证的部分：

- 逐项条件投影不保证完整共享 encoder 总方差必然下降，各项协方差会改变。
- 容易的 pair 在全部 rollout 中都赢、很难的 pair 全部都输时，LOO advantage 为零，辅助信号退化。
- 二值 pair reward 不直接强调 top-K 位置，不能替代主 listwise 指标。
- 更多 pair 有奖励计算、投影、通信和梯度汇总成本，低 rank 不等于低总耗时。
- 相比直接 InfoNCE，随机 pair reward 是否有额外收益，没有证据。

已有 `top_weighted_pairwise` reward 依据 teacher order 加权并输出聚合 reward，见 [rewards.py](../src/rewards.py)；它不是已实现的“大池逐 pair CP”估计器，不能仅切换 reward 名称就完成本设计。

## 5. 其他方案的讨论结果

| 方案 | 对投影问题的作用 | 目标与当前选择 |
|---|---|---|
| 大池挖掘，固定小 shortlist 做 RL | 控制本次 span | 子集排序目标；可行，但不是全池指标无偏估计 |
| 全池 reward + 直接 top-K/SVD 截断投影 | 强制低 rank | 通常有偏，可能丢真实梯度信号 |
| 安全条件通过才 top-K，否则回退 | 保持期望 | 检查保守，预计通过率有限；未测量 |
| 小池 CP + 全池 reward 差值修正 | 主要项低维，残差完整方向 | 可保持全池目标，暂因复杂度放下 |
| 小池 CP + 大池 InfoNCE 直接 loss | 大池不进入主 projector | 已有实现，但当前优先讨论纯 RL |
| Triplet margin 直接 loss | 与 InfoNCE 类似接入 | 没有明确理由优先替换现有 InfoNCE |
| 大池 Gaussian 分数扰动 + 条件积分 | 不依赖低维文档 span 降噪 | 改变 vMF 随机目标，留作后续工作 |

差值修正的恒等式是 R_full=R_C+ΔR；分别计算 LOO 后，使用 A_CΠ_Cq+A_Δq。小池项 CP 保持期望，残差保留完整方向，故可保持原全池梯度。有效性依赖残差 reward 的波动，当前不推进。

距离差也可保持 pair 结构：单位向量满足 ||q−d₋||²−||q−d₊||²=2qᵀ(d₊−d₋)。但固定 κ、两侧独立且同浓度采样时，线性分数差的期望为 a_D(κ)²h_qᵀ(h₊−h₋)，可以直接求导，没有必要引入 SF 噪声。采样动作上的非线性 margin reward 是另一个候选，但不是已决定的辅助 reward。

## 6. 现有 InfoNCE 作为替代方案与后续对照

InfoNCE 辅助项使用未扰动均值，形式为：

\[
L_q=\frac1{|P_q|}\sum_{p\in P_q}
\log\left(1+\sum_{n\in N_q}\exp((s_{qn}-s_{qp})/\tau)\right).
\]

多正例由二值 `positive_mask` 定义，正例之间不互相竞争。候选不做额外 top-K mining：过滤后的全部 negatives 进入 loss，通过指数权重强调高分 negatives。

- 默认仅自有 negatives；普通 in-batch 模式追加同卡其他 query 的代表正例，这些跨 query 文档 detach。
- Strong 模式追加跨卡其他 query 的全部候选，使用可微 all-gather；启用的文档分支接收其他 query 的负例梯度。
- 两种模式都遵守 `action_components` 的直接梯度范围，未启用分支 detach。`negative` 分支指代表正例之后的槽位，不等于这些槽位全部具有负例身份。
- 不使用历史 queue，也不合并 gradient accumulation 中不同 microbatches 的候选池。
- 辅助项不进入 RL reward、advantage 或 CP；其梯度与 RL 梯度在参数上相加。

默认 `aux_infonce_coef=0`。已有 [strong 辅助草案](../configs/experiments/iclr2027/g1_r2_aux_infonce_strong.yaml) 使用系数 0.1、温度 0.03，尚未注册到 suite/队列，也不能据此认定用户已有实验启用了它。实现见 [contrastive.py](../src/contrastive.py)、[grpo.py](../src/grpo.py)，说明见 [aux_infonce.md](aux_infonce.md)。

## 7. 相关论文与方法定位

[Variance Reduction in Gradient Exploration for Online Learning to Rank](https://arxiv.org/abs/1906.03766)，SIGIR 2019，提出在交错排序反馈之后，将线性 ranker 的探索更新投影到用户浏览文档的特征空间。

其 Lemma 3.2 给出更新二阶矩上界 rank(P)/d；在固定投影、均匀球面方向下 E||Pu||²=rank(P)/d。这个上界不是实际梯度方差比的等式，也不能直接迁用到 vMF product rollout 和共享 Transformer。论文的关键结构是浏览文档数远小于特征维度；原文构建投影空间时对浏览范围有实践近似，不能据此证明当前 sampled top-K 裁剪无偏。

因此，“投影到文档空间降低梯度方差”不是新的核心主张。候选设计需要围绕以下问题组织证据：在 vMF 联合策略中，能否通过可加局部 reward 的逐项条件投影，扩大负例覆盖，同时避免单个 query score 项的全池满秩退化。

纯 RL 便于统一目标和梯度路径，但不自动优于直接监督；二值 pair reward 本身也是已有的偏好监督形式。方法价值应落在估计器、无偏条件、总方差/耗时和检索效果，而不只在“纯 RL”名称上。

后续分数积分可参考 [StochasticRank](https://proceedings.mlr.press/v119/ustimenko20a.html) 与 [METHOD_REDESIGN.md](../paper/METHOD_REDESIGN.md) 的分数空间条件积分推导；当前不推进该路线。

## 8. 后续实现前需要明确的事项

1. 辅助 pairs 使用二值已知 positive，还是 teacher grade 严格偏好；是否只扩充跨 query 比较。
2. 跨 query 文档沿用 fixed pool，还是采用 sampled joint pool。两者对应不同随机目标。
3. 等权/固定标签权重、pair 抽样数量和 λ；按 query 的 pair 平均归约，避免池越大辅助梯度越大。
4. 二值 reward、同分约定与 pair 退化监控；是否需要非线性 margin reward。
5. 继承哪个已选主基线的 alignment、G、预算与评分协议。对话中的 0.90 是机制讨论背景，不覆盖 [paper/README.md](../paper/README.md) 已记录的主结果 0.80 选择。

若以后决定实现，先核对逐项期望与 query/document 梯度路径，再验证共享模型总方差和训练效果。必要监控包括主/辅助 reward、pair LOO 退化比例、各项投影 rank、辅助梯度规模和耗时。现有单进程配对诊断无法直接复现八卡候选池；平均八个独立 microbatches 不等于构造八卡大池。

本次只归档讨论，不修改训练配置、启动队列、代码行为或既有实验结论。
