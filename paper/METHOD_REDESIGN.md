# 方法改进：模型原生协议、排序目标与有效动作

日期：2026-09-16。承接 [整体审查](METHOD_DESIGN_REVIEW.md)。本文给出已经核实的差异、可实现的估计器推导和有边界的实验建议；没有改动训练入口、模型配置或历史结果，也没有启动训练。

## 结论与推荐

1. **末尾重复只在已核对的 Embedding 路径成立。** 普通 Qwen3 路径没有重复，但截断可能丢失末尾读出 token。需要按最终 token IDs 定义协议。
2. **核心假设应改成“已有检索几何上的奖励驱动改进”。** 需要实测随机奖励的改善能否传递到确定性全库检索，以及能否超过同监督信息的强排序方法。
3. **优先验证保留 vMF 分布、对无关动作坐标做条件投影的估计器。** 它改变梯度估计方式，保持声明的条件随机目标。比继续扩大 G 更能直接检验高维噪声是否是瓶颈。
4. **若需要真正更换动作，候选分数扰动是一个清晰的备选。** 对 nDCG 可解析积分掉单个文档的扰动；但这与已有排序平滑研究重合，应作为强控制或明确的改造路线，不能直接包装成全新原理。

## 1. 先把两类 Qwen 模型的差别说清楚

### 1.1 tokenizer：相同配置不代表相同输入

固定缓存 revision，直接运行真实 tokenizer 得到：

| 检查项 | 项目普通模型 `Qwen/Qwen3-0.6B` | `Qwen/Qwen3-Embedding-0.6B` |
|---|---|---|
| 原生 `test` | `[1944]` | `[1944, 151643]` |
| 手动追加 `<\|endoftext\|>` 后默认 tokenize | `[1944, 151643]` | `[1944, 151643, 151643]` |
| 自动后处理 | 只有 ByteLevel | ByteLevel + TemplateProcessing 追加 151643 |
| 长输入先手动追加、再截断 | 末尾 151643 被裁掉 | 自动追加的 151643 仍保留 |
| 当前短输入是否重复末尾 | 否 | 是 |

`151643` 对应 `<|endoftext|>`，恰好也是两者的 `pad_token_id`；其作为有效末尾时 attention mask 应为 1。两者的 `eos_token_id` 是 `151645`，所以也不能仅凭论文写了 “EOS” 就把配置改成 `append_token: eos`。论文抽象符号与部署 tokenizer 的字段名不是一回事。

复现：[compare_qwen_tokenizers.py](method_audit/compare_qwen_tokenizers.py)、[tokenizer_comparison.json](method_audit/tokenizer_comparison.json)。本地 revision 分别为 `c1899de...` 和 `97b0c614...`，完整值在 JSON 中。历史远端 checkpoint 仍需按其实际保存的 tokenizer 检查。

### 1.2 模型训练：相同末尾位置不代表相同检索能力

Qwen3 Embedding 保留 causal attention，取末尾 token 的最后层隐藏状态；模型经过专门的对比学习、多阶段数据训练与模型合并。它不是只替换了 pooling 的普通语言模型，也不是通过改成双向 attention 获得 embedding 能力。[Qwen3 Embedding 技术报告 §2–3](https://arxiv.org/html/2506.05176v1)

还有一个命名细节：当前配置使用的是 `Qwen/Qwen3-0.6B`，官方标注其包含 pretraining 和 post-training；严格的纯预训练版本另有 `Qwen/Qwen3-0.6B-Base`。现有 G2 应准确描述为“从尚未专门做 embedding 训练的通用 LLM 开始”，不能写成已经验证了纯预训练模型的结果。[当前模型卡](https://huggingface.co/Qwen/Qwen3-0.6B)、[Base 模型卡](https://huggingface.co/Qwen/Qwen3-0.6B-Base)

因此，D/W/E 分支的差异至少包括初始检索几何与训练历史。修正 token 协议后，这些差异仍存在。相同 alignment 只规定相同角度扰动强度；各模型的分数间隔不同，实际名次翻转率、top-K 变化和 reward 多样性不一定相同。

### 1.3 推荐的最终协议

对这两个模型的**原始文本 embedding 路径**，明确期望：

```python
content_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
input_ids = content_ids[:max_length - 1] + [151643]
attention_mask = [1] * len(input_ids)
# 然后进行 left padding；补齐位置的 mask 为 0，末尾读出位置的 mask 为 1。
```

这定义的是输入正文之后的一个读出 token；带特殊控制 token 的原始内容需要另有输入约定。原生 Embedding tokenizer 可以继续直接使用自动后处理；普通模型需要显式保留截断后的读出位置。上面统一写法在本次空文本、短文本、中文与长文本检查中，与 Embedding 原生 token IDs 完全一致。

实施时必须把 `add_special_tokens`、`terminal_token_id`、截断后追加规则、pooling 与协议版本一起存入 checkpoint；训练 collator、离线评测和在线回调共用实现。原来的 `append_token: pad` 元数据不足以表达这些行为。普通模型的这个读出协议是本项目的建模选择，不应称为官方原生 embedding 协议。其他模型族、chat template 和历史 checkpoint 不自动套用。

## 2. 改写核心假设：让结果能指出是哪一环失效

建议主假设写成：

> 对具有可用初始检索表示的双塔编码器，在候选集合覆盖主要排序错误时，能否利用低方差的随机排序奖励梯度，在保留确定性检索能力的同时，提高目标任务效用，并获得超过匹配监督排序方法的收益？

这句话包含三个需要分别检验的假设：

| 假设 | 应观测什么 | 被否定后意味着什么 |
|---|---|---|
| 随机目标可用于改进确定性检索 | 相同 checkpoint/query/candidates 上，sampled reward 的变化是否伴随 deterministic reward 改善 | 优先处理扰动尺度或目标混合；增加 rollout 无法修复目标失配 |
| 训练候选覆盖部署错误 | 小池改善能否传递到大池/全库，错误 top-K 文档是否出现在训练候选里 | 优先扩大和刷新候选；精确优化旧小池可能已经没有价值 |
| RL 比强监督排序有额外价值 | 同标签、候选、初始化下，对比 LambdaLoss/CL，并报告训练成本 | 若未超过，就不能保留“RL 普遍更适合排序”的主张 |

诊断要固定 query 与候选，逐 query 比较，而不是只对齐两条来自不同 batch 的平均曲线。随机与确定性比较使用相同标签；确定性部署分数用未做 frozen-candidate alignment 校准的 cosine。

MRR 可以保留为消融；当前数据多数 query 有多个正例，主任务若是 multi-positive 检索，应明确选择 binary nDCG 或与标签一致的排序效用。Teacher grades 属于另一种监督来源，不能无说明地替换 binary labels 后归因于 RL。

对于普通模型，先经过 CL/排序训练的 W 分支应成为主要机制检验；D 分支保留用于回答“是否需要 embedding warm-up”。直接从 D 训练失败不能单独否定 E/W 上的奖励微调。对于 E0，不需要假定必须再做一轮 warm-up，应由确定性能力和梯度诊断决定。

## 3. 首选动作改进：保留 vMF，积分掉奖励看不到的方向

### 3.1 直观解释

固定 20 篇文档时，query action 有 1024 个坐标，排序只看到它与这些文档的内积。许多方向不会改变任何一个内积，却仍被当前 policy-gradient 当作更新方向。它们在无限次采样中抵消，在有限 G 下变成噪声。

**先保持原动作与 reward 不变，只替换 score-function 梯度里的随机向量。** 对奖励不可见的正交方向，利用分布对称性直接取条件均值。这属于 Rao–Blackwell 化，即解析消去可以不采样估计的随机部分。

### 3.2 保持目标的条件

令 query 均值为单位向量 \(h_q\)，第 i 个动作为 \(e_i^q\)；第 j 个 document bundle 中第 m 篇的动作为 \(e_{jm}^d\)，均值为 \(h_m^d\)。固定候选集、掩码与标签，κ 为常数或外部预定 schedule。

定义奖励表 \(R_{ij}=R(e_i^q,e_{j1}^d,\ldots,e_{jM}^d;F)\)，F 包括参与该奖励的固定干扰文档。严格保持当前 reward 的所有输入。

逐 cell 的 leave-one-out 权重为：

\[
A^q_{ij}=R_{ij}-\frac{1}{G-1}\sum_{i'\ne i}R_{i'j},\qquad
A^d_{ij}=R_{ij}-\frac{1}{G-1}\sum_{j'\ne j}R_{ij'}.
\]

不做 empirical standard-deviation normalization。这些式子在不投影时展开后，恰好等于当前 row/column marginal 的 LOO 更新；不需要改 advantage 的目标定义。

令 \(J_q=\partial h_q/\partial\theta\)，\(J_m^d=\partial h_m^d/\partial\theta\)。query 侧取：

\[
S_j^q=\operatorname{span}(h_q,e_{j1}^d,\ldots,e_{jM}^d,F),\qquad
\widehat g_q=\frac{\kappa}{G^2}\sum_{ij}A^q_{ij}J_q^T\Pi_{S_j^q}e_i^q.
\]

document 侧取：

\[
S_{im}^d=\operatorname{span}(h_m^d,e_i^q),\qquad
\widehat g_d=\frac{\kappa}{G^2}\sum_{ijm}A^d_{ij}(J_m^d)^T\Pi_{S_{im}^d}e_{jm}^d.
\]

均值包含在 S 内，因此 vMF 密度在 S 的正交补上有反射对称性；奖励与 baseline 只依赖已保留的投影和其余条件变量。于是：

\[
E[e\mid\Pi_S e]=\Pi_S e.
\]

原 score 项与替换后的 score 项具有相同期望。所有 projector、sample 和 advantage 均 detach；仅通过 live mean 的 Jacobian 反传。这保持的是**当前固定候选、固定跨 query 文档条件下的目标**，不会补回原实现省略的跨 query encoder 梯度。

### 3.3 实现不需要为每个 cell 保存一个大投影矩阵

- Query：对每个 document bundle j 建一个小基底，先计算 \(v_j=\sum_i A^q_{ij}e_i^q\)，再投影 \(\Pi_{S_j^q}v_j\)。利用线性性，省掉 G² 个 d×d 矩阵。基底应使用 rank-revealing QR/SVD，处理重复文档、零秩与 padding；本次 toy 仅检查了满秩 QR。
- Document：归一化均值的梯度没有径向分量。定义 \(u_{im}=e_i^q-\langle e_i^q,h_m^d\rangle h_m^d\)，所需切向投影只是 \(u_{im}\langle u_{im},e_{jm}^d\rangle/\|u_{im}\|^2\)。近零范数需要稳定处理；严格为零时切向项为零。
- 把累加后的向量作为 detached 系数，与 live means 点积形成负的梯度 surrogate，再做一次 encoder backward。实际单位化与 score 计算使用显式 fp32 区域。
- 不把全部 G 组文档先合成一个全局子空间。该空间可能铺满 1024 维，损失降噪机会。

以当前最多 20 个自有候选与 15 个固定候选计，query 的相关线性空间最多 36 维（包含均值），而每个 document/cell 的有效切向方向只有 1 维。它仍未解决所有文档共用 reward 的信用分配问题，只是删除其中可以解析消除的噪声。

### 3.4 不能扩大宣称的边界

逐项条件投影的方差性质，不能直接推出 product estimator 在共享 Transformer 参数上的总方差必然降低；不同 cell 的协方差也会变化。共享 encoder 的完整梯度或可靠的参数方向投影，是下一步的必要检查。

不要把它写成“在低维球面重新采样，再继续用原 vMF log-prob”。那会改变分布；若低维子空间还依赖 θ，忽略其变化通常也不再是原目标的正确梯度。

对 G3 的全库动态检索，所有可能影响 top-K 边界的文档都影响奖励；不能只用这次检索出来的 top-K 文档建立投影并宣称无偏。全库张成空间也可能已经满维。因此本方案首先针对 G1/G2 的有限候选排序计算。

## 4. 已完成的数值验证

复现代码：[action_design_probes.py](method_audit/action_design_probes.py)；聚合证据：[action_design_evidence.json](method_audit/action_design_evidence.json)。复用了项目真实 vMF sampler、LOO、log-prob 与 nDCG 函数。

### 4.1 与现有实现及解析梯度对照

逐 cell 未投影公式与现有 log-prob + row/column LOO 的 autograd 梯度，最大绝对差为 **7.15×10⁻⁷**。

对双线性奖励 \(R=(e^q)^T\sum_m c_m e_m^d\)，有明确解析目标：

\[
E[R]=A_d(\kappa)^2h_q^T\sum_m c_mh_m^d.
\]

在 d=32、G32、4,096 次独立重复中，完整共享线性 encoder 梯度的均值相对解析误差为：原估计 **4.40%**，投影估计 **0.97%**；由各自方差估算的 Monte Carlo RMS 相对误差分别为 **4.78%**、**1.04%**。数值符合预期采样误差，支持推导与实现；无偏性的论证来自前面的条件期望，不来自有限次“看起来相近”。

### 4.2 联合排序奖励的方差

使用一个 query、8 篇采样文档、3 篇固定干扰文档、3 个 binary positives、nDCG@5、G32、256 次重复。共享模型为 \(h_i(W)=\operatorname{normalize}(Wx_i)\)，在 W=I 处计算包含 query/document 交叉协方差的完整参数梯度方差。

| 维度 | 原估计噪声方差 | 条件投影噪声方差 | 投影 / 原值 |
|---|---:|---:|---:|
| 128 | 365.20 | 3.92 | 1.072% |
| 1024 | 16075.04 | 23.36 | 0.145% |

这是合成几何、固定模型点上的结果，**不能当作 Qwen Transformer 方差比、训练加速比或 BRIGHT 增益**。离散奖励的完整解析均值未知；固定、与采样独立的参数方向上，投影前后均值差小于一个标准误。原始高维估计的样本均值范数还受噪声抬高，不能直接用两个均值范数之比判断偏差或有效信号。

## 5. 真正更换动作的备选：候选分数扰动与条件积分

### 5.1 定义新的随机目标

直接从确定性 encoder 分数出发：

\[
s_m(\theta)=h_q^Th_m^d,\quad z_m=s_m+\tau\epsilon_m,\quad
\epsilon_m\sim\mathcal N(0,1),\qquad
J_\tau(\theta)=E_\epsilon[R(\operatorname{rank}(z),y)].
\]

此时一个候选只对应一个随机分数，梯度经 \(s_m\) 同时传给 query 和 document encoder。部署继续使用双塔 cosine；训练中的随机分数不要求能同时表示为一组球面向量的内积。

这明确改变了 vMF 诱导的分数相关性与随机目标。它仍是平滑后的排序目标，不会自动等于确定性全库 nDCG。

### 5.2 不必再用一次奖励乘上整组高维噪声

固定其他文档的随机分数 \(z_{-m}\) 时，第 m 篇文档只有越过某个对手分数，奖励才会变化。对手分数形成有限个边界 b；设从下向上越过 b 的奖励变化为 \(\Delta R_m(b)\)，则：

\[
\frac{\partial E[R\mid z_{-m}]}{\partial s_m}
=\sum_b\Delta R_m(b)\frac{1}{\tau}\phi\!\left(\frac{b-s_m}{\tau}\right),
\]

其中 φ 是标准正态密度。它解析积分掉了当前文档的扰动，其他文档的扰动仍通过采样平均。

对 nDCG，若越过该对手后由第 r+1 名升到第 r 名：

\[
\Delta R_m=\frac{(g_m-g_{\mathrm{opponent}})[D(r)-D(r+1)]}{\mathrm{IDCG}},
\]

\(D(r)=1/\log_2(r+1)\)，超出 cutoff 则为 0。所有 gain、IDCG 与固定标签一致。没有正例时按预先定义的零奖励处理。将估计的分数梯度 detach，形成 \(-\sum_m\widehat g_m s_m\) 即可反传。

直观上，更新由“跨过某个排序边界能得到多少收益”和“当前噪声下处在该边界附近的密度”共同决定。它给每篇文档单独分配更新，不要求等待本轮采样恰好跨过边界。

### 5.3 验证、已有工作与限制

本次两文档 nDCG toy 的解析梯度为 `[2.5624669, -2.5624669]`。G32、8,192 次重复下：

| 估计方法 | 梯度均值 | 梯度噪声方差 |
|---|---|---:|
| Gaussian score-function + LOO | `[2.5632342, -2.5679618]` | 0.97117 |
| 单坐标条件积分 | `[2.5603856, -2.5628529]` | 0.06924 |

方差比为 **7.13%**。另用 5 文档、graded labels、cutoff=3，独立枚举区间奖励并对区间概率求导，检查所有文档方向；与边界公式最大差 **1.78×10⁻¹⁵**。这些是公式验证，不是检索结果，也不意味着在真实数据上它一定优于条件投影。

Gaussian 分数平滑已有 [SoftRank](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/SoftRankWsdm08Submitted.pdf)；部分积分估计器与 [StochasticRank](https://proceedings.mlr.press/v119/ustimenko20a.html) 直接相关。这里是针对当前双塔计算的具体适配建议，需要进一步核对创新边界。不能迁用这些论文对其他优化算法的收敛保证。

τ 使用固定值或预先确定的 schedule 最清楚。τ 很小时，远离边界的信号仍可能极小；若令 τ 随当前分数变化却停止其梯度，必须承认优化的是每步固定 τ 的条件目标。用同一数据集上的 rank-flip rate 和 top-K 变化做校准比照搬角度 alignment 更可解释，但不能把 reward variance 越大当作越好。

对于 G3，若只能对少量召回候选扰动分数，该方案只能学候选内选择，无法发现未进入候选的文档；不能据此替代全库 query-action 检索。

## 6. 推荐的训练设计与最小验证顺序

### 6.1 方法配方

**表示协议明确化 → 获得可用的确定性检索表示 → 覆盖当前错误的候选 → 低方差 reward 微调 → 检查确定性全库收益。**

候选方面，先修复每个 epoch 的固定同伴分组与永久丢尾；再用预定 encoder 快照挖掘接近 top-K 的困难负例，合并原正例和一部分旧/随机负例。挖掘只用训练资源，显式处理已知正例、重复文档与未标注相关文档。同一阶段 CL/LL/RL 应共享候选 artifact、掩码和刷新时间，避免把访问更多候选的收益记到 RL 上。

若 sampled reward 上涨、deterministic reward 不涨，可采用已经存在的直接辅助 InfoNCE：

\[
L=L_{\mathrm{det}}+\lambda L_{\mathrm{PG}},\qquad
E[\nabla L_{\mathrm{PG}}]=-\nabla J.
\]

这里 \(L_{\mathrm{PG}}\) 是对应估计器的负梯度 surrogate；不是把不可微 reward 直接塞进 autograd。用固定训练校准 batch 观察两部分梯度范数和夹角，再预定 λ；当前 0.1 不能仅靠 loss 值大小论证。加入辅助项是明确改变训练目标，应与相同 \(L_{\mathrm{det}}\) 单独训练对照。

暂不同时加入 learned κ、任意低秩动作、多个 reward 和复杂 reference loss。若观测到能力漂移，再考虑对独立训练 anchor 的 E0 query-document 分数做保留约束；分数约束比锁定 embedding 每个坐标更符合检索的旋转不变性。

### 6.2 分关验证，避免一次混合所有改动

| 顺序 | 最小比较 | 通过或转向标准 |
|---|---|---|
| 1. 协议与目标 | E0 原生/历史末尾协议；同 query 上 sampled-slate → deterministic-slate → large-pool/corpus | token 可追溯，并定位主要失配发生在哪一级 |
| 2. 估计器 | 固定真实 E0/CL/RL 状态，配对随机数，原 vMF vs 条件投影 | 真实 encoder 的均值检查、总梯度方差与单位时间方差改进；不能只看输出 embedding 范数 |
| 3. 固定候选短训练 | 同初始化、同标签的强 LL、原 RL、投影 RL | 区分“估计器变好”与“最终指标变好”；先用训练域 held-out 做机制判断 |
| 4. 候选与辅助目标 | 若大池迁移失败，统一刷新候选后复核；若确定性目标失配，再比较直接监督与监督+投影 RL | 每次只检验已定位的瓶颈，所有方法获得相同数据访问条件 |
| 5. 最终证据 | 冻结方案后做匹配重复，并在未用于本轮开发的确认集验证 | 报告均值、波动、成本和能力保留，判断是否值得相对强 LL 的额外复杂度 |

若投影显著降低真实梯度方差，但效果仍停滞，结果本身很有信息：主要问题更可能是候选/监督/随机目标与部署的差异，而不是继续加 G 可以解决的估计噪声。此时再用分数条件积分作为目标与动作空间的强对照。

## 7. 本次可复现产物

```bash
.venv/bin/python paper/method_audit/compare_qwen_tokenizers.py
.venv/bin/python paper/method_audit/action_design_probes.py
```

均使用本地 CPU；tokenizer 固定缓存 revision，动作实验只使用合成输入。没有训练或外部模型下载。推导成立的数学条件、toy 数值证据和真实训练待验证内容已分别注明。
