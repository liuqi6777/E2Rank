# G2 RL 实验计划（2026-09-14）

本轮建议：沿用 G1 的 `G1-A-MRRAlign090` 配方，先完成 E2Rank 的三条 RL，
再在 BGE-M3 数据上复现三种初始化的 CL/RL 比较。主计划共六条 RL，暂不增加搜索或消融。
研究问题是 **同一 RL 配方相对 CL 的收益如何依赖初始化，能否跨训练数据复现**。

状态：用户确认 G1 全部完成、E2Rank CL 已完成；本计划按 E2Rank 的 D/E/W 三条 CL
均完成安排。CL 分数、实际运行配置与 W0 权重尚未同步到本地；BGE-M3 CL 完成状态未确认。
六条 RL 的配置与批量脚本已按本计划落地，旧 `g2_rl_recipe` 启动阻塞已解除；未启动训练。
与旧计划冲突的 G2 待决策项和执行顺序以本文为准；运行能力仍以实际配置和预检为准。

## 1. G1 已经回答了什么

证据来自 [run_summary.csv](_summary/g1_bright/run_summary.csv) 和
[subset_summary.csv](_summary/g1_bright/subset_summary.csv)：本地共 46 行（45 次训练及 E0），
每行 12 个领域。下表为 BRIGHT 宏平均 nDCG@10，单位为百分点。

| 配置 | 分数 | 对 G2 的启示 |
|---|---:|---|
| MRR@10 / G=32 / alignment 0.90 | 22.01 | 统一起始配方 |
| 关闭 frozen-candidate rescaling | 21.72 | 保留现有实现；0.29 的单 seed 差异不足以宣称该组件普遍必要 |
| Paired rollout | 20.28 | 主线保留 product，不为节省 rollout 更换配方 |
| 仅 query / 仅 document policy | 13.00 / 17.85 | 主线保留双侧 action |
| Advantage norm / document mean / 同时开启 | 18.58 / 15.24 / 13.81 | 保留无标准化及 document sum |
| Binary nDCG@10，alignment 0.90 | 19.84 | 本轮优先 MRR，不重复 reward sweep |
| 原始 E0 / G1 CL | 15.07 / 16.96 | 22.01 相对 E0 +6.94、相对 CL +5.05，仅为 G1 证据 |

| group size | alignment 0.80 | alignment 0.90 | alignment 0.95 |
|---:|---:|---:|---:|
| 16 | 19.45 | 17.46 | 17.13 |
| 32 | 19.14 | **22.01** | 19.88 |
| 64 | 19.17 | 20.04 | 19.80 |

G=64 未超过 G=32 的最佳格，且 product 组合数增加四倍，因此不扩大 G。
G=16 下 0.80 更好，说明探索强度与 G 有交互，不能把 0.90 解释为与配置无关的最优值。
全部为单 seed、经过 BRIGHT 配置开发的结果；它们支持选择迁移配方，不保证 B0/W0/E0
在新数据上的表现，也不证明直接 RL 的最优参数与成熟 embedding 相同。

## 2. 主实验矩阵与执行顺序

| 顺序 | E2Rank run | 初始化 | 必须配对的 CL | 所回答的问题 |
|---:|---|---|---|---|
| 1 | `G2-E-RL` | 原始 `Qwen/Qwen3-Embedding-0.6B`（E0） | `G2-E-CL` | G1 配方迁移到更大训练集后，RL 是否仍有收益？ |
| 2 | `G2-W-RL` | `G2-D-CL` 最终权重（W0） | `G2-W-CL` | 同一个 CL warm-up 后，改用 RL 是否优于继续 CL？ |
| 3 | `G2-D-RL` | `Qwen/Qwen3-0.6B`（B0） | `G2-D-CL` | 该统一 RL 配方能否直接建立有效检索表示？ |

三条 RL 之间没有权重依赖；上述顺序用于尽早获得可解释的迁移结果，资源充足时可独立运行。
W-RL **使用 D-CL 的最终模型，不使用 W-CL 的最终模型**；仅加载模型权重，
重置 optimizer、scheduler、step counter 与数据迭代，完整执行后续预算。
E-RL 使用未经 G1 微调的原始 E0。每对 CL/RL 固定模型 revision、表示协议、数据版本与顺序、
候选构造和实际 device-local microbatch。

第二批按 E → W → D 执行 `G2-BGE-E-RL`、`G2-BGE-W-RL`、`G2-BGE-D-RL`，
分别与同前缀的 CL 配对。这里 BGE-M3 指训练数据，模型仍为上述 Qwen B0/E0，
不是更换为 BGE-M3 encoder。W0 只来自 `G2-BGE-D-CL`，不跨数据集复用。
若 BGE-M3 CL 尚未完成，先补齐 D-CL/E-CL，再运行 W-CL；其 CL 工作可与 E2Rank RL 重叠。

预算：新增主 RL 为 E2Rank 三次 × 1200 steps = **3600 optimizer steps / 460,800 query exposures**，
以及 BGE-M3 三次 × 1 epoch。BGE-M3 的实际样本数、steps 与 GPU hours 在数据核验后填写，
不根据目录名估算；尚缺的 CL 执行另计。完整 G2 仍是 12 次训练，已完成的 CL 直接复用。
W 路线另含 D-CL 前缀，需同时报告后续阶段和端到端成本。

## 3. 六条 RL 共用的首轮配方

| 项目 | 计划值 |
|---|---|
| Reward | Binary MRR@10：`reward_type=mrr_in_batch`、`reward_ndcg_k=10`；无混合 reward |
| 标签 | E2Rank 读取 1-based document `pos_index`；BGE-M3 从标注 pos/neg 构造 binary 标签 |
| 候选 | 沿用 G2 slate 与同设备 in-batch 代表正例；`ndcg_in_batch_include_negatives=false`；跨 query 文档 detach、去重及已知正例过滤与 CL 一致 |
| Policy | `sampling_law=vmf`；query + bundled positive/negative document 双侧 action；joint encoder |
| Rollout | `rollout=product`、`group_size=32`，每 query 32×32 个组合 |
| 探索 | `target_alignment=0.90`、`exploration_schedule=fixed`、`final_alignment=null`、`sigma_learnable=false`；按实际维度反解 κ |
| 更新 | `advantage_baseline=leave_one_out`、`advantage_norm=none`、`document_log_prob_reduction=sum` |
| 冻结候选 | `frozen_doc_rescale=true`、`in_batch_use_sampled_documents=false` |
| 优化器 | Full FT、AdamW、LR 5e-6、weight decay 0.01、linear schedule、warmup ratio 0.03、`kl_coef=0` |
| Batch / seed | Global batch 128、microbatch 16、seed/data seed 42；默认 8 GPU |
| 长度与预算 | `q_max_len=512`、`d_max_len=1024`；E2Rank 1200 steps，BGE-M3 1 epoch |

LR 与预算沿用对应 CL，六条 RL 不按初始化单独调参。相同 LR 不等价于相同有效更新尺度，
相同 steps / 样本暴露也不等价于相同算力；效果与成本分别报告。
保留 G1 选定的更新规则，但 G2 候选长度及表示分布发生变化，梯度尺度必须从实际日志观察。
平台上已完成 CL 的 launch 配置是配对核验依据，不能只凭当前仓库默认值认定历史运行一致。

## 4. 诊断嵌入正式运行，不新增一轮短训练

每条正式运行观察前 50 steps 与首个保存点；随后按保存间隔记录。保持正式 scheduler
与总预算，不另设 `max_steps=50` 后拿短训权重继续正式实验，不根据 callback 分数早停。

现有代码可记录 reward 均值、`reward/mrr_in_batch/n_distinct`、query/document 两个
component 的 `group_std` 与 `degenerate_frac`，以及 `train/grad_norm`。
日志中记录实际 alignment/κ、LR、有效候选数（若已有）、steps/s；wall time 与 peak GPU
memory 由平台采集，训练和 callback/最终评测耗时分开。没有采集的字段明确留空。

解释顺序：先检查数值有限、标签/候选正确，再检查两个 policy 分量的奖励差异，最后结合
外部曲线判断是否迁移。RL loss 的绝对值不直接代表检索质量；低方差或高退化率本身不作为
删除 B0 结果或提前改配方的依据。NaN/Inf、数据错误等实际故障停止并保留日志；修复后
使用可追溯的新运行，不覆盖已产生的结果。

若数值正常但 B0 长期缺少排序信号，保留 D-RL 的固定预算结果。首轮不自动加训练：
先用日志决定后续是否需要一个 alignment 对照（0.80 或 0.95）或 LR 对照，单独记录诊断
依据、唯一改变量与新增预算。不要因 MTEB 分数较低就启动 reward sweep，也不把修订配方
的最佳结果混入首轮统一配方矩阵。G1 已覆盖的 policy、norm、document reduction 消融不重跑。

## 5. 评测与结果判读

沿用现有 callback：`checkpoint-0` 和每次保存后运行 `MTEB(eng, v1, subset)`；
E2Rank 间隔 200 steps，BGE-M3 间隔 1000 steps。最终模型统一运行完整 `MTEB(eng, v2)`。
配对的 CL/RL 应使用同一评测包版本、解析后的任务清单、数据 revision 与表示配置。
已有 CL 结果若缺任务或协议不一致，补齐相同模型的评测，不因此重训 CL。

主指标建议预先固定为 **最终 MTEB(eng, v2) 中全部 `Retrieval` 任务的 main score 等权宏平均**，
附逐任务结果与任务数；各任务具体 metric 随任务清单记录，不把所有 main score 无条件称为
nDCG@10。全 MTEB 的 `mean_task_score` 与 `mean_type_score` 作为辅助结果分别呈现。
有缺失/失败任务时结果标记不完整并补评，不能静默缩小平均分分母。v1 subset 与 v2 最终结果
不混算，前者只用于学习曲线，报告最终 checkpoint，不选择最佳中间 checkpoint。

当前 `eval_mteb/summary.py` 的 OOD 名单硬编码排除 NQ/HotpotQA/FEVER，来自旧数据设定。
它不能直接作为 E2Rank、尤其 BGE-M3 的已核验 OOD 指标。按两套数据来源分别审计并冻结
任务分组后才报告 OOD；完整 Retrieval 均值仍可报告，但不将其自动解释为 zero-shot 泛化。

每个数据集填写下表，所有分数采用同一尺度：

| 初始化 | 初始 Retrieval | 最终 CL | 最终 RL | RL−CL | CL/RL 训练 GPU hours |
|---|---:|---:|---:|---:|---:|
| B0 | 待填 | 待填 | 待测 | 待测 | 待填 |
| W0 | 待填 | 待填 | 待测 | 待测 | 待填 |
| E0 | 待填 | 待填 | 待测 | 待测 | 待填 |

初始 Retrieval 若仅有 v1 subset callback，单独呈现，不能填入 v2 表；需要时补评初始
模型的同一 v2 任务集。W0 的 v2 初始结果可复用对应 D-CL 最终评测。

主要比较为同一数据集内三个 `Δ_init = RL_init − CL_init`，其中 W 分支直接回答共同
warm-up 后替换目标的价值。跨初始化比较三个 Δ，但不把差异完全归因于表示质量；E0
另有预训练历史，W0 另有 CL 前缀。再检查第二套数据各分支收益方向是否一致。
W/E 有益而 D 无益，支持该配方适用于已有检索表示的后训练；不能据此断言直接 RL
原则上不可行。训练 reward 提高、外部分数未提高则报告迁移不足。单 seed 不作显著性主张。

## 6. 从计划到启动的交付项

1. 核对三条 E2Rank CL 的实际配置、最终 checkpoint 与评测目录，确认使用修复后的
   pos_index 和无 ID 候选过滤协议；固定 W0 权重及同一训练 artifact。
2. 已在 `suite.yaml` 的共享 `&g2_rl_recipe` overrides 中显式写入第 3 节 RL 配方，
   六行均复用，覆盖通用 runner 的旧 nDCG 默认值；stage 顺序与说明已同步，配方阻塞已解除。
3. RL 入口为 `bash scripts/run_g2_rl.sh check 8` / `train 8` 和
   `bash scripts/run_g2_bge_rl.sh check 8` / `train 8`。两者预检各自全部三行及 W0 后，
   按 E → W → D 训练及评测，失败即停且不覆盖旧输出。执行平台完成数据、权重与评测协议
   核验后启动；W0 未就绪或批量部分完成时，用 `experiment.py` 单独启动可运行的剩余行。
4. 首批提交三条 E2Rank RL；回收最终结果、逐任务差值与诊断。按同一配方完成 BGE-M3
   矩阵，不因第一套数据的收益方向删掉第二套数据中的不利分支。

本轮完成标准是六条主 RL 的结果或有日志的失败状态、各自 CL 配对、固定任务覆盖与三种
初始化差值均可追溯，不以必须超过 CL 某个分数作为完成条件。
