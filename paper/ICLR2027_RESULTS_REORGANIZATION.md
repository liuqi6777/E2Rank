# ICLR 2027：BRIGHT 论文表格草稿

更新：2026-09-24。本文件直接给出拟放入论文的表格形态。**假定主配方为 `K=0`、graded nDCG + pairwise `λ=0.5`、RLOO+CMP、`G=64`、`ρ=0.70`。** 尚未完成的实验在表中留空，不推测数值。这里仅整理 BRIGHT；RAG 正在运行，暂不纳入。

0.6B 的补实验已汇入[独立 suite 与导入脚本](../docs/iclr2027_final_k0_suite.md)，输出根目录为 `checkpoints/iclr2027-final-k0/`；新训练仍待执行，旧结果只能在存有 checkpoint 的训练机上复制。

配套的 LaTeX 表体在本地 `iclr2027_table_drafts/proposed_*.tex`，已由 Git 忽略，目前未接入 `main.tex`，不会让未完成结果出现在现有 PDF。0.6B 数字取自 [`g1_r2_bright`](_summary/g1_r2_bright/run_summary.csv)，4B 数字取自 [`g1_r2_bright_4b`](_summary/g1_r2_bright_4b/run_summary.csv)及[4B calibration](_summary/g1_r2_bright_4b_calibration/run_summary.csv)，各领域使用对应目录的 `subset_summary.csv`。分数按 nDCG@10 × 100；消融表的 StackExchange、Coding、Theorem-based 分别是 7、2、3 个 subset 的等权均值，再对三个 seed 取均值；Avg. 保持原始 12-subset 宏平均，`±` 为三个 seed 的样本标准差，**不是三个类型均值的简单平均**。E0 只评测一次。空单元格表示需要补实验；表格定稿后才加最优值粗体。

## 正文表 1：BRIGHT 主结果（两个模型 × 两种 query）

**拟用表注：** ReasonRank 适配后的 BRIGHT nDCG@10（×100）。每个区块固定 backbone 与评测 query 形式；领域列为三个 seed 的均值，Avg. 为逐 seed 的 12 领域宏平均及样本 SD。E0 仅评测一次。GPT-reasoning query 将 BRIGHT examples 中的 GPT reasoning 附到原 query，缺失时使用原 query；两种 query 使用相同语料和 qrels。0.6B 全参数微调，4B 使用 LoRA；训练均为 113 steps，但两者学习率和可训练参数范围不同。`K=0` 是拟定主配方，`K=7` 的历史结果移至附录。

| Model / query / method | Bio. | Earth. | Econ. | Psy. | Rob. | Stack. | Sus. | Leet. | Pony | AoPS | TheoQ. | TheoT. | Avg. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Qwen3-Embedding-0.6B / Original query** |  |  |  |  |  |  |  |  |  |  |  |  |  |
| E0 | 13.4 | 27.8 | 18.0 | 16.6 | 12.0 | 12.8 | 13.2 | 14.2 | 0.7 | 3.3 | 17.1 | 32.1 | 15.10 |
| InfoNCE | 34.3 | 34.1 | 26.2 | 33.4 | 21.0 | 28.8 | 27.4 | 10.2 | 1.2 | 3.1 | 19.7 | 32.3 | 22.64 ± 0.20 |
| LambdaLoss | 24.3 | 25.5 | 23.9 | 29.0 | 17.6 | 22.9 | 21.9 | 12.4 | 1.4 | 3.3 | 21.0 | 29.0 | 19.37 ± 0.28 |
| RELER, K=0 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| **Qwen3-Embedding-0.6B / GPT-reasoning query** |  |  |  |  |  |  |  |  |  |  |  |  |  |
| E0 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| InfoNCE |  |  |  |  |  |  |  |  |  |  |  |  |  |
| LambdaLoss |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RELER, K=0 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| **Qwen3-Embedding-4B / Original query** |  |  |  |  |  |  |  |  |  |  |  |  |  |
| E0 | 17.1 | 34.6 | 16.4 | 23.0 | 12.9 | 15.9 | 17.1 | 21.2 | 2.0 | 3.7 | 19.6 | 40.9 | 18.70 |
| InfoNCE | 48.5 | 47.0 | 32.0 | 42.3 | 26.7 | 34.7 | 35.9 | 15.2 | 6.6 | 3.8 | 23.5 | 44.7 | 30.08 ± 0.16 |
| LambdaLoss | 32.3 | 35.5 | 27.5 | 36.0 | 19.9 | 27.0 | 30.8 | 18.5 | 3.6 | 3.8 | 23.3 | 42.0 | 25.02 ± 0.37 |
| RELER, K=0 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| **Qwen3-Embedding-4B / GPT-reasoning query** |  |  |  |  |  |  |  |  |  |  |  |  |  |
| E0 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| InfoNCE |  |  |  |  |  |  |  |  |  |  |  |  |  |
| LambdaLoss |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RELER, K=0 |  |  |  |  |  |  |  |  |  |  |  |  |  |

[LaTeX 表体](iclr2027_table_drafts/proposed_bright_main.tex)。主表每个区块仅含 E0、InfoNCE、LambdaLoss 和拟定的 `K=0 + pairwise` RELER；reward 与 RL 组件变体分别放正文表 2、表 4。已测 `K=7` 的 0.6B 23.04 见附录表 A，4B 29.00 以及经 BRIGHT 校准的 `K=7, LR=2e-4` 29.84 ± 0.07 见[4B 附录配方表](iclr2027_table_drafts/proposed_4b_recipes.tex)，均不填入主表的 `K=0` 行。4B `K=0` 草稿沿用原 LoRA rank 16、LR `1e-4` 和其余训练协议，只改变 `K`；目前尚未训练。GPT-reasoning 的分数目前均未进入本地汇总，故对应两个区块保持空白。

## 正文表 2：reward 消融（固定 `K=0`）

**拟用表注：** BRIGHT nDCG@10（×100），固定 Qwen3-Embedding-0.6B、`K=0`、RLOO+CMP、`G=64`、`ρ=0.70`、`T=1`、113 steps 和三个配对 seed。Graded nDCG 使用 teacher grades；binary nDCG/MRR 使用原始已知正例。Pairwise 仅加在 graded nDCG 上，权重 `λ=0.5`，其正例来自原始 `positive_mask`。本表只改变训练 reward，不改变 CMP 或候选数。

| Reward | StackExchange | Coding | Theorem-based | Avg. |
|---|---:|---:|---:|---:|
| Graded nDCG@10 + pairwise (`λ=0.5`) |  |  |  |  |
| Graded nDCG@10 | 29.80 | 6.21 | 17.24 | 22.72 ± 0.17 |
| Binary nDCG@10 |  |  |  |  |
| Binary MRR@10 |  |  |  |  |

[LaTeX 表体](iclr2027_table_drafts/proposed_reward_ablation_k0.tex)。第一行与 0.6B original-query 主表的 RELER 使用同一组三 seed；两种 binary reward 各补三 seed，不加 pairwise。旧 local-pool 的 MRR/binary nDCG 分数和 `K=7` 的 binary 混合分数不能填入。Binary nDCG 在 shortlist 路径中将原始正例的 `reward_shortlist_binary_weight` 设为 1；binary MRR 使用原始 binary 标签和仅自有候选的 `reward_type=mrr`，需在小批次核对 cutoff 和正例身份一致。`mrr_in_batch` 即使不纳入全部负例，也会纳入跨 query 代表正例，不能作为此处的 K0 对照。正文表 4 在纯 graded reward 下比较 RL 组件，不能把它的 CMP 差异外推为组合 reward 下的 CMP 效应。

## 正文表 3：额外候选数

**拟用表注：** 固定纯 graded nDCG、RLOO+CMP、`G=64`、`ρ=0.70`、一个候选集 `T=1`；每个 query 的自有候选始终保留，`K` 只计额外跨 query 候选。每行使用三个配对 seed 的最终 checkpoint。此表比较候选数，不能直接推出组合 reward 下的最佳 `K`。

| Additional negatives `K` | StackExchange | Coding | Theorem-based | Avg. |
|---:|---:|---:|---:|---:|
| 0 | 29.80 | 6.21 | 17.24 | 22.72 ± 0.17 |
| 1 | 29.74 | 6.01 | 17.34 | 22.69 ± 0.11 |
| 3 | 29.50 | 6.17 | 17.48 | 22.61 ± 0.07 |
| 7 | 29.10 | 6.29 | 17.82 | 22.48 ± 0.10 |
| 15 | 28.72 | 6.35 | 17.72 | 22.25 ± 0.27 |

[LaTeX 表体](iclr2027_table_drafts/proposed_candidates_k0.tex)。`K=0` 相对 `K=7` 的逐 seed 提升为 +0.33/+0.16/+0.24；`K=0` 与 `K=1` 的平均差仅 +0.04。这张紧凑表直接解释为何改变默认候选构造；原来 `K=15` 下的 `T`、hard fraction 和候选来源搜索仍放附录。

## 正文表 4：CMP、rollout 与 policy role 的 RL 组件消融

**拟用表注：** BRIGHT nDCG@10（×100），Qwen3-Embedding-0.6B、仅自有候选（与 shortlist 的 `K=0` 同义）、**纯 graded nDCG**、`ρ=0.70`、`G=64`、113 steps、三个配对 seed。五行均不含 pairwise，且保持相同候选、reward 定义、数据和训练预算；只改变估计器、rollout 或启用的 policy role。单侧 policy 仍训练共享 encoder，只移除一类随机动作。Paired 每组只有 `G` 个 reward，而 product 有 `G²` 个 reward，不能称为等计算量比较。

| Configuration | StackExchange | Coding | Theorem-based | Avg. |
|---|---:|---:|---:|---:|
| RLOO+CMP, product, query + document | 29.80 | 6.21 | 17.24 | 22.72 ± 0.17 |
| RLOO (w/o CMP), product, query + document |  |  |  |  |
|  paired rollout |  |  |  |  |
|  query policy only |  |  |  |  |
|  document policy only |  |  |  |  |

[LaTeX 表体](iclr2027_table_drafts/proposed_rollout_policy_controls.tex)。第一行复用已完成的 `K=0` 纯 graded CMP 三 seed；**第二行只把同一个 `K=0` shortlist 配方的估计器切为 RLOO**，因此前两行是匹配的 CMP 对照。后三行以 RLOO product 为参照，分别检验 paired rollout 和两侧随机 policy；其余四行各需三个新训练。后三行走非 shortlist 实现路径，因为现有[shortlist 路径](../src/config.py)即使 `reward_shortlist_size=0` 也要求 product 与双侧动作。非 shortlist 路径没有 `K` 参数：设 `reward_shortlist_count=0`、`reward_type=ndcg`、`reward_cross_device_negatives=false`，只对自有候选评分，语义上对应 `K=0`；不能沿用会加入 batch 内候选的 `ndcg_in_batch`。提交后三行训练前，用小批次核对两条路径的纯 graded reward 定义及 RLOO product 梯度；若不等价，后三行不能被称作仅改变 rollout/policy role 的消融。旧 [`mechanism.tex`](iclr2027/tables/mechanism.tex) 的 `ρ=0.90` 分数留附录，不填新表。这张表不能证明主配方中 pairwise 分量的 CMP 收益。

## 梯度与训练过程分析：改用图

固定状态梯度诊断、训练 reward 曲线和检索行为不再规划为新的数字表，而按[分析图计划](ANALYSIS_PLAN.md)制作。正文优先两张图：在 `K=0 + pairwise` 的匹配动作与 reward 下画完整参数梯度方差和方差×耗时；以及 `K=0` 主配方与纯 graded 对照的训练 reward 曲线。已有 local-pool、`ρ=0.90`、standalone reward 的梯度探针改为附录图，明确其与新主配方的条件不同。新探针需要补诊断实现和固定状态测量，但不属于本文件统计的训练 run。

## 附录表 A：组合 reward 下的候选数（固定 reward）

**拟用表注：** 固定 Qwen3-Embedding-0.6B、graded nDCG@10 + pairwise `λ=0.5`、RLOO+CMP、`G=64`、`ρ=0.70`、`T=1`、113 steps 和三个配对 seed；只改变额外候选数 `K`。每个 query 的自有候选始终保留。随着 `K` 改变，pairwise 中跨 query 负例的数量也随之改变，这是候选数干预的一部分。

| Additional negatives `K` | StackExchange | Coding | Theorem-based | Avg. |
|---:|---:|---:|---:|---:|
| 0 |  |  |  |  |
| 1 |  |  |  |  |
| 7 | 30.20 | 6.18 | 17.57 | 23.04 ± 0.21 |
| 15 | 29.78 | 6.12 | 17.75 | 22.83 ± 0.30 |

[LaTeX 表体](iclr2027_table_drafts/proposed_pairwise_k_sweep.tex)。`K=0` 行共用主表结果，`K=1` 三 seed 尚待填；`K=7/15` 是已有匹配 reward 的结果。原 `K=15, λ=0.25` 的探索结果 22.51 ± 0.23 可在附录文字中报告，不放进固定 reward 的 `K` 表。

## 附录表 B：主配方 alignment 敏感性

**拟用表注：** 固定 Qwen3-Embedding-0.6B、`K=0`、graded nDCG@10 + pairwise `λ=0.5`、RLOO+CMP、`G=64`、`T=1`、113 steps 与三个配对 seed；只改变 vMF 目标 alignment `ρ`。报告最终 checkpoint 的 BRIGHT nDCG@10（×100），Avg. 是逐 seed 的 12 领域宏平均及样本 SD。

| Alignment `ρ` | StackExchange | Coding | Theorem-based | Avg. |
|---:|---:|---:|---:|---:|
| 0.60 |  |  |  |  |
| 0.70 |  |  |  |  |
| 0.80 |  |  |  |  |
| 0.90 |  |  |  |  |

[LaTeX 表体](iclr2027_table_drafts/proposed_alignment_k0_pairwise.tex)。`ρ=0.70` 行与 0.6B 主配方共用三 seed；其余三点各需三 seed。旧 [`alignment.tex`](iclr2027/tables/alignment.tex) 虽覆盖更宽的 `ρ=0.40–0.98`，但使用纯 graded、device-local 候选，不能回答新 `K=0 + pairwise` 配方的敏感性。本表用于检查 `ρ=0.70` 的稳健性；若结果促使主配方改用其他 `ρ`，正文 reward/RL 消融的比较基座和表注也要按最终配方重审，不能混用旧 `ρ` 的结果。

## 附录表 C：主配方 pairwise 权重敏感性

**拟用表注：** 固定 Qwen3-Embedding-0.6B、`K=0`、graded nDCG@10、RLOO+CMP、`G=64`、`ρ=0.70`、`T=1`、113 steps 和三个配对 seed；只改变 pairwise reward 系数 `λ`。`λ=0` 即正文 reward 表的纯 graded 行；`λ=0.5` 即主配方。报告最终 checkpoint 的 BRIGHT nDCG@10（×100）。

| Pairwise weight `λ` | StackExchange | Coding | Theorem-based | Avg. |
|---:|---:|---:|---:|---:|
| 0 | 29.80 | 6.21 | 17.24 | 22.72 ± 0.17 |
| 0.5 |  |  |  |  |
| 1.0 |  |  |  |  |

[LaTeX 表体](iclr2027_table_drafts/proposed_pairwise_lambda_k0.tex)。`λ=0/0.5` 分别复用正文 reward 表的现有纯 graded 三 seed 和待完成主配方三 seed；只为 `λ=1.0` 新增三 seed。旧 `K=15, λ=0.25` 不填入此表。此表与附录表 A 分开，避免混合 `K` 与 reward 权重两个变量。

## 其余现有表格的定稿位置

| 论文位置 | 表格 | 定稿处理 |
|---|---|---|
| 正文表 4 | [RL 组件消融草稿](iclr2027_table_drafts/proposed_rollout_policy_controls.tex) | 固定 `K=0` 纯 graded reward；复用已有 CMP product 三 seed，核对路径等价后补 RLOO product/paired/单侧 policy 四变体 × 三 seed；旧机制分数不填入 |
| 附录 | [固定状态梯度探针](iclr2027/sections/additional_results.tex) | 保留现有诊断及协议，不搬入正文 |
| 附录 | [`reward_controls.tex`](iclr2027/tables/reward_controls.tex)、[`reward_domains.tex`](iclr2027/tables/reward_domains.tex) | 保留旧 local-pool 的 reward×estimator 表，作为 reward 依赖性的独立证据；不要用它填正文表 2 |
| 附录 | [`alignment.tex`](iclr2027/tables/alignment.tex)、[`mechanism.tex`](iclr2027/tables/mechanism.tex) | 宽范围旧 alignment 曲线可由新配方的附录表 B 替代；旧 RL 机制表如保留，须明确其 local-pool、graded、`ρ=0.90` 设置与正文不同 |
| 附录 | [`candidates.tex`](iclr2027/tables/candidates.tex) | 保留 `K=15` 为主的 `T`/hard/候选来源搜索；与正文表 3 分开，避免混淆 `ρ` 和 `T` |
| 附录 | [`initialization.tex`](iclr2027/tables/initialization.tex) | E2Rank/MTEB 的单 seed 研究独立呈现，不作为 BRIGHT `K=0` 的复现 |
| 重建后保留 | [`seed_scores.tex`](iclr2027/tables/seed_scores.tex) | 更新 0.6B 主配方逐 seed 行；12 领域已在四区块主表中展示，不重复 `ablation_domains.tex` |
| 被新表替代 | [`bright_main.tex`](iclr2027/tables/bright_main.tex)、[`recipe_ablation.tex`](iclr2027/tables/recipe_ablation.tex)、[`reward_pairwise.tex`](iclr2027/tables/reward_pairwise.tex) | 不手工覆盖现有生成表；等空格填齐后更新 `build_results.py` 并从 CSV 重建 |

主表待填项按区块分别是：0.6B `K=0 + pairwise` 三 seed 的 original-query 结果；4B LoRA 在匹配协议下的 `K=0 + pairwise` 三 seed 训练与 original-query 结果；以及四个区块各方法的 GPT-reasoning-query 评测（复用相应 checkpoint，仅改变评测输入）。正文表 2 的 pairwise 行共用主表结果，binary nDCG/MRR 各缺三 seed；正文表 4 的纯 graded CMP 基座已填，其余四个变体各缺三 seed，无需实现 pairwise 的 RLOO 估计器。`K=1 + pairwise` 三 seed 填附录表 A，检查主配方的 `K=0` 是否只是边界点；`ρ=0.60/0.80/0.90` 各三 seed 填附录表 B，检查主配方的 alignment 敏感性。其他旧配置消融保留其原有条件，不为凑齐表格自动重跑。当前 `results_audit.json` 仍记录 196 个 run，而最新 0.6B BRIGHT CSV 有 205 个 run；定稿时需同步结果生成器与审计。

## 0.6B 待补工作量（按当前表格去重）

| 用途 | 尚缺配方 | 新训练 run |
|---|---:|---:|
| 主表：`K=0 + pairwise` | 1 | 3 |
| 正文 reward 表：binary nDCG、binary MRR | 2 | 6 |
| 正文 RL 组件表：RLOO product、paired、query-only、document-only | 4 | 12 |
| 附录表 A：`K=1 + pairwise` | 1 | 3 |
| 附录表 B：`ρ=0.60/0.80/0.90` | 3 | 9 |
| 附录表 C：`K=0, λ=1.0` | 1 | 3 |
| **合计** | **12** | **36** |

其中正文共 7 个配方、21 次训练；附录共 5 个配方、15 次训练。已测的 `K=0` 纯 graded CMP 和其他基线不重复计算；同一 `K=0 + pairwise` checkpoint 在主表、reward 表、附录 `K` 表、alignment 表和 `λ` 表中共用。另需 **10 次 GPT-reasoning-query 评测**：E0 一次，InfoNCE、LambdaLoss、RELER 各三个 seed；这是复用 checkpoint 的评测，不是新的训练。新训练按原流程还需评测 original query。若两条 `K=0` 纯 graded reward 路径的核对不通过，正文 RL 组件表可能额外需要 3 次 CMP 基座重训；若 alignment 或 `λ` 结果改变最终主配方，则上述复用关系及其他消融的匹配条件需要重新核算。

## 其他实验缺口审查（不计入上面的 36 次）

`λ=1.0` 已纳入附录表 C 和上面的 36 次训练；目前不扩展 `K × reward × alignment` 全网格。4B 学习率交互先不安排新增实验。

| 优先级 | 问题 | 最小新增工作 | 呈现位置与判据 |
|---|---|---|---|
| 中，若正文继续强调完整组合 reward 下的 CMP 实证效益 | 现有全参数梯度探针用旧 local-pool、`ρ=0.90` 和单独 reward；正文 RL 组件表则是纯 graded。因此两者都不能直接量化 `K=0 + pairwise` 主配方的 CMP 方差或检索增益。 | 优先在固定权重、同动作/同 reward 下补一个 `K=0 + pairwise` 的配对梯度探针；若要声称训练收益，再补匹配的无 CMP 三 seed。 | 探针放附录；若不补，则把 CMP 的实证结论明确限制在已测的纯 graded 条件。 |

还有两项先做**不需新训练**的核查：统计 GPT-reasoning query 的 reasoning 缺失率、回退率、截断率和长度，并固定与 original query 相同的 qrels、语料、checkpoint；从现有日志汇总训练成本，尤其区分 product 的 `G²` 个 reward cell 与 paired 的 `G` 个，避免把等 steps 写成等计算量。[BRIGHT 官方数据说明](https://github.com/xlang-ai/BRIGHT/blob/main/Dataset_documentation.md)把全部数据放在 test split，因此额外 sweep 应作为敏感性分析，避免在 BRIGHT 上反复选出“最优”超参数后又把同一分数当独立测试结果。
