# ICLR 2027 分析实验与论文图计划

更新：2026-09-25。本计划与[结果表规划](ICLR2027_RESULTS_REORGANIZATION.md)采用同一主配方：0.6B、`K=0`、graded nDCG + pairwise `λ=0.5`、RLOO+CMP、`G=64`、`ρ=0.70`。本文件记录图及分析数据。正文按用户决定同时呈现检索与固定索引 RAG，RAG 采用结果规划中的紧凑表，不额外增加曲线。论文已填入新 `K=0` 主配方的 0.6B 结果；64 次 original-query 与 10 次 GPT-reasoning-query 评测及全部子集数据已齐。以下区分已接入的 F1 正文图、F2 附录训练曲线、F3 的 12 子集三 seed 领域差值图和 F3/F5 单 seed 四领域附录案例，以及尚未完成的跨 seed 查询行为分析。

新训练及已完成结果的统一输出见[0.6B 独立 suite](../docs/iclr2027_final_k0_suite.md)。代表 checkpoint 配置已指向该新根目录；旧结果需先在训练机执行 `import` 才能由分析入口读取。

## 图稿清单与优先级

| 位置 | 图 | 问题与最终画法 | 当前证据 / 缺口 |
|---|---|---|---|
| 正文，优先 | **F1：匹配条件下的梯度方差** | **正文仅保留紧凑单面板梯度云**：代表 E0 probe 的 64 次配对 rollout，CMP 放大插图；两轴是原有归一化随机投影，图注说明定量降噪 290–590 倍来自完整梯度、覆盖全部 9 probe。图与短图注左右排列。原“梯度云 + SNR”双面板移至附录 E.3，并在那里解释 SNR 定义及参考线。 | 已有 local-pool、单 reward 的探针可作附录历史对照；新主配方 `K=0 + pairwise` 的完整估计器对照已于 2026-09-25 运行完毕（三个固定状态 × 3 microbatch × 64 draws，结果见 [`gradient_probe_k0_pairwise`](analysis_results/gradient_probe_k0_pairwise/README.md)），V_CMP/V_RLOO ≈ 0.002–0.003；同日完成逐 draw 投影补采并按此设计出图。 |
| 附录 E.5 | **F2：训练 reward 动态** | 两面板：A，`K=0` 的 RELER 与纯 graded 两组三 seed 的**同定义** graded nDCG，横轴 optimizer step；B，仅 RELER 的 graded 项、`0.5×pairwise` 项与 combined reward。三 seed 均值加样本 SD；聚合与单 seed 原线均放附录。 | 六条 `K=0` history 已于 2026-09-24 导出至 [`reward_curves_k0`](analysis_results/reward_curves_k0/README.md)（主配方三 seed 当日完成，最终 BRIGHT Avg. 23.38 ± 0.17）；`scripts/export_reward_curves_k0.py` 可复算。 |
| 附录 E.1/E.5 | **F3：收益落在哪些领域/查询** | A，12 个 BRIGHT subset 上 `RELER−InfoNCE` 的三个配对 seed 点和均值；B，固定 query ID 的最佳正例 rank 迁移，分 1、2–10、11–100、101+ 桶，对照 E0/InfoNCE/RELER。领域图用于完整结果展开；查询级案例独立标明分析范围。 | 12 subset 的三 seed 配对差值图已从主表来源绘制并放附录（`bright_domain_deltas_k0/`）；四个预定领域的 seed 3407 query-level 排名缓存与案例图也已完成。原定跨 12 subset、三 seed 查询迁移图仍待补；不能从领域均值倒推。 |
| 附录 | **F4：旧条件下的梯度证据** | 用已有 local-pool、纯 graded/MRR、`ρ=0.90` 的配对探针，画各 source microbatch 的 `V_CMP/V_RLOO` 与时间比；另可画旧 G16/32/64 的同目标方差比。图注清楚区分与 F1 的 reward、候选池和 alignment。 | 原始 JSON 已有；先复核 `outputs/g1_r2_gradient_probe/` 和 `outputs/rollout_gradients/` 的协议、输入 hash 与逐 probe 数值。旧 G 结果按[订正分析](../docs/rollout_gradient_reanalysis.md)解释，不引用旧报告的错误噪声比推断。 |
| 附录，按需要 | **F5：检索行为** | 四个预定领域（Biology、Earth Science、Robotics、TheoremQA theorems）的相同 query/文档对 margin 或正例 rank 变化；如做扰动图，注明 candidate reranking 和固定的 query-side vMF 扰动。 | [代表 checkpoint 配置](analysis/bright_representatives.json)已指向新 `K=0` 权重，分析管线与官方逐领域分数最大绝对差 0.0017；四领域单 seed 分析及附录案例图已完成，见 `analysis_results/iclr2027_representatives_k0_s3407/`；图同时显示优势与劣势领域，不把单 seed 当三 seed。 |

正文实验仅保留 F1 紧凑单面板梯度云，完整双面板位于附录 E.3；F2 聚合与单 seed 曲线均放附录 E.5，F3 的 12 子集差值图放 E.1，四领域单 seed F3/F5 案例图放 E.5。F4 的历史表格集中于 E.7。这里的 F1–F5 是分析任务编号，不是论文 Figure 编号；以 LaTeX 交叉引用为准。

## F1：主配方固定状态梯度探针

目的：检验 CMP 对**完整 encoder 梯度**的方差作用，而不是只展示单个 score-function 项的理论保证。预定状态为 E0、0.6B 主配方 seed 3407 的 step 50 与 step 113；若 step 50 权重未保存，记录缺失，不重训取点。固定三个训练 microbatch（沿旧 probe 位置 0/100/200，记录实际 source、query ID 和 tensor hash）；每个状态/批次使用相同的 `K=0` 自有候选、64 次独立 rollout，且同一 draw 的动作、reward 与 shortlist 身份在两估计器间完全配对。先固定候选集合测 rollout 条件方差；`K=0` 没有额外候选重采样，所以此处不需要 K7 式的第二层 shortlist 方差实验。关闭 dropout，不做 optimizer 更新。

需要先扩展 `scripts/diagnose_rollout_gradients.py`：完整捕获 shortlist reward 表；为 graded 项与逐 pair 项分别提供正确的未投影 RLOO 对照，然后组成与主方法**相同**的 `graded + 0.5×pairwise` reward。只切换现有 `gradient_estimator` 仍会留下 pairwise CMP，因此不能算完整的无 CMP 对照。正式测量前，在小张量上核对两估计器使用逐 cell 同一 reward/动作、均值梯度在 MC 误差内相容，并保存实现版本及配置。

实现状态（2026-09-25）：扩展已完成。`--compare-gradient-estimators` 遇 `reward_shortlist_count>0` 自动走 shortlist 配对模式（`probe_paired_shortlist_gradients`），逐 draw 捕获并 hash 校验动作、shortlist 身份、graded reward 表与 pairwise 输入；RLOO 对照为脚本内 `rloo_pairwise_shortlist_loss`：graded 项沿用训练 score-function 分支，pairwise 项与 CP 版同 pair 集、同 0/0.5/1 reward、同逐轴 LOO 与归一化，仅把梯度载体换成 vMF log-prob（document 侧端点局部，对应 CP 的端点局部性；各候选独立采样保证无偏）。`--self-check` 合成张量校验通过：reward 逐对一致、两估计器均值梯度在 5×MC 误差内相容、RLOO 均值梯度与有限差分参考一致（CP 对 FD 的残差仅报告不门限——两估计器同坐标同向偏离而彼此一致，指向参考噪声而非估计器）。分项梯度经 objection pass 记录（同 seed 重放，pairwise 置零 / graded reward 置零，记范数、夹角与 `g_total−(g_graded+λ·g_pairwise)` 分解残差），`--no-component-passes` 可跳过。E0 状态 2-draw CPU smoke 全部校验通过（配对 hash 一致、pairwise reward 差恰为 0、均值梯度差在 MC 误差内）。运行入口为本机脚本 `scripts/f1_probe_local.sh`（不入库；显存+利用率双低才认空闲 GPU，支持 `--state` 单状态并行）。**2026-09-25 已完成正式测量**：E0/step50/step113 三状态各 3 microbatch（位置 0/100/200）× 64 draws、含分项 pass，本机三卡并行（每进程峰值 ~33 GB）；结果与图由 `scripts/analyze_f1_probe.py` 汇总至 `paper/analysis_results/gradient_probe_k0_pairwise/`——`V_CMP/V_RLOO` 几何均值 0.0019/0.0029/0.0031（降噪 300–600 倍），`V×t` 比值仍低两个半数量级；9 probe 中 5 个均值梯度差未分辨、其余偏差 ≤ signal² 的 3%；pairwise reward 逐对差为 0；分项夹角从 E0 的 ~0.6 收缩到 step 113 的 ≈0，分解残差 ≤1.4%。同日完成逐 draw 随机投影补采（`--state proj-*`，三状态并行；8 条固定 ±1 轴、种子 20260925，与正式 probe 同种子同 microbatch，汇总脚本逐 draw 比对 hash 一致后才绘图），主图改为"梯度云 + 单步信噪比"双面板；坐标为与范数 √d 符号向量的点积，单轴方差期望即全空间 `noise_variance`，最大实测/期望比 1.19×/1.28×（n=64 的 χ² 波动内），云的半径比 20–23× 与 √方差比吻合。

每个 probe 记录 `V`（完整参数样本方差迹）、`V_CMP/V_RLOO`、`t`（配对前后向计时）、`V×t` 比、均值梯度差相对 MC RMS、两分项梯度范数与夹角，以及峰值显存。先以完整和作 F1，分项及协方差放附录。逐 probe 点与配对值都保留，不把不同 batch 的梯度合并求 rollout 方差。若 64 draw 仍无法分辨均值梯度，标注未分辨；不要把 `noise_rms/‖ḡ‖` 的小样本 plug-in 比值当作主要证据。固定状态方差不等于训练 seed 方差，也不证明端到端提速或 BRIGHT 增益。

已有旧探针见[论文现有附录](iclr2027/sections/additional_results.tex)、[原始诊断](../docs/rollout_rng_diagnostics.md)与[统计订正](../docs/rollout_gradient_reanalysis.md)。旧 `G16/32/64` 在相同目标下的方差约按 `1/G` 缩放；不能引用首次报告中“远慢于 `1/√G`”的推断。只用旧 standalone reward 时应制作 F4，图标题和图注必须明示旧 local-pool 协议。

## F2：训练 reward 曲线

[运行清单](analysis/reward_curve_runs.csv)改为 `K=0` 的 RELER 与纯 graded 各三 seed，均为 113 optimizer steps、CP/G64、`ρ=0.70`、T1，只有 pairwise 系数不同。清单中的六条运行及其 history 均已有导出；已有 K7 清单不是主图数据源。取得 history 时逐一核对唯一 W&B run ID 或本地 `trainer_state.json`、seed、配置、checkpoint 路径和实际 global step；不能用 W&B `_step` 代替 optimizer step。

- **面板 A 的共同量**：`train/reward/mean` 对应 graded nDCG@10；两组均用 `K=0` 自有候选，才可以横比训练 reward 的定义。图注仍说明各 step 的训练 batch 变化，reward 曲线不是固定验证集学习曲线，也不直接代表 BRIGHT 泛化。
- **面板 B 的分项**：从 RELER history 取 graded、pairwise 和 `combined_mean`；验证 `combined ≈ graded + 0.5×pairwise`。Pairwise 项须画加权后的数值，纵轴注明 reward 范围及尺度。若旧日志字段与训练版本不同，以代码版本与原始字段定义为准并写入 manifest。
- **聚合**：以 seed 为重复单位，画逐 step 原始均值和样本 SD；不把 rollout 内 `reward/std` 当 seed 阴影。不对缺步补零，缺任一 seed 就标注 `n`。如用 5-step trailing mean，仅作为显示层，先逐 seed 平滑再聚合，并保留原始图及 CSV。
- **对照最终效果**：图注或正文紧邻处报告相同三 seed 的最终 BRIGHT Avg.；不把训练目标的绝对值排序直接解释成检索质量。InfoNCE/LambdaLoss 的 loss 不放在 reward 纵轴。

现有 `paper/g1_gradient_analysis/raw/` 只含旧 G1 日志；新 K0 六条逐步 history 已导出到 `paper/analysis_results/reward_curves_k0/`，包括：逐运行 `history.csv`、`run_manifest.json`、聚合 `curves.csv`、图注、矢量 PDF 和预览 PNG。图中每个点必须可从 CSV 复算；缺失 history 时先查日志，不为了补图自动重训。`λ=1.0` 是附录的**最终 BRIGHT** 敏感性点，只有其完整可比 history 也齐全且图意确有收益时才加入扩展曲线，不改变主图的两组对照。

若想展示 reward 与检索质量的关系，另在附录对预定的 step 25/50/75/100/113 checkpoint 画 BRIGHT Avg. 轨迹，并标注这是同一 test benchmark 的重复评测；必须用同一评测协议及可用的真实 checkpoint，不从训练 reward 反推缺失的 BRIGHT 点，也不把该轨迹当作独立验证集选模。

## F3/F5：检索行为与 checkpoint 分析

[代表 checkpoint 配置](analysis/bright_representatives.json)用于 seed 3407 的案例分析：E0、InfoNCE、LambdaLoss、`K=0 + pairwise` RELER、纯 graded `K=0`。固定四个领域覆盖优势与劣势；图要报告真实 query 数和筛选规则。先用 `scripts/analyze_embeddings.py` 的无扰动 retrieval 复现各 checkpoint 逐领域 nDCG@10，统一 0–1 与 0–100 尺度，再做 rank、margin 或扰动分析。不得沿用旧 `K=7` 权重填新方法曲线。

如果做训练过程行为图，预定 step 25/50/113，模型与 query ID 固定；缺中间 checkpoint 就删该面板，不用不同 tokenization 的旧评测拼接。若做跨 seed 查询分析，再补 42/2026 的最终权重；单个 seed 3407 只能作为案例。二维 t-SNE/UMAP 不作为主要几何证据，固定文档对的排名和 margin 更接近检索机制。扰动的默认候选重排不能称为全库鲁棒性；若要声称全库效果，另做 corpus scope 重搜。

## 数据交付与图稿验收

F2 已于 2026-09-24 导出（`paper/analysis_results/reward_curves_k0/`）。F1 已于 2026-09-25 完成测量与汇总（`paper/analysis_results/gradient_probe_k0_pairwise/`）。F3/F5 的 seed 3407 四领域最终 checkpoint 案例已运行并绘图（`iclr2027_representatives_k0_s3407/`）；跨 seed 查询行为分析仍待补。附录 F4 可随时从已有 JSON 重绘，但必须采用订正后的统计解释。

所有图保存源数据、run/checkpoint manifest、绘图脚本、矢量 PDF 和预览 PNG；图注注明候选池、reward、alignment、`G`、状态/seed 数及误差棒含义。正文图只使用已核实协议的数据，不绘制空白曲线或预测走势。分析所需的固定状态 probe 与日志导出**不计入**[结果表规划](ICLR2027_RESULTS_REORGANIZATION.md)中已完成的 36 次新训练；若后续决定增加完整训练对照，再单独调整该计数。
