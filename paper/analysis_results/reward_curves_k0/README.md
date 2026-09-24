# F2：K=0 训练 reward 曲线

[分析计划 §F2](../../ANALYSIS_PLAN.md) 的交付目录。生成命令（仓库根目录）：
`python scripts/export_reward_curves_k0.py`；生成于 2026-09-24 22:36，git `85c2077c3ef93ca2e7532eefba6e9ce4f0d05979` (dirty)。

## 图

- `reward_curves_k0_main.pdf` / `.png` — 正文 F2。面板 A：RELER（`K=0` + pairwise `λ=0.5`）与纯 graded 对照各三 seed 的**同定义** graded nDCG@10 训练 reward（optimizer step 1–113）；面板 B：RELER 的 graded 项、`0.5×pairwise` 项（加权后数值）与 combined reward。带为逐 step 三 seed 样本 SD（**不是** rollout 内 `reward/std`）。
- `reward_curves_k0_per_seed.pdf` / `.png` — 附录单 seed 原线；颜色 = 组/分项（与主图一致，线端直接标组名），seed 身份由 marker 编码（○ s42、□ s3407、△ s2026）。

## 拟用图注要点

- 协议：Qwen3-Embedding-0.6B 全参数微调，shortlist 路径 `K=0`（仅自有候选，`T=1`），graded nDCG@10（teacher grades）+ pairwise `λ=0.5`，RLOO+CMP（`conditional_projection`），`G=64`，`ρ=0.70`，113 optimizer steps，global batch 128，paired seeds 42/3407/2026；两组只有 pairwise 系数不同。
- 每个 step 的训练 batch 不同：曲线是训练目标值的轨迹，不是固定验证集学习曲线，也不直接代表 BRIGHT 泛化。
- 面板 B 纵轴为 reward 原值：graded 项与 pairwise 项同为 [0,1] 尺度但语义不同（nDCG vs 配对指示 reward）；combined = graded + 0.5×pairwise。
- 相同三 seed 最终 checkpoint 的 original-query BRIGHT 12-subset 宏平均（×100）：Graded-only 22.72 ± 0.18 · RELER 23.38 ± 0.17。SD 为全精度逐 seed 值的样本标准差；若与正文表对数，注意结果表的 ± 来自两位小数逐 seed 值（本组即 0.17/0.17），与全精度在舍入边缘可能差 0.01。

## 数据核对

- 步数完整：六条运行 log_history 均为 step 1–113 无缺步；`trainer_state.global_step` = 113，`exploration_state.step` = 113。
- combined 恒等式 `combined = graded + 0.5×pairwise`：全部成对 step 成立，最大绝对偏差 1.00e-06（G1-R2-RL-GradedNDCG64-CP-Align070-ShortlistUniform-K0-T1-Pairwise050-s42）。
- 字段映射：W&B 键 `train/reward/mean` 等对应本地 `trainer_state.json` 的 `reward/mean` 等（trainer 加 `train/` 前缀）；语义以 `src/grpo.py`、`src/pairwise_projection.py` 的当前版本为准（sha256 见各 `run_manifest.json`）。这些运行无本地 W&B 工件，本地 `trainer_state.json` 为逐步数据权威来源。
- 纯 graded 组为从 `suite_g1_shortlist_small_k.yaml` 导入的已完成运行，RELER 组为新 suite 训练；两组配置除 `reward_shortlist_pairwise_coef` 外逐字段一致（见 manifest）。

## 文件

- `runs/<run>/history.csv`、`runs/<run>/run_manifest.json` — 逐步原始字段与来源核对。
- `curves.csv` — 面板/系列聚合（mean、sample SD、n_seeds）；`curves_per_seed.csv` — 图中每个点的原值，可复算全图。
- `bright_final.csv` — 六条运行最终 checkpoint 的 BRIGHT 12-subset nDCG@10（×100）宏平均。
- 图中不补缺步、不做平滑；如后续加 5-step trailing mean 仅作显示层并另行保留原始图。
