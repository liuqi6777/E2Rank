# F1：K=0 + pairwise 主配方固定状态梯度探针

[分析计划 §F1](../../ANALYSIS_PLAN.md) 的交付目录。数据源：`outputs/f1_k0_probe/pairwise050_seed3407_{e0,step50,step113}_micro0-100-200.json`（探针由 `scripts/diagnose_rollout_gradients.py --compare-gradient-estimators` 生成，本目录由 `scripts/analyze_f1_probe.py` 汇总，2026-09-25 12:49，git `183bd5ec` (dirty)）。

## 图

- `gradient_probe_k0_pairwise.pdf` / `.png` — 正文 F1。面板 A：**单 draw 梯度云**（E0、microbatch 位置 100 的同一批 64 次 rollout：完整参数梯度在 8 条固定随机 ±1 轴上的坐标；横轴取 CMP 均值分量最大的轴、纵轴取分量最小者，均按投影信号归一；灰线连接同一 draw 的两个估计器）。RLOO 的云把原点罩在里面（单步方向被噪声淹没），CMP 收拢成与原点分离的小团；右上插图按 CMP 自身尺度放大该团（64 个 draw 加两估计器均值，RLOO 均值落在 CMP 团簇内部——无偏的视觉版）。面板 B：**单步信噪比**（`√signal²/noise_rms`，对数轴，"signal = noise" 参考线）：CMP 4–6.5，RLOO 0.23–0.32。耗时与比值不单独成面板：CMP 每趟前后向仅慢 ~13%，`V_CMP/V_RLOO` 0.002–0.003（降噪 300–600 倍，等效于把 ~300–600 个独立 rollout 梯度平均才追上 CMP 单步），`V×t` 比值 0.002–0.004，见 `probes.csv` 的 `variance_ratio`/`variance_time_ratio` 列。

## 结果

| 固定状态 | V_CMP/V_RLOO（几何均值） | (V·t)_CMP/(V·t)_RLOO | 降噪倍数 | signal² CMP / RLOO |
|---|---:|---:|---:|---:|
| E0 | 0.0019 | 0.0022 | 524× | 93.6 / 102.1 |
| step 50 | 0.0030 | 0.0033 | 339× | 46.7 / 47.2 |
| step 113 | 0.0031 | 0.0035 | 323× | 27.4 / 27.6 |

- **CMP 把完整 encoder 梯度的 rollout 方差降低约 300–600 倍**，三个固定状态、九个 probe 一致；扣除前后向耗时后（CMP 慢 ~13%）仍低约两个半数量级。
- 无偏性：5/9 个 probe 的两估计器均值梯度差**未分辨**（无偏平方差 ≤ 0）；其余分辨出的偏差 ≤ signal² 的 3%。两估计器的 signal² 估计逐 probe 相符。
- 配对完整性：所有 probe 的 pairwise reward 逐对差 = [0.0]（bit 级一致）；动作/shortlist/graded reward hash 逐 draw 相等（探针内强制校验）。
- 分项（附录素材，见 `probes.csv`）：graded 与 pairwise 梯度范数、夹角随状态变化（夹角从 E0 的 ~0.6 收缩到 step 113 的 ≈0）；分解残差 `‖g_total−(g_graded+0.5·g_pairwise)‖/‖g_total‖` 最大 1.4%。

## 协议

0.6B 主配方 seed 3407（`K=0` + pairwise `λ=0.5`、RLOO+CMP、`G=64`、`ρ=0.70`、`T=1`）的固定权重：E0、step 50、step 113。每状态三个训练 microbatch（位置 0/100/200，实际 source/query ID/tensor hash 见 JSON）；每批次 64 次独立 rollout，同一 draw 的动作、reward 与 shortlist 身份在两估计器间完全配对（种子重置 + hash 校验）。RLOO 对照 = 训练 score-function 分支（graded 项）+ `rloo_pairwise_shortlist_loss`（pairwise 项，端点局部 score-function；reward/LOO/归一化与 CP 逐项一致），即完整的无 CMP 对照。关闭 dropout、无 optimizer 更新。前置校验：`--self-check` 合成张量（reward 一致、均值梯度相容、与有限差分参考一致）与 E0 32-draw precheck。

梯度云面板的数据来自同协议的投影补采（`pairwise050_seed3407_e0_micro0-100-200_proj.json`，不入库）：同种子、同 microbatch、关分项 pass，逐 draw 记录完整梯度在 8 条固定 ±1 随机轴（种子 20260925，两估计器与各 probe 共用）上的点积坐标；汇总脚本逐 draw 校验动作/shortlist/graded reward/pairwise 输入 hash 与正式 probe 完全一致后才绘图。坐标为与范数 √d 的符号向量的点积，故单轴方差期望即全空间 `noise_variance`；各轴最大实测/期望比：CMP 1.19× / RLOO 1.28×（n=64 的 χ² 波动内）。2D 投影按期望保持相对散布，定量结论以面板 B 与 `probes.csv` 为准。

## 文件

- `probes.csv` — 每个 probe 一行：方差比、V×t、两估计器的 V/noise_rms/mean norm/signal²/耗时、均值差分辨标记与偏差占比、分项统计、来源与 hash。
- `gradient_cloud_e0_micro100.csv` + `_meta.json` — 面板 A 的逐 draw 数据：64 draw × 两估计器的归一化坐标与所选两轴的原始点积；meta 记录轴选择/定向、投影信号尺度、两估计器均值、CMP σ、插图窗口与缩放、轴方差校验、配对与 hash 校验说明（同 draw 序号 = 同一 rollout）。
- `snr_probes.csv` — 面板 B 的 9 个点（state / microbatch / snr_cp / snr_rloo）。
- 原始 JSON 不入库（`outputs/` 被忽略）；`probes.csv` 的 `git_commit` 与 `diagnostic_sha256` 可定位生成版本。

## 边界（图注须带上）

固定状态方差不等于训练 seed 方差，也不证明端到端提速或 BRIGHT 增益；方差×时间的比值假设探针 pass 的前后向成本与训练步等价（CMP 的投影计算发生在 loss 侧，训练步差异见 RL 组件表的 wall-clock）。每状态仅 3 个 probe，点不合并、不当独立 seed 用。梯度云为单一代表 probe 的 2D 随机投影，只按期望保持相对散布，不新增定量结论。
