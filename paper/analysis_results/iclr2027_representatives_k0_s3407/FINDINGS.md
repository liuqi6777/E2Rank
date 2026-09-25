# 代表 checkpoint embedding 分析结论（F3/F5 数据基础）

更新：2026-09-25。seed 3407 案例分析，`paper/analysis/bright_representatives.json`（E0、InfoNCE、LambdaLoss、Listwise=`K=0` 纯 graded、RELER=`K=0`+pairwise λ=0.5）× 4 个 BRIGHT 领域（biology、earth_science、robotics、theoremqa_theorems，original query）。产物由 `scripts/analyze_embeddings.py` 生成，报告快照 `reports/0cbf2c8d82934128/`。本文件只汇总已实测数字；所有值可从 `retrieval/queries.csv`、`paired_queries.csv`、`rank_bins.csv`、`perturb.csv`、`geometry.csv` 复算。

## 协议复现验证

分析管线（fp16、领域 instruction、trec_eval 同分规则、top_k=1000 外正例精确 rank）与官方 MTEB 逐领域评测对照，20 格最大绝对偏差 **0.0017**（LambdaLoss biology），平均 **0.0008**，量级与 fp16 近同分抖动一致。协议可信，图可用此数据。

| nDCG@10 | biology | earth_science | robotics | theoremqa_theorems |
|---|---|---|---|---|
| E0 | 0.1332（官方 0.1341） | 0.2782（0.2781） | 0.1197（0.1196） | 0.3209（0.3210） |
| InfoNCE | 0.3347（0.3360） | 0.3289（0.3295） | 0.2077（0.2093） | 0.3259（0.3259） |
| LambdaLoss | 0.2401（0.2418） | 0.2502（0.2498） | 0.1760（0.1757） | 0.2925（0.2925） |
| Listwise | 0.3998（0.3985） | 0.3795（0.3799） | 0.1799（0.1791） | 0.2891（0.2898） |
| RELER | 0.4659（0.4674） | 0.3814（0.3800） | 0.1816（0.1804） | 0.2908（0.2898） |

4 领域宏平均：RELER 0.3299 > Listwise 0.3121 > InfoNCE 0.2993 > LambdaLoss 0.2397 > E0 0.2130，与主表（original query）排序一致。

## 结论 1：收益集中在 E0 的"边缘查询"

`rank_bins.csv`，按 E0 最佳正例 rank 分桶的 ΔnDCG（4 领域均值；桶内 query 数占比固定）：

| 模型 | 1（16.7%） | 2–10（27.8%） | 11–100（25.3%） | 101+（30.3%） |
|---|---|---|---|---|
| InfoNCE | −0.134 | +0.119 | +0.239 | +0.055 |
| LambdaLoss | −0.186 | +0.031 | +0.192 | +0.020 |
| Listwise | −0.077 | +0.123 | +0.261 | +0.058 |
| RELER | −0.113 | +0.128 | **+0.297** | **+0.096** |

- 收益主要来自 rank 11–100 桶（RELER +0.297）；rank 101+ 深水区改善有限；E0 已排第一的查询回退（该桶已近天花板）。
- **pairwise 项（RELER−Listwise）的增量集中在 11–100 与 101+ 两个难桶**（+0.297/+0.096 vs +0.261/+0.058），2–10 桶两者相同。

## 结论 2：领域分布不均匀（劣势领域需如实报告）

逐领域差值（分析值）：

- RELER−InfoNCE：biology **+0.131**、earth_science +0.053、robotics **−0.026**、theoremqa_theorems **−0.035**。
- RELER−Listwise（同协议仅差 λ）：biology **+0.066**，其余三领域均 +0.002——**pairwise 的宏平均增益几乎全部来自 biology**。

F3 若进正文，robotics/theoremqa 的负值必须在图中可见，不得只展示优势领域。

## 结论 3：query 扰动下 RELER 最稳健

`perturb.csv`（scope=candidates，每强度 64 samples，逐 query 固定种子；干净基线为隐式 alignment=1.0）。nDCG 相对自身干净基线的相对下降（4 领域均值，%）：

| alignment | 1.00 | 0.99 | 0.95 | 0.90 | 0.80 | 0.70 |
|---|---|---|---|---|---|---|
| E0 | 0 | −0.17 | −0.38 | −0.64 | −1.20 | −1.86 |
| InfoNCE | 0 | +0.15 | +0.09 | −0.14 | −0.80 | −1.81 |
| LambdaLoss | 0 | +0.04 | −0.10 | −0.44 | −1.18 | −2.06 |
| Listwise | 0 | +0.06 | 0.00 | −0.14 | −0.48 | −1.05 |
| RELER | 0 | −0.06 | −0.03 | −0.12 | −0.39 | **−0.82** |

- 最强档（0.70）RELER 下降最小；严格正确 pair 翻转率（0.90 档）同为 RELER 最低：5.2%（Listwise 6.1%、InfoNCE 8.2%、E0 8.7%、LambdaLoss 10.4%）。
- 0.95/0.99 弱档差异在采样噪声量级（逐 query `ndcg_mc_se` ≈ 0.003，见各模型 `perturb/queries.csv`），**不作为信号**；聚合表 `perturb.csv` 未含 SE 列，画图需自算。

## 结论 4：检索改进不依赖表示几何重塑

`geometry.csv`（4 领域均值）：

| 模型 | effective rank（query） | uniformity | ‖均值向量‖ | 邻域保留率（相对 E0） |
|---|---|---|---|---|
| E0 | 168.7 | −2.83 | 0.519 | 1.000 |
| InfoNCE | 129.7 | −1.88 | 0.717 | 0.660 |
| LambdaLoss | **76.0** | **−0.94** | 0.870 | 0.624 |
| Listwise | 155.3 | −3.21 | 0.396 | 0.679 |
| RELER | 152.6 | −3.12 | 0.421 | 0.638 |

- LambdaLoss 出现典型各向同性坍缩（effective rank 76、向量范数 0.87）；**RELER/Listwise 的几何与 E0 基本持平**——CMP 训练的增益不是靠重塑表示空间。
- `paired_queries.csv`：RELER 在 E0 选定的固定文档对上 margin 提升最大（Δfixed margin **+0.156**，Listwise +0.102、InfoNCE +0.125），与最稳健的扰动表现一致。
- 代价：所有微调模型只保留 E0 query–query 邻域的 63–72%（RELER 0.668），排序重组是实质性的，不是微调微扰。

## 一句话结论

RELER（`K=0` graded + 0.5×pairwise，RLOO+CMP）的收益主要来自把 E0 排名 11–100 的边缘查询推进前 10（biology 域贡献大头），并在 query 侧 vMF 扰动下最稳健、表示几何不坍缩；代价是 E0 已解决查询的小幅回退、robotics/theoremqa 略降。

## 局限与图注要求

- **单 seed（3407）案例分析**，不构成 seed 间显著性；跨 seed 查询分析需另补 42/2026 最终权重。
- 4 个预定领域，不是 12 领域全量；F3 面板 A 的正式数据仍是 12 subset × 3 seed 配对。
- 扰动为候选池内重排（`scope=candidates`），**不能解释为全库鲁棒性**；未标注候选不是已验证负例。
- 邻域保留率、rank 桶、固定对均以 E0 为参考（`reference=E0`）。
- E0 官方对照取自单次评测。

## 复现

```bash
# 编码+检索（已有缓存会校验复用）
python scripts/analyze_embeddings.py all --config paper/analysis/bright_representatives.json
# 注意：perturb 需用去掉 alignments 中 1.0 的配置副本（脚本要求开区间 (0,1)，1.0 基线为隐式生成）
```

已知问题：① `paper/analysis/bright_representatives.json` 的 `perturb.alignments` 含 1.0，按原配置跑 perturb 必报 `ValueError`，本次用临时副本修正；② 并行跑同一 config 时同 subset 的共享 `data/` 发布存在竞态（Errno 39），需按 (model, subset) 分区并行；③ `docs/embedding_analysis.md` 引用的 CPU 回归测试 `scripts/tests/test_embedding_analysis.py` 本机 checkout 不存在。
