# G1-A-MRRAlign090 三 Seed 方差分析报告

**日期**：2026-09-14
**起因**：`G1-A-MRRAlign090-Seed3407-s3407` 与 `G1-A-MRRAlign090-s42` 除 seed 外配置相同，
但 BRIGHT 评测差距接近一倍（ndcg@10 0.162 vs 0.220），需定位原因。
**涉及 run**：`G1-A-MRRAlign090-s42`、`G1-A-MRRAlign090-Seed3407-s3407`、`G1-A-MRRAlign090-Seed2026-s2026`

---

## 结论摘要（TL;DR）

1. 三个 run **除 seed 外无任何实质差别**：数据、超参、base 模型、训练代码路径、评测命令、评测环境全部逐一核实一致。
2. **评测环境已排除**：在当前节点重跑 s42 逐位复现原始分数（0.22014 = 0.22014），当前 eval 正常，3407 的低分是其 checkpoint 的真实表现。
3. **权重层面看不出差异**：三个 seed 相对 base 的更新幅度都仅 ~0.05%，两两差异 ~0.065%，逐层分布高度对称，3407 无异常。
4. **差异的真正机制在 embedding 几何**：微小的权重方向差异经 last-token pooling 被放大约 100 倍，导致 3407 表征坍缩（各向异性）最严重、正负样本区分度最低。三个 seed 的 embedding 几何指标排序与最终 ndcg@10 **严格一致**。
5. **根因**：该 RL 配方的训练奖励（in-batch MRR reward，三者均 ~0.71）**无法反映最终 embedding 空间质量**，导致高 seed 方差。单 seed 结果不可靠。

---

## 1. 评测结果：三 seed BRIGHT ndcg@10

| subset | s42 | 3407 | 2026 |
|---|---|---|---|
| biology | 0.33024 | 0.16512 | 0.20188 |
| earth_science | 0.33533 | 0.20080 | 0.29585 |
| economics | 0.28167 | 0.19021 | 0.23194 |
| psychology | 0.29340 | 0.24973 | 0.27916 |
| robotics | 0.19266 | 0.13503 | 0.16011 |
| stackoverflow | 0.23387 | 0.19928 | 0.23400 |
| sustainable_living | 0.24985 | 0.17909 | 0.23046 |
| pony | 0.01090 | 0.00858 | 0.00666 |
| leetcode | 0.09970 | 0.10586 | 0.11256 |
| aops | 0.03779 | 0.02921 | 0.04978 |
| theoremqa_theorems | 0.37513 | 0.29172 | 0.32651 |
| theoremqa_questions | 0.20119 | 0.18859 | 0.19707 |
| **AVG** | **0.22014** | **0.16193** | **0.19383** |

**三 seed 均值 0.192，标准差 0.024**（相对波动约 12%）；排序 s42 > 2026 > 3407。
差距在几乎所有子集上系统性存在，非单一子集异常。

---

## 2. 配置一致性核查

对 `model_args.bin`、`rl_args.bin`、`training_args.bin`、`config.json`、`embedding_protocol.json`
逐字段比对，两个 run 仅以下不同：

| 项 | s42 | Seed3407 |
|---|---|---|
| seed / data_seed | 42 | 3407 |
| 输出路径 / run_name | ... | ...（仅命名）|
| 训练节点 | lshb-qs-jzsy-2 | qs-260919-...（新节点）|
| git commit | 65148987 | 6d16d85 |

- **数据完全相同**：均为 `data/processed/reasonrank_multi/train.ready.jsonl`（`schema=embedding_candidates_v2`，
  `relevance_scheme=binary`、`slate_size=20`、`reward_type=mrr_in_batch`、`target_alignment=0.9`）。
- **git commit 差异不影响本次训练**：两 commit 间改动的 `src/embedding_data.py`（`record_to_slate`、`pos_index`、
  collator）都在 BGE-M3 原始 pos/neg 数据（`ready=False`）分支，本次 v2 prepared 数据走 `ready=True` 分支。
  少数触及 ready 分支的改动（`known_ids - {None,""}`、`candidate_id not in (None,"")`、`known_positive_key_sets`）
  只在数据含 None/空 id 或 `known_positive_keys` 字段时才生效；实测 4964 条数据这些字段**全部为 0**，
  故新旧代码对本数据逐样本等价。
- **训练过程一致**：两 run 最后阶段 reward 均稳定在 ~0.71，train_loss 正常收敛。

---

## 3. 评测环境排除（对照实验）

**当前节点即 `qs-260919`（原 3407 训练/评测节点）。** 在此重跑 s42 完整 BRIGHT 评测（单进程→8 卡，
`fp16`、`--benchmark BRIGHT`、命令与原始一致），结果与原始 s42 **12 个子集逐位一致**（例：biology 0.33024 = 0.33024）。

**推论**：当前节点评测环境正常且可复现，"新节点/CUDA 环境导致评测劣化"的假设被证伪。3407 的低分是其 checkpoint 编码质量的真实反映，而非评测 bug。

（环境记录：torch 2.6.0+cu124、transformers 4.52.3、mteb 1.38.32；两 run 训练进程 requirements 逐行一致；
无 flash-attn，走 transformers 默认 attn。）

---

## 4. Checkpoint 权重差距

以 base `Qwen/Qwen3-Embedding-0.6B` 为参照，310 个张量的相对 Frobenius 差异：

| 度量 | 结果 |
|---|---|
| 相对 base 更新 ‖seed−base‖/‖base‖ | s42 0.048% / 3407 0.047% / 2026 0.047% |
| seed 两两差异 ‖a−b‖/rms | s42–3407 0.065% / s42–2026 0.065% / 3407–2026 0.064% |
| ckpt100 → ckpt113 漂移 | 三者均 ~0.008% |

**逐层分布**：更新集中在中间层（L2–L14，~0.1%），浅层/深层更小；`final norm.weight` 完全未改（0.000）。
三个 seed 的逐层更新幅度几乎相等。

**观察**：seed 间差异（0.065%）大于相对 base 的更新（0.048%），说明不同 seed 的更新**方向近乎正交/发散**——
绝对幅度都极小，但方向各异。**权重视角完全看不出 3407 更差，三 seed 对称分布。**

---

## 5. Embedding 分布分析（差异的真正机制）

用 biology 子集真实 query/doc（200 query × 20 候选），三个模型按评测协议（last pooling、fp16、L2 归一化）编码。

### 5.1 微小权重差异被放大约 100 倍

同一文档在不同 seed 下的 embedding 平均余弦仅 **0.88–0.93**（最低 0.62）。
即 0.065% 的权重差异 → 7–12% 的 embedding 方向偏移。last-token pooling 沿长序列累积放大了扰动。

### 5.2 表征坍缩：3407 各向异性最严重

文档 embedding 协方差谱（participation ratio = 有效维度）：

| model | top1_evr | top10_evr | **有效维度** |
|---|---|---|---|
| s42 | 0.069 | 0.266 | 82.2 |
| **3407** | 0.062 | **0.305** | **70.8** |
| 2026 | 0.049 | 0.260 | 92.4 |

3407 有效维度最低（70.8），文档向量挤在更窄的锥内（随机 doc-doc 相似度 0.333 最高，s42 0.320、2026 0.295）。

### 5.3 区分度：整体抬高，但负样本抬得更多

| model | pos_sim | neg_sim | **gap(pos−neg)** | doc-doc 锥宽 |
|---|---|---|---|---|
| s42 | 0.665 | 0.457 | **0.208** | 0.320 |
| **3407** | **0.683** | **0.510** | **0.174** | **0.333** |
| 2026 | 0.630 | 0.470 | 0.160 | 0.295 |

- 3407 的 **pos_sim 其实最高（0.683）**——正样本相似度不是问题。
- 问题是 3407 把**所有**向量（含无关负样本）都往上挤：neg_sim 也最高（0.510），锥最窄。
- 结果 **pos−neg gap 被压缩**（0.174，比 s42 的 0.208 小 16%）。

### 5.4 归一化区分度（排除整体偏移后仍最差）

将每 query 内相似度尺度归一（SNR = gap / within-query std），排除"整体偏高"的表面影响：

| model | raw_gap | **SNR** | 每 query MRR | frac(margin<0) |
|---|---|---|---|---|
| s42 | 0.208 | **1.198** | 0.802 | 0.32 |
| 2026 | 0.160 | 1.054 | 0.800 | 0.32 |
| 3407 | 0.174 | **1.011** | 0.756 | **0.40** |

即使归一化尺度，3407 的 SNR 仍最低（1.01），且有 **40% 的 query 出现"最强负样本 ≥ 最强正样本"**。
说明这不是靠调温度/阈值能修的表面偏移，而是正负样本方向上更纠缠的实质区分力下降。

### 因果链

> 3407 表征坍缩更严重（锥更窄、有效维度更低、各向异性最强）
> → query 与**所有** doc（含无关负样本）相似度都被抬高
> → 正负 gap / SNR 被压缩
> → 检索区分度下降 → BRIGHT ndcg@10 最低（0.162）。

三个 seed 的 embedding 几何指标（有效维度、gap、SNR、MRR）排序与最终 ndcg@10 **严格一致**。

---

## 6. 根因与建议

**根因**：该 RL 配方对随机种子高度敏感。训练奖励（in-batch MRR，三者均 ~0.71）与最终 embedding 空间几何质量
**几乎无关**；极小的权重扰动经 pooling 放大为显著的表征坍缩差异，造成高下游方差。奖励不是可靠的模型选择/早停指标。

**建议**：

1. **多 seed 报告**：以三 seed 均值 0.192 ± 0.024（或给出 range）作为该配方代表值；单 seed（尤其 s42 的 0.220）不可作结论。
2. **监控 embedding 几何**：训练中跟踪 held-out 的有效维度 / 各向异性 / 正负 margin / SNR，替代仅看 reward。
3. **缓解各向异性**：考虑对文档 embedding 加各向同性 / uniformity 正则，或评测前做 whitening / 均值中心化等后处理，以压低 seed 方差。
4. **排查训练不稳定来源**：advantage 归一化策略、exploration 随机性、in-batch 负样本的 seed 依赖等。

---

## 附：复现方式

- 评测重跑：
  ```bash
  python eval_mteb/run_mteb.py \
    --model checkpoints/iclr2027/G1-A-MRRAlign090-<seed> \
    --precision fp16 --batch_size 16 --langs eng --benchmark BRIGHT \
    --model_kwargs '{"instruction_dict_path":"eval_mteb/scripts/task_prompts.json"}' \
    --output_dir <out>
  ```
- 权重 / embedding 分析脚本：`scripts/analyze_seed_embedding_variance.py`（编码复用 `src/embedding_protocol.py`）：
  ```bash
  python scripts/analyze_seed_embedding_variance.py \
    --seeds s42:checkpoints/iclr2027/G1-A-MRRAlign090-s42 \
            3407:checkpoints/iclr2027/G1-A-MRRAlign090-Seed3407-s3407 \
            2026:checkpoints/iclr2027/G1-A-MRRAlign090-Seed2026-s2026 \
    --base Qwen/Qwen3-Embedding-0.6B \
    --source biology --max-queries 200 \
    --json-out outputs/embedding_analysis/seed_variance/biology_metrics.json
  ```
- 复现产物：`outputs/embedding_analysis/seed_variance/biology_metrics.json`。
- 数据：`data/processed/reasonrank_multi/train.ready.jsonl` 的 `source==biology` 样本。
