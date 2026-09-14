# G1 rollout 方差实验结果

> **2026-09-15 原始 JSON 重分析订正：**下文保留首次报告。其“G 收益强烈次于 √G”“换 reward 没有改善”及“已定位为 vMF 固有噪声”等判断过强；实际 G 翻倍时噪声方差约减半，原 `noise/mean` 分母含有 16 次采样均值的噪声，压缩了条件差异。逐文档 baseline 的负结果仍有直接方差证据支持。修订结论与逐 probe 数值见[原始梯度重分析](rollout_gradient_reanalysis.md)。

日期：2026-09-14。本文汇总[诊断使用文档](rollout_rng_diagnostics.md)中 Section 1（固定状态梯度诊断）与 Section 2（独立 rollout 训练对照）的实测结果，并按[实验计划](rollout_variance_experiment_plan.md) §2 给出判断。证据边界沿用[讨论复盘](rollout_variance_discussion.md)：固定状态梯度噪声与完整训练 seed 差距是**两个独立层面**的证据，本文不把二者合并成单一因果结论。

## 结论摘要（TL;DR）

1. **固定模型 + 固定 batch**：只换 rollout seed，reward 几乎不变（16 draw 极差 ≤0.015），但梯度方向近乎正交（`mean_pairwise_cosine` ≈ 0.004–0.020），噪声/均值比 ≈ 3.5–3.9。E0 与 step-100 一致；按 global-batch-128 结构平均 8 个 microbatch 后该比值仍为 3.68–3.85，未见衰减。
2. **固定 data、仅换 rollout seed 的完整训练**：三行 BRIGHT 宏平均 0.16992 / 0.18517 / 0.18591，均值 0.18033，样本标准差 0.00903，极差 0.016。分散**真实存在但明显小于**历史三行（同时变数据流程与动作采样：均值 0.19197、标准差 0.02915、极差 0.058）。
3. **判断**：rollout 随机性足以造成可测的训练分化（与固定状态高噪声方向一致），但**单因素不足以复现原始 seed 差距的量级**；宏平均比单领域稳，单领域仍剧烈波动（如 biology 0.132↔0.282）。reward 不是可靠选模信号（三行后期 reward 0.704/0.713/0.722，几乎不可区分）。
4. **固定状态换 reward / 换逐文档 baseline（§3）**：graded nDCG 的可分辨 reward 档位是 MRR 的约 60 倍（223 vs 3.9），但固定状态的相对噪声 `noise/mean` 几乎不变（E0 3.72↔3.72，final 3.65→3.43 仅小幅），方向仍近乎正交。逐文档反事实 baseline 无改善、在 final 模型上反而更差。8-mb 平均对全部 5 档也不降该比值。**降低固定状态梯度噪声的瓶颈不在 reward 粗粒度或文档信用分配**，更像是 vMF score-function 估计器在当前探索尺度下的固有噪声。
5. **固定状态换探索尺度（§3.3）**：alignment/κ 对 `noise/mean` 的影响跨 E0/final 不一致，不可靠；**只有加大 group size G 在两个状态都单调降噪、增方向一致性（G16→64：~3.9→3.3）**，但**强烈次于 √G**（翻倍仅降 ~8–10%），单一超参难以把噪声压到 O(1)。

---

## 1. Section 1：固定状态梯度诊断

单进程、无 optimizer、无 W&B，复用真实 `GRPOModel`、prepared-v2 loader、collator 与训练长度；关闭 dropout，仅 rollout RNG 改变。每个 probe 固定一个 size-16 microbatch，做 16 次独立 rollout（seed 42/3407/2026/0…12）。bf16 autocast、fp32 参数、启用 gradient checkpointing。诊断代码记录 `git=418bd5c`（dirty），Section 1 不涉及逐文档 baseline 代码路径。

产物：`outputs/rollout_gradients/{e0,step100,e0_batch128}.json`。

| 状态 | probe | 来源 | reward 极差(16 draw) | `gradient_norm_mean` | `‖ḡ‖` | `noise_rms` | **noise/mean** | `pairwise_cosine` |
|---|---|---|---:|---:|---:|---:|---:|---:|
| E0 | 0 | leetcode | 0.0097 | 174.2 | 50.06 | 173.1 | **3.457** | 0.02040 |
| E0 | 1 | biology | 0.0111 | 136.2 | 35.16 | 136.3 | **3.878** | 0.00425 |
| E0 | 2 | biology | 0.0152 | 196.4 | 51.26 | 196.1 | **3.826** | 0.00593 |
| step-100 | 0 | leetcode | 0.0094 | 116.6 | 31.27 | 116.5 | **3.726** | 0.00952 |
| step-100 | 1 | biology | 0.0073 | 134.1 | 35.28 | 134.4 | **3.810** | 0.00617 |
| step-100 | 2 | biology | 0.0136 | 136.9 | 38.65 | 136.2 | **3.524** | 0.01785 |
| E0, 8-mb 平均 | 0 | leetcode | 0.0048 | 72.0 | 19.28 | 71.7 | **3.717** | 0.00972 |
| E0, 8-mb 平均 | 1 | economics | 0.0048 | 52.9 | 14.30 | 52.6 | **3.680** | 0.01134 |
| E0, 8-mb 平均 | 2 | biology | 0.0035 | 62.1 | 16.12 | 62.0 | **3.847** | 0.00517 |

`noise/mean = noise_rms / ‖ḡ‖`。若 16 个梯度完全无一致方向，该比值理论上趋近 √N=√16=4；实测 3.5–3.9 已接近该上限，与 `pairwise_cosine≈0`（近乎正交）一致，说明**单次 rollout 梯度基本是采样噪声，reward 标量掩盖了更新方向的随机性**。

**8-microbatch 平均并未降低相对噪声**：`‖ḡ‖` 与 `noise_rms` 同步缩小（probe0：50→19、173→72），比值几乎不变。这是因为 8 个 microbatch 是 8 个不同数据 batch，各自"真实方向"也不对齐、相互抵消，分子分母等比例缩小。含义：即使放大到 global-batch-128 结构，更新方向仍被 rollout 噪声主导。

**边界**：这是固定状态的 Monte Carlo 重复，16 次不是完整训练；8-mb 平均近似 global batch 128 的平均结构，但不等于逐位重放八卡采样。固定状态噪声大**不自动证明**它造成了最终 BRIGHT 差距。

---

## 2. Section 2：独立 rollout 训练对照

三行都从 E0 独立训练，训练/data seed 固定 42，仅 rollout seed 不同，使用新隔离 RNG（提交 `418bd5c`）；113 steps、LR 5e-6、global batch 128 / microbatch 16、8 GPU、完整 MRR@10 配方。评测沿用既有 BRIGHT 命令（fp16、12 subset）。**历史全局 RNG 的 `s42=0.220` 不作为其中一行复用**；新 RNG 路径使 `Rollout42` ≠ 历史 `s42`。

产物：`checkpoints/iclr2027/G1-A-MRR090-Rollout{42,3407,2026}-s42/`（含 `mteb_eval/bright/`）。

### 2.1 BRIGHT ndcg@10（subset 顺序沿用 `paper/G1_RESULTS.md`）

| subset | Rollout42 | Rollout3407 | Rollout2026 |
|---|---:|---:|---:|
| biology | 0.13234 | 0.28170 | 0.18979 |
| earth_science | 0.19893 | 0.29406 | 0.27166 |
| economics | 0.20724 | 0.24315 | 0.22153 |
| psychology | 0.24233 | 0.27591 | 0.27803 |
| robotics | 0.13501 | 0.13259 | 0.14077 |
| stackoverflow | 0.22692 | 0.18573 | 0.19174 |
| sustainable_living | 0.17509 | 0.19137 | 0.20821 |
| pony | 0.00621 | 0.00869 | 0.00519 |
| leetcode | 0.16916 | 0.11403 | 0.12796 |
| aops | 0.04294 | 0.02788 | 0.03696 |
| theoremqa_theorems | 0.30288 | 0.27323 | 0.36723 |
| theoremqa_questions | 0.19993 | 0.19368 | 0.19188 |
| **宏平均** | **0.16992** | **0.18517** | **0.18591** |

- 三行宏平均：均值 **0.18033**，样本标准差 **0.00903**，min 0.16992，max 0.18591，极差 **0.01600**。
- 后期训练 reward（step 94–113 等权均值）：**0.704 / 0.713 / 0.722**，几乎不可区分。
- 历史三行（数据流程 + 动作采样一起变）：均值 0.19197，标准差 0.02915，极差 0.05821。

### 2.2 判断（实验计划 §2 决策表）

结果落在决策表第 2、3 行之间，**最接近第 3 行**：

1. **固定 data 后，最终差距明显小于历史重复。** 独立 rollout 标准差 0.009 / 极差 1.6 分，历史三行 0.029 / 5.8 分。→ **单靠 rollout 随机性不足以复现原始 seed 差距的量级**；其余部分很可能来自数据流程随机性（sampler/组批）及其与 rollout 的交互——这正是历史三行混在一起、无法拆分的部分。后续应考虑数据顺序效应，**不先认定 rollout 是主要来源**。
2. **但分散绝非可忽略，尤其在单领域。** 宏平均较稳是各领域涨跌部分抵消所致；仅换动作 RNG，单领域仍剧烈波动：biology 0.132↔0.282、earth_science 0.199↔0.294、leetcode 0.114↔0.169。这与 Section 1 的固定状态高噪声方向一致。
3. **排序翻转的旁证**：原始数据里 s42 最好（0.220）；隔离 RNG 后 `Rollout42` 反而最差（0.170）。历史"最好 seed"在隔离 RNG 下不可复现，进一步说明单 seed 结果不可靠。

### 2.3 保留的证据边界

- 未复用历史 22.01 当控制组；三行自成一体。
- 未设事后"显著波动"阈值；未把标准差之比（0.009/0.029）解释为 rollout 的方差贡献率。
- n=3，标准差本身高度不确定（2 自由度）；这是**初步**稳定性证据，不是方差分解。
- 固定状态噪声与完整训练差距分层报告，不互相充当对方的证明。

---

## 3. 固定状态 reward / 逐文档 baseline 梯度对比

按实验计划 §3–§4，在**两个固定状态**上做诊断，无需重新训练：**E0**（run 初始化）与**预定最终模型** `G1-A-MRR090-Rollout42` 的 113-step 权重（`model.safetensors` sha256 `a2080075…`）。每条件复用同一采样脚本、默认 batch（sampler 顺序 probe 0/1/2 = leetcode/biology/biology）与默认 16 个 rollout seed，单 microbatch。全部 10 条运行于同一代码 `git=c918bd8`（dirty）。graded 条件仅按设计改变 `relevance_labels`，其余输入一致。

产物：`outputs/rollout_gradients/reward_baseline/{e0,final}_{R0,R1,R2,C0,C2}_*.json`。

比较矩阵：R0 MRR@10 binary（当前控制）、R1 nDCG@10 binary（隔离"指标"）、R2 nDCG@10 graded（隔离"分级标签方案"）；C0/C2 在 R0/R2 目标上把文档 advantage 从 `shared`（组级 LOO）换成 `counterfactual`（逐文档单位方向反事实差）。

### 3.1 结果（3 个 probe 的均值）

| 条件 | reward `n_distinct` | E0 `noise/mean` | E0 `cosine` | E0 `‖ḡ‖` | final `noise/mean` | final `cosine` | final `‖ḡ‖` | cf `zero_frac` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 MRR-bin / shared | 3.2–3.9 | 3.720 | 0.0102 | 45.5 | 3.651 | 0.0129 | 41.5 | — |
| R1 nDCG-bin / shared | ~38 | 3.629 | 0.0135 | 27.3 | 3.573 | 0.0158 | 22.4 | — |
| R2 nDCG-grad / shared | ~220 | 3.716 | 0.0101 | 20.1 | 3.431 | 0.0224 | 18.1 | — |
| C0 MRR-bin / counterfactual | 3.2–3.9 | 3.812 | 0.0062 | 61.1 | 3.896 | 0.0036 | 79.7 | 0.69–0.70 |
| C2 nDCG-grad / counterfactual | ~220 | 3.581 | 0.0154 | 21.8 | 3.845 | 0.0052 | 26.9 | 0.29–0.32 |

### 3.2 判断

**换 reward（R0/R1/R2）：相对噪声几乎不随 reward 分辨率改善。** graded nDCG 的可分辨档位约为 MRR 的 60 倍（≈220 vs ≈4），但两个状态下 `noise/mean` 都停在 3.4–3.7、方向近乎正交（cosine ≈ 0.01–0.02）。`‖ḡ‖` 随更丰富 reward 缩小（45→27→20），这是**目标尺度缩小，而非信噪比改善**——正是计划提醒要区分的陷阱。final 模型上 graded 有一点边际优势（3.43 vs 3.65），但 E0 上不存在（3.72 vs 3.72），不构成跨状态稳健的改善。

**换逐文档 baseline（C0/C2）：无改善，final 上反而更差。** 反事实 baseline 的 `noise/mean` 与 `cosine` 在 E0 基本持平或略差，在 final 模型上明确更差（C0 3.90 vs R0 3.65；C2 3.85 vs R2 3.43；cosine 同步下降）。MRR 下约 70% 的逐文档反事实 advantage 严格为零（多数文档换成自身均值方向不改变 MRR），baseline 无法在 reward 平坦处注入信号。这是**当前 baseline 选择（单位均值方向、不再中心化）**的结论，不否定该 baseline 家族。

**综合**：跨 reward、标签方案、文档 baseline，固定状态相对噪声（`noise/mean`≈3.4–3.9，近 √N=4）与近正交性都异常稳定。**降低固定状态梯度信噪比的瓶颈不在 reward 粗粒度，也不在文档信用分配**，更像 vMF score-function 估计器在当前探索尺度（κ≈4846、alignment 0.9）下的固有采样噪声。

**8-mb 平均已核对（边界闭合）**：对 R1/R2/C0/C2 补跑 `--microbatches-per-probe 8 --batch-indices 0 8 16`（E0），`noise/mean` 相对单 microbatch 变化 ≤0.14，全部仍在 3.4–3.8、方向仍近乎正交：R0 3.72→3.75、R1 3.63→3.50、R2 3.72→3.44、C0 3.81→3.84、C2 3.58→3.48。远达不到 iid 噪声下 √8≈2.8 倍的衰减。即"换 reward / 换 baseline 也不能靠 global-batch 平均压低相对噪声"在全部 5 档成立，不只 MRR。产物：`outputs/rollout_gradients/reward_baseline/e0_*_batch128.json`（R0 的 8-mb 见 Section 1 `e0_batch128.json`）。

**其余边界**：N=16、`‖ḡ‖` 小使比值本身不精确，只读粗量级差异；final 模型的 8-mb 未跑（E0 已足以闭合结论）；不要求不同 reward 的期望梯度方向一致。

### 3.3 探索尺度与采样预算（alignment / G）

在同一份权重上（E0 与 final Rollout42），改 `target_alignment`（0.40/0.65/0.90/0.95，G=32）或 group size（16/32/64，align=0.9），MRR reward、shared baseline，单 microbatch。**无需训练**，只改采样。产物：`outputs/rollout_gradients/exploration_scale/{e0,final}_*.json`（align0.9/G32 = R0，见 `reward_baseline/`）。

**alignment/κ（noise/mean，E0 | final）**：0.40 `3.725 | 3.291`、0.65 `3.503 | 3.342`、0.90 `3.720 | 3.651`、0.95 `3.644 | 3.809`。final 模型上更强探索（低 alignment）略降噪声，但 **E0 上无此趋势**——跨状态不一致，不是可靠杠杆。低 alignment 只是增加 reward 多样性（`n_distinct` 从 ~4 升到 ~10），与信噪比无关。

**group size G（noise/mean，E0 | final）**：G=16 `3.895 | 3.792`、G=32 `3.720 | 3.651`、G=64 `3.453 | 3.308`；`cosine` 同步单调上升（final 0.007→0.013→0.029）。这是**唯一在两个状态都单调、方向一致的杠杆**：加大 G 确实降相对噪声、增方向一致性。

**但强烈次于 √G**：G 翻倍（32→64）只降约 8–10%（×1.08–1.10），远小于 iid 1/√G 预期的 ×1.41——共享的 query 动作与边际化带来递减收益。所以 G 有效但要把 `noise/mean` 压到 O(1) 需要不现实的大 G。

**小结**：探索尺度诸杠杆中，只有 G 给出稳健但温和的改善；alignment/κ 不可靠。这进一步支持"固定状态噪声是 vMF score-function 估计器的固有属性",单一超参难以根治。

## 4. 下一步

reward、逐文档 baseline、alignment/κ 均未在固定状态稳健改善信噪比；只有加大 G 给出温和（次于 √G）的改善。因此**不优先**据此安排新的完整训练重复（计划 §5 进入条件未满足）。若继续：

- **G 是唯一有正向信号的方向**，但收益次于 √G。可评估"加大 G 的噪声—成本比"是否值得，或用**多组独立 rollout 平均（R 份）**在同一 batch 上摊噪声——两者都是加采样预算，需与算力成本权衡；后者需给诊断脚本加一个"对 rollout 平均 R 份"的模式（当前脚本不保留逐 draw 完整梯度）。
- 若继续 baseline 路线：换独立替代动作或条件均值近似，而非单位均值方向；先在可枚举小例上验证期望一致。
- graded nDCG 可作为**目标函数对照**保留（reward 信息更丰富是事实），但不宣称它降低了梯度噪声。
- **边界**：以上都在固定权重上隔离估计器方差；不能据此把某个完整训练成绩归因于 G/alignment（那需各自训练，计划 §6）。
