# Qwen3-Embedding-4B LoRA RELER 参数校准

## 目的与边界

现有 4B 主矩阵已经完整回答受控迁移与组件消融问题，不修改、不覆盖。本轮只校准完整
RELER 的学习率、均匀 shortlist 大小 K 和 vMF target alignment；CP、G64、pairwise 0.5、
frozen-document rescale、LoRA rank 16、113 steps、global batch 128 及数据/评测协议保持不变。
新输出根目录为 `checkpoints/iclr2027-qwen3-embedding-4b-lora-calibration`。

当前控制为 `LR=1e-4, K=7, alignment=0.70`，其三 seed BRIGHT 均值为 29.00；同协议
InfoNCE 为 30.08。控制直接复用现有结果，不在新目录重复训练。

## 第一阶段：seed 42 单因素校准

| recipe | LR | K | alignment | 相对控制的唯一变化 |
|---|---:|---:|---:|---|
| `lr050` | 5e-5 | 7 | 0.70 | LR 减半 |
| `lr200` | 2e-4 | 7 | 0.70 | LR 加倍 |
| `k3` | 1e-4 | 3 | 0.70 | 更短 shortlist |
| `k15` | 1e-4 | 15 | 0.70 | 更长 shortlist |
| `align065` | 1e-4 | 7 | 0.65 | 增强探索 |
| `align080` | 1e-4 | 7 | 0.80 | 减弱探索 |

不运行三因素全网格，也不在第一阶段同时改变两个参数。先检查并运行六条 seed 42：

```bash
python scripts/run_qwen3_embedding_4b_lora_calibration.py check
python scripts/run_qwen3_embedding_4b_lora_calibration.py train
```

入口默认 `--seeds 42`。已有非空输出目录不会被覆盖；评测失败时用相同选择执行 `eval`。

## 第二阶段：配对复验

第一阶段只用于筛选。仅当某点相对 seed-42 当前 RELER（28.85）提高至少 0.50 分，且收益
不由单个领域独占时，才冻结该 recipe 并补 3407/2026。示例：

```bash
python scripts/run_qwen3_embedding_4b_lora_calibration.py check \
  --recipes lr050 --seeds 3407 2026
python scripts/run_qwen3_embedding_4b_lora_calibration.py train \
  --recipes lr050 --seeds 3407 2026
```

最终报告保留三个 seed 原始分数、均值、样本标准差及相同 seed 对当前 RELER 的配对差。
若六点均未通过门槛，本轮停止，不继续组合搜索。若某个单因素点通过，先完成其三 seed
确认；参数交互属于后续独立实验，不能把 seed-42 BRIGHT 最优点直接改写为新的主结果。

## 后续：LR × K 交互

单因素三 seed 结果中，`lr200` 为 29.84 ± 0.07，相对原 RELER 的配对均值差为 +0.84；
`k3` 为 29.34 ± 0.10，配对均值差为 +0.33。新增 `lr200k3` 同时设置
`LR=2e-4, K=3`，其余沿用完整 RELER 配方。该点用于测量交互，不能把两个单因素
收益直接相加当作实测结果。先运行 seed 42，再根据与同 seed `lr200` 的比较决定是否补
3407/2026：

```bash
python scripts/run_qwen3_embedding_4b_lora_calibration.py check --recipes lr200k3
python scripts/run_qwen3_embedding_4b_lora_calibration.py train --recipes lr200k3
```

新 recipe 需显式通过 `--recipes` 选择；不改变入口默认的原六条单因素队列。

## 后续：K=3 的 pairwise 消融

新增 `lr200k3_nopairwise`，与 `lr200k3` 同为 `LR=2e-4, K=3`，只将
`reward_shortlist_pairwise_coef` 从 0.5 改为 0；shortlist 采样、跨卡候选池、
CP/G64 和其余协议保持一致。因此同 seed 的两者之差可用于衡量 K=3 下 pairwise 项的
作用。两条均先跑 seed 42：

```bash
python scripts/run_qwen3_embedding_4b_lora_calibration.py check \
  --recipes lr200k3 lr200k3_nopairwise
python scripts/run_qwen3_embedding_4b_lora_calibration.py train \
  --recipes lr200k3 lr200k3_nopairwise
```

## 后续：关闭 shortlist

新增 `lr200k3_noshortlist`，与 `lr200k3_nopairwise` 同为 `LR=2e-4` 且关闭
pairwise，仅将 `reward_shortlist_count` 从 1 改为 0，走普通跨卡候选 nDCG 路径。
配置保留 `K=3`，但 shortlist 关闭时 K 不参与计算。两者的同 seed 差值衡量从 K=3
shortlist 路径切换为完整候选池路径的效果；CP/G64、frozen-document rescale、训练预算
和评测协议保持一致。先跑 seed 42，再视结果决定是否补其余两个 seed：

```bash
python scripts/run_qwen3_embedding_4b_lora_calibration.py check --recipes lr200k3_noshortlist
python scripts/run_qwen3_embedding_4b_lora_calibration.py train --recipes lr200k3_noshortlist
```
