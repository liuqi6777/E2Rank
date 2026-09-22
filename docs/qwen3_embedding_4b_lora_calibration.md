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
