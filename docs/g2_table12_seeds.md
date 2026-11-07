# Rebuttal Table 12：通用 LLM 初始化的三个 seed

Table 12（`tab:initialization`）补齐 **42、3407、2026**，默认复用两条 seed42，
只新增 **4 次训练**。InfoNCE 对应 `G2-R2-D-CL-Strong`，不是普通 `D-CL`；
RELER 对应 `G2-R2-D-RL-Align070-ShortlistUniform-K0-T1-Pairwise050`，
不是旧纯 graded reward 或 K7 配方。与 RAG 的三 seed、Group size 消融一起，
本轮预计新增训练数从17变为 **21**，复用失败的必要重跑另计。

两种方法均直接从 `Qwen/Qwen3-0.6B` 开始，全参数训练，使用原 E2Rank
`data/train.jsonl`、1200 steps、LR5e-6、global batch128、micro batch16、8卡。
InfoNCE 保留完整跨卡负例池及文档梯度；RELER 保留 own-query K0、graded
nDCG@10 + pairwise0.5、RLOO+CMP、product、G64、rho0.70。训练、数据和
RELER 独立 rollout RNG 都使用对应 seed，数据准备与评测协议保持原设置。
入口会逐字段核对重复配置，只允许 seed、data_seed、rollout_seed 和运行身份变化。

在仓库根目录运行：

```bash
# 仅查看配置矩阵，不要求训练数据或 GPU
python scripts/run_g2_table12_seeds.py show
# 预检默认四条新运行
python scripts/run_g2_table12_seeds.py check
# 按 seed 配对，顺序训练 InfoNCE、RELER；每条完成后自动评测最终 MTEB
python -u scripts/run_g2_table12_seeds.py train

# 分机器运行时，每台仍使用八卡
python -u scripts/run_g2_table12_seeds.py train --seeds 3407
python -u scripts/run_g2_table12_seeds.py train --seeds 2026
# 仅补失败的最终评测
python -u scripts/run_g2_table12_seeds.py eval --methods reler --seeds 3407
# 核验 seed42 或必要时按原协议重新训练：显式指定 --seeds 42
python scripts/run_g2_table12_seeds.py show --seeds 42

# 在含 mteb==1.38.32 的评测环境汇总完整的两方法 × 三 seed
python scripts/run_g2_table12_seeds.py summary
```

统一通过 `--config /path/to/settings.yaml` 调整训练机路径，默认复用
`configs/experiments_g2_rl_r2.yaml` 的共享 G2 设置。输出保留原根目录
`checkpoints/iclr2027-g2-cl-r2/`；seed42 使用原 run ID，新行追加
`-Seed3407` / `-Seed2026`（解析器另追加 `-s3407` / `-s2026`）。已有非空目录
拒绝覆盖；部分完成的队列通过 `--methods`、`--seeds` 精确选择剩余项。
旧训练入口的默认方法和 seed 不变。

复用 seed42 前在训练机核对原 resolved config、最终 checkpoint、相同训练数据、
模型 revision 与完整原始评测。D 初始化的历史配置没有固定 revision；重复运行应
使用与原 seed42 相同的模型快照/缓存，不把不同 revision 的结果混报。原论文的
四舍五入 CSV 只用于确认方法身份，汇总不以它替代原始评测。

`summary` 始终读取六条运行的 `mteb_eval/final/`，忽略方法/seed选择参数，
输出到 `outputs/g2_table12_seeds/`（`--summary-dir` 可修改）：

- `per_run.csv`：逐seed的七类任务均值、Task avg.、Type avg.及来源路径；
- `per_task.csv`：全部41个任务的逐seed main score；
- `grouped.csv`：两方法各指标的三seed均值和样本SD；
- `paired.csv`：各seed的RELER − InfoNCE及三seed差值均值/样本SD；
- `results.json`：上述数据的未格式化数值。

分数均为 main score ×100。先对同一任务的 subsets 等权，再按原论文分别计算
41任务等权与七类型等权。缺失结果留空，只有每个 seed 的41任务均完整才纳入；
聚合必须三seed齐全，损坏、非有限值或多个含任务结果的目录会报错。
三seed完成前不改论文表格或将计划写成已完成证据。
