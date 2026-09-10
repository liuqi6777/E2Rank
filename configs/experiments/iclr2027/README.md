# 三组实验配置指南

## 修改与运行

日常只改 `configs/experiments.yaml`。从仓库根目录运行：

```bash
python scripts/experiment.py prepare G1
python scripts/experiment.py list
python scripts/experiment.py show G1-J-RL
python scripts/experiment.py show G1-J-RL --verbose
python scripts/experiment.py check G1-J-RL
python scripts/experiment.py train G1-J-RL --gpus 4
```

`--config PATH` 选择另一份公共配置；`--gpus` 控制单机进程数。
`list G1` / `check G1` 可按组查看；`check` 遇到尚未就绪的训练行返回 2。
`show` 和 `check` 不写文件、不下载模型。没有 train-all，也不会自动训练依赖。

## 配置职责

| 文件 | 内容 | 何时修改 |
|---|---|---|
| `configs/experiments.yaml` | 模型、数据目录、学习率、步数、保存间隔、batch | 日常运行 |
| `suite.yaml` | run IDs、目标、训练范围、消融、依赖、选模与实现缺项 | 改实验设计或实现新能力 |
| `g1.yaml` / `g2.yaml` / `g3.yaml` | 各组共享的底层数据和训练设置 | 改组内公共协议 |
| `configs/train/posttrain.yaml` | 通用 full FT 参数 | 改通用训练配方 |
| `configs/model/*.yaml` | 模型表示协议 | 增加或替换模型 |
| `configs/grpo/posttrain.yaml` | 当前 embedding RL 共享策略 | 改 RL 公共参数 |

公共配置被映射到 suite protocol / runtime overrides，再与 profile 和目标配置合并。
`show --verbose` 中的 `config` 是最终 trainer 参数，`protocol` 是实验约定；协议字段本身不会实现训练行为。
内部模块 `scripts/experiments/iclr2027.py` 不作为另一套日常入口。

## G1 运行参数与数据

| 参数 | 值 |
|---|---|
| 初始模型 | Qwen/Qwen3-Embedding-0.6B |
| 更新方式 / seed | full FT / 42 |
| 数据 | 4,963 train / 0 dev，候选数 2–20 |
| 训练预算 | 450 optimizer steps，固定最终 checkpoint |
| Learning rate | 5e-6 |
| Global batch / 每卡 microbatch | 32 / 8 |
| 1 / 2 / 4 卡 accumulation | 4 / 2 / 1 |
| Optimizer / scheduler | AdamW / linear，warmup ratio 0.03 |
| Weight decay / max grad norm | 0.01 / 1.0 |
| 保存 | 每 100 步，保留最近 2 个中间 checkpoint，另存最终模型 |
| RL group size / kappa | 32 / 755 |
| CL / RankNet temperature | 0.03 |
| LambdaLoss | LambdaRank `|ΔnDCG@10|` weighting，sigma 1.0 |

这是一版固定试跑参数，尚未完成正式 GPU 实验；不用 BRIGHT 选模。
`--gpus` 改变时自动维持 global batch，GPU 数与 microbatch 的乘积须整除 batch。

`prepare G1` 调用 `scripts/prepare_reasonrank.py`：排除 MSMARCO、隔离 BRIGHT 重叠、清理冲突与重复 query，
用 seed 42 每个 query 固定抽一个正例，移除其他已知正例，保留负例。生成：

- `train.jsonl`：`id/query/positive/negatives/source`，方便查看。
- `train.metadata.jsonl`：文档 ID、teacher 排序与审计信息。
- `train.ready.jsonl`：训练唯一使用的文件，包含候选、标签、去重标识和已知正例 ID。

预处理一次完成静态转换。Loader 不关联 sidecar，collator 只做 tokenization、动态 padding/mask 与当前 batch 过滤。
已有 public + metadata 时 `prepare` 只编译；已有 ready 时直接提示已准备。
JSONL 不存 padding，模型和 loss/reward 均排除 padding。

## 实验行与依赖

共 24 个逻辑行：21 个训练执行、2 个 E0 评测、1 个复用。

- G1：`J` / `Q` 表示 joint / query-only；CL、RN、LL、RL 是四类目标。
  `A-Paired`、`A-Cal`、`A-MRR` 的对照均为 `G1-J-RL`。
- G2：`D` 从 base LLM 直接训练，`W` 从共同 CL warm-up 继续。
  `G2-D-CL` 训练到 T，在固定 W 保存 W0；`G2-W-CL` 复用 D-CL 的后半程结果。
  `G2-W-LL/RL` 从 `G2-D-CL-s42/checkpoint-W` 开始，只训练 T-W 步。
  W 必须落在保存 schedule 上，不能退回到旧 Stage-1 或最终模型。
- G3：从 E0 开始，比较 CL、LL、检索 RL、答案 RL；不自动采用其他组的最优 checkpoint。

当前 G1 joint CL / RN / LL / RL 与 RL 消融已接入。
LL 固定为 LambdaRank variant：pairwise logistic 乘当前排序交换产生的 `|ΔnDCG@10|`，
使用 `gain=2^rel-1` 和 sigma 1.0。G1 query-only 仍需要真正冻结 document 分支，仅设置 query action sampling 不够。
G2 仍需数据划分加载、确定性 dev 选模和 continuation 状态处理，并补齐 base 模型表示协议与预算。
G3 仍需候选访问控制、答案 F1 选模；检索 RL 需要 nDCG，不能用 source-aware MRR 代替。

运行时以 `check RUN` 为准。当前入口不要求 model revision 或 dataset manifest/hash 作为启动门槛；
审计记录保留供复现。数据存在、实现缺项、必要参数、依赖 checkpoint 和输出目录检查仍生效。

## 输出

模型保存在 `checkpoints/iclr2027/RUN-s42/`，展开后的配置和启动记录在
`checkpoints/iclr2027/.launches/RUN-s42/`。公共配置 `output_dir` 可更改根目录。
既有输出或启动记录会阻止重复运行；自动恢复尚未接入。

## G1 排序监督

主设置为 graded：保留候选的 teacher 排名第 1 名 → 3，第 2–5 名 → 2，
第 6–10 名 → 1，其余 → 0。CL 仍使用已知正例；RankNet 使用完整 teacher 排序，
LambdaLoss 与 RL nDCG 使用 graded。ready 文件分别保存 relevance（已知正例）、
graded_relevance 与 rank_labels。teacher 排序不被强行改为已知正例第一。
G1-A-Binary 只将 RL nDCG 标签改为已知正例 binary；G1-A-MRR 也使用 binary。
CL 与排序方法的监督信息不同；LL 与 RL 才是相同 graded 目标下的主要对照。
