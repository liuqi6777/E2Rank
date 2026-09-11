# 三组实验配置指南

## 修改与运行

日常只改 `configs/experiments.yaml`。从仓库根目录运行：

```bash
python scripts/experiment.py prepare G1
python scripts/experiment.py encode G1 --gpus 8
python scripts/experiment.py list
python scripts/experiment.py show G1-J-RL
python scripts/experiment.py show G1-J-RL --verbose
python scripts/experiment.py check G1-J-RL
python scripts/experiment.py train G1-J-RL --gpus 8
```

`--config PATH` 选择另一份公共配置；`--gpus` 控制单机进程数，默认为 8。
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
| 训练预算 | 113 optimizer steps / 14,464 query exposures，固定最终 checkpoint |
| Learning rate | 5e-6 |
| Global batch / 每卡 microbatch | 128 / 16 |
| 1 / 2 / 4 / 8 卡 accumulation | 8 / 4 / 2 / 1 |
| Optimizer / scheduler | AdamW / linear，warmup ratio 0.03 |
| Weight decay / max grad norm | 0.01 / 1.0 |
| 保存 | 每 25 步，保留最近 2 个中间 checkpoint，另存最终模型 |
| RL group size / kappa | 32 / 755 |
| CL / RankNet temperature | 0.03 |
| LambdaLoss | LambdaRank `|ΔnDCG@10|` weighting，sigma 1.0 |

这是一版面向单机 8×80GB 的固定试跑参数，尚未完成正式 GPU 实验；不用 BRIGHT 选模。
113 steps 与原 450 steps × global batch 32 的 query exposure 预算近似相同（14,464 vs. 14,400）。
`--gpus` 改变时自动维持 global batch，GPU 数与 microbatch 的乘积须整除 batch。

`prepare G1` 调用 `scripts/prepare_reasonrank.py`：排除 MSMARCO、隔离 BRIGHT 重叠、清理冲突与重复 query，
用 seed 42 每个 query 固定抽一个正例，移除其他已知正例，保留负例。生成：

- `train.jsonl`：`id/query/positive/negatives/source`，方便查看。
- `train.metadata.jsonl`：文档 ID、teacher 排序与审计信息。
- `train.ready.jsonl`：训练唯一使用的文件，包含候选、标签、去重标识和已知正例 ID。

预处理一次完成静态转换。Joint loader 不关联 sidecar，collator 做 tokenization、动态 padding/mask 与当前 batch 过滤。
已有 public + metadata 时 `prepare` 只编译；已有 ready 时直接提示已准备。
JSONL 不存 padding，模型和 loss/reward 均排除 padding。

G1-Q-* 在 prepare 后先运行 `encode G1`。它使用同一 E0、document prompt、pooling、append token
和 document 最大长度，按 `document_key` 首次出现顺序生成冻结 ordinal。query-only collator 只输出
`candidate_ordinals` 和 mask，不 tokenize 文档；训练输出记录全部冻结 artifact 的前后 hash。
最终 BRIGHT 评测同样保持 document side 为 E0：使用仓库根目录 README 中的 fixed-corpus 命令，
按 BRIGHT subset 分别建立和复用索引；训练后 checkpoint 仅编码 query。

## 实验行与依赖

共 23 个逻辑行：21 个训练执行、2 个 E0 评测。

- G1：`J` / `Q` 表示 joint / query-only；CL、RN、LL、RL 是四类目标。
  `A-Paired`、`A-Cal`、`A-MRR` 的对照均为 `G1-J-RL`。
- G2：`D` 从初始 LLM 训练；`W` 从 `G2-D-CL-s42/` 最终模型重新训练。
  D-CL 训练 1200 步，W-CL/LL/RL 各新建 optimizer/scheduler 和数据迭代，再训练 1200 步。
  W-CL 是独立训练行，不复用 D-CL；只加载模型权重，不恢复 trainer 状态。
- G3：从 E0 开始，比较 CL、检索 RL、答案 RL；不自动采用其他组的最优 checkpoint。

当前 G1 joint 与 query-only CL / RN / LL / RL，以及 joint RL 消融均已接入。
LL 固定为 LambdaRank variant：pairwise logistic 乘当前排序交换产生的 `|ΔnDCG@10|`，
使用 `gain=2^rel-1` 和 sigma 1.0。G1 query-only 通过离线冻结索引训练，仅更新 query encoder。
G2 使用预处理生成的固定训练文件，采用固定预算和最终 checkpoint，不要求 manifest loader 或 dev 选模。
模型协议和预算已填写；第二阶段只需先完成 D-CL 以提供初始化权重。
G3 仍需候选访问控制、答案 F1 选模；检索 RL 需要 nDCG，不能用 source-aware MRR 代替。

运行时以 `check RUN` 为准。G1 query-only 的 source、corpus、offset、mapping、vector shard hash
和 embedding protocol 是启动门槛；joint 不加载冻结索引。数据存在、其余实现缺项、必要参数、
依赖 checkpoint 和输出目录检查仍生效。

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

## G2 第一版 scratch 参数

参照已退役的 `posttrain_ablations.sh scratch`，使用 `Qwen/Qwen3-0.6B`、full FT、
learning rate 5e-6、AdamW、linear scheduler、warmup ratio 0.03、weight decay 0.01。
表示协议继承 `configs/model/qwen3_0.6b.yaml`：last pooling、left padding、append pad，
query 使用 Instruct/Query 模板，document 为原文；query/document 上限 512/1024。
模型名称沿用旧配方，不据此声称它是纯预训练 checkpoint。

全局 batch 128、每卡 microbatch 16；1/2/4/8 卡 accumulation 为 8/4/2/1。
每次运行 1200 步，约 153,600 query exposures，每 200 步保存。
D-CL 最终权重保存在 `checkpoints/iclr2027/G2-D-CL-s42/`。
W-CL/LL/RL 从此目录初始化，各独立运行 1200 步，不读取 optimizer/scheduler 或数据游标。
两阶段路线计入 CL 前缀后为 2400 步；直接路线为 1200 步，不作为等总预算比较。
不沿用旧数据配置的 per-source cap 或自动 dev 划分。

G2 读取 `data/train.jsonl`，全量训练。每个 G2 训练行成功保存最终模型后，实验入口自动运行一次
完整的 `MTEB(eng, v2)`，输出到该模型目录的 `mteb_eval/final/`；不会在训练前或中间 checkpoint
重复运行，任一 MTEB 任务失败则整个命令返回失败。当前本地尚缺该文件。
先运行 D-CL，再分别启动 W-CL/LL/RL；入口不会隐式训练依赖。
