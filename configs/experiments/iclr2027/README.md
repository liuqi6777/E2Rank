# 三组实验配置指南

## 修改与运行

日常只改 `configs/experiments.yaml`。从仓库根目录运行：

```bash
python scripts/experiment.py prepare G1
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
保留全部去重后的正负例，seed 42 每个 query 固定抽一个 in-batch 代表正例，放在候选首位。
新版输出至 `data/processed/reasonrank_multi/`，旧数据保留。生成：

- `train.jsonl`：`id/query/positives/negatives/source`，`positives` 为列表。
- `train.metadata.jsonl`：文档 ID、teacher 排序与审计信息。
- `train.ready.jsonl`：训练唯一使用的文件，包含候选、标签、去重标识和已知正例 ID。

预处理一次完成静态转换。Joint loader 不关联 sidecar，collator 做 tokenization、动态 padding/mask 与当前 batch 过滤。
已有 public + metadata 时 `prepare` 只编译；已有 ready 时直接提示已准备。
新版 schema 为 `embedding_candidates_v2`；旧单正例 public 数据须从原始 parquet 重新生成。
JSONL 不存 padding，模型和 loss/reward 均排除 padding。

G1 主对照与 RL 消融使用 joint encoder；`G1-A-QPolicy` 只移除 document policy action，
`G1-A-DPolicy` 只移除 query policy action；两者仍更新共享 encoder，并在最终 BRIGHT 评测前
用训练后 checkpoint 编码 query 和 passage。`G1-A-Gaussion` 保持 joint action，但改用
projected-Gaussian 采样替代 vMF。
这些 run 不需要先执行 `encode G1`。`G1-DR` 则按 ReasonRank source 动态检索完整 corpus，
冻结 E0 document encoder/index，只更新 query encoder，因此必须先执行 `encode G1`。
索引准备导出各 source 的完整 `id_doc`，不再仅导出训练候选涉及的文档；新版目录需重新编码。

## 实验行与依赖

共 28 个逻辑行：**22 个核心训练、4 个可选训练、2 个 E0 评测**。
仅增加探索成对对照时为 24 次训练；四项可选全部运行时为 26 次。

- G1：四个 `J` run 是主对照；新增 `A-Norm`、`A-DocMean`、`A-NormDocMean` 与 `J-RL`
  组成相同 leave-one-out baseline 的标准化 × 文档权重 2×2。
  `A-Paired`、`A-QPolicy`、`A-DPolicy`、`A-Cal`、`A-Binary` 是其余五个核心消融。
  `G1-DR` 是独立的 full-corpus 动态检索行。
  `A-Anneal` / `A-FixedSmall` 为成对可选对照；`A-Gaussion` / `A-MRR` 也为可选，
  MRR 直接比较 Binary，其余行复用主 RL 为控制。
- G2：`D` 从初始 LLM 训练；`W` 从 `G2-D-CL-s42/` 最终模型重新训练。
  D-CL 训练 1200 步，W-CL/LL/RL 各新建 optimizer/scheduler 和数据迭代，再训练 1200 步。
  W-CL 是独立训练行，不复用 D-CL；只加载模型权重，不恢复 trainer 状态。
- G3：从 E0 开始，比较 CL、检索 RL、答案 RL；不自动采用其他组的最优 checkpoint。

当前 G1 joint CL / RN / LL / RL、八个核心 RL 消融和四个可选 RL 消融均已注册。
LL 固定为 LambdaRank variant：pairwise logistic 乘当前排序交换产生的 `|ΔnDCG@10|`，
使用 `gain=2^rel-1` 和 sigma 1.0。`G1-A-QPolicy` 仍使用 joint encoder，只将 RL action
限制为 query；`G1-A-DPolicy` 则仅采样 positive/negative document slots。二者都不是冻结
document encoder 的 query-only 训练。`G1-A-Gaussion` 设置 `sampling_law=gaussian`：它以
归一化高斯采样 action，但沿用 vMF log-density surrogate，因而是采样一致性消融。
`G1-DR` 每个 sampled query action 在对应 source 的完整冻结 corpus 中检索 top-20，使用已知
teacher grades 的 nDCG@10；不在已知 qrels 中的检索结果 gain 为 0。它以 `G1-A-QPolicy`
作为最近控制，但因同时改变 candidate access 和 document update scope，不作为单因素消融。
G2 使用预处理生成的固定训练文件，采用固定预算和最终 checkpoint，不要求 manifest loader 或 dev 选模。
模型协议和预算已填写；第二阶段只需先完成 D-CL 以提供初始化权重。
G3 的监督 CL 使用离线候选；RL action 动态检索冻结的完整 corpus，不设置共享候选控制。
当前仍需接入答案 F1 选模；检索 RL 需要 nDCG，不能用 source-aware MRR 代替。

运行时以 `check RUN` 为准。只有 `G1-DR` 加载冻结索引；缺少时 `check` 会提示先运行
`python scripts/experiment.py encode G1 --gpus N`。

## 推荐执行顺序

`list` 按 suite 的 `execution_stages` 排序，并显示 `core` / `optional` / `reference` 与 stage；
`show RUN` 同时显示阶段名称。阶段是建议次序，不会自动串行执行、等待、创建依赖或启动训练。
`control` 表示比较对象，只有 `init_from` 表示模型权重依赖。就绪检查仍以实际数据/实现/输出状态为准。

| Stage | 建议顺序 | 核心训练数 |
|---|---|---:|
| 0 | G1-E0，复用原始模型评测 | 0 |
| 1 | G1-J-LL → G1-J-RL → G1-J-CL → G1-J-RN | 4 |
| 2 | G1-A-Norm → G1-A-DocMean → G1-A-NormDocMean | 3 |
| 3 | G1-A-Paired → G1-A-QPolicy → G1-A-DPolicy → G1-A-Cal → G1-A-Binary | 5 |
| 4 | G2-D-CL → D-LL / D-RL / W-LL / W-RL / W-CL | 6 |
| 5 | 索引就绪后 G1-DR | 1 |
| 6 | G3-E0 → G3-CL → G3-AnsRL → G3-RetRL；先解决对应 blocker | 3 |
| 7 | 可选：Anneal + FixedSmall，然后 Gaussian mismatch / MRR | 0（另计最多 4） |

第一批优先完成 stage 1–2 共 7 次训练，得到同标签主对照与完整 2×2。
G2 的 W 系列只依赖 D-CL 的最终模型；D-CL 就绪后这些分支可以独立安排。
G2-D-CL 或索引/QA 准备可与独立 G1 运行重叠；顺序不增加人为的硬依赖。
所有正式消融从 E0 初始化，不从主 RL 的最终模型继续训练。主结果只复用为控制。

四格仅改变 `advantage_norm` 与 `document_log_prob_reduction`，其余配置一致；
`NormDocMean` 不是旧版完整复现，因为它仍使用 leave-one-out。
可选 Anneal 起点使用当前 1024 维模型的 `A(755)=0.5302373892742263`，终点为 0.80；
FixedSmall 全程固定 0.80。命名行已显式设置参数，覆盖公共配置中的调度值，且限定当前模型。
若改变主方法的探索或初始化，应重新声明整套对照，不能沿用原有可比性解释。
两个可选探索结果必须一起报告，不能用其中表现更好的配方替换预定主方法。
最终测试分数不用于选择训练顺序、预算、checkpoint 或消融保留范围。

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
MRR 与 Binary 直接比较指标变化；Binary 与主 RL 比较标签变化。
CL 与排序方法的监督信息不同；LL 与 RL 才是相同 graded 目标下的主要对照。

本轮决定保留 LL 当前 sigma=1.0 与其余配方先跑。G1 普通训练与 DR 均保留全部已知正例。
InfoNCE 每个正例分别对有效负例计算损失，其他正例不进入分母，再按正例/query 两级平均。
独立 positive_mask 不受 teacher grades 影响。In-batch 仅使用每条 query 的一个代表正例。
DR 使用新版 ready 中的完整 teacher qrels 计算奖励和 IDCG；奖励有效性由运行实测确定。

## G2 第一版 scratch 参数

参照已退役的 `posttrain_ablations.sh scratch`，使用 `Qwen/Qwen3-0.6B`、full FT、
learning rate 5e-6、AdamW、linear scheduler、warmup ratio 0.03、weight decay 0.01。
表示协议继承 `configs/model/qwen3_0.6b.yaml`：last pooling、left padding、append pad，
query 使用 Instruct/Query 模板，document 为原文；query/document 上限 512/1024。
G2 定位为从未经 embedding 专项训练的通用 LLM 学习检索表示；允许通用后训练，
不据此声称它是纯预训练 checkpoint。

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


## 第一轮方法修订（2026-09-12）

Embedding RL 主配置现为 leave-one-out baseline、不做 advantage normalization、文档 log-density 求和。
固定索引与 RAG 使用相同 baseline 语义；query-only 不存在文档权重。
`configs/grpo/legacy.yaml` 保留旧版配置；`configs/grpo/annealed.yaml` 是底层独立调度预设。
默认 κ=755 保持不变。可在公共 `configs/experiments.yaml` 的某个组中显式添加：

```yaml
target_alignment: 0.53
final_alignment: 0.80
exploration_schedule: linear
```

这些值仅传入 RL 行；G3 自动映射为 `rag_*` 参数。该配方是预定示例，未经效果验证。
Alignment 优先于 κ，按实际 embedding 维度反解；线性调度从第一次更新到最后一次更新收缩探索。
它只支持 exact vMF，不能同时用于 G1-A-Gaussion 或 learnable sigma；Gaussian 消融应保持固定 κ。
对单条 run 的调度可在 suite 的该 run `overrides` 中声明，避免改变整组。
改变策略配方应使用新的 output_dir。新版 checkpoint 中的 `exploration_state.json` 校验估计器与调度配置；
旧 checkpoint 缺少新版契约时不会被静默作为新版续训。Trainer 恢复的 global_step/max_steps 控制后续调度，
不会在每个 accumulation microbatch 推进。

日志中 `exploration/kappa` 和 `exploration/mean_alignment` 描述采样分布；joint 路径另记录
`exploration/own_boundary_pairs` 与 `exploration/own_boundary_flip_rate`。后者只涵盖 own-list 的
不同 grade、非平局且涉及 top-K 的相邻比较，不应解释为完整 corpus 或答案变化率。
不归一化时近退化统计仍有效，但非零 advantage 不会被日志阈值截断。
