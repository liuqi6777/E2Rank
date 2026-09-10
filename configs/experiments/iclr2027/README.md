# ICLR 2027 三组实验配置

## 数据加载已简化

`prepare G1` 一次完成单正例化、候选重排、标签生成、元数据关联和文档去重标识计算，
自动产出内部训练文件 `train.ready.jsonl`。现有 4,963 条数据已完成该步骤。
配置只需指定数据目录，入口自动选择 ready 文件。

- Loader：读取 ready 记录、按模型协议格式化 query；不再关联 sidecar 或解析 teacher 标签。
- Collator：tokenize、按 batch 补齐、生成 mask、对当前 batch 的预计算标识做集合过滤。
- `train.jsonl` 和 metadata 保留用于查看/审计，训练不依赖它们。

此前 sidecar 自动读取的描述由本节替代。若已有 public 数据但尚无 ready 文件，
同一个 `prepare G1` 命令会完成编译；ready 文件存在时提示已准备，不会隐式重写。

## G1 第一版运行参数

Joint CL、RankNet、RL 和对应 RL 消融已通过配置启动检查。模型用
`Qwen/Qwen3-Embedding-0.6B`，full FT，seed 42，4,963 条全量训练，最终 checkpoint 评测。

| 参数 | 初始值 |
|---|---:|
| Learning rate | 5e-6 |
| 总更新步数 | 450（约 14,400 query exposures，约 2.9 遍） |
| 全局 batch | 32 |
| 每卡 microbatch | 8 |
| Gradient accumulation | 1/2/4 卡分别为 4/2/1 |
| Optimizer / schedule | AdamW / linear，warmup ratio 0.03 |
| Weight decay / max grad norm | 0.01 / 1.0 |
| 保存间隔 | 100 步，保留最近 2 个中间 checkpoint，另存最终模型 |
| RL group size / kappa | 32 / 755 |
| CL / RankNet temperature | 0.03 |

这是一版预先确定的试跑参数，不是经过 dev 调参得到的最优配置。bf16 与 gradient
checkpointing 开启。改变 GPU 数会自动调整 accumulation，保持 global batch=32；
GPU 数与 microbatch 的乘积须整除 batch_size。

按当前决定，训练入口不再要求 model revision、manifest 或文件 hash 校验。
保留文件存在、数据字段、元数据 ID 关联、mask 维度和输出目录保护。
审计文件可继续保存，但不是启动门槛。下文哈希说明仅为可选历史审计工具。
LambdaLoss、固定 document 分支及 G2/G3 的未完成实现仍会阻止相关运行。

## 日常入口

现在只需编辑 `configs/experiments.yaml` 中的模型、数据目录、learning rate 和 steps，使用：

```bash
.venv/bin/python scripts/experiment.py list
.venv/bin/python scripts/experiment.py show G1-J-RL
.venv/bin/python scripts/experiment.py check G1
.venv/bin/python scripts/experiment.py train G1-J-RL --gpus 4
```

新数据用 `prepare G1`，已生成的数据不会覆盖。`show --verbose` 才展开底层参数。
下文 suite/profile 与旧命令为高级入口，日常无需编辑。

当前 G1 数据目录是 `data/processed/reasonrank_simple/`，4,963 条全量 train。
`train.jsonl` 只含 `id/query/positive/negatives/source`。
`train.metadata.jsonl` 保存 IDs、teacher 排序和审计信息，loader 按 id 自动关联并验证
元数据 ID；移动数据时保持这两个文件在同一目录。JSONL 不存 padding 或重复标签。
ReasonRank 训练只接受 ready 格式，旧版完整单正例格式不再兼容；数据量与抽样未改变。

对应 [实验计划](../../../paper/EXPERIMENT_PLAN.md)，修订于 2026-09-11。
统一入口为 `scripts/experiments/iclr2027.py`，只依赖 Python 和 PyYAML。

## 使用

从 repository root 运行：

```bash
# 列出三组全部逻辑行与 READY/BLOCKED/REUSE/EVAL 状态
python scripts/experiments/iclr2027.py list

# Dry-run：打印合并配置、实验协议、启动命令预览和阻塞项，不创建文件、不下载模型
python scripts/experiments/iclr2027.py resolve G1-J-RL
python scripts/experiments/iclr2027.py resolve G1-Q-RL
python scripts/experiments/iclr2027.py resolve G2-W-RL
python scripts/experiments/iclr2027.py resolve G3-AnsRL

# 就绪检查：任意训练行未就绪则 exit 2，便于 GPU 调度前调用
python scripts/experiments/iclr2027.py check --group G1
python scripts/experiments/iclr2027.py check G2-D-CL

# 仅在前置条件全部完成后启动一个明确的 run
python scripts/experiments/iclr2027.py launch G1-J-RL --nproc 4
```

不提供 `launch all`。`--nproc` 默认 1，只控制单机进程数，不读取 NODE_RANK 作为 GPU 数。
多机调度尚未接入此入口。训练环境沿用 repository 的 torch/transformers 依赖；此入口
不安装运行时。可用 `--suite PATH` 指向同目录下的校准副本，配置路径相对于 suite 文件。

## 文件职责

- `suite.yaml`：run IDs、更新范围、objective、控制/复用关系、选模指标、最终评测、预算和实现状态。
- `g1.yaml`：E0 / ReasonRank 的共享参数，禁用旧 loader 自动切 dev 和 per-source cap。
- `g2.yaml`：E2Rank 共享参数，B0 和 pooling/prompts 需明确填写，不继承旧 instruction 模型假设。
- `g3.yaml`：E0 / 单个 NQ QA 任务的初始配置，index/generator/candidates 需按 manifest 核实。

这些组配置是**协议草案**。当前 `resolve` 生成的 `config` 部分会展示计划中的参数，
包括尚未支持的 `lambdaloss` 等值；不要把它直接传给旧 trainer 绕过 `check`。
`protocol` 保存实验语义；冻结文档和 pinned revision 等仍需进一步接入训练器。
修改这些元数据并不会自动实现训练行为，suite 中的 implementation blockers 需要配套代码与验证。

## 当前可执行程度

G1 joint 的 CL/RankNet/RL 已可启动，其余行按具体缺项显示 BLOCKED。已完成的是
配置展开、计划映射、依赖和预算检查、启动前检查与日志保存；没有完成训练器改造。

- G1：变长单正例 schema、候选 mask、去重候选池、CL/binary RankNet 和 RL 已接入。
  仍需 LambdaLoss、真正 frozen document branch；G1 不再用 dev 选模。
- G2：需 manifest loader、deterministic dev selection、LambdaLoss；共同 warm-up
  还需 data cursor、optimizer/scheduler 状态及 T/W 预算整合。
- G3：需 LambdaLoss、retrieval nDCG、候选访问公平控制、answer-F1 选模。
- 模型 revision 不再是强制配置或启动阻塞。

G1-Q-RL 的 query action sampling 已可表达，但旧 GRPO wrapper 仍会使用训练中的 document
backbone；因此配置它不等于 fixed-index 训练，`frozen_document` 检查不可省略。
G3 原有 `source_aware_mrr` 不是计划中的 nDCG，G3-RetRL 明确阻塞等待对应 reward。
现有 `scripts/validate_experiments.py` 检查旧 shell 配置，不代表本方案可执行。

## 预算和依赖

23 个逻辑行 = 20 个训练执行 + 2 个 E0 评测 + 1 个复用行。

- `G2-D-CL` 训练到总预算 T，并在预先固定 W 步保存 W0。
- `G2-W-CL` 是 D-CL 后半程的结果别名，入口不创建新的输出目录、不自动重训。
- `G2-W-LL/RL` 只从 `G2-D-CL-s42/checkpoint-W` 初始化，预算为 T-W。
  W 必须在保存 schedule 上；不能从 D-CL 最终模型或旧 LoRA Stage 1 自动回退。
- G1 ablation `control` 指向 G1-J-RL，仅描述对照关系，不自动启动 control。
- G3 从 E0 开始，不自动采用其他组的获胜 checkpoint。

`protocol.max_steps`、`learning_rate`、`eval_steps` 默认为 null，要求校准后填写。
G2 还要填写 B0、表示协议、W 和 continuation optimizer 规则。T/W 以 optimizer steps
编码，manifest/训练日志仍须记录 global batch、实际 examples/tokens，不能只报告 steps。
可用每行 `protocol: {learning_rate: ...}` 覆盖目标专属校准结果；直接对照的总预算应保持一致。

## 数据 manifest 契约

默认位置为 `data/protocols/g1_reasonrank.json`、`g2_e2rank.json`、`g3_qa.json`。
这是启动审计契约，不是已实现的 trainer split API。至少包含：

```json
{
  "version": 1,
  "split_seed": 20260911,
  "artifacts": [
    {"role": "train", "path": "data/processed/EXAMPLE/train.jsonl", "sha256": "ACTUAL_SHA256"},
    {"role": "dev", "path": "data/processed/EXAMPLE/dev.jsonl", "sha256": "ACTUAL_SHA256"}
  ],
  "evaluation_protocol": {"tasks": "FROZEN_TASK_LIST", "revisions": "ACTUAL_REVISIONS"},
  "split_audit": {"report": "PATH_TO_GROUP_AND_OVERLAP_AUDIT"}
}
```

检查器验证引用文件和哈希、必填 train/dev 及协议字段；不把这些字段非空当成语义去污染
证明。实际 split disjointness、G3 corpus/index/generator 对齐仍需数据准备和 trainer 检查。
最终 test、QA corpus/index、去重清单等也应加入 artifacts，并记录实际样本/来源数量。
不要复制示例占位符当作就绪 manifest。

## 输出与恢复

检查通过后，启动前将 resolved runtime config、协议、命令与哈希写入
`checkpoints/iclr2027/.launches/RUN-s42/`，模型输出独立为 `RUN-s42/`。
既有输出或 launch receipt 会阻止重复启动。不会凭 `config.json` 就判断训练已完成，
也不会覆盖旧实验目录。失败运行的恢复尚未接入，需按 trainer checkpoint 状态显式实现，
不要删除 provenance 后把新运行伪装成原运行的续训。

## 本地验证

```bash
python -m unittest discover -s scripts/tests -p test_iclr2027_runner.py -v
```

覆盖预算/别名复用、采样消融、frozen-document 启动阻止、warm-up 剩余预算、哈希变化、
已有输出保护和配置循环。无需 GPU；不声称完成 trainer 参数解析或训练数值验证。

## G1 数据处理与全量训练（2026-09-11 更新）

```bash
uv run --no-project --with pyarrow python scripts/prepare_reasonrank.py
```

默认读取已下载且 SHA-256 校验通过的 ReasonRank train parquet，应用 BRIGHT 保守隔离
清单，排除 MSMARCO、标签冲突、无负例和重复 query。每个 query 用 seed 42 固定随机选择
一个正例，移除其余已知正例，保留实际负例。训练 JSONL 使用单正例接口；配套 metadata 保存文档 ID、原始正例 IDs 和独立的
teacher_ranking，不伪造 teacher 排序作为正例。

当前输出在 `data/processed/reasonrank_simple/`：**4,963 train / 0 dev**，候选数 2–20。
G1 改为提前固定配置和训练预算、报告最终 checkpoint，不做 dev 网格调参、不用 BRIGHT
选模。G2/G3 的开发集协议不变。当前 suite 已指向该目录内的 manifest；其最终评测协议
仍待填写；trainer mask 集成已完成 CPU 验证，尚未进行正式 GPU 训练。

默认 `--dev-size 0`；此前 4,463 train / 500 dev 保留在 `data/processed/reasonrank/`。
若要复现旧划分，用 `--dev-size 500 --output-dir NEW_DIRECTORY`；拒绝覆盖已有目录。
脚本生成 decisions、summary、split audit 和带文件 hashes 的 manifest。

`EmbeddingDataCollator` 在拼 batch 时用空字符串补齐，并输出候选 mask 和标签。JSONL 不存 padding，补齐 label=0
只是占位，不能作为真实负例。原始正例 IDs 保留，用于以后屏蔽跨 query 池里重新出现的
已知正例。训练 collator 已接受该 schema，将抽取正例放到第一个有效位置，生成
candidate mask 和两个 cross-query masks。编码只处理有效文档；CL/RankNet、RL reward、
采样、document log-prob 与 KL 排除 padding。显式 binary 标签不再被 teacher 排序覆盖。

可运行 `.venv/bin/python -m unittest discover -s scripts/tests -p test_candidate_masks.py -v`
检查补齐不变性、零 padding 梯度、不同 rollout/采样角色以及新 schema 的 loader/model 链路。
