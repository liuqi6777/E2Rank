# RL Training and Evaluation for Embedding

Embedding RL、InfoNCE / RankNet / LambdaLoss 与固定索引 RAG。当前实验按 G1 / G2 / G3 组织。

## 环境

```bash
uv sync
source .venv/bin/activate
```

训练需要 GPU；配置展开与检查不下载模型、不启动训练。

## 实验入口

[Reasoning E0 核心实验](docs/g1_reasoning_e0.md)已配置：DIVER-0.6B / ReasonEmbed-4B
各自比较 E0、CL、graded LL、graded RL（G64＋CP），复用 ReasonRank 与最终 BRIGHT。
入口：`python scripts/run_g1_reasoning_e0.py check`；GPU 运行去掉 `check`。
默认两模型×三方法×三 seed，支持 `--models diver` / `--models reasonembed` 和 `--seeds 42` 拆分。

[ReasonEmbed 数据上的 G2 CL](docs/g2_reasonembed_cl.md)已接入独立 D/E/W 队列：每次 1,200 步、最终 BRIGHT，暂不包含 RL。训练机器上先运行 `scripts/prepare_reasonembed.py`，再执行 `bash scripts/run_g2_reasonembed_cl.sh check` / `train`。

G2-CL 新轮次设计见 [G2_CL_R2_PLAN.md](paper/G2_CL_R2_PLAN.md)：沿用原 D/E/W、seed 42、训练与 MTEB 评测设置，应用昨晚的新协议和 sampler 修复，使用独立输出目录并重建 W0。新配置已准备，尚未启动 GPU 训练；下文 G1 夜跑队列不包含这批任务。

2026-09-16 的[新实验计划](paper/EXPERIMENT_PLAN.md)已接入[整批夜跑脚本](docs/g1_r2_overnight.md)：
3 个监督基线 + MRR / binary nDCG / graded nDCG 各自的 SF/CP，RL 统一 G64，共 9 方法 × 3 seed = 27 次训练。
沿用现有数据与丢尾规则，每个 epoch 在 source 内重新组合 microbatch，不另划 dev；固定 113 步、最终 BRIGHT 评测，G2/G3 暂缓。

在已激活训练环境的 8 卡 GPU 机器上，一条命令运行全部：

```bash
nohup python -u scripts/run_g1_r2.py > g1_r2_night.log 2>&1 &
```

新批次配置是 [configs/experiments_r2.yaml](configs/experiments_r2.yaml)。先只预检用 `python scripts/run_g1_r2.py check`；
结果在 `checkpoints/iclr2027-r2/.r2_batch/summary.md`，重启同一命令会跳过已完成项，评测失败只补评测。
三台机器可分别添加 `--seeds 42`、`--seeds 3407`、`--seeds 2026`，每台跑九个方法；E0 和公共诊断只由 seed 42 执行。
分机汇总位于 `.r2_batch/queues/seeds-<seed>/summary.md`；结果汇集后用 `python scripts/run_g1_r2.py summary` 生成三 seed 总表。

### 历史实验入口

下列命令对应 [configs/experiments.yaml](configs/experiments.yaml) 和旧注册矩阵：

```bash
python scripts/experiment.py prepare G1
python scripts/experiment.py list
python scripts/experiment.py show G1-J-RL
python scripts/experiment.py show G1-J-RL --verbose
python scripts/experiment.py check G1-J-RL
python scripts/experiment.py train G1-J-RL --gpus 8
```

| 组 | 目的 | 当前状态 |
|---|---|---|
| G1 | 开源 embedding model → ReasonRank reasoning 训练，BRIGHT 主评测 | Joint CL / RankNet / LambdaLoss / RL 与 query-policy-only RL 消融已接入 |
| G2 | 两套数据上 B0 / CL warm-up W0 / 原始 embedding E0 的 CL/RL 比较 | E2Rank CL 已完成；六条 RL 配方和批量入口已接入 |
| G3 | 固定索引 RAG 的检索与答案目标 | 保留现有 RAG 工具；论文实验的选模和奖励协议尚待接入 |

`list` 展示 READY / BLOCKED / EVAL / REUSE；`check G1` 检查整组，因此包含未实现行时返回非零。
READY 表示启动前检查通过，不代表已完成 GPU 训练。入口只启动指定的一行，不自动运行依赖或覆盖输出。
`experiment.py` 默认使用单机 8 卡；其他规模用 `--gpus N` 显式覆盖。默认 batch 配方按 8×80GB 节点设置。

详细参数、run IDs、预算与输出规则见 [实验配置指南](configs/experiments/iclr2027/README.md)。
研究设计见 [实验计划](paper/EXPERIMENT_PLAN.md)。

G2 RL 使用固定 MRR@10 / G=32 / alignment 0.90 配方，按 E → W → D 执行：

```bash
bash scripts/run_g2_rl.sh check 8
bash scripts/run_g2_rl.sh train 8
# BGE-M3 数据的对应入口：scripts/run_g2_bge_rl.sh
```

批量预检要求同数据 D-CL 的最终权重已就绪；W-RL 只从该权重初始化。
完整矩阵与诊断安排见 [G2 RL 计划](paper/G2_RL_PLAN.md)。

## 数据

G1 使用 `data/processed/reasonrank_multi/train.ready.jsonl` 中的 4,963 条训练输入记录，无 dev。
保留全部已知正例；现有采样器丢尾后的实际索引为 4,896 条，数据哈希及运行约定见[新实验计划](paper/EXPERIMENT_PLAN.md)。
`prepare G1` 每个 query 固定抽一个正例作为 in-batch 代表，同时保留其余已知正例和变长负例，生成 `train.ready.jsonl`。
Loader 直接读 ready 文件；collator 动态补齐和生成 mask。公开文本与审计 sidecar 不参与训练加载。
已有 ready 文件不会自动重写。
G1 主对照和 RL 消融使用 joint encoder，不需要预先运行 `encode G1`。`G1-DR` 是独立的
query-only full-corpus 动态检索行：运行前须用 `encode G1` 从 `xlangai/BRIGHT` 的
`documents` 配置构建分领域 E0 只读索引并校验 ReasonRank document ID。

原始数据下载、BRIGHT 重叠审计和预处理各脚本的职责见 [脚本索引](scripts/README.md)。

G2 的原始 E2Rank 数据可下载为：

```bash
hf download Alibaba-NLP/E2Rank_ranking_datasets train.jsonl --local-dir data --repo-type dataset
```

原始格式是 `{query, document, ranking, source}`，`ranking` 为从 1 开始的 teacher permutation。
G2 同时注册 E2Rank 与 BGE-M3 两套 D/W/E × CL/RL 实验。E2Rank 直接读取
`data/train.jsonl` 并训练 1200 steps；BGE-M3 从 `configs/experiments.yaml` 的
`G2.bge_m3_data` 目录读取各 source 数据并训练 1 epoch。两套实验除数据路径与训练预算外一致，
都不另留内部 dev/test，最终评测使用同一组固定外部检索任务。
全部 G2 训练会在 `checkpoint-0` 及每次 checkpoint 保存后运行
`MTEB(eng, v1, subset)` callback，用于记录训练曲线；E2Rank 每 200 steps、BGE-M3
每 1000 steps 保存并评测，该结果不用于选模。
通过 `python scripts/experiment.py train G2-* --gpus N` 启动的 G2 训练在最终模型保存成功后，
会自动用可见 GPU 跑一次完整的 `MTEB(eng, v2)`；结果写入
`checkpoints/iclr2027/G2-*-s42/mteb_eval/final/`。任一任务失败会让评测阶段返回非零，
已经保存的训练模型不受影响；训练前和中间 checkpoint 不跑这项完整评测。
通用 loader 仍支持 BGE-M3 的 `query/pos/neg` 格式。
该路径只支持 binary relevance：每次选择一个标注正例和固定数量的负例，保留全部
`pos` 的文本去重标识，过滤跨 query 的已知正例。`pos_scores/neg_scores` 只用于选样，
不转换成 graded 等级；无 teacher 排序时负例的 RankNet 标签并列。
无文档 ID 时按文本过滤，不把缺失 ID 当作相同文档。

## 通用工具

论文以外的单次试验可直接组合底层配置：

```bash
NPROC_PER_NODE=1 bash scripts/run.sh configs/exp/template.yaml
NPROC_PER_NODE=1 bash scripts/run_baseline.sh \
  --base-train configs/train/posttrain.yaml \
  --base-dataset configs/dataset/e2rank_listwise.yaml \
  --base-model configs/model/qwen3_embedding_0.6b.yaml \
  --base-baseline configs/baseline/ranknet.yaml \
  --base-eval configs/eval/default.yaml
```

这些工具不应用 G1/G2/G3 的实验检查。模型配置定义 pooling、padding、append token 和 query/document 模板；
更换模型时需要同步表示协议。`configs/train/posttrain.yaml` 为共享 full FT 参数，
`configs/train/default.yaml` 是显式选择的 LoRA 示例。

表示协议现使用 `tokenization_version=2`：`append_token: pad/eos` 对正文关闭自动 special
tokens，预留一个位置并在截断后追加指定 token；`append_token: none` 保留模型原生处理。
因此 Qwen3 普通模型与 Embedding 模型均只有一个有效末尾读出 token，长输入也不会丢失它。
训练、MTEB、RAG 和固定语料编码共用实现；所有训练 checkpoint 都保存解析后的 token 协议，
不依赖 MTEB callback。旧协议 checkpoint 不兼容，旧语料索引须重新编码；MTEB 结果标识新增
`__tokens-v2__pool-fp32`，不会复用旧协议结果。细节和验证见 [模型协议](docs/embedding_protocol.md)。

CL/LL/RL 的归一化和评分统一为 FP32；RL 可用 `gradient_estimator: conditional_projection`
启用条件投影。使用固定状态配对诊断验证全参数梯度后再做训练对照，见 [梯度估计器](docs/gradient_estimators.md)。

现有固定索引 RAG 工具保留在 `scripts/rag_pipeline.sh`，支持 prepare / encode / candidates / train / tune-eval / eval。
`configs/rag/` 是底层 RAG 配方；G3 正式运行仍通过 `experiment.py` 检查。
RAG 数据准备需要 `hf`，索引需要兼容 CUDA 的 FAISS；答案生成需要单独部署 generator 服务。
每个命令的参数可通过 `scripts/rag_pipeline.sh COMMAND --help` 查看（train 的参数入口是配置文件）。

评测入口：

```bash
bash eval_mteb/scripts/run_mteb.sh CHECKPOINT 'MTEB(eng, v2)' configs/model/qwen3_embedding_0.6b.yaml
bash eval_mteb/scripts/run_mteb.sh CHECKPOINT BRIGHT configs/model/qwen3_embedding_0.6b.yaml
```

通过 `python scripts/experiment.py train G1-* --gpus N` 启动时，最终模型保存成功后会自动运行
BRIGHT：joint run 用训练后 checkpoint 编码 query 和 passage；`G1-DR` 用训练后 query encoder
配合固定 E0 passage index。结果写入各 run 的 `mteb_eval/bright/`；训练前和中间 checkpoint
不运行 BRIGHT，也不能把 12 个 subset 的 corpus 合并搜索。

```bash
bash eval_mteb/scripts/run_mteb.sh CHECKPOINT BRIGHT \
  configs/model/qwen3_embedding_0.6b.yaml \
  --fixed_corpus_model Qwen/Qwen3-Embedding-0.6B \
  --fixed_corpus_model_revision E0_COMMIT_SHA \
  --fixed_corpus_index_dir data/eval/bright_qwen3_e0
```

`--fixed_corpus_model_revision` 应填写训练所用 E0 的 immutable commit SHA；未显式填写时，
评测器会使用模型加载后解析出的 revision。每个 subset 的目录都包含 `index_manifest.json`
和独立 shards；已完成索引的语料、顺序、分块或 E0 表示协议不匹配时会停止而不是重建。
普通 joint-eval 与 fixed-corpus eval 使用不同结果标识，不会互相复用已有结果。

BRIGHT 通过 MTEB 的 `BrightRetrieval` 任务运行，但显式从 `xlangai/BRIGHT` 的 `documents`
配置和仓库固定 revision 加载 corpus，复用 MTEB 的 exact retrieval、nDCG@10 和结果格式；
评测器会按 12 个领域分别应用 query instruction。已有同路径结果默认复用，
需要重跑时在上述命令末尾添加 `--run_kwargs '{"overwrite_results":true}'`。
默认 `--bright_query_set original` 使用原始 query；也可选择官方 examples 中的 GPT reasoning
作为 query 扩展（缺少有效 reasoning 的样本自动保持原 query）：

```bash
bash eval_mteb/scripts/run_mteb.sh CHECKPOINT BRIGHT \
  configs/model/qwen3_embedding_0.6b.yaml \
  --bright_query_set gpt-reasoning
```

`gpt-reasoning` 的结果写入输出目录下的 `query-gpt-reasoning/`，不会与原始 query 结果互相复用。
逐领域 nDCG@10 和宏平均可用同一汇总器读取：

```bash
python eval_mteb/summary.py results/mteb BRIGHT --views run,subset
```

## 表示空间离线分析

使用 `scripts/analyze_embeddings.py` 对 E0 / CL / RL 做编码缓存、全库排序诊断、
几何统计和 query vMF 扰动实验。配置示例在 `configs/analysis/bright.json`，需先填写
实际 checkpoint 路径；命令、输出和指标定义见 [分析指南](docs/embedding_analysis.md)。

## 目录

- `configs/experiments.yaml`：日常参数。
- `configs/experiments/iclr2027/`：三组实验定义和共享 profile。
- `configs/{train,dataset,model,grpo,reward,baseline,eval,rag}/`：通用参数组件。
- `scripts/experiment.py`：唯一推荐的论文实验入口。
- `scripts/experiments/iclr2027.py`：入口调用的内部解析与运行模块。
- `scripts/README.md`：数据、训练、RAG 和诊断脚本索引。
- `src/`：训练与模型实现；`eval_mteb/`：离线评测。

旧 Stage/Phase、posttrain shell 实验集合及其专用配置已移除，可从 Git 历史查阅。
论文 LaTeX 源码与图稿仅保留在本地，Markdown 和审计记录可提交。
