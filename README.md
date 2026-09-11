# RL Training and Evaluation for Embedding

Embedding RL、InfoNCE / RankNet / LambdaLoss 与固定索引 RAG。当前实验按 G1 / G2 / G3 组织。

## 环境

```bash
uv sync
source .venv/bin/activate
```

训练需要 GPU；配置展开与检查不下载模型、不启动训练。

## 实验入口

日常只编辑 [configs/experiments.yaml](configs/experiments.yaml)，使用一个入口：

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
| G2 | Base LLM 上大规模 CL / RL，以及共同 CL warm-up 后的比较 | 预算、表示协议和部分训练能力待补齐 |
| G3 | 固定索引 RAG 的检索与答案目标 | 保留现有 RAG 工具；论文实验的选模和奖励协议尚待接入 |

`list` 展示 READY / BLOCKED / EVAL / REUSE；`check G1` 检查整组，因此包含未实现行时返回非零。
READY 表示启动前检查通过，不代表已完成 GPU 训练。入口只启动指定的一行，不自动运行依赖或覆盖输出。
`experiment.py` 默认使用单机 8 卡；其他规模用 `--gpus N` 显式覆盖。默认 batch 配方按 8×80GB 节点设置。

详细参数、run IDs、预算与输出规则见 [实验配置指南](configs/experiments/iclr2027/README.md)。
研究设计见 [实验计划](paper/EXPERIMENT_PLAN.md)。

## 数据

G1 使用 `data/processed/reasonrank_simple/` 中的 4,963 条全量训练数据，无 dev。
`prepare G1` 每个 query 固定抽一个正例，移除其余已知正例，保留变长负例，生成 `train.ready.jsonl`。
Loader 直接读 ready 文件；collator 动态补齐和生成 mask。公开文本与审计 sidecar 不参与训练加载。
已有 ready 文件不会自动重写。
当前 G1 主对照和 RL 消融均使用 joint encoder，不需要预先运行 `encode G1`。已有
`encode G1` 能从 `xlangai/BRIGHT` 的 `documents` 配置构建分领域只读索引并校验
ReasonRank document ID；该能力保留给尚未冻结设计的 G1 full-corpus 动态检索扩展，
当前没有对应正式 run。

原始数据下载、BRIGHT 重叠审计和预处理各脚本的职责见 [脚本索引](scripts/README.md)。

G2 的原始 E2Rank 数据可下载为：

```bash
hf download Alibaba-NLP/E2Rank_ranking_datasets train.jsonl --local-dir data --repo-type dataset
```

原始格式是 `{query, document, ranking, source}`，`ranking` 为从 1 开始的 teacher permutation。
G2 直接读取 `data/train.jsonl`，全量训练，不另留内部 dev/test；最终评测使用固定的外部检索任务。
通过 `python scripts/experiment.py train G2-* --gpus N` 启动的 G2 训练在最终模型保存成功后，
会自动用可见 GPU 跑一次完整的 `MTEB(eng, v2)`；结果写入
`checkpoints/iclr2027/G2-*-s42/mteb_eval/final/`。任一任务失败会让评测阶段返回非零，
已经保存的训练模型不受影响；训练前和中间 checkpoint 不跑这项完整评测。
通用 loader 仍支持 BGE-M3 的 `query/pos/neg` 格式。

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
BRIGHT：当前所有 G1 run 都使用训练后 checkpoint 编码 query 和 passage。结果写入各 run 的
`mteb_eval/bright/`；训练前和中间 checkpoint 不运行 BRIGHT。下面的 fixed-corpus 命令保留给
未来的动态检索扩展；不能把 12 个 subset 的 corpus 合并搜索。

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
逐领域 nDCG@10 和宏平均可用同一汇总器读取：

```bash
python eval_mteb/summary.py results/mteb BRIGHT --views run,subset
```

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
