# 脚本索引

ReasonEmbed 当前使用 [每阶段 1 epoch 的九条训练方案](../docs/g2_reasonembed_cl.md)：普通 CL 用 `run_g2_reasonembed_cl.sh`；Strong CL 用 `run_g2_reasonembed_cl_strong.py check/train/eval`（支持 `--branches D E W`）；binary CP 用 `run_g2_reasonembed_rl_r2.sh`。W 统一依赖新目录下普通 D-CL 的最终权重，每条最终评测 BRIGHT。

2026-09-18：E2Rank G2-R2 九条结果已同步，见 [结果文档](../paper/G2_R2_RESULTS.md)。脚本保持原行为：`run_g2_cl_r2.sh` 运行 D → E → W，`run_g2_rl_r2.sh` 运行 E → W → D，Strong CL 支持 D/E/W；W 保留同数据 D-CL 初始化依赖。本次没有启动训练，以下入口说明不表示需要重跑已完成实验。

当前 G1 新协议批次：`python scripts/run_g1_r2.py check` 预检，`python scripts/run_g1_r2.py` 一次完成
E0、配对梯度诊断和 27 次训练及最终 BRIGHT；支持跳过完成项和只补失败评测。见[夜跑说明](../docs/g1_r2_overnight.md)。
三台机器分别传 `--seeds 42`、`--seeds 3407`、`--seeds 2026` 可同时运行；E0/公共诊断由 seed 42 负责，日志和汇总按所选 seed 隔离。

强化版 CL 使用独立入口 `run_g1_cl_strong_r2.py` / `run_g2_cl_strong_r2.py`，支持 `check/train/eval`，直接启动训练和最终评测，不走 hash、manifest 或夜跑 receipt 校验；G1 可选 `--seeds`，G2 可选 `--branches D E W`。功能实现位于现有 `src/contrastive.py` 与 `src/train_baseline.py`。

大负例池纯 RL 使用 `run_g1_rl_large_pool_r2.py`，默认 CP / alignment 0.90 / seeds 42、3407、2026，逐个训练并评测最终 BRIGHT；`check` 预检，`--seeds` 选择种子。配方与服务器命令见[说明](../docs/g1_r2_large_pool.md)。

跨 query 文档联合策略使用 `run_g1_cross_query_policy_r2.py`，默认 CP/G64/alignment 0.90、seeds 42/3407/2026，先普通池再跨卡大池，共六次训练；支持 `check/train/eval`、`--pool local/large/both` 和 `--seeds`，每条训练后自动评测。见[运行说明](../docs/cross_query_document_policy.md)。

历史 G1 稳定性过夜批次：`python scripts/run_g1_stability.py check` 预检，`train` 顺序完成 21 次训练与
BRIGHT，`summary` 汇总三 seed 的均值/标准差/最差成绩。无时间自动停止，失败行记录后继续。
矩阵与命令见[过夜实验计划](../docs/g1_stability_overnight_plan.md)。

原始 rollout 梯度 JSON 的有限样本修正与配对核查使用 `python scripts/analyze_rollout_gradients.py`；
统计含义和最新结论见[梯度重分析](../docs/rollout_gradient_reanalysis.md)。

从仓库根目录运行。论文实验只使用 `experiment.py`，其余是明确用途的辅助工具。

| 脚本 | 职责 |
|---|---|
| `experiment.py` | G1/G2/G3 的 prepare、encode、list、show、check、train |
| `experiments/iclr2027.py` | 实验定义解析、预算与依赖检查、实际启动；由公共入口调用 |
| `run_qwen3_embedding_4b_lora.py` | 8 卡 Qwen3-Embedding-4B LoRA：原 Table 1 矩阵、三个核心 RL 消融、单独的 `--set k0` 主表补实验、E0、adapter 合并和最终 BRIGHT；见 [实验说明](../docs/qwen3_embedding_4b_lora.md) |
| `run_g1_mrr_seed_repeats.sh` | 预检并顺序运行最终 G1 MRR 配方的 seed 3407、2026；每次训练后自动评测 BRIGHT |
| `run_g1_cl_seed_repeats.sh` | 预检并顺序运行 G1 joint InfoNCE 的 seed 3407、2026；复用已有 seed 42 |
| `diagnose_rollout_gradients.py` | 固定 checkpoint 和训练 batch，仅改变 rollout，统计完整参数梯度的噪声与方向一致性 |
| `run_g1_rollout_seed_repeats.sh` | 固定训练 seed 42，预检并运行独立 rollout seed 42/3407/2026 的对照 |
| `download_reasonrank_audit.py` | 下载审计使用的 ReasonRank / BRIGHT 原始数据 |
| `audit_reasonrank_bright.py` | 检查数据来源、标签和 BRIGHT 重叠，产出审计记录 |
| `prepare_reasonrank.py` | 多正例保留、去污染与 ready 数据编译，也提供审计共用的原始格式解析函数 |
| `run.sh` / `run_baseline.sh` | 通用 RL / supervised torchrun 包装器，显式指定 `NPROC_PER_NODE` |
| `run_g2_cl.sh` / `run_g2_bge_cl.sh` | 分别串行预检并启动 E2Rank / BGE-M3 的三条 G2 CL |
| `run_g2_rl.sh` / `run_g2_bge_rl.sh` | 分别预检 E2Rank / BGE-M3 的三条 RL（含 W0），再按 E → W → D 训练与最终 MTEB 评测 |
| `run_g2_rl_r2.sh` / `run_g2_reasonembed_rl_r2.sh` | 新协议 E2Rank / ReasonEmbed 的单 seed 42 CP/G64/0.80 队列；各三条 RL，与对应 D/E/W CL 配对，见 [G2-RL-R2 计划](../paper/G2_RL_R2_PLAN.md) |
| `rag_pipeline.sh` | RAG 数据准备、编码、NQ/HotpotQA qrels 构造、候选挖掘、底层训练与评测 |
| `rag_acceptance.sh` / `rag_toy_distributed.py` | RAG 分布式索引和端到端验收 |
| `measure_score_gaps.py` | embedding 分数间隔诊断 |
| `analyze_embeddings.py` | 缓存编码、全库排序、几何、query vMF 扰动与报告；见 [分析指南](../docs/embedding_analysis.md) |
| `merge_lora.py` | 显式 LoRA 实验的 adapter 合并，支持固定 base-model revision |
| `zero3.json` | 模型配置共用的 DeepSpeed ZeRO-3 参数 |
| `tests/` | CPU 数据、mask 和实验解析测试 |

原始 parquet 处理依赖 `pyarrow`；重叠审计额外依赖 `rapidfuzz`（当前本地环境未安装）。

旧 `stage1.sh`、`phase*.sh`、`posttrain_*.sh`、对应 shell validator 和固定 16 候选 ReasonRank converter 已退役，
不再维护兼容入口。原始 ReasonRank 解析函数已合入 `prepare_reasonrank.py`，有效候选不因不足 16 而被丢弃。

```bash
python scripts/experiment.py list
python scripts/experiment.py check G1-J-RL
python -m unittest discover -s scripts/tests -v
```

测试检查实现，不代表完成 GPU 训练。缺少数据、初始化权重或协议参数的实验行仍会显示 BLOCKED。

## ReasonRank 数据准备

三个步骤分别负责下载、审计和转换，共用 `--data-dir`（默认 `data/audit_reasonrank_bright`）：

```bash
python scripts/download_reasonrank_audit.py --include-reasonrank-documents
uv run --no-project --with pyarrow --with scikit-learn --with rapidfuzz python scripts/audit_reasonrank_bright.py
python scripts/experiment.py prepare G1
```

目录内 `reasonrank/` 与 `bright/` 存放原始 parquet，`download_manifest.json` 记录下载来源，
`results/` 存放隔离清单和审计结果。所有产物位于被 Git 忽略的 `data/`，不依赖 `paper/` 下的文件。
下载器复用已有且有记录的文件，缺失文件逐个下载并保存进度；导入下载/审计模块不会执行任务。
审计会重新生成 `results/` 中的报告，筛选阈值保持现有规则。

`prepare G1` 使用公共实验配置中的输出目录。单独准备到其他目录：

```bash
python scripts/prepare_reasonrank.py --data-dir data/audit_reasonrank_bright --output-dir data/processed/reasonrank_new
```

默认全量 train、seed 42、排除 MSMARCO，不读取用于 dev 分组的 internal matches。
`--dev-size` 和 `--split-seed` 只用于明确要求的非默认划分；`--compile-only` 只编译已有 public + metadata。
原始数据、隔离清单和最终 ready 文件的职责分开，准备脚本拒绝覆盖已有输出目录。
`--include-reasonrank-documents` 会额外下载 pinned `liuwenhan/reasonrank_data_13k` 的 `id_doc`
映射。G1 主对照和 RL 消融不需要冻结索引；`G1-DR` 运行前必须执行 `encode G1`，为每个训练
source 构建 `<G1.data>/reasonrank_frozen_document_indices/<source>/` 和路由 manifest。
索引、corpus、offset、document-ID 映射及 vector shards 均带 SHA256。

## RAG train qrels

先用 `prepare` 冻结 FlashRAG `wiki18_100w` corpus，随后一条命令同时构造 NQ 和
HotpotQA train 的二值 passage qrels：

```bash
scripts/rag_pipeline.sh qrels
```

该命令会复用 `data/rag/flashrag_manifest.json` 中的 HotpotQA train，缺少时自动下载；
也会在缺少时下载 DPR `biencoder-nq-train.json.gz`。NQ 只保留 DPR `score == 1000`
的人工 positive context，HotpotQA 则要求所有标注 supporting facts 都映射到冻结 corpus。
主产物是 `data/rag/qrels/nq_hotpotqa_train.jsonl`，同时写出带输入哈希和覆盖率的
manifest，以及未能完整映射的 query 清单。脚本不会在检索时重新构造正例。
训练时检索 reward type 只保留 `mrr` 和 `ndcg`；两者读取同一份固定二值 qrels，
并与 generator 统一使用 top-10。
冻结 E0 index 后运行 `scripts/rag_pipeline.sh candidates`，候选挖掘会读取上述 qrels、
插入全部已知正例，并校验 qrels、corpus、FlashRAG source manifest 与 index 的哈希关系。

## G1 rollout 随机性诊断

`diagnose_rollout_gradients.py` 在固定 checkpoint 和真实训练 batch 上重复 rollout，输出完整参数
梯度的噪声、方向一致性和输入审计信息。`run_g1_rollout_seed_repeats.sh` 启动固定训练 seed 42、
独立 rollout seed 42/3407/2026 的完整对照。用法见 [诊断文档](../docs/rollout_rng_diagnostics.md)。

加 `--document-advantage-baseline counterfactual` 可比较逐文档反事实 baseline；默认保持原有
共享文档 advantage。公式、支持范围和对比命令见[实现文档](../docs/document_counterfactual_baseline.md)。
