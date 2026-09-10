# 脚本索引

从仓库根目录运行。论文实验只使用 `experiment.py`，其余是明确用途的辅助工具。

| 脚本 | 职责 |
|---|---|
| `experiment.py` | G1/G2/G3 的 prepare、list、show、check、train |
| `experiments/iclr2027.py` | 实验定义解析、预算与依赖检查、实际启动；由公共入口调用 |
| `download_reasonrank_audit.py` | 下载审计使用的 ReasonRank / BRIGHT 原始数据 |
| `audit_reasonrank_bright.py` | 检查数据来源、标签和 BRIGHT 重叠，产出审计记录 |
| `prepare_reasonrank.py` | 单正例化、去污染与 ready 数据编译，也提供审计共用的原始格式解析函数 |
| `run.sh` / `run_baseline.sh` | 通用 RL / supervised torchrun 包装器，显式指定 `NPROC_PER_NODE` |
| `rag_pipeline.sh` | RAG 数据准备、编码、候选挖掘、底层训练与评测 |
| `rag_acceptance.sh` / `rag_toy_distributed.py` | RAG 分布式索引和端到端验收 |
| `measure_score_gaps.py` | embedding 分数间隔诊断 |
| `merge_lora.py` | 显式 LoRA 实验的 adapter 合并；当前 full FT 主实验不需要 |
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

测试检查实现，不代表完成 GPU 训练。G2/G3 尚未实现的实验行仍会显示 BLOCKED。

## ReasonRank 数据准备

三个步骤分别负责下载、审计和转换，共用 `--data-dir`（默认 `data/audit_reasonrank_bright`）：

```bash
python scripts/download_reasonrank_audit.py
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
