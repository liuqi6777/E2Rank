# ReasonEmbed 数据上的 G2 CL

使用 ReasonEmbed 公开数据，自行训练 Qwen3-0.6B 的 G2 CL，并在每次训练完成后评测 BRIGHT。独立队列不包含 RL；这是沿用本项目训练协议的数据实验，不是作者模型的训练复现。

| Run | 初始化 | 预算 |
| --- | --- | --- |
| G2-ReasonEmbed-D-CL | Qwen/Qwen3-0.6B（现有 B0） | 1,200 步 |
| G2-ReasonEmbed-E-CL | Qwen/Qwen3-Embedding-0.6B（固定 revision） | 1,200 步 |
| G2-ReasonEmbed-W-CL | 本队列 D 的最终权重；重置 optimizer、step 和数据 epoch | 再 1,200 步 |

沿用 E2Rank G2 新轮次的 joint full FT、seed 42、8 GPU、global batch 128、microbatch 16、LR 5e-6、query/document 长度 512/1024、InfoNCE temperature 0.03、local in-batch negatives、tokenization v2 和 FP32 pooling/scoring。每 200 步保存，不划 dev，不按验证指标选择 checkpoint。关闭训练期间的 MTEB 回调；仅在最终模型保存后运行 BRIGHT，结果位于每个模型输出下的 `mteb_eval/bright/`（共用评测工具的目录名）。

## 训练机器上运行

激活项目训练环境后下载和转换（下载需要 Hugging Face CLI）：

```bash
hf download hanhainebula/reason-embed-data --repo-type dataset \
  --revision 8dada2e649a21913df303510980ebb8ed1de541e \
  --include 'reason-embed-data-0928/*.jsonl' --local-dir data/raw/reasonembed
python scripts/prepare_reasonembed.py \
  --input-dir data/raw/reasonembed --output-dir data/processed/reasonembed_g2
bash scripts/run_g2_reasonembed_cl.sh check
bash scripts/run_g2_reasonembed_cl.sh train
```

默认依次 D → BRIGHT → E → BRIGHT → W → BRIGHT；任一步失败就停止。`check` 检查 D/E，展开 W；W 的初始化权重在 D 完成后才检查。输出根目录为 `checkpoints/iclr2027-g2-reasonembed-cl`，不覆盖 E2Rank 模型。

修改数据位置、输出目录或训练预算使用 `configs/experiments_g2_reasonembed_cl.yaml`；`G2.data` 指向转换后的目录。独立数据配置为 `configs/dataset/reasonembed.yaml`，实验定义为 `configs/experiments/iclr2027/suite_g2_reasonembed_cl.yaml`。单独运行可用：

```bash
python scripts/experiment.py train G2-ReasonEmbed-D-CL --gpus 8 \
  --suite configs/experiments/iclr2027/suite_g2_reasonembed_cl.yaml \
  --config configs/experiments_g2_reasonembed_cl.yaml
```

不自动跳过、续训或重跑已有输出。如果训练成功但 BRIGHT 失败，仅重跑 `.launches` receipt 中记录的 `post_train_commands`，不要重跑训练队列。

## 数据处理约定

转换脚本仅使用 Python 标准库，输出 `train.jsonl`、`manifest.json` 和 `decisions.jsonl`，已有输出目录会拒绝覆盖。固定 seed 42，每条保留一个 positive 和最多 15 个 negatives，形成最多 16 个候选；去掉空文档、重复候选及与 positive 重合的 negatives。全部已知 positives 的 key 都保留，用于过滤 in-batch false negatives。处理不划分 dev，训练 loader 按现有 G2 规则在每个 source 内逐 epoch shuffle 并丢弃不足 microbatch 的尾部。

公开数据是 `pos`/`neg`，没有原始 graded 分数；输出使用 binary relevance，也不给 negatives 伪造排序。训练输入使用原始 `query`，不用 `reasoning_query`；原始 prompt 仅保留为 metadata，实际使用现有 G2 loader 的任务 instruction（这些 domain 走 generic fallback）。默认不要启用 `--all-positives`，它会扩大候选数。

manifest 记录输入/输出 hash、候选选择和清洗统计，revision 是本地文件来源的声明，无法仅靠 JSONL 验证。现有 G2 preflight 检查数据是否存在，并不自动校验该 manifest 的 hash；跨机器传输时应连同 manifest 保留。这里只在本地验证转换与配置，不启动 GPU 训练或完整 BRIGHT。
