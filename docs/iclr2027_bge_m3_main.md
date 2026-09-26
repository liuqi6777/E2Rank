# BGE-M3：BRIGHT 主表补充实验

2026-09-26 更新：官方 GPT-4 query 重评已替换论文中的旧 reasoning 结果。BGE-M3 的 E0 / InfoNCE / LambdaLoss / RELER 分别为 **21.45 / 20.62±0.09 / 19.13±0.10 / 23.84±0.07**，RELER 相对 InfoNCE 为 **+3.22**。原 query 分数不变。新结果以 [重评导出](../paper/_summary/bright_gpt4_reasoning/) 和 [核验记录](../paper/BRIGHT_GPT4_REEVAL_REVIEW_2026-09-26.md) 为准；下方 2026-09-25 表格及其 reasoning 领域解读仅保留为历史记录。

使用与 0.6B 主表相同的 prepared ReasonRank 数据、113 steps、global batch 128、学习率 `5e-6`、seed `42/3407/2026`。训练矩阵为 InfoNCE、graded LambdaLoss、RELER（graded nDCG@10 + pairwise `λ=0.5`，K=0，CMP，G=64，`ρ=0.70`）各三次，共 **9 次训练**。原始 BGE-M3（E0）只评估一次。每个模型都评估 BRIGHT 的 original query 和 GPT-reasoning query，分别需要 10 次评估。

模型采用 [BAAI/bge-m3 的固定提交](https://huggingface.co/BAAI/bge-m3/tree/5617a9f61b028005a4858fdac845db406aefb181)，沿用仓库已有的 BGE-M3 dense 协议：CLS pooling、右侧 padding、无末尾附加 token、无 query/document 前缀。数据和主表其他模型相同，**不是** G2 的 BGE-M3 多 source 训练数据。结果写入 `checkpoints/iclr2027-bge-m3-main/`；不会读取 0.6B checkpoint。

配置：[suite](../configs/experiments/iclr2027/suite_g1_bge_m3_main.yaml)、[模型与训练](../configs/experiments/iclr2027/g1_bge_m3_main.yaml)、[路径设置](../configs/experiments_iclr2027_bge_m3_main.yaml)。入口：[训练和完整评估](../scripts/run_iclr2027_bge_m3_main.py)，以及[仅 GPT-reasoning 评估](../scripts/eval_iclr2027_gpt_reasoning_bge_m3.py)。

在仓库根目录执行：

```bash
# CPU：查看 9 条训练及 E0 的解析配置和两套评估命令
python scripts/run_iclr2027_bge_m3_main.py plan

# 校验 prepared ReasonRank 文件和 manifest 的 SHA256；不下载模型或启动 GPU
python scripts/run_iclr2027_bge_m3_main.py check

# 8 张可见 BF16 GPU：先评估 E0，再顺序执行 9 条训练；每条训练后评估两种 query
python scripts/run_iclr2027_bge_m3_main.py train

# 训练和评估可按方法、seed 拆分；E0 无训练
python scripts/run_iclr2027_bge_m3_main.py train --methods reler --seeds 42

# 只评估 E0 和已有的训练权重；缺失结果会补跑，完整结果跳过
python scripts/run_iclr2027_bge_m3_main.py eval
python scripts/run_iclr2027_bge_m3_main.py status
```

`train` 遇到已有完整最终权重时跳过训练并补评估；已有不完整训练目录会报错，不自动覆盖或续训。`eval` 不需要 ReasonRank 训练文件，只需模型权重及 BRIGHT 评估依赖。原始 query 结果写在各 run 的 `mteb_eval/bright/`，reasoning 结果写在 `mteb_eval/bright_gpt_reasoning/query-gpt4-reasoning/`。更换机器上的数据路径或输出根目录时，复制 settings 文件并通过 `--config` 指定。

## 历史结果（2026-09-25，reasoning 已被新重评替代）

已收到并审计[结果导出](../paper/_summary/bge_m3_main/)：九次训练及 E0，共 10 个 original-query、10 个 GPT-reasoning-query 评测、240 个 subset 分数。三种训练方法均覆盖 seed 42/3407/2026；E0 仅评估一次。分数为 nDCG@10 × 100，训练行报告三 seed 均值和样本 SD。

| Method | Original query | GPT-reasoning query |
| --- | ---: | ---: |
| E0 | 10.66 | 21.41 |
| InfoNCE | 12.63 ± 0.11 | 22.33 ± 0.32 |
| LambdaLoss | 11.40 ± 0.08 | 18.74 ± 0.20 |
| RELER | **14.49 ± 0.04** | **26.07 ± 0.05** |

RELER 相对 InfoNCE 分别为 **+1.86/+3.74**，两种 query 的三个配对 seed 差均为正。领域收益不一致：Coding 类在两种 query 下均下降 3.49。论文表 1 已加入 BGE-M3 的 original/GPT-reasoning 两个区块，完整展示 12 领域与 Avg.；逐 seed 结果位于附录 E.1，编码协议位于实验协议附录。

[论文表格生成器](../paper/iclr2027/build_k0_results.py)直接读取三份 CSV，核对 seed 覆盖、12-subset 聚合及 grouped 均值/SD，并生成[BGE-M3 审计](../paper/iclr2027/bge_m3_results_audit.json)。复算全部论文表格：

```bash
python paper/iclr2027/build_results.py
```

本轮论文更新仅接入已有结果，未重新启动训练或评估。
