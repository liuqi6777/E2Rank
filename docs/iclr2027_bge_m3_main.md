# BGE-M3：BRIGHT 主表补充实验

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

`train` 遇到已有完整最终权重时跳过训练并补评估；已有不完整训练目录会报错，不自动覆盖或续训。`eval` 不需要 ReasonRank 训练文件，只需模型权重及 BRIGHT 评估依赖。原始 query 结果写在各 run 的 `mteb_eval/bright/`，reasoning 结果写在 `mteb_eval/bright_gpt_reasoning/query-gpt-reasoning/`。更换机器上的数据路径或输出根目录时，复制 settings 文件并通过 `--config` 指定。

本配置仅准备实验，未启动 9 次 GPU 训练。公开结果与论文主表需要训练和两套评估完成后再汇总。
