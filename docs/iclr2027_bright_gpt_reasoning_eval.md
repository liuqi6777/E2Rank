# ICLR 2027 主表：BRIGHT GPT-reasoning query 评测

主表中每个模型各需 10 次只评测：E0 一次，InfoNCE、LambdaLoss、`K=0 + pairwise` RELER 各三个 seed。评测复用训练后的 checkpoint、语料和 qrels，仅将 query 改为 BRIGHT 官方 examples 中的原 query 加 GPT reasoning；reasoning 缺失时沿用原 query。`eval_mteb/run_mteb.py` 使用固定的 BRIGHT 数据 revision，并在日志中报告各 subset 的回退数。

从仓库根目录执行。`plan` 只解析配置并列出命令，不读取权重；`check` 检查本地最终权重及已有 reasoning 结果；`eval` 先检查全部所选 checkpoint，再顺序运行缺失项，完整的 12-subset 结果会跳过。每个脚本默认选全套 10 次，也可用 `--methods`、`--seeds` 拆分。E0 只随 seed 42 运行；选择 seed 3407/2026 时只安排三个已训练方法。

```bash
# 0.6B：先在新 suite 输出根目录导入 E0/InfoNCE/LambdaLoss，并完成 K0 主配方训练
python scripts/eval_iclr2027_gpt_reasoning_06b.py plan
python scripts/eval_iclr2027_gpt_reasoning_06b.py check
python scripts/eval_iclr2027_gpt_reasoning_06b.py eval

# 4B：复用原 LoRA 基线及 --set k0、LR 2e-4 的三个训练结果
python scripts/eval_iclr2027_gpt_reasoning_4b.py plan
python scripts/eval_iclr2027_gpt_reasoning_4b.py check
python scripts/eval_iclr2027_gpt_reasoning_4b.py eval

# 例如只评测刚跑完的 4B RELER seed 3407
python scripts/eval_iclr2027_gpt_reasoning_4b.py eval --methods reler --seeds 3407
```

0.6B 默认读取 `checkpoints/iclr2027-final-k0/`；4B 默认读取 `checkpoints/iclr2027-qwen3-embedding-4b-lora/`。不同机器可通过 `--config` 指向对应 suite settings 文件。4B 评测读取训练脚本生成的 `-merged` 模型；若 merged 权重尚未生成，先对相应运行执行原 4B 脚本的 `eval`，完成 adapter 合并与原 query 评测。

GPT-reasoning 结果独立写在各运行目录下的 `mteb_eval/bright_gpt_reasoning/query-gpt-reasoning/`，不会覆盖 `mteb_eval/bright/` 的原 query 结果。脚本拒绝把不完整的 12-subset 文件当成已完成结果，也不会自动删除部分输出。两个脚本都只做评测，不启动训练。
