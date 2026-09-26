# ICLR 2027 主表：BRIGHT GPT-reasoning query 评测

2026-09-26 修复：旧 `query-gpt-reasoning/` 结果误将 `examples.reasoning`（人工 gold reasoning 标注）拼接到原 query，不能作为 GPT reasoning 评测结果使用。原结果保留供追溯，修复后的入口不会读取或覆盖它们；需复用现有 checkpoint 重新评测。

每个模型各需 10 次只评测：E0 一次，InfoNCE、LambdaLoss、`K=0 + pairwise` RELER 各三个 seed；0.6B、4B、BGE-M3 合计 30 次。正确输入为官方 `xlangai/BRIGHT` 的 **`gpt4_reason` 配置中的 `query` 字段原文**，不再拼接原 query 或 `reasoning`。语料和 qrels 仍使用相同固定 revision `3066d29c9651a576c8aba4832d249807b181ecae`。缺失 query、重复 ID 或与原 query 集合不一致时直接报错，不静默回退。

已只读核对该版本 12 个子集：1,384 条 GPT-4 query 均非空，ID、`gold_ids` 和 `excluded_ids` 与 `examples` 对齐；LeetCode/Pony 均有完整 GPT-4 query。官方配置中 biology 的两条 query 本身与原 query 相同，按发布文本原样使用。协议依据：[官方 run.py](https://github.com/xlang-ai/BRIGHT/blob/main/run.py)、[固定版本数据卡](https://huggingface.co/datasets/xlangai/BRIGHT/blob/3066d29c9651a576c8aba4832d249807b181ecae/README.md)。


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

# BGE-M3：同样复用原 checkpoint
python scripts/eval_iclr2027_gpt_reasoning_bge_m3.py plan
python scripts/eval_iclr2027_gpt_reasoning_bge_m3.py check
python scripts/eval_iclr2027_gpt_reasoning_bge_m3.py eval
```

0.6B 默认读取 `checkpoints/iclr2027-final-k0/`；4B 默认读取 `checkpoints/iclr2027-qwen3-embedding-4b-lora/`。不同机器可通过 `--config` 指向对应 suite settings 文件。4B 评测读取训练脚本生成的 `-merged` 模型；若 merged 权重尚未生成，先对相应运行执行原 4B 脚本的 `eval`，完成 adapter 合并与原 query 评测。

正确的 GPT-4 reasoning 结果独立写在各运行目录下的 `mteb_eval/bright_gpt_reasoning/query-gpt4-reasoning/`，不会覆盖原 query 或旧错误协议结果。launcher 使用 `--bright_query_set gpt4-reasoning`；旧参数值 `gpt-reasoning` 兼容映射到同一新协议和新目录。脚本拒绝把不完整的 12-subset 文件当成已完成结果，也不会自动删除部分输出。三个脚本都只做评测，不启动训练。

重新评测后，论文 CSV/表格需从新目录重新导出；当前已存的 GPT-reasoning 分数不会因代码修复自动变成正确结果。

## 16 个独立 8 卡节点同时启动

本次只修正 GPT-4 query 数据源并重评主表 30 个 reasoning setting。按用户要求，`excluded_ids` 的独立问题暂不处理，原 query 入口、结果及评测安排保持原状。因此此处修复的是 query 输入，不代表全部评测细节已与官方协议对齐。

每个节点使用已安装项目依赖的 Python 环境，并能读取分配到的 checkpoint。16 个节点使用同一版本代码和相同配置，只改变 slot。不要使用 torchrun：评测入口会自行使用当前节点可见的 8 张 GPU。

```bash
# 查看完整分配，不读取权重、不启动评测
python scripts/launch_bright_repair.py plan

# 16 个节点分别执行，编号依次为 0、1、…、15
bash scripts/launch_bright_repair.sh 0
```

分配：0–9 号节点各跑一个 4B checkpoint；10–15 号节点分摊 0.6B 和 BGE-M3，各跑 3–4 个。使用 4B 相对成本 4、其他模型成本 1 的调度估计，并非实测耗时。

中断后使用完全相同的命令重启：完整结果自动跳过，部分结果只补缺失子集。旧 query-gpt-reasoning 结果不会被当作新结果。日志位于 logs/bright-repair/slot-XX/，失败清单为 failures.json；单个 checkpoint 失败后继续该节点其他任务，最后以非零退出码报告失败。每个 checkpoint 有文件锁，防止重复节点同时写入。

```bash
# settings 路径可覆盖；16 个节点必须传相同配置参数
bash scripts/launch_bright_repair.sh 0 \
  --config-06b configs/experiments_iclr2027_final.yaml \
  --config-4b configs/experiments_qwen3_embedding_4b_lora.yaml \
  --config-bge-m3 configs/experiments_iclr2027_bge_m3_main.yaml

# 在对应节点只检查分配的权重，不启动评测
python scripts/launch_bright_repair.py check --slot 0
```

如需指定解释器，设置 PYTHON_BIN=/path/to/python。脚本不会训练或合并模型，4B 的 merged checkpoint 需已存在；启动前会检查该节点的全部 checkpoint。评测完成后需从新 reasoning 目录重新导出论文表格。
