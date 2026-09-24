# ICLR 2027：0.6B K=0 论文实验 suite

本 suite 对应[最终表格草稿](../paper/ICLR2027_RESULTS_REORGANIZATION.md)和[分析图计划](../paper/ANALYSIS_PLAN.md)。[suite 配置](../configs/experiments/iclr2027/suite_iclr2027_final_k0.yaml)列出 **36 个新训练（12 配方 × 3 seed）**和 **28 个已完成结果导入项**；[独立 settings](../configs/experiments_iclr2027_final.yaml)将全部输出放到 `checkpoints/iclr2027-final-k0/`，不改旧 `checkpoints/iclr2027-r2/`。入口为 [`scripts/run_iclr2027_final_k0.py`](../scripts/run_iclr2027_final_k0.py)。RAG 和 4B 不在此 suite。

新训练包括：`K=0 + pairwise λ=0.5` 主配方、`λ=1.0`、binary nDCG/MRR、纯 graded 的 RLOO product/paired/query-only/doc-only、`K=1 + pairwise`、以及 `ρ=0.60/0.80/0.90` 的 `K=0 + pairwise`。每项 42/3407/2026 三 seed；旧 K0 纯 graded CMP、K sweep、K7/15 pairwise，以及 E0/InfoNCE/LambdaLoss 从原输出导入。导入项与旧 suite 的解析训练配置逐字段相同，仅输出目录不同；导入不是重训。Binary MRR 使用原始 binary 标签和自有候选 `reward_type=mrr`；`mrr_in_batch` 即使关闭负例全集，也可能增加跨 query 代表正例，不符合这里的 K0 条件。

导入映射中的 28 个旧目录名均已与本地 BRIGHT 汇总的 `result_dir` 核对。E0 是原始模型评测，旧目录名为 `G1-R2-E0`（没有训练 seed 后缀），导入后使用新 suite 统一的 `G1-R2-E0-s42` 目录名；其余运行名保持原样。

## 在训练机上执行

先确认 `configs/experiments_iclr2027_final.yaml` 的 `G1.data` 与旧运行使用的 prepared ReasonRank 数据相同；若旧根目录不是默认值，给导入命令加 `--source-root /旧根目录`。本地仓库不含旧 checkpoint，因此只能在保存旧结果的机器上正式复制。

```bash
# CPU：检查 suite、旧/新解析配置一致性以及当前完成数
python scripts/run_iclr2027_final_k0.py check

# CPU：预览可导入的旧结果；只认最终模型及 12-subset BRIGHT 完整评测
python scripts/run_iclr2027_final_k0.py import --dry-run

# 将完整的旧运行目录复制到新根目录；保留旧目录
python scripts/run_iclr2027_final_k0.py import

# 仅看新根目录的完成数；无 GPU 操作
python scripts/run_iclr2027_final_k0.py status

# 训练主配方一个 seed，再补其余 seed；每条训练后自动进行原 query BRIGHT 评测
python scripts/run_iclr2027_final_k0.py train --recipes main --seeds 42
python scripts/run_iclr2027_final_k0.py train --recipes main --seeds 3407 2026

# 选择其余配方；不选时 train 默认执行全部 12 配方 × 3 seed
python scripts/run_iclr2027_final_k0.py train --recipes lambda1 binary_ndcg binary_mrr
```

`import` 对每个旧运行先检查旧/新配置匹配、最终权重、embedding protocol 和 12 个 BRIGHT 子集；将完整运行目录复制到新根目录的临时目录并复核，然后原子改名。已有完整且有复制回执的目标会跳过；已有不完整或无回执的目标会报错，不覆盖。旧来源缺失会报告 `missing_source`，正式导入命令以非零状态退出，不会在新根目录伪造结果。完成后 `.imports/` 写入每条复制回执。`train` 只调度新训练，已完成目标自动跳过；存在不完整输出时保留现场并报错，需人工确认后再处理。`eval --recipes ...` 可对已有新权重重跑原 query BRIGHT 评测。GPT-reasoning query 使用[单独的主表评测脚本](iclr2027_bright_gpt_reasoning_eval.md)；梯度/曲线分析仍按论文图计划另行运行。

## 运行前必须核对的两个 reward 路径

纯 graded CMP 与 RLOO product 使用 shortlist 路径的 `K=0`；paired/query-only/doc-only 使用非 shortlist 路径的 `reward_type=ndcg`，都只保留自有候选。配置解析与 RL 参数验证已通过，且固定分数张量上的 `K=0` shortlist reward 与 own-only nDCG 一致。正式启动后三种 policy/rollout 消融前，仍需在同一小批次上核对两个训练路径的 reward 和 RLOO product 梯度；若不一致，不能将后三行解释为只改变 rollout/policy role。Binary MRR 也需核对原始正例身份和 @10 cutoff。

输出根目录还供[代表 checkpoint 分析配置](../paper/analysis/bright_representatives.json)使用。训练曲线使用的六条运行名见[运行清单](../paper/analysis/reward_curve_runs.csv)；导入的旧结果与新训练统一在一个目录下，后续汇总时仍应根据配置和回执区分来源。
