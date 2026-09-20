# Qwen3-Embedding-4B LoRA 核心实验

## 目标与范围

这组实验只回答两个问题：G1-R2 的主结果能否迁移到 Qwen3-Embedding-4B 的 LoRA 训练；G1-R2 中最关键的 RL 组成在 4B 上是否仍有贡献。完整矩阵为 6 个训练配方 × 3 seeds，共 18 次训练，另做一次未训练 E0 的 BRIGHT 评测。它不重复 alignment、K/T、hard-negative、reward 类型和 LoRA rank 的大网格。

所有训练使用同一份清洗后的 ReasonRank、113 optimizer steps、global batch 128、8 卡、最终 checkpoint 和 BRIGHT nDCG@10。训练 seed、数据顺序 seed 和 RL rollout seed 均配对为 42、3407、2026。

## 实验矩阵

| 组别 | 配方 | 直接回答的问题 |
|---|---|---|
| Table 1 | InfoNCE | 4B LoRA 下的强对比学习基线 |
| Table 1 | LambdaLoss | graded 排序监督基线 |
| Table 1 | RELER | G1-R2 最终主配方能否迁移到 4B |
| RL 消融 | RELER-NoPairwise-CP | pairwise reward 在 4B 上是否贡献收益 |
| RL 消融 | RELER-NoPairwise-RLOO | 在相同 standalone graded reward 和 shortlist 下，CP 是否优于 RLOO |
| RL 消融 | RELER-NoRescale | frozen-document rescaling 是否仍是 CP 配方的重要组成 |

RELER 固定使用 graded nDCG@10、CP、G=64、alignment=0.70、跨卡全部候选、Uniform K7/T1、pairwise coefficient 0.5。`NoPairwise-RLOO` 不带 pairwise 项，因为当前 item-local pairwise 实现只支持 CP；因此 CP 的严格对照是 `NoPairwise-CP` 与 `NoPairwise-RLOO`，不是 RELER 与 RLOO。

## LoRA 与显存配置

- base model 固定为 `Qwen/Qwen3-Embedding-4B` revision `5cf2132abc99cad020ac570b19d031efec650f2b`。
- LoRA 使用 rank 16、alpha 32、dropout 0，覆盖 attention 和 MLP 的七类 projection。
- 学习率固定为 `1e-4`，不做 LR sweep；这是本轮唯一的 4B LoRA 训练配方。
- 每卡 microbatch 4、gradient accumulation 4，在 8 卡上保持 global batch 128。
- 使用 BF16、gradient checkpointing 和 ZeRO-3。训练结束后自动把 adapter 合并到固定 revision 的 base model，再按统一 embedding protocol 运行 BRIGHT。

## 推荐执行顺序

先预检并跑 seed 42 的六个配方：

```bash
python scripts/run_qwen3_embedding_4b_lora.py check --set pilot
python scripts/run_qwen3_embedding_4b_lora.py train --set pilot
```

pilot 的作用是确认显存、吞吐、loss/reward 数值和 LoRA 合并评测链路。若六个 run 均完成，再补剩余两个 seed；这样不会重复 seed 42：

```bash
python scripts/run_qwen3_embedding_4b_lora.py train --set main --seeds 3407 2026
python scripts/run_qwen3_embedding_4b_lora.py train --set ablations --seeds 3407 2026
```

若已在训练机确认配置，可一次跑完整 18 次：

```bash
python scripts/run_qwen3_embedding_4b_lora.py check --set all
python scripts/run_qwen3_embedding_4b_lora.py train --set all
```

E0 单独评测：

```bash
python scripts/run_qwen3_embedding_4b_lora.py eval --set e0
```

训练结束后如只需重试合并或 BRIGHT 评测，使用 `eval` 和相同的 set/seeds。已完整合并的 checkpoint 会被复用：

```bash
python scripts/run_qwen3_embedding_4b_lora.py eval --set main --seeds 42
```

## 报告口径

主表报告 E0、InfoNCE、LambdaLoss 和 RELER。RL 消融分别报告三组配对差：RELER − NoPairwise-CP、NoPairwise-CP − NoPairwise-RLOO、RELER − NoRescale。总体分数使用 12 个 BRIGHT subset 的宏平均，三 seed 报均值和样本标准差，同时保留逐 seed 差值。pilot 单 seed 只用于运行门槛和早期异常检查，不单独形成论文结论。
