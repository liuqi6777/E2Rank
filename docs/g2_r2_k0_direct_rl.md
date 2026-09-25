# G2-R2：从 Qwen3-0.6B 直接使用当前 K=0 配方训练 RL

新增一条 E2Rank 训练：`G2-R2-D-RL-Align070-ShortlistUniform-K0-T1-Pairwise050`，seed 42。从 `Qwen/Qwen3-0.6B` 初始化，不加载 CL 或其他 RL checkpoint。该模型是通用后训练 LLM；本实验不使用纯预训练的 `Qwen/Qwen3-0.6B-Base`。

旧 `G2-R2-D-RL` 使用纯 graded nDCG、G64、CP、alignment 0.80 和跨 query 候选。新行迁移当前 [G1 论文主配方](../paper/ICLR2027_RESULTS_REORGANIZATION.md)：graded nDCG@10 + 原始正例 pairwise `λ=0.5`、K=0（仅自有候选）、RLOO+CMP、双侧 vMF product、G64、固定 alignment 0.70、frozen-document rescaling。新旧两行共同使用 E2Rank 数据、模型初始化、表示协议、LR `5e-6`、global batch 128、1200 optimizer steps 和最终 MTEB English v2 评测。由于一次改变了 reward、候选和探索强度，结果比较衡量整套配方，不能归因于单一组件。G1 的 113 步预算不转移到此实验。

输出仍位于 `checkpoints/iclr2027-g2-cl-r2/`，但使用独立 run ID，不覆盖旧 D-RL。配对参照为同数据的 `G2-R2-D-CL` / `G2-R2-D-CL-Strong` 和旧 D-RL；主指标为最终模型的 MTEB English v2 Retrieval main-score 等权均值，同时保留逐任务与全任务均值。单 seed 只作这一次配方迁移比较。

在项目根目录、训练环境中运行：

```bash
python scripts/run_g2_rl_r2_k0.py check
python -u scripts/run_g2_rl_r2_k0.py train
# 若训练成功、最终评测失败，仅重跑评测：
python -u scripts/run_g2_rl_r2_k0.py eval
```

`train` 要求 8 张可见的 BF16 CUDA GPU，训练后自动执行完整 MTEB English v2 评测。设置沿用 `configs/experiments_g2_rl_r2.yaml`；需要指定训练机路径时使用 `--config /path/to/settings.yaml`。输入为 `data/train.jsonl`，应与旧 G2 使用同一份带有效 `pos_index` 的数据。首次在训练机运行时核对 `Qwen/Qwen3-0.6B` 的实际 revision 与旧 D-RL 一致；现有 D 配置未固定 revision。已有非空输出目录会拒绝覆盖。

配置：[suite](../configs/experiments/iclr2027/suite_g2_rl_r2_k0.yaml)、[训练预设](../configs/experiments/iclr2027/g2_rl_r2_k0.yaml)、[入口](../scripts/run_g2_rl_r2_k0.py)。本地只验证配置解析与训练参数；没有启动训练或生成成绩。
