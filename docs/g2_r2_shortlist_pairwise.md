# G2-R2：迁移 G1 K7 / Pairwise050 配方

2026-09-20。新增 E2Rank 的 E/W/D 三条 RL，seed 42；用户自行在训练机运行。

## 设计与比较

采用 `G1-R2-RL-GradedNDCG64-CP-Align070-ShortlistUniform-K7-T1-Pairwise050`
的完整 RL 参数：graded nDCG@10 + 0.5 × binary pairwise reward，exact vMF、
query/document product rollout、CP、G64、alignment 0.70、Uniform K7/T1。
候选池为跨卡全部候选，自有候选全部保留，额外均匀抽取 7 个跨 query 候选；
沿用身份去重、已知正例过滤和 frozen-document rescaling，跨 query 文档 detach。
Pairwise 使用原始 `positive_mask`，逐 pair LOO/CP；binary nDCG 混合权重为 0。
完整定义与 G1 结果见 [G1 shortlist 实验](g1_shortlist_improvements.md)。

此次检验整套配方能否改善旧 G2 的检索结果，不分离 alignment、候选池和辅助奖励的单因素贡献。
沿用 G2 的数据、表示、裁剪与 DeepSpeed、LR 5e-6、8 卡 × microbatch 16、global batch 128、
1200 optimizer steps、每 200 steps 保存、固定最终模型。三条合计 3600 steps、460800 query exposures；
不把 G1 的 113 steps 和无裁剪搬过来。实际耗时以训练记录为准。

| 分支 | 初始化 | 配对对照 |
|---|---|---|
| E | Qwen3-Embedding-0.6B，revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | E 的旧 RL / CL / CL-Strong |
| W | `G2-R2-D-CL-s42` 最终权重，重新建立 optimizer/scheduler | W 的旧 RL / CL / CL-Strong |
| D | Qwen3-0.6B | D 的旧 RL / CL / CL-Strong |

D 继承既有配置的模型版本设置；训练机应核对与旧 D-CL 的实际 revision 一致。
W 不加载 W-CL 或旧 RL，也不自动重训 D-CL。
新 run ID 为 `G2-R2-{E,W,D}-RL-Align070-ShortlistUniform-K7-T1-Pairwise050`，
均使用独立输出目录；旧 RL 配置与结果保留。ReasonEmbed 不包含在本轮。

主指标沿用最终完整 `MTEB(eng, v2)` 的 Retrieval main-score 等权宏平均，
同时保留逐任务和全任务均值。训练中的 v1 subset callback 仅作诊断，不用于选模。
比较新 RL−旧 RL、新 RL−CL、新 RL−CL-Strong；固定 seed 42 不报告跨 seed 显著性。
旧结果见 [G2-R2 结果](../paper/G2_R2_RESULTS.md)。

## 运行

在项目根目录、已激活训练环境中执行；需恰好 8 张可见且支持 BF16 的 CUDA GPU。
准备 `data/train.jsonl`，全队列还需要
`checkpoints/iclr2027-g2-cl-r2/G2-R2-D-CL-s42/` 的完整最终模型。

```bash
# 预检全部输入；默认 action 也是 check
python scripts/run_g2_rl_r2_shortlist_pairwise.py check

# 按 E → W → D 运行，每条训练后自动完整评测
python -u scripts/run_g2_rl_r2_shortlist_pairwise.py train

# 只跑指定分支；没有 W0 时可先跑 E、D
python -u scripts/run_g2_rl_r2_shortlist_pairwise.py train --branches E D
python -u scripts/run_g2_rl_r2_shortlist_pairwise.py train --branches W

# 训练成功而评测失败，只补评指定分支
python -u scripts/run_g2_rl_r2_shortlist_pairwise.py eval --branches E
```

机器路径与预算设置复用 `configs/experiments_g2_rl_r2.yaml`；可复制该文件并通过
`--config /path/to/settings.yaml` 指定。改变 output_dir 后，W0 也必须能在该根目录的
`G2-R2-D-CL-s42` 下找到。保持配对预算时不要改变 steps/batch/learning_rate。
脚本预检全部选中分支，失败即停；非空训练输出拒绝覆盖，重试用 `--branches` 选择剩余分支。
复用项目现有直接运行入口，展开后的配置写在输出根目录的 `.cl_strong_configs/` 下。
最终评测位于各新 run 的 `mteb_eval/final/`。

配置：[suite](../configs/experiments/iclr2027/suite_g2_rl_r2_shortlist_pairwise.yaml)、
[训练预设](../configs/experiments/iclr2027/g2_rl_r2_shortlist_pairwise.yaml)。

## 本地验证与状态

三条配置解析和 `RLArguments` 构造通过；所有 RL 参数与指定 G1 行逐字段相同。
与旧 G2 行相比，仅 RL 探索/候选/奖励配方、缓存和输出标识改变；W0 依赖与最终 MTEB 命令正确。
G2 launch 与通用入口现有测试通过：9 passed、9 subtests passed。
本地缺少 `data/train.jsonl` 与 W0，输入预检会报缺失；没有启动训练或生成新成绩。
