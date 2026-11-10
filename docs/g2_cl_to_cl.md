# G2-R2：CL → CL 续训对照

从 seed **42、3407、2026** 各自的 `G2-R2-D-CL-Strong` 最终 checkpoint
继续原 Strong CL 配方，共新增 **3 次训练**。与已完成的 CL → RELER
使用相同起点、数据和第二阶段预算，最终仍评测 **MTEB English v2**。

| 设置 | 第二阶段 |
| --- | --- |
| 训练目标 | InfoNCE，temperature 0.03 |
| 负例及梯度 | 包含负文档的完整跨卡负例池，文档梯度不 detach |
| 预算 | 1200 steps；global batch 128；micro batch 16；8 张 BF16 CUDA GPU |
| 优化日程 | LR 5e-6；linear scheduler；warmup ratio 0.03；weight decay 0.01 |
| 初始化 | 各自 seed 的 CL-Strong 最终权重 |
| 训练状态 | 新 optimizer/scheduler，step 和数据 epoch 从第二阶段开始 |
| 评测 | 固定第二阶段最终 checkpoint；MTEB v2，41 任务、7 类型 |

入口逐字段检查续训配置与原 Strong CL 配置相同，仅模型路径和运行身份改变；
并检查与 CL → RELER 的共同配置及预算一致，允许训练目标、binary/graded
监督和候选处理的既定差异。两个续训阶段的总预算均为 1200 + 1200 steps。

## 运行

在仓库根目录执行，默认使用 [共享 G2 设置](../configs/experiments_g2_rl_r2.yaml)：

```bash
# 查看三条新增运行及对应的 CL 起点
python scripts/run_g2_cl_to_cl.py show
# 预检全部输入 checkpoint、表示协议、数据和新输出目录
python scripts/run_g2_cl_to_cl.py check
# 顺序训练三个 seed，每条结束后自动评测 MTEB v2
python -u scripts/run_g2_cl_to_cl.py train
# 更新现有 G2 汇总，以及 CL→RELER 与 CL→CL 的配对差值
python scripts/run_g2_cl_to_cl.py summary
```

训练机路径通过 `--config /path/to/settings.yaml` 配置。`G2.data` 指向原
CL 与 CL → RELER 使用的 E2Rank 数据目录，`output_dir` 指向原 CL 权重的
共享根目录。沿用已有实验的预算值，以保持比较一致。

默认权重根目录为 `checkpoints/iclr2027-g2-cl-r2/`，例如 seed42：

```text
输入：G2-R2-D-CL-Strong-s42/
输出：G2-R2-CLStrong-CL-s42/
```

其他 seed 的输入为 `G2-R2-D-CL-Strong-Seed<seed>-s<seed>/`，输出为
`G2-R2-CLStrong-CL-Seed<seed>-s<seed>/`。展开配置保存到共享根目录的
`.cl_strong_configs/<run-directory>.json`。训练前检查完整队列；缺少输入或
任一新输出目录非空时停止，防止部分队列已经启动才发现问题。

单独运行和补评测：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u scripts/run_g2_cl_to_cl.py train --seeds 42
python -u scripts/run_g2_cl_to_cl.py train --seeds 3407 2026
python -u scripts/run_g2_cl_to_cl.py eval --seeds 42
# 仅在已有对照确实需要补评测时使用；不会重训对照
python -u scripts/run_g2_cl_to_cl.py eval --methods cl cl_reler --seeds 42
```

`train` 只支持 `cl_cl`，默认跑三次新增训练。`eval` 可选择 `cl`（CL-Strong）、
`reler`（direct RELER）、`cl_reler` 或 `cl_cl`；默认仅评测 `cl_cl`。

## 统一汇总

`summary` 读取四种方法 × 三个 seed 的最终 `mteb_eval/final/`，忽略
`--methods` / `--seeds` 选择，更新
[`paper/_summary/g2_r2_mteb_v2/`](../paper/_summary/g2_r2_mteb_v2/README.md)
的四张总表和 `seeds/` 明细。保留历史 run，以及已归档但本机没有原始文件的
完整成绩；缺失成绩不填 0，三 seed 未齐全时不生成对应均值和样本 SD。

新增 `cl_cl` 方法，并在 `seeds/paired.csv` / `seeds/results.json` 中加入：

- `cl_cl_minus_infonce`：CL → CL 相对原 CL-Strong 的变化；
- `cl_reler_minus_cl_cl`：同样第二阶段预算下，CL → RELER 相对 CL → CL 的差值。

原有三组配对比较仍保留。先按同一个 seed 相减，再计算三 seed 均值和样本 SD；
使用完整精度分数，不从两位小数反推。重点查看 Retrieval，同时保留其他类型与
总体指标。

只同步评测 JSON 时，按 `<root>/<run-directory>/` 放置每条最终结果，然后执行：

```bash
python scripts/run_g2_cl_to_cl.py summary --results-root /path/to/final_results
```

允许保留 MTEB 的 model/revision 子目录，每条 run 只放一份最终结果。
该对照比较同预算的两种续训配方，监督和候选处理仍有差异。

配置：[suite](../configs/experiments/iclr2027/suite_g2_cl_to_cl.yaml)；
入口：[run_g2_cl_to_cl.py](../scripts/run_g2_cl_to_cl.py)。
