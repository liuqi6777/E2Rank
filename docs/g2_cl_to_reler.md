# G2-R2：从 Strong CL 继续训练 RELER

新增 **3 次训练**：分别从 seed **42、3407、2026** 的
`G2-R2-D-CL-Strong` 最终 checkpoint 加载权重，在同一份 E2Rank
`train.jsonl` 上继续 RELER。复用原 CL，不新增 CL → CL。

续训使用现有 direct RELER 的 K0 配方：graded nDCG@10 + pairwise 0.5、
RLOO + CMP、vMF product、G64、alignment 0.70。LR `5e-6`、global batch
128、micro batch 16、8 卡、1200 steps，训练/数据/rollout seed 均与各自
CL 起点对应。只加载模型权重，optimizer、scheduler、step 和数据 epoch
从本阶段重新开始；最终 checkpoint 固定为本阶段第 1200 步。

最终评测沿用 **MTEB English v2（41 个任务、7 个类型）**。保留原保存点
callback 设置；汇总只读最终模型的 `mteb_eval/final/`。

## 运行

在仓库根目录、已安装项目依赖的环境执行：

```bash
# 查看三个 seed 的初始化、数据和输出路径，不需要 GPU 或 checkpoint
python scripts/run_g2_cl_to_reler.py show
# 检查所有输入 checkpoint、表示协议、训练数据和输出目录
python scripts/run_g2_cl_to_reler.py check
# 顺序续训三个 seed，每条训练完成后自动评测最终 MTEB v2
python -u scripts/run_g2_cl_to_reler.py train
# 合并到现有 G2 汇总目录，并计算三 seed 配对差值
python scripts/run_g2_cl_to_reler.py summary
```

默认沿用 [G2 设置](../configs/experiments_g2_rl_r2.yaml)。通过
`--config /path/to/settings.yaml` 修改训练机路径；`G2.data` 指向与 CL 阶段
相同的 E2Rank 数据目录，`output_dir` 指向原 CL checkpoint 所在根目录。
训练入口不会自动重训缺失的 CL，也不会覆盖已有非空训练目录。

分机器执行或只补失败的评测：

```bash
# 每台训练机仍需暴露 8 张 BF16 CUDA GPU
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u scripts/run_g2_cl_to_reler.py train --seeds 42
python -u scripts/run_g2_cl_to_reler.py train --seeds 3407 2026
python -u scripts/run_g2_cl_to_reler.py eval --seeds 42
# CL 已有完整 MTEB v2 原始结果可直接复用；缺失时只补评测
python -u scripts/run_g2_cl_to_reler.py eval --methods cl --seeds 42 3407 2026
```

## 运行身份和汇总

所有权重继续放在 `checkpoints/iclr2027-g2-cl-r2/`。新 run 基名为
`G2-R2-CLStrong-RL-Align070-ShortlistUniform-K0-T1-Pairwise050`；
3407/2026 分别追加 `-Seed3407` / `-Seed2026`，所有目录再追加 `-s<seed>`。
例如 seed42 的输入与输出：

```text
G2-R2-D-CL-Strong-s42/
G2-R2-CLStrong-RL-Align070-ShortlistUniform-K0-T1-Pairwise050-s42/
```

3407/2026 必须加载各自的 `D-CL-Strong-Seed<seed>-s<seed>/`，
不会共用 seed42。展开后的运行配置保存到同一根目录的
`.cl_strong_configs/<run-directory>.json`。

`summary` 默认把完整结果按 run 合并进
[`paper/_summary/g2_r2_mteb_v2/`](../paper/_summary/g2_r2_mteb_v2/)
现有的 `run_summary.csv`、`type_summary.csv`、`task_summary.csv` 和
`retrieval_summary.csv`，保留其他 run。只替换原始 41 任务结果完整的行；
缺失或尚未完成的运行不会覆盖已有成绩，损坏结果会报错。

同目录额外生成：

- `cl_to_reler_grouped.csv`：CL、CL → RELER 的各指标三 seed 均值和样本 SD；
- `cl_to_reler_paired.csv`：各 seed 的 `(CL → RELER) − CL`，及差值的均值和样本 SD；
- `cl_to_reler_results.json`：未四舍五入的逐任务成绩、来源、缺失任务和聚合结果。

三 seed 汇总始终检查两方法 × 三 seed，只有三条完整时才填写对应均值和
SD；不会把缺失成绩填成 0。差值先逐 seed 相减，再求均值和 SD。分数使用
`main_score × 100`；沿用原汇总中任务内 subsets、任务与类型的平均方式。
配对统计从原始 MTEB JSON 计算，不使用已有 CSV 的两位小数反推。

只同步评测 JSON 到本机时，可以把每条最终结果放到
`/path/to/final_results/<run-directory>/`，然后执行：

```bash
python scripts/run_g2_cl_to_reler.py summary --results-root /path/to/final_results
```

每条目录只放最终结果，允许保留 MTEB 的 model/revision 子目录。
`--summary-dir` 可改汇总目标，但默认就是现有 G2 目录。

这组比较回答已有 CL 表示在继续 RELER 后如何变化；未做 CL → CL，
不能单独归因于训练目标而排除额外训练预算。1200 × 128 = 153,600
是第二阶段的 query exposures，不能当作不同 query 的数量。

配置：[suite](../configs/experiments/iclr2027/suite_g2_cl_to_reler.yaml)；
入口：[run_g2_cl_to_reler.py](../scripts/run_g2_cl_to_reler.py)。
