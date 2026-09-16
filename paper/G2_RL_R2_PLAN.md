# G2-RL-R2 实验计划（2026-09-17）

本轮新增 **E2Rank、ReasonEmbed 各 D/E/W 三条 RL，共六次训练，只有 seed 42**。目标是检查已选定的 CP 配方相对同初始化 CL 的收益，以及收益是否随初始化和训练数据变化。仅交付计划、配置和两个运行脚本，不启动 GPU 训练。

## 1. 最新提交的依据与配方选择

依据最新提交 `f716580` 的 [G1-R2 结果](G1_R2_RESULTS.md)：

| G1-R2 配方 | BRIGHT 均值 ± SD | 对本轮的启示 |
| --- | ---: | --- |
| graded nDCG / CP / G64 / alignment 0.80 | 20.76 ± 0.20 | 固定迁移起点，不再沿用历史 MRR/0.90 |
| 同 reward、G、alignment 的 SF | 18.81 ± 0.30 | CP 对 SF +1.94；本轮只比较 CL，不重复估计器矩阵 |
| LL-Graded / CL | 19.37 ± 0.28 / 17.72 ± 0.62 | CP 对 LL +1.39、对 CL +3.04；不当作 G2 的预期增益 |
| binary nDCG / CP 对 SF，alignment 0.90 | +0.48，2/3 seed 正收益 | binary 迁移证据较弱，不保证 ReasonEmbed 复现 graded 收益 |
| MRR / CP 对 SF，alignment 0.90 | −0.02 | 不继续把 MRR 作为统一主 reward |
| CP 关闭 rescaling，alignment 0.90 | −1.01，3/3 seed 下降 | 保留 frozen-candidate rescaling |

G32 的已有质量结果仍有优势，但没有完整训练成本证据；本轮保留主结果 G64，不新增 G 扫描、退火、SF、LL 或 seed 重复。0.80 是根据已有 BRIGHT 消融选择的固定点，本轮不根据中间成绩再改探索强度。

**E2Rank 用 graded nDCG@10；ReasonEmbed 用 binary nDCG@10。** E2Rank 的 teacher ranking 可生成 3/2/1/0 grades；ReasonEmbed 公开文件只有 pos/neg，现有转换文件也只保存 binary ties，不能把 negative 的布局顺序当 teacher 排序。这是数据标签条件下的迁移实验，不是两套数据完全相同 reward 信息量的比较。

## 2. 六条 RL 与 CL 的逐行配对

| 数据 | RL run ID | 初始化 | CL 对照 |
| --- | --- | --- | --- |
| E2Rank | `G2-R2-E-RL` | 原始 E0 | `G2-R2-E-CL` |
| E2Rank | `G2-R2-W-RL` | 本轮 `G2-R2-D-CL` 最终权重 W0 | `G2-R2-W-CL` |
| E2Rank | `G2-R2-D-RL` | 原始 B0 | `G2-R2-D-CL` |
| ReasonEmbed | `G2-R2-ReasonEmbed-E-RL` | 原始 E0 | `G2-ReasonEmbed-E-CL` |
| ReasonEmbed | `G2-R2-ReasonEmbed-W-RL` | `G2-ReasonEmbed-D-CL` 最终权重 W0 | `G2-ReasonEmbed-W-CL` |
| ReasonEmbed | `G2-R2-ReasonEmbed-D-RL` | 原始 B0 | `G2-ReasonEmbed-D-CL` |

B0 为 `Qwen/Qwen3-0.6B`；E0 为 `Qwen/Qwen3-Embedding-0.6B`，revision 固定为 `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`。E-RL 不加载任何 G1 微调模型。B0 的 immutable revision 尚未在现有 CL 固定；训练机需记录 D-CL 实际模型/config/tokenizer revision，并让 D-RL 使用同一版本（必要时在 D-RL overrides 写入该 SHA）。

W-RL 与 W-CL 从同一个 D-CL **最终模型**独立出发，仅加载模型权重；新建 optimizer/scheduler、step counter 和阶段内数据 epoch，均再训练 1200 步。不从 W-CL 最终模型继续 RL，不跨数据集使用 W0，不用历史协议 D-CL。数据 seed 与训练 seed 均为 42，rollout seed 也固定 42。

每个脚本先预检全部三条 RL（包括 W0），再按 **E → W → D** 训练和最终评测；顺序用于先观察成熟表示迁移，并不是 E/W/D 的额外依赖。两套队列可在独立资源上分别运行；本轮脚本不调度 CL。D-CL 尚未完成时可先单独运行 E-RL/D-RL。

本地尚未同步这六条 CL 的最终分数和实际训练 receipt；此处提供的是可追溯的配对设计，不宣称 CL 已完成或已核验历史运行配置。

## 3. 配置与 CL 对照

| 项目 | CL | 本轮 RL |
| --- | --- | --- |
| 更新范围 | joint full FT | 相同 |
| 目标 | InfoNCE，temperature 0.03 | nDCG@10，conditional projection |
| E2Rank 标签 | `pos_index` binary 正例 | teacher ranking 3/2/1/0；已知正例仍用于 in-batch 代表与过滤 |
| ReasonEmbed 标签 | annotated pos/neg binary | 相同 binary 标签；不伪造 graded 排序 |
| 候选 | slate 16、device-local in-batch 代表正例、去重和已知正例过滤 | 相同候选构造；跨 query 文档 detach，不加入 in-batch negatives 的 negative slots |
| 数据 | E2Rank `data/train.jsonl`；ReasonEmbed `data/processed/reasonembed_g2/train.jsonl` | 复用各自 CL 同一份数据，不重新转换或抽样 |
| 分组 | source 内按 epoch 重分组，固定保留集合和丢尾规则 | 相同；microbatch 16，不做 length buckets |
| 长度与表示 | query/document 512/1024；last pooling、left padding、append pad、原 prompt | 相同；tokenization v2、FP32 pooling/scoring，评测上限 8192 |
| Batch/seed | 8 GPU × 16，accumulation 1，global batch 128；seed 42 | 相同，另有固定 rollout seed 42 |
| 优化器 | AdamW，LR 5e-6、weight decay 0.01、linear、warmup ratio 0.03 | 相同；不引入每初始化 LR 搜索 |
| Backbone/engine | BF16、gradient checkpointing、`scripts/zero3.json`，trainer 默认 max_grad_norm 1.0 | 相同；实际 DeepSpeed clipping 以运行日志为准 |
| 预算与保存 | 每阶段 1200 steps，每 200 steps 保存，固定最终模型 | 相同，不按中间评测选模 |

本轮沿用 CL 的裁剪设置，**不把 G1-R2 的关闭裁剪、113 steps 等额外变化搬到 G2**。因此是选定 reward/CP/探索配方在 G2 预算和优化器条件下的迁移，不是 G1 训练合同逐项复制。E2Rank 的 graded RL 与 binary CL 所用监督信息不同；RL−CL 衡量整套训练目标替换收益，不能拆成 CP 单因素因果效应。ReasonEmbed 的 binary 标签则与其 CL 匹配。

RL 共同设置由两份 suite 的显式 overrides 覆盖 resolver 的 GRPO 默认值：exact vMF、query + bundled documents 双侧 action、product rollout、G64（每 query 64×64 reward cells）、固定 alignment 0.80、不可学习 sigma、leave-one-out baseline、无 advantage normalization、document shared baseline、document log probability sum、frozen rescaling 开启、in-batch 使用冻结文档、reward combine sum、KL/辅助 InfoNCE 系数为 0。κ 按实际 embedding 维度反解，不把旧 κ=755 当实际探索强度。

新增 RL 预算为 **6 × 1200 = 7200 optimizer steps / 921,600 query exposures**，每个数据集各 3600 steps / 460,800 exposures。CL 如需补齐另计；每条 W 路线另有 1200-step D-CL 前缀，分别报告阶段与端到端成本。相同步数不代表 CL/RL 算力相同。

## 4. 文件与运行命令

| 数据 | 日常配置 | Suite | 预设 | 批量入口 |
| --- | --- | --- | --- | --- |
| E2Rank | [experiments_g2_rl_r2.yaml](../configs/experiments_g2_rl_r2.yaml) | [suite_g2_rl_r2.yaml](../configs/experiments/iclr2027/suite_g2_rl_r2.yaml) | [g2_rl_r2.yaml](../configs/experiments/iclr2027/g2_rl_r2.yaml) | [run_g2_rl_r2.sh](../scripts/run_g2_rl_r2.sh) |
| ReasonEmbed | [experiments_g2_reasonembed_rl_r2.yaml](../configs/experiments_g2_reasonembed_rl_r2.yaml) | [suite_g2_reasonembed_rl_r2.yaml](../configs/experiments/iclr2027/suite_g2_reasonembed_rl_r2.yaml) | [g2_reasonembed_rl_r2.yaml](../configs/experiments/iclr2027/g2_reasonembed_rl_r2.yaml) | [run_g2_reasonembed_rl_r2.sh](../scripts/run_g2_reasonembed_rl_r2.sh) |

沿用对应 CL 的输出根目录，使现有 `init_from` 自动解析本数据集的新 W0；每条 RL 通过全新 run ID 获得独立模型、评测和 `.launches` 目录，不覆盖 CL。两套 RL 索引缓存单独隔离。Suite 中三条 CL 是 `kind: reuse` 的引用，不能从 RL suite 启动训练，也不会自动核验这些 CL 的成绩。修改 RL `output_dir` 时，必须将对应 D-CL 最终模型放在新根目录下原 ID 的目录中；默认无需复制。

在已有训练环境运行：

```bash
# E2Rank: W0 must be G2-R2-D-CL-s42.
bash scripts/run_g2_rl_r2.sh check 8
bash scripts/run_g2_rl_r2.sh train 8

# ReasonEmbed: W0 must be G2-ReasonEmbed-D-CL-s42.
bash scripts/run_g2_reasonembed_rl_r2.sh check 8
bash scripts/run_g2_reasonembed_rl_r2.sh train 8
```

如果 CL/W0 尚未准备，使用原来的两个 CL 入口补齐，不更改其配方：

```bash
bash scripts/run_g2_cl_r2.sh train 8
bash scripts/run_g2_reasonembed_cl.sh train 8
```

ReasonEmbed 数据下载、冻结转换和版本见 [ReasonEmbed CL 说明](../docs/g2_reasonembed_cl.md)。已有数据直接复用。部分 CL 已完成时用对应 CL suite 单独运行缺失行，不能重复运行完整队列。

单独运行独立 RL 或补齐批量剩余行：

```bash
python scripts/experiment.py train G2-R2-E-RL --gpus 8 \
  --suite configs/experiments/iclr2027/suite_g2_rl_r2.yaml \
  --config configs/experiments_g2_rl_r2.yaml
python scripts/experiment.py train G2-R2-ReasonEmbed-E-RL --gpus 8 \
  --suite configs/experiments/iclr2027/suite_g2_reasonembed_rl_r2.yaml \
  --config configs/experiments_g2_reasonembed_rl_r2.yaml
```

脚本失败即停，不自动跳过、恢复或覆盖。训练成功而最终评测失败时，单独执行该 `.launches` receipt 的 `post_train_commands` 补评，不重训模型。

## 5. 评测、诊断与判读

**严格沿用各自 CL 的评测协议。** E2Rank 在 checkpoint-0/每次保存后使用 `MTEB(eng, v1, subset)` callback，最终模型运行完整 `MTEB(eng, v2)`，结果在 `mteb_eval/final/`；主指标为最终 v2 全部 Retrieval 任务 main score 的等权宏平均，附逐任务成绩与全 MTEB 辅助均值，不把各任务 main score 统一叫 nDCG@10。ReasonEmbed 无中间 MTEB callback，仅最终模型评测 BRIGHT，结果在 `mteb_eval/bright/`；主指标为全部 12 个领域 nDCG@10 的等权宏平均，并列逐领域差值。

CL/RL 固定评测包、任务清单、数据 revision 和表示协议；缺项标记不完整并补评，不缩小分母，也不混算 v1/v2。ReasonEmbed 原始数据含 BRIGHT 同名领域，且 BRIGHT 已参与 G1 配方开发；未完成重叠审计前，不把其 BRIGHT 结果宣称为未见测试泛化。E2Rank 的 OOD 分组也不能直接复用旧 summary 的硬编码任务排除名单。

每个数据集独立填表，统一使用 0～100 分数尺度：

| 初始化 | 同协议初始分数 | 最终 CL | 最终 RL | RL−CL | RL−初始 | CL/RL GPU hours |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D / B0 | 待填 | 待填 | 待测 | 待测 | 待测 | 待填 |
| E / E0 | 待填 | 待填 | 待测 | 待测 | 待测 | 待填 |
| W / W0 | 同数据集 D-CL 最终成绩 | 待填 | 待测 | 待测 | 待测 | 待填 |

不因初始分数尚缺而新增训练；必要时补评原始模型。E2Rank v1 callback 的初始成绩不能填入 v2 主表。RL 优于 CL 但低于初始化仍须如实报告，不把相对训练目标优势等同于继续训练有益。

六条正式训练保留原 scheduler/完整预算，记录 reward mean、query/document `group_std`/`degenerate_frac`、实际 alignment/κ、grad norm、LR、实际 step、wall time 和显存。成本分别报告训练、callback、最终评测；没有采集的字段留空。NaN/Inf 或数据错误按故障停止并保存日志，不按 benchmark 成绩早停，也不删除低收益的 D/ReasonEmbed 分支。

主要结论按同数据集的三个 `RL−CL` 判断，再观察第二套数据的收益方向；两套 benchmark 的绝对分数不直接比较。仅 seed 42，不计算训练 seed SD 或提出跨 seed 显著性主张。完成标准是六条 RL 的固定预算最终结果（或有日志的失败状态）、完整 CL 配对、任务覆盖与成本记录可追溯，不以必须超过 CL 为条件。

## 6. 本地验证范围

配置验证应解析两份 RL suite 的全部行，逐对对照实际展开的 CL：检查模型/revision、数据、表示、optimizer/engine、batch/seed、保存间隔和评测；确认只存在六条 RL train 行，以及两条 W 的依赖均指向对应 D-CL 最终目录。两个 shell 入口检查语法和非法参数；预检只读，不下载模型或启动训练。

本地已通过上述六条配置对照、最终评测命令生成、shell 语法和非法参数检查。本地没有配置要求的两份 `train.jsonl` 和新 D-CL 最终模型，因此两份 suite 的 `check G2` 均按预期返回 2，仅报告缺失输入/W0。引用行显示 `READY reference` 只代表引用定义可解析，并不表示 CL 模型或成绩已经存在。配置展开通过不代表训练机已经就绪。G2 preflight 不自动校验训练 manifest 的 hash 或历史 CL 训练合同，训练机应以实际数据/launch receipt 为配对依据。
