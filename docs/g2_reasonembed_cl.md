# ReasonEmbed：D/E/W × CL、CL-Strong、RL（每阶段 1 epoch）

2026-09-18 更新：使用 ReasonEmbed 公开 pos/neg 数据，进行 **三个初始化 × 三种方法，共九条 seed 42 训练，每条 1 个 epoch**。这是本项目协议下的数据实验，不是 ReasonEmbed 作者模型的训练复现。每条训练结束后评测最终模型的 BRIGHT，不按中间成绩选模。此前 1200-step 预算被本轮 1-epoch 设置替代；E2Rank 的已完成实验保持不变。

## 矩阵与共同初始化

| 分支 | 初始化 | CL | CL-Strong | RL |
| --- | --- | --- | --- | --- |
| D | 原始 `Qwen/Qwen3-0.6B`（B0） | `G2-ReasonEmbed-D-CL` | `G2-ReasonEmbed-D-CL-Strong` | `G2-R2-ReasonEmbed-D-RL` |
| E | 原始 `Qwen/Qwen3-Embedding-0.6B`（E0） | `G2-ReasonEmbed-E-CL` | `G2-ReasonEmbed-E-CL-Strong` | `G2-R2-ReasonEmbed-E-RL` |
| W | 本轮 **普通 D-CL** 的 1-epoch 最终权重（W0） | `G2-ReasonEmbed-W-CL` | `G2-ReasonEmbed-W-CL-Strong` | `G2-R2-ReasonEmbed-W-RL` |

W 的三种方法共享同一个 W0，均只加载权重，重置 optimizer、scheduler、step 和阶段内数据 epoch，再各训练 1 epoch。W-CL-Strong 不从 D-CL-Strong 出发，W-RL 不从 W-CL 出发，也不复用 E2Rank 或旧 1200-step W0。D/E 不依赖其他训练；E 不加载 G1 微调模型。E0 revision 固定为 `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`；B0 需记录实际模型/config/tokenizer revision，三种 D 方法保持相同版本。

九条训练共九个独立的单 epoch 阶段；D-CL 前缀只需训练一次。每条 W 路线端到端包含 1 epoch CL warm-up + 1 epoch 后续训练，成本需分别报告。实际 steps 和 query exposures 根据 loader 保留样本、分组/丢尾与全局 batch 计算，不沿用 1200 steps 或预估数据规模。

## 方法与训练协议

| 设置 | CL | CL-Strong | RL |
| --- | --- | --- | --- |
| 标签与目标 | binary InfoNCE，temperature 0.03 | 相同 | binary nDCG@10，CP |
| in-batch 文档 | device-local 代表正例 | 全部候选，跨卡负样本池 | 与普通 CL 相同的候选构造 |
| 跨 query 文档梯度 | detach | 完整梯度 | 冻结跨 query 文档 |
| 去重与已知正例过滤 | 开启 | 开启 | 开启 |
| 预算 | 1 epoch | 1 epoch | 1 epoch |

统一 joint full FT、seed 42、8 GPU、global batch 128、microbatch 16、LR 5e-6、AdamW、linear scheduler、warmup ratio 0.03、query/document 长度 512/1024、tokenization v2 和 FP32 pooling/scoring。采用原 G2 BF16、gradient checkpointing 和裁剪设置；不将 G1 的关闭裁剪搬入本组。

`num_train_epochs: 1`、`max_steps: -1`，两份日常配置的 `G2.steps` 也为 `-1`；suite 标记 `training_budget: one_epoch`，因此不能再用正数 steps 覆盖这轮预算。保存间隔仍为每 200 步，完成时保存最终模型；关闭训练期 MTEB 回调，仅评测最终 BRIGHT。无内部 dev，不按 benchmark 分数早停或选择 checkpoint。

RL 保留 CP / G64 / alignment 0.80、exact vMF 双侧 product、LOO、无 advantage normalization、document log probability sum、frozen rescaling、KL/辅助 InfoNCE 系数 0。ReasonEmbed 只有正负标签，使用 binary nDCG，不将 negative 顺序伪造成 graded 标签。这里沿用既定 RL 配方，不追加 SF/LL 或超参数搜索。

CL-Strong 同时改变负例池和梯度路径，不是单因素消融。三方法共享 binary 标签，但候选访问和目标仍有差别；结果用于比较实际训练配方，不单独归因于 CP。该 BRIGHT 评测也不等同于 E2Rank 的 MTEB v2，不能将两组绝对分数直接比较或把差异只归因于训练数据。

## 数据与训练机器上的执行

激活项目环境后，仅在尚未准备数据时下载和转换：

```bash
hf download hanhainebula/reason-embed-data --repo-type dataset \
  --revision 8dada2e649a21913df303510980ebb8ed1de541e \
  --include 'reason-embed-data-0928/*.jsonl' --local-dir data/raw/reasonembed
python scripts/prepare_reasonembed.py \
  --input-dir data/raw/reasonembed --output-dir data/processed/reasonembed_g2
```

新输出根目录为 **`checkpoints/iclr2027-g2-reasonembed-1epoch`**。CL、Strong CL、RL 共用此根目录以解析同一个 W0，run ID 各自独立；旧 `iclr2027-g2-reasonembed-cl` 的 1200-step 产物不混入本轮。修改路径时同步两份日常配置：`configs/experiments_g2_reasonembed_cl.yaml` 与 `configs/experiments_g2_reasonembed_rl_r2.yaml`。

先预检无需 W0 的部分：

```bash
bash scripts/run_g2_reasonembed_cl.sh check 8
python scripts/run_g2_reasonembed_cl_strong.py check --branches D E
python scripts/experiment.py check G2-R2-ReasonEmbed-E-RL --gpus 8 \
  --suite configs/experiments/iclr2027/suite_g2_reasonembed_rl_r2.yaml \
  --config configs/experiments_g2_reasonembed_rl_r2.yaml
python scripts/experiment.py check G2-R2-ReasonEmbed-D-RL --gpus 8 \
  --suite configs/experiments/iclr2027/suite_g2_reasonembed_rl_r2.yaml \
  --config configs/experiments_g2_reasonembed_rl_r2.yaml
```

全新目录上依次完成九条训练，每条自动评测 BRIGHT，任一步失败即停止：

```bash
bash scripts/run_g2_reasonembed_cl.sh train 8 &&
python -u scripts/run_g2_reasonembed_cl_strong.py train &&
bash scripts/run_g2_reasonembed_rl_r2.sh train 8
```

三个队列分别执行普通 CL 的 D → E → W、Strong CL 的 D → E → W、RL 的 E → W → D。先完成普通 CL 队列即可满足后两个队列的 W0 依赖；Strong CL 和 RL 彼此没有权重依赖。RL suite 的普通 CL 行仅是 reference，不自动调度 CL。

已完成部分时只运行缺失行，不再次启动完整队列。Strong CL 支持 `--branches D E W` 任意子集与 `eval`；普通 CL/RL 用 `experiment.py` 指定 run ID 和对应 suite/config。Strong CL 复用已有直接启动器，不写 `.launches` receipt，展开配置位于 `.cl_strong_configs/`；普通 CL/RL 继续保存 `.launches`。

```bash
# 只补评已有 Strong CL 模型，不重训。
python scripts/run_g2_reasonembed_cl_strong.py eval --branches E
```

普通 CL/RL 若训练成功而 BRIGHT 失败，只补跑 `.launches` receipt 中的 `post_train_commands`。三个队列均不自动覆盖、跳过或恢复已有训练输出；更换输出根目录时，W0 必须来自匹配的本轮 D-CL。

## 结果记录

主指标为 BRIGHT 全部 12 个领域 nDCG@10 的等权宏平均，同时保存逐领域结果、最终 checkpoint、实际 steps 与成本。每个分支并列初始模型、CL、CL-Strong、RL；W 初始成绩使用已核验的 D-CL，B0/E0 缺少同协议成绩时标缺，不以公开分数代替受控比较。单 seed 不计算训练 seed SD，不根据结果筛掉某个分支。

BRIGHT 已参与配方开发，且 ReasonEmbed 数据含同名领域；未完成重叠审计前，不称未见测试泛化。当前本地缺少转换后的 `train.jsonl` 和本轮 W0；本次准备配置、入口与本地验证，尚未启动 GPU 训练。

## 数据处理约定

转换脚本仅使用 Python 标准库，输出 `train.jsonl`、`manifest.json` 和 `decisions.jsonl`，已有输出目录会拒绝覆盖。固定 seed 42，每条保留一个 positive 和最多 15 个 negatives，形成最多 16 个候选；去掉空文档、重复候选及与 positive 重合的 negatives。全部已知 positives 的 key 都保留，用于过滤 in-batch false negatives。处理不划分 dev，训练 loader 按现有 G2 规则在每个 source 内逐 epoch shuffle 并丢弃不足 microbatch 的尾部。

公开数据是 `pos`/`neg`，没有原始 graded 分数；输出使用 binary relevance，也不给 negatives 伪造排序。训练输入使用原始 `query`，不用 `reasoning_query`；原始 prompt 仅保留为 metadata，实际使用现有 G2 loader 的任务 instruction（这些 domain 走 generic fallback）。默认不要启用 `--all-positives`，它会扩大候选数。

manifest 记录输入/输出 hash、候选选择和清洗统计，revision 是本地文件来源的声明，无法仅靠 JSONL 验证。现有 G2 preflight 检查数据是否存在，并不自动校验该 manifest 的 hash；跨机器传输时应连同 manifest 保留。这里只在本地验证转换与配置，不启动 GPU 训练或完整 BRIGHT。
