# G1：Reasoning E0 核心实验

这两组检验现有 post-training 方法能否改善已训练的 reasoning retriever。
模型作为初始化 E0；每组只比较自身 E0、CL、graded LambdaLoss、graded RL，不做公开模型榜单横向比较。
数据保持现有 ReasonRank ready/manifest，评测保持最终 BRIGHT original-query 12 域，无新增 benchmark。

## 模型与协议

| E0 key | 权重 | 固定 revision | hidden size |
|---|---|---|---:|
| `diver` | `AQ-MedAI/Diver-Retriever-0.6B` | `9ce2a1e8acae4342c453e1a18b71d468c4c81e39` | 1024 |
| `reasonembed` | `hanhainebula/reason-embed-qwen3-4b-0928` | `048461e2e4106012afb52ebae037ed9a172ef5c3` | 2560 |

DIVER 来源为 Qwen3-Embedding-0.6B；ReasonEmbed 来源为 Qwen3-4B→MS MARCO 微调→RI-InfoNCE。
核查了两者固定 revision 的模型卡、config、tokenizer config、tokenizer.json 和 Pooling config。
二者使用 last-token pooling、cosine；tokenizer.json 的 TemplateProcessing 均在正文后添加
`<|endoftext|>`（151643）。因此复用项目 `append_token: pad` 的 tokens-v2 实现：关闭正文自动
special tokens，预留截断位置后手动追加一个读出 token。ReasonEmbed 的 `add_eos_token: true`
字段不应被误读为要改成 `append_token: eos`；实际 fast tokenizer 后处理添加的是 endoftext。

DIVER query 格式为 `Instruct: {task_description}\nQuery:{query}`；ReasonEmbed 的 `Query:` 后有空格。
document 不加 instruction。保留项目逐领域 task instruction、训练长度 512/1024 与评测上限 8192。
每个模型自身 E0 和三种训练方法统一表示协议，不承诺复现作者公开分数。

官方入口：[DIVER](https://huggingface.co/AQ-MedAI/Diver-Retriever-0.6B)、
[ReasonEmbed 4B](https://huggingface.co/hanhainebula/reason-embed-qwen3-4b-0928)。

## 核心矩阵

每个 E0 使用 seeds 42/3407/2026，三方法共 9 次训练；两模型合计 18 次训练，加两次 E0 评测。
E0 只归 seed 42 队列，不重复算 seed SD。可通过 `--seeds 42` 先运行单 seed 核心组（6 次训练＋2 次 E0）。

| 方法 | 标签/目标 | 其他条件 |
|---|---|---|
| CL | 全部已知正例，InfoNCE，temperature 0.03 | joint full FT |
| LL-Graded | teacher 3/2/1/0，LambdaLoss@10，sigma=1/0.03 | joint full FT |
| RL-Graded | 同 graded labels，nDCG@10，conditional projection | G64、vMF product、alignment 0.90、LOO、无 advantage norm、shared document baseline、document log-prob sum |

所有训练从各自原始 E0 独立初始化；不串行 warm-up，不继承 Qwen G1 微调权重。
沿用 G1-R2：113 steps、LR 5e-6、8 卡×microbatch16、global batch128、AdamW、无裁剪、
无 KL/aux loss；保存 25/50/75/100/113，最终 checkpoint 评测 BRIGHT；不额外跑 gradient probe。
RL 使用 dimension-aware target-alignment 配置，不把 1024 维的固定 κ 搬到 2560 维。
相同 seed 的三种方法使用相同 source 内采样/丢尾规则。固定 LR 为受控起点，不解释为充分调优。

4B 仍按 8×80GB 节点配方准备，真实 peak memory 尚未验证。若需调整 microbatch，会同时改变
device-local in-batch 候选；必须整组同改并使用新输出目录，不能只降低 RL 的 microbatch。

## 执行

设置文件：`configs/experiments_reasoning_e0.yaml`；可改 G1.data 指向训练机器实际 ready/manifest 对。
模型与 revision 分别位于 `configs/model/diver_0.6b.yaml`、`configs/model/reason_embed_4b.yaml`。

在项目根目录、已激活 GPU 训练环境中运行：

```bash
python scripts/run_g1_reasoning_e0.py check
nohup python -u scripts/run_g1_reasoning_e0.py > g1_reasoning_e0.log 2>&1 &
```

只跑指定模型或 seed：

```bash
python scripts/run_g1_reasoning_e0.py check --models reasonembed --seeds 42
python -u scripts/run_g1_reasoning_e0.py --models diver --seeds 42
python -u scripts/run_g1_reasoning_e0.py --models reasonembed --seeds 3407
```

每个 run 仍需要 8 卡；不同 model/seed 可分机执行，共享目录下用逐 run 锁防止重复启动。
结果根目录 `checkpoints/iclr2027-reasoning-e0` 与现有 G1-R2 分离，数据索引缓存也按模型隔离。
合同记录展开配置、固定模型 revision、数据/源码哈希、DeepSpeed、命令、日志、硬件与时间。
已完整结束的任务跳过；训练成功但评测失败只补评测；失败/中断训练保留现场，重训更换输出根目录。
独立任务失败后队列继续；预检发现合同冲突则整体拒绝启动。

汇集各机结果后：

```bash
python scripts/run_g1_reasoning_e0.py summary
```

总表位于 `.reasoning_e0/summary.md` 和 `summary.json`。分机运行时实时汇总在
`.reasoning_e0/queues/<models>/seeds-<seeds>/`。只有完整的所选 seed 组且数据/训练源码一致时
计算均值、样本 SD、最差分数；另保留逐 seed、逐领域与训练耗时，按模型自身 E0 解释提升。

入口开发时已完成本地 CPU 配置预检。2026-09-17 已汇集 DIVER 的 9 次训练和 E0 评测结果，分析见 [G1-R2 与 DIVER 结果文档](../paper/G1_R2_RESULTS.md#6-diver-作为-e0相对方法优势保留绝对改善消失)；本轮本地汇总未包含 ReasonEmbed 结果或完整运行显存/耗时记录。

论文纳入决策（2026-09-17）：DIVER 本组仅保留为内部文档与数据记录，不纳入论文正文、附录、实验计数或主张。当前优先解释为数据与强初始化的匹配问题，尚未排除共同训练配方的影响；不将该组诊断列为当前论文必做任务。
