# 实验计划：协议修订后的重新验证

更新日期：2026-09-16。输入协议修订：`64d2e7d`；条件投影实现基线：`54e9fbf`。

**本轮执行 9 个方法 × 3 个 seed，共 27 次全新训练。RL 沿用历史较稳定的 G64，覆盖 MRR、binary nDCG、graded nDCG，每种奖励配对比较原估计器与条件投影。**

按用户的整批运行安排，不再在单 seed 结束后等待决策。通过 `--seeds` 将 42/3407/2026 分配给三台机器，每台独立完成九方法对照；不传参数则按上述 seed 顺序单机执行全部。
不另划 dev，保留现有丢尾规则与实际训练样本，每个 epoch 在 source 内重新组合 microbatch。固定训练 113 步，统一评测最终 checkpoint。
G2/G3 延后，不复刻整个历史搜索矩阵。本次只更新配置、脚本与 CPU 验证，没有启动真实 GPU 实验。

运行入口：[夜跑说明](../docs/g1_r2_overnight.md)、[脚本](../scripts/run_g1_r2.py)、[日常配置](../configs/experiments_r2.yaml)、[28 行注册矩阵](../configs/experiments/iclr2027/suite_r2.yaml)（27 训练 + 1 E0 评测）。
旧 93 行 suite 和[归档计划](EXPERIMENT_PLAN_2026-09-15.md)继续保留为历史记录，不作为新队列。

## 1. 重跑与复用范围

| 对象 | 本轮处理 | 原因与边界 |
|---|---|---|
| 原始 E0、通用 Qwen 权重与 tokenizer 资产 | 固定 revision 后复用；重新编码与评测 | 项目输入/计算协议改变，原始模型权重不需要重训 |
| 新主表的 CL、LambdaLoss、RL | 从原始 E0 重新训练、评测 | 单末尾 token、截断边界、FP32 pooling/评分也影响监督基线，不能只重跑 RL |
| G2 的旧 D-CL、E-CL、W-CL/RL | 若后续保留，重建相应训练依赖 | 普通 Qwen 也有截断边界/精度变化；新 W0 必须来自新 D-CL |
| 原始语料、qrels、去污染与清理后的文本 | 哈希和语义一致时复用 | 不重复下载、标注、清理或划分数据 |
| 当前训练数据和采样器 | 复用数据与丢尾规则，不设 dev；修复跨 epoch 固定批内同伴 | 同 data seed、同 epoch 的方法间保持实际样本、批内同伴和 batch 顺序一致 |
| 旧 embedding、索引、评测结果缓存 | 在新协议目录重新构建 | 不能复用旧向量作为新协议结果 |
| 历史参数搜索、G/alignment 网格、消融 | 保留作配方选择与诊断证据 | 只在新结果指出具体问题时补最小对照 |

尚未逐个审计历史远端运行的 tokenizer/autocast，不能断言所有历史运行具有同一种错误。
但新主表必须遵守统一的新协议；将旧微调权重换 tokenizer 后重新评测，不能替代新训练。

训练输入为 `data/processed/reasonrank_multi/train.ready.jsonl`，4,963 条，SHA256：
`f5beddc0cdf47de1b7cc05d8e22fc225ad33c9f1ca1796a387f6cc9ac8dfca39`。
当前 microbatch 16 配置保留 4,896 条，丢弃 67 条（约 1.35%）。记录实际索引数量，不把输入数量写成实际覆盖数量；此事不阻塞本轮训练。
历史本地 G1 的 75 行分数保留在 [run_summary.csv](_summary/g1_bright/run_summary.csv)，不拼入新 seed 均值。

## 2. 本轮检验什么，沿用哪些旧结论？

| 问题 | 直接证据 |
|---|---|
| 条件投影能否保留随机目标梯度均值，并改善方差/成本？ | 同一模型状态、实际动作、奖励下的全参数配对诊断 |
| 估计器的改变能否提升最终检索或稳定性？ | 每个 reward 内，同 G、标签、LR、预算与 seed 的 CP−SF 比较 |
| reward/监督信息有什么影响？ | binary MRR vs binary nDCG；binary vs graded nDCG；各自匹配的 LambdaLoss 参考 |
| RL 是否值得额外成本？ | 与 CL/LL 同数据预算比较质量、seed 波动、运行时间；同 steps 不解释为同算力 |

[历史稳定性实验](../docs/g1_stability_overnight_results.md)中，MRR64 相对 MRR32、graded nDCG64 相对 G32，均在 3/3 个配对 seed 上提高 BRIGHT；平均提升分别约 1.18、1.19 分。
MRR64 与 graded nDCG64 的宏平均接近，尚未决出 reward 优劣；LR 减半没有形成有价值的质量与稳定性折中。
因此沿用 G64、alignment 0.90、LR 5e-6，并保留两个 reward 方向。Binary nDCG64 是标签匹配的新增对照，其收益尚未由旧结果验证。
这些历史结果用于选择起点，不作为新协议下结论成立的证明。初版仅以 binary nDCG/G32 开始、总计 15 次训练的排期已被本版替代。

## 3. 固定运行条件

### 3.1 模型和数值协议

- 原始 E0：`Qwen/Qwen3-Embedding-0.6B`，模型、config、tokenizer 共用 revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`。
- shared joint encoder，full FT；原始权重分别初始化，不加载旧微调权重/optimizer 状态。
- 单末尾读出 token，截断后保留边界，`tokenization_version=2`；FP32 pooling/归一化与评分，评分关闭 autocast/TF32。
- 训练 backbone BF16；外部评测 backbone 统一 FP16，与历史评测精度一致。评测仍使用新 token/FP32 pooling 协议。
- query/document 训练限长 512/1024，评测上限 8192；保留既有 prompt，不同时做长度或 instruction 搜索。

原始模型的 last-token pooling 与 instruction 示例见[固定 revision 的官方说明](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/blob/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3/README.md)。项目的截断边界和 FP32 区域单独记录。

### 3.2 数据、预算和优化器

| 项目 | 固定值 |
|---|---|
| 数据 | 现有多正例 ready 文件；`dev_samples_per_source=0`；不生成新 split |
| 采样 | 建数据集前按 data seed 重置 RNG，按现有规则一次性选样和丢尾；每个 epoch 在 source 内重排样本、重组完整 microbatch，再重排整批；启用长度桶时同时保持桶内分组 |
| seed | training/data：42、3407、2026；RL rollout seed 同值但独立管理；预处理代表正例 seed 固定 42 |
| 预算 | 113 optimizer steps；8 卡 × microbatch 16，global batch 128；每 run 14,464 次 query 呈现 |
| 优化器 | AdamW，LR 5e-6，weight decay 0.01，linear schedule，warmup ratio 0.03 |
| 梯度裁剪 | 显式 `max_grad_norm=0` 和 DeepSpeed `gradient_clipping=0`；实际 engine 设置见训练日志 |
| RL | G64，fixed alignment 0.90，vMF product，LOO，advantage norm none，document log-prob sum，共享 document baseline |
| 候选 | 保留全部已知正例和变长候选；相同 masked device-local 代表正例，跨 query 文档 detach；RL 保留 frozen-candidate rescaling |
| 其他损失 | KL=0，aux InfoNCE=0，单一 reward，不混合目标 |
| 保存/选模 | 模型 checkpoint：25/50/75/100/113；只保存模型，不保存 optimizer；外部主表只用最终 113 步 |

同 seed 的方法读取相同数据顺序；模型加载/不同 wrapper 对 RNG 的消耗不再改变数据分组。
SF/CP 完整训练第一次更新后权重会分叉，后续动作无需相同；训练 seed 配对不能替代同状态梯度诊断。
单一 LR 的比较是受控起点，不能声称各方法均已充分调优。若优化尺度异常，另立等预算校准并记录选择依据，本批不自动改配方。

### 3.3 外部评测的角色

BRIGHT 已多轮用于配置开发，历史不会因协议修复或换 seed 消失。新结果可报告，但须披露选择过程。
用户已看过部分 G2 MTEB 结果；具体任务/版本及用于决策的聚合分数尚待核对，不能把整套 MTEB 当作未见测试。
方法最终冻结后，再从确实未用于决策的任务/查询中指定独立确认集合，首次查看结果前保存任务清单与指标。
若没有这样的集合，就收窄为已知 benchmark 上的受控比较；新的 training seed 不替代独立泛化证据。

## 4. 已接入的夜跑队列

### P0/P1：E0 与配对梯度诊断

夜跑先用 E0、binary MRR@10、G64、alignment 0.90，在固定 epoch-0 microbatch 位置 0/100/200 做配对诊断，然后重新评测 E0 BRIGHT。
每个位置使用 16 个独立 rollout seed，共 48 对、96 次 microbatch forward/backward；没有 optimizer 更新。
三个位置预先固定，报告实际样本 ID、source 和 batch 哈希，不称为全部 source 的代表性抽样或 dev。
seed 42 的新 LL-Binary 完成后，在其 step 25 权重上重复同样诊断。两种估计器必须加载同一份权重。

诊断校验实际动作和完整奖励表哈希相同、梯度有限，并输出完整参数梯度方差、配对均值差及 Monte Carlo 误差尺度、时间和内存。
分析时比较 `variance × mean_compute_time`，把 CPU 统计开销与 GPU 计算耗时分开；该乘积只是固定计算预算下的近似，不能直接视为训练收敛速度。
若均值差可疑或信号未分辨，后续按 16→64→256 个不重叠 seed 增加诊断样本，保留全部结果；256 后仍不明确则记录“未分辨”。
夜跑只执行初始 16 draws，不自动根据噪声大小追加计算或挑 batch。

按整批执行要求，诊断失败会记录并继续独立训练；没有自动根据方差大小批准/取消后续方法。诊断结果仍须在解释论文结论前审查。
实现边界与内存开销见 [gradient_estimators.md](../docs/gradient_estimators.md)。本机无 CUDA，真实预训练模型的诊断尚未完成。

### P2/P3：九方法、三个 seed

| seed 42 run ID | 标签/目标 | G | 角色 |
|---|---|---:|---|
| `G1-R2-CL` | 全部已知正例，InfoNCE，temperature 0.03 | — | 通用监督基线 |
| `G1-R2-LL-Binary` | binary LambdaLoss@10，sigma=1/0.03 | — | binary nDCG 匹配监督 |
| `G1-R2-LL-Graded` | teacher 3/2/1/0 LambdaLoss@10，同 sigma | — | graded nDCG 匹配监督 |
| `G1-R2-RL-MRR64-SF` / `-CP` | binary MRR@10 | 64 | MRR 内估计器对照 |
| `G1-R2-RL-GradedNDCG64-SF` / `-CP` | graded nDCG@10 | 64 | graded 监督内估计器对照 |
| `G1-R2-RL-BinaryNDCG64-SF` / `-CP` | binary nDCG@10 | 64 | binary 排序内估计器对照 |

其他 seed 的 run ID 在上述完整名称后加 `-Seed3407` 或 `-Seed2026`。
每个 seed 按表中顺序完成九个方法，共 27 次训练，每次结束后评测最终 BRIGHT。三个 seed 可跨机器并行；E0 评测与两次公共诊断只分配给 seed 42。
不因中途分数涨跌改变目标、LR、G 或训练步数；失败的独立 run 记录后继续，不能把缺失 seed 当作低分而删掉。

仅完整的三 seed 组报告均值、样本 SD、最差分数，同时给出各 reward 内的逐 seed CP−SF 差、逐领域分数和耗时。
n=3 是最低限度的稳定性证据；query bootstrap 不替代训练重复，跨 reward/监督标签的差值也不能单独归因于估计器。

### 后续机制分析：收益在哪一级消失？

三层检索诊断尚未自动接入夜跑，不再作为本批训练启动前提。保留 checkpoint 后，在固定训练 query、同一批候选/标签上追加：

1. sampled 小候选池 reward（固定诊断 seeds，报告采样波动）；
2. 同池、未经 RL 均值校准缩放的 deterministic cosine 排序；
3. 同 source 大池或可核实完整 corpus 的 deterministic nDCG@10 / Recall@10 / Recall@100。

共享固定文本/ID 的 corpus，处理重复、已知正例与未标注文档；仅有 source 文档并集时明确称“大池”，不能写成全库。
这些 query 仍在训练中，结果只解释机制，不作为未见 query 泛化证据。泛化看最终外部评测。
sampled 升而同池不升时检查随机/部署目标差异；同池升而大池不升时检查候选覆盖；对匹配 LL 没有质量或成本优势时收窄主张，不盲目增加网格。

## 5. 不自动追加的消融

| 需要回答的问题 | 最小后续对照 |
|---|---|
| G64 收益在新协议是否仍成立，投影能否减少采样成本？ | 在选定 reward 下补 SF/CP × G32/G64；复用本批 G64，另报算力 |
| 候选覆盖限制收益？ | LL/SF/CP 共享同一 E0 挖掘候选与刷新时间 |
| 随机目标与 deterministic 目标失配？ | CL vs CL+CP，对齐数据/候选，用梯度范数与夹角校准混合系数 |
| 某个 source 退化或 seed 效应接近噪声？ | 预先指定局部分析或额外 seed，在查看确认集前决定 |

MRR vs binary nDCG、binary vs graded nDCG 已进入本批，不另排重复网格。
不自动追加 alignment、norm/doc-mean 全组合、counterfactual baseline、Gaussian、learned κ 或 reference loss。

## 6. G2、G3 如何安排

G2 延后到 G1 有可解释结果后，再用 E2Rank 检查初始化依赖：新 D-CL→W0，W-CL/W-RL 从同一新 W0 出发；E-CL/E-RL 从 E0 出发；D-RL 从通用 Qwen 出发。单 seed 最小六次训练，warm-up 成本计入。
若宣称估计器收益，补匹配 SF；若宣称优于强排序监督，补标签匹配 LL。不能从 B0 统一配方下失败推断 RL 无法训练 embedding。
当前 `Qwen/Qwen3-0.6B` 官方标注包含 pretraining/post-training，不能称纯 Base；若研究纯 Base，另定义对应 checkpoint 与整组初始化。[官方模型卡](https://huggingface.co/Qwen/Qwen3-0.6B)
BGE-M3 后续启动前需修复未 seed 的候选 RNG 或冻结实际候选文件；不复用旧 W0。G2 历史完整结果尚未同步本地。

G3 不在本轮预算。当前条件投影只覆盖有限候选的 joint vMF；不能对动态全库返回的 top-K 局部投影后宣称保持原目标梯度期望。
后续先统一 corpus、generator、真实上下文装载和候选访问条件，再比较检索/答案奖励，并记录真实进入 context 的文档、调用数、F1 和检索指标。

## 7. 执行、成本与恢复

在 GPU 机器项目根目录、激活训练环境后：

```bash
nohup python -u scripts/run_g1_r2.py > g1_r2_night.log 2>&1 &
```

三台机器分别添加 `--seeds 42`、`--seeds 3407`、`--seeds 2026`，各执行九次训练；每台仍按 8 卡运行，不改变单 run 的 global batch。
各自事件/汇总写到 `.r2_batch/queues/seeds-<seed>/`，状态仍按 run ID 隔离。共享输出目录或汇集结果后，执行 `python scripts/run_g1_r2.py summary` 生成三 seed 总表。

运行根目录 `checkpoints/iclr2027-r2`，数据索引缓存 `.cache/dataset_index/g1-r2`；文本文件直接复用。
每条运行保存展开配置、数据/源码哈希、DeepSpeed 配置、命令、训练/评测日志和状态；汇总在 `.r2_batch/summary.md` 与 `summary.json`。
同一命令重启时只跳过身份一致且完整的结果；训练成功但评测失败只补评测。失败训练保留现场，不自动覆盖或续训；需重训时用新的输出根目录。
同根目录按 seed 加进程锁，不同 seed 可并行；任何失败最终返回非零，后续独立任务继续。单 seed 报告不计算 SD，三 seed 总表只聚合完整组。完整规则见[夜跑说明](../docs/g1_r2_overnight.md)。

预算为 27 次训练 + 28 次最终 BRIGHT（含 E0）+ 两个固定模型状态的配对诊断。G2/G3、独立确认集评测和三层机制分析单列。
不根据历史耗时承诺一夜完成；队列无时长上限。G64 的 4,096 个 product reward cell 来自 64 个 query 动作和 64 个文档组，不是 4,096 个独立梯度样本。

## 8. 证据与历史

- [方法审查](METHOD_DESIGN_REVIEW.md)、[方法重设计](METHOD_REDESIGN.md)：协议差异、投影条件与目标失配。
- [G1 稳定性结果](../docs/g1_stability_overnight_results.md)：选择 G64、两个 reward 方向与 LR 的历史证据。
- [梯度与裁剪核查](G1_GRADIENT_ANALYSIS.md)：历史推断及边界。
- [旧实验计划](EXPERIMENT_PLAN_2026-09-15.md)、[旧 G2 计划](G2_RL_PLAN.md)：历史批次与当时决策。
