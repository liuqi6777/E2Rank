# G2-R2：E2Rank / MTEB English v2 结果与呈现建议

更新日期：2026-09-18。用户已完成实验并同步[总分 CSV](_summary/g2_r2_mteb_v2/run_summary.csv)。本次只记录结果和论文呈现建议；脚本、suite 和训练配置恢复为本轮讨论前的版本，保留 D/E/W、W0 依赖及 Strong CL。没有启动或重跑训练，也没有改写输入结果。

## 结论

**本批 E、D、W 三个初始化分支中，RL 的 Retrieval 均值均低于普通 CL 和 CL-Strong。** D/W 的全任务均值局部领先不构成检索优势；这批结果不支持把 G2 写成 G1 检索收益的正向迁移验证。

当前呈现建议是：若正文需要讨论从通用 LLM 直接训练 embedding，可聚焦 D，并在附录保留 E/D/W 完整结果与正文引用；若正文只讨论 G1 后训练机制，可将整个 G2 放附录。此处记录建议，不宣称论文 LaTeX/PDF 已同步，也不因正文取舍删除实验定义或结果。E2Rank 当前不新增训练。后续另有 [ReasonEmbed 单 epoch 三方法实验](../docs/g2_reasonembed_cl.md)，不改变本表结果。

## 已同步结果

九条运行均为 s42；CSV 每行报告 41 个任务、7 个类型、0 个缺失任务和 0 个错误。这是汇总文件报告的完整性，尚未核验逐任务 JSON、训练 receipt 或实际 checkpoint。RL 按已有计划对应 graded nDCG / CP / G64 / alignment 0.80，但 CSV 本身不能证明实际训练配方。

| 初始化 | 方法 | 全任务均值 | 类型均值 | Retrieval |
| --- | --- | ---: | ---: | ---: |
| E | CL | 69.45 | 63.91 | 59.12 |
| E | CL-Strong | 69.61 | 64.07 | 59.28 |
| E | RL | 68.85 | 63.63 | 57.28 |
| D | CL | 60.64 | 57.04 | 51.47 |
| D | CL-Strong | 60.99 | 57.27 | 51.52 |
| D | RL | 61.0 | 56.89 | 49.26 |
| W | CL | 59.92 | 56.35 | 50.71 |
| W | CL-Strong | 60.4 | 56.6 | 50.89 |
| W | RL | 61.55 | 57.48 | 49.96 |

数值为 0–100 尺度。全任务均值对任务等权，类型均值先在类型内平均再对类型等权；Retrieval 只对检索任务的 main score 等权。不能用较高的全任务均值替换原定检索主指标。

| 配对比较 | 全任务均值差 | 类型均值差 | Retrieval 差 |
| --- | ---: | ---: | ---: |
| E: RL−CL | -0.60 | -0.28 | -1.84 |
| E: RL−CL-Strong | -0.76 | -0.44 | -2.00 |
| D: RL−CL | +0.36 | -0.15 | -2.21 |
| D: RL−CL-Strong | +0.01 | -0.38 | -2.26 |
| W: RL−CL | +1.63 | +1.13 | -0.75 |
| W: RL−CL-Strong | +1.15 | +0.88 | -0.93 |

## D 的正文表述范围

D 从 `Qwen/Qwen3-0.6B` 通用后训练 LLM 开始，不称纯预训练 Base。RL 的全任务均值为 61.00，CL-Strong 为 60.99，基本持平；Retrieval 则为 49.26 vs 51.52，低 2.26 分。对普通 CL，RL 的全任务均值高 0.36，但 Retrieval 低 2.21。

可用表述：在本次单 seed、固定配方和预算下，直接 RL 在全任务均值上与强化 CL 接近，但检索均值低于两种 CL 对照。不能写成 RL 更适合直接训练 embedding。原始 B0 没有同协议成绩，因此当前表也不能量化相对 B0 的表示建立收益。

## E0 外部参考与 W0

[Qwen 官方 MTEB English v2 表](https://github.com/QwenLM/Qwen3-Embedding/blob/main/README.md#evaluation)（2026-09-18 查阅）报告 Qwen3-Embedding-0.6B 的全任务均值 **70.70**、类型均值 **64.88**、Retrieval **61.83**。使用 English v2 embedding 表，不混入 Multilingual 表或 top-100 reranking 表。

E-CL / E-CL-Strong / E-RL 的 Retrieval 分别比这个公开参考低 2.71 / 2.55 / 4.55 分。公开 E0 不属于本项目同协议复测，不能把这些差值写成严格控制的训练前后退化；模型 revision、任务版本、instruction、tokenization 和精度仍须匹配。E-RL 低于本批两个 CL 的比较不依赖该公开参考。

如果实际 W0 确实是表中的 D-CL，W0 Retrieval 为 51.47，W-CL 为 50.71、W-CL-Strong 为 50.89、W-RL 为 49.96；三者分别下降 0.76、0.58、1.51 分。W-RL 的全任务均值相对 W0 增加 0.91。这项解释以实际 checkpoint 来源核验为前提，不能只凭 run 名称确认。

## 证据边界

- 只有总分，没有逐任务/子集结果；不能定位退化领域、推断所有任务均退化或检验差异显著性。单 seed 不计算训练 seed SD。
- `retrieval_ood` 与 `retrieval` 在本表各行相同；汇总脚本使用历史固定排除名单，不能据该列认定所有任务都已通过 G2 的 OOD/重叠审计。
- CL 使用 binary 正例监督，graded RL 使用 teacher grades，CL-Strong 还改变负例池和梯度路径。本组比较的是训练配方，不能分离 CP 单因素效应；没有匹配 SF/LL 也不能验证其跨环境机制优势。
- 同时下降不能确定是数据不匹配、遗忘或训练过久；结果没有提供这些机制的因果证据。
- 原始模型同协议评测、逐任务结果及训练记录是后续解释所需证据，不在本次自动启动评测或补训练。
