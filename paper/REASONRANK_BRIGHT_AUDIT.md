# ReasonRank → BRIGHT 数据审计

审计日期：2026-09-10。结论：可以推进 BRIGHT 为主的研究方向，但不能直接用原始 ReasonRank train 训练后报告 BRIGHT 泛化；须先去除测试题重叠、重建开发集、修正数据转换。数据规模足以启动已有 embedding checkpoint 的受控 post-training 实验，是否足以获得稳定提升仍须由训练曲线和独立开发集判断。

## 1. 范围与复现

实际读取 [ReasonRank RL 数据](https://huggingface.co/datasets/liuwenhan/reasonrank_data_rl) 的全部 train/val，以及 [BRIGHT](https://huggingface.co/datasets/xlangai/BRIGHT) 全部 12 个领域的 1,384 条 example/query 记录。没有只依赖数据卡标称规模。

- ReasonRank revision：`28c5836408857149b80dc352ed362eadc1199a50`
- BRIGHT revision：`3066d29c9651a576c8aba4832d249807b181ecae`
- 文件哈希：[download_manifest.json](audits/reasonrank_bright/download_manifest.json)
- 完整数值：[summary.json](audits/reasonrank_bright/summary.json)
- 保守隔离行号：[quarantine_manifest.json](audits/reasonrank_bright/quarantine_manifest.json)，行号为原始 parquet 的 **0-based** 索引。

```bash
python scripts/download_reasonrank_audit.py
uv run --no-project --with pyarrow --with scikit-learn --with rapidfuzz python scripts/audit_reasonrank_bright.py
```

原始文件和含完整问题文本的匹配证据保存在 `data/audit_reasonrank_bright/`（git ignored）；脚本输出在其 `results/` 下。扫描方法为 NFKC/casefold/空白规范化、去标点后的文本比较、字符 TF-IDF cosine（3–5 gram）、整段问题包含关系。**这是可复现的文本重叠筛查，不是语义去污染完毕的认证。** 未下载 BRIGHT 全部文档，未审计基础模型预训练污染、所有候选文档中的答案泄漏或完整跨数据集语义同题关系。

## 2. 实际规模和监督结构

| 项目 | Train | 原始 Val |
|---|---:|---:|
| Query/listwise records | 6,721 | 50 |
| 候选文档出现次数 | 120,882 | 1,000 |
| 按规范化文本去重的文档 | 86,305 | 767 |
| `relevant_docids` 正例出现次数 | 18,746 | 126 |
| 候选数 | 1–20 | 20 |
| 全部候选都为正例的记录 | 19 | 0 |
| 候选列表内重复文档文本的记录 | 644 | 0 |
| 相同文档文本对应相反 relevance 标签的记录 | 75 | 0 |

全部 6,771 条记录成功解析，候选 ID 唯一、teacher permutation、`final_list`、`extra_info`、正例 ID 包含关系等结构检查通过。结构合法并不代表训练监督无冲突。

**原始 Val 50 条全部来自 math-theorem。** 不能用其选择面向整个 BRIGHT 的 checkpoint 或超参数。Train/val 无规范化空白后的完全相同 query；去标点后 train 内有一组重复（5350、5614）。内部 cosine ≥0.9 有 18 对，其中唯一跨 train/val 对共享立方体绘图但问的量不同，人工检查不认定为同题。Train/val 共享 671 种候选文档文本；共享定理/语料不自动等于 query 泄漏，开发集应按问题/同题簇划分，不能单凭共享文档把所有样本并成一组。

## 3. 与 BRIGHT 测试题的重叠

- 空白规范化后的完全相同问题：0 对；仅做此检查会漏检末尾多一个句号等情况。
- 去标点和格式规范化后相同：81 对，全部来自 train。
- 字符 cosine ≥0.9：115 对（含上述 81 对）。近似匹配中可见标题增删、图片说明省略等同题情况。
- cosine ≥0.7：132 个 train records。该阈值包含误报，例如相似模板但问题不同的 LeetCode 题，不能将 132 称为“确认泄漏数量”。
- 独立整段包含检查额外找到 1 条：train row 5799 与 BRIGHT `aops/math_train_intermediate_algebra_2060`，都是求三根为正整数的多项式系数最小值，BRIGHT 多了选择项。

建议先保守隔离上述并集 **133 条**，保留完整 BRIGHT 测试集，而非删除测试题改善结果。隔离清单目前是筛查产物，并非最终人工标注的去污染集合；进一步的改写/数学表达式/代码题同题审查仍可能改变可用量。

| ReasonRank 来源 | 原始 train | 隔离 133 条后 |
|---|---:|---:|
| biology | 866 | 831 |
| earth_science | 283 | 254 |
| economics | 393 | 366 |
| robotics | 225 | 215 |
| sustainable_living | 73 | 54 |
| stackoverflow | 870 | 870 |
| leetcode | 816 | 810 |
| math-qa | 863 | 859 |
| math-theorem | 786 | 783 |
| msmarco | 1,546 | 1,546 |
| **总计** | **6,721** | **6,588** |

去掉 MSMARCO 后为 **5,042 条**。“只用 ReasonRank 文件”不等于“只用 reasoning 数据”。若主张面向 reasoning retrieval 的适配，建议主实验使用非 MSMARCO 部分；保留 MSMARCO 可作为混合训练选择，但论文必须准确描述。

## 4. 现有 converter 不适合原样沿用

已退役的 `scripts/convert_reasonrank.py` 曾只取原始检索顺序前 16 篇，并丢弃候选不足 16 篇的记录；仅输出文本和 teacher ranking，主动忽略 `relevant_docids`。

实测后果：

- 仅保留 5,399/6,721 条，丢失 **1,322 条（19.7%）**；StackExchange 领域受影响更大，改变了训练领域分布。
- 在保留记录中，正例出现次数由 16,016 降至 14,072，丢掉 1,944 个正例。
- 125 个保留列表失去所有提供的正例；326 个失去 teacher rank-1 文档。
- Teacher rank-1 在 **2,200 条（32.7%）**记录中不属于 `relevant_docids`。这是两种监督不一致，不能直接解释为 teacher 排错。

建议保留可变长度列表、文档 ID、source、完整 teacher permutation 和原始 relevance，并支持 padding/mask；至少需要 2 个有效候选。对相同文本的重复候选去重；75 条正负冲突记录先隔离，或采用显式且可复现的标签合并策略。不要静默把 teacher 第一名等同于唯一正例。

需要明确训练目标：若直接优化 binary nDCG/MRR，应以 `relevant_docids` 构造 reward；若优化 teacher-order agreement，应明确其蒸馏性质，不能将伪等级解释为人工 relevance。两条路线均可研究，但主方法和对照必须使用相同候选、标签和样本。全正例列表对 binary ranking reward 没有排序区分度，建议剔除；teacher-order 路线则未必无用。

## 5. 数据量是否够

6,588 条（含 MSMARCO）或 5,042 条（不含）是**仅做上述 BRIGHT 保守隔离之后**的数量，尚未扣除冲突、退化列表、内部重复和开发集；不是最终训练规模。附带 summary 的 `conservative_usable_estimate` 进一步将 75 条冲突、全正例列表及规范化重复保守剔除，并计算并集，避免重复扣减：**6,498 条（含 MSMARCO），4,963 条（不含 MSMARCO）**。后者再留出 500 条开发集，约有 **4,463 条**训练样本。

我的判断是：**足够开展已有 embedding checkpoint 的方法验证，不足以在训练前承诺效果或宣称数据覆盖充分。** 几千条 query 是独立监督单元，12 万次候选出现和每个 query 的多次 RL rollout 不能视为新增独立问题。

建议从清理后的非 MSMARCO 数据中按来源和同题簇保留约 500 条开发集，其余约 4.4k 条用于 full FT，单 seed 42。原始 50 条 val 可以作为额外数学诊断集，不能承担全局模型选择。Sustainable living 清理前就仅余 54 条，应保留适量开发样本并明确其估计不稳定，不能机械地每领域等额抽 50 条。

约 4.4k 条训练数据在 global batch 32/64/128 下分别约为 138/69/35 个更新/epoch（示例估算，实际取决于最终 manifest、梯度累积和 drop-last）。因此沿用大数据集的一轮训练默认值未必合适；先用开发集曲线检查欠拟合、过拟合、reward 方差和有效 advantage 比例，再固定统一预算。无需因为本次审计改成多 seed，也不需要先扩充海量数据。

## 6. 对新实验设计的影响

1. **Joint 与 query-only 使用同一清理后的 train/dev、初始化、候选列表和 reward。** Query-only 冻结初始 document encoder/cache；joint 更新两侧并用最终 document encoder 重建评测向量。明确两者都允许 full FT，只是可训练分支不同。
2. Query-only 不要求训练 query 来自 BRIGHT，也不要求 corpus 永久不变；固定的是所比较实验中的文档表示版本。BRIGHT query/qrels 不参与梯度、超参数或 checkpoint 选择。
3. 核心结论必须来自同数据、同预算的 RL 与强监督对照，而非只和未经该数据适配的公开模型比较。按 BRIGHT 领域报告效果，并区分训练覆盖/未覆盖领域；BRIGHT 提升不能单独证明模型学会了推理。
4. RAG 另用含答案的 QA train/dev/test 数据。ReasonRank 排序标签不直接提供 downstream answer-F1 训练监督；它不能替代 RAG 数据集。固定 generator 和索引，隔离 query adaptation 效果。
5. 在正式训练前完成剩余同题复核、数据转换修改和固定 split manifest。本文档完成的是审计和规模估算，**没有启动训练、改写原始数据、实施最终划分或改动训练代码**。
