# RAG qrels：HotpotQA 对齐损失诊断与阈值方案

本文记录对 G3 RAG 训练 qrels 中 **HotpotQA supporting-fact → 冻结 corpus** 对齐的
诊断结论、阈值敏感性、样本质量核验，以及生成候选 qrels 版本的工具与推荐。
NQ 部分几乎无损（58,879 / 58,880 保留），本文只针对 HotpotQA 的 45% 丢弃问题。

## 1. 问题

严格版 qrels（`data/rag/qrels/`，`--hotpot-minimum-f1 0.8`）对 HotpotQA：
- 输入 90,398 条可用 query
- **kept 49,851（55.1%）/ dropped 40,547（44.9%）**

对齐规则（`rag.build_qrels`）：每个 supporting fact `(title, sentence)` 需
①`normalize_answer(title)` 与某 corpus passage 标题**完全相等**，②该 passage 正文对
sentence 的**滑窗 token-F1 ≥ 0.8**；一条 query 的**所有** fact 都命中才保留（全或无）。

## 2. 丢弃成因拆解（全量 corpus 扫描）

对 147,905 个唯一 supporting fact 分类：

| 类别 | 数量 | 占比 | 含义 |
|---|---:|---:|---|
| matched（命中，F1≥0.8） | 109,972 | 74.4% | 标题相等且句子 F1≥0.8 |
| title_no_sent（B，可救） | 17,295 | 11.7% | 标题在 corpus，但句子 F1<0.8 |
| title_missing（A，硬失败） | 20,638 | 14.0% | 归一化标题在 corpus 里不存在 |

Query 级（全或无）丢弃的 40,547 条中：
- **含 ≥1 个 title_missing（A，救不回）：20,812 条**（占丢弃 51%）
- **失败全是 title_no_sent（B，潜在可救）：19,735 条**（占丢弃 49%）

**结论：近一半被丢的 query 仅仅因为 F1 阈值 0.8 偏高，其证据词条确实在 corpus 里。**

## 3. 阈值敏感性（单次扫描推算，NQ 固定 58,879）

| hotpot F1 阈值 | HotpotQA kept | HP 保留率 | vs 0.8 | 总 qrels(NQ+HP) |
|---|---:|---:|---:|---:|
| 1.0（精确子串） | 24,718 | 27.3% | −25,133 | 83,597 |
| 0.9 | 37,486 | 41.5% | −12,365 | 96,365 |
| **0.8（现行严格版）** | 49,851 | 55.1% | +0 | 108,730 |
| 0.7 | 57,729 | 63.9% | +7,878 | 116,608 |
| 0.6 | 63,067 | 69.8% | +13,216 | 121,946 |
| **0.5（推荐）** | 66,223 | 73.3% | +16,372 | 125,102 |

## 4. 样本质量核验（决定安全下限）

对不同 F1 段各抽样人工判读，并统计标题歧义度（同一归一化标题对应的 corpus passage 数）：

- **所有匹配的标题歧义度 amb%=100%**：并非“同名不同实体”，而是 wiki18 把长词条按
  100 词切成多段、每段复用同一 title。所以低 F1 主要来自“证据句落在词条的另一段”，
  **title 完全相等仍是可靠的实体锚点**。
- **[0.5,0.6)**：抽样全部为正确同实体（villanova university、grupo modelo、averroes…），
  仅是 supporting sentence 落在词条另一段。**降到 0.5 安全。**
- **[0.4,0.5)**：仍基本同实体（victoria university、hulu、marsha blackburn），但匹配到的
  passage 不一定含该句本身；用于二值检索监督尚可，严格性下降。
- **[0.3,0.4) 及以下**：开始语义错位（harry shearer 匹配到 SNL 段、gift 匹配到通用“礼物”
  词条而非 Paul Brandt 专辑、haile selassie 匹配到战役段）。**不应纳入。**

**安全下限 = 0.5**：0.8→0.5 挽回的都是真同实体；<0.4 引入错误正例。

## 5. 候选版本与推荐

| 版本 | 阈值 | 总 qrels | 定位 |
|---|---|---:|---|
| strict（已产出，`data/rag/qrels/`） | 0.8 | 108,730 | 最保守，证据段与句子高度一致 |
| balanced | 0.6 | 121,946 | 折中，+13k |
| **recommended** | **0.5** | **125,102** | 最大化召回且经样本验证无假正例，+16k（总量 +15%） |

**推荐 0.5 版本作为训练主用**，strict 0.8 版本保留作对照/消融。理由：
- 检索器训练是二值相关性监督，只要正例段确属该证据实体即可；0.5 挽回的正是这类样本。
- 保留 0.8 严格版可做“标签严格度”消融，量化放宽是否影响 BRIGHT/QA 下游。

A 类（title_missing，14%）是数据源天花板（wiki18 2018 快照缺词条或标题命名体系不同），
放宽 F1 无法挽回；如需进一步挖掘，可另做“标题去括号/前缀变体匹配”，但收益与假正例风险
需单独评估，不在本轮。

## 6. 生成工具（避免重复扫 14GB）

新增两个脚本，把“任意阈值 qrels”从“每次全量扫 corpus（约 8 分钟且与 GPU 争 CPU）”
变成“一次扫描 + 秒级重建”：

- `scripts/rag_export_alignment_cache.py`：**一次**扫描 corpus，导出
  `data/rag/qrels_alignment_cache/`（`nq_matches.json` + `hotpot_best.jsonl` +
  `cache_manifest.json`，记录 corpus/输入 sha256）。
- `scripts/rag_build_qrels_from_cache.py`：从缓存 + 指定 `--hotpot-minimum-f1` 秒级产出
  完整 qrels（格式、NQ 处理、manifest、unresolved 与 `build_qrels` 一致）。

已离线验证：缓存路径的 nq/hotpot 匹配与直接扫描**完全等价**；用 0.8 可复现严格版。

### 使用（务必在 GPU encode 空闲后再运行，避免争 CPU）
```bash
# 1) 一次性导出对齐缓存（全量扫描一次）
E2RANK_QRELS_WORKERS=180 PYTHONPATH=src python scripts/rag_export_alignment_cache.py

# 2) 从缓存秒级生成各阈值版本
PYTHONPATH=src python scripts/rag_build_qrels_from_cache.py \
  --hotpot-minimum-f1 0.5 --output-dir data/rag/qrels_f1_0.5
PYTHONPATH=src python scripts/rag_build_qrels_from_cache.py \
  --hotpot-minimum-f1 0.6 --output-dir data/rag/qrels_f1_0.6
```

## 7. 对 build_qrels 的性能改动（本轮附带）

`src/rag/build_qrels.py`：
- `iter_json_array`：DPR 2.3GB gz 从逐字符流式解析改为整体解压 + `json.loads`
  （快 10× 以上；`E2RANK_QRELS_STREAM_JSON=1` 可回退旧流式解析）。
- `align_to_corpus`：14GB corpus 单线程扫描改为按字节分块的多进程并行
  （`E2RANK_QRELS_WORKERS` 控制并行度）；保留 ordinal 校验、NQ/hotpot 匹配、
  tie-break 与 corpus sha256 语义。已验证并行结果与串行完全一致。
- 效果：qrels 构造从约 110 分钟降到约 8 分钟（192 核）。

**注意**：任何全量 corpus 扫描（qrels/cache 导出）都会与 GPU encode 的 CPU 端
tokenization 争抢 CPU，导致 GPU 挨饿掉速。实测两者并行会把 encode GPU 利用率打到个位数。
**务必错峰**：encode 期间不要跑全量扫描；把并行度压到很低（如 16）仍会拖慢，最稳是等 encode 完成。

## 8. 偏差声明建议（写入论文时）

无论选哪个阈值，HotpotQA 都存在**非随机丢弃**：被丢样本偏向冷门实体、标题带消歧后缀、
证据句跨段者。保留样本偏“易”，可能高估多跳检索能力。建议在论文中：
- 报告所用阈值、HotpotQA 保留率、A/B 两类占比；
- 若用 0.5，说明挽回样本经同实体核验、未放宽标题匹配；
- 将 strict 0.8 作为标签严格度消融，观察下游是否稳健。

---
生成时间：随本轮诊断；encode（21M passage E0 索引）在单张 H20 上并行进行，约 21 小时。
