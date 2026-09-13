# Embedding 离线分析

入口是 `scripts/analyze_embeddings.py`。训练代码不变；各 checkpoint 编码一次，随后从
FP32、单位长度的 `.npy` memory map 读取向量。第一版实现 query-side vMF 扰动，
不包含 document/joint 扰动或文本改写实验。输出报告为 Markdown 与 CSV，暂不生成图。

## 开始运行

编辑 `configs/analysis/bright.json` 中 E0、CL、RL 的真实 checkpoint 路径和计算设置。
配置是 JSON，所有相对路径均相对命令工作目录，以下命令从仓库根目录执行。
E0 示例路径是占位符；也可指定 Hugging Face model ID，但必须提供 `revision`，值为
40 位不可变 commit SHA。编码会下载该模型/数据；已有本地缓存时使用缓存。

```bash
# 先跑一个领域；其余领域可用相同配置稍后补齐。
python scripts/analyze_embeddings.py all --config configs/analysis/bright.json --subsets biology

# 独立执行阶段。encode/retrieval 可按模型分别运行，后续对比阶段应提供完整模型集合。
python scripts/analyze_embeddings.py encode --config configs/analysis/bright.json --models E0
python scripts/analyze_embeddings.py encode --config configs/analysis/bright.json --models CL RL
python scripts/analyze_embeddings.py retrieval --config configs/analysis/bright.json
python scripts/analyze_embeddings.py geometry --config configs/analysis/bright.json
python scripts/analyze_embeddings.py perturb --config configs/analysis/bright.json
python scripts/analyze_embeddings.py report --config configs/analysis/bright.json
```

编码使用 `AutoModel` 和项目的 `embedding_protocol`，支持 CPU/CUDA；一次加载一个模型，
不启动多 GPU worker。`encoding.precision` 支持 `fp32/fp16/bf16`，默认 `fp32`。
`max_query_length/max_document_length` 默认使用模型协议长度。与正式评测比较时，保持
instruction、query_set、精度、截断长度和模型协议相同；不要为了省显存静默缩短长度。
后续几何、检索和采样默认在 CPU 执行，NumPy BLAS 可使用多个 CPU 线程。

## 数据与缓存

BRIGHT 使用现有 `load_official_bright()`、固定 revision、领域 instruction 和 qrels。
12 个领域独立检索。每个领域的 query 按 ID 升序、document 按 ID 降序固定行顺序。
所有已知正例参与分析，不把训练 collator 的代表正例当作唯一正例。
每条 query 必须有至少一个在 corpus 内的正例；缺失引用、重复 ID/qrel、非法等级均报错。

也支持本地数据：把配置中的 `data` 改为：

```json
{
  "kind": "local",
  "subsets": {"example": "data/analysis/example"},
  "instructions": {"example": "Retrieve relevant passages."}
}
```

该目录包含三个 JSONL 文件：

```text
queries.jsonl: {"query_id":"q1","text":"query text"}
corpus.jsonl:  {"doc_id":"d1","text":"document text"}
qrels.jsonl:   {"query_id":"q1","doc_id":"d1","relevance":1}
```

输出结构：

```text
<output_dir>/<subset>/data/                         # 固定文本、qrels、manifest
<output_dir>/<subset>/<model>/embeddings/           # query/document_embeddings.npy
<output_dir>/<subset>/<model>/retrieval/            # queries、rankings、positive_ranks
<output_dir>/<subset>/geometry/                     # 固定对、谱、邻域、候选并集
<output_dir>/<subset>/<model>/perturb/              # 逐 query、逐强度结果
<output_dir>/reports/<content_id>/report.md          # 跨模型/领域报告
```

每个产物目录在临时目录内完成后发布。已有且输入身份一致时校验 SHA256 后复用；
输入改变或缓存损坏会报错，不覆盖结果。模型文件、数据顺序/文本、协议和阶段参数纳入身份。
重新训练同路径 checkpoint 也会被权重校验识别。修改分析参数后，最简单的方式是使用新
`output_dir`；可复用旧 `data/` 和各模型 `embeddings/` 的目录副本，然后重新计算分析阶段。
不需要复制旧 retrieval/geometry/perturb。报告按输入内容产生新快照，允许先生成仅检索
报告，再追加 geometry/perturb 后生成完整报告。运行阶段和报告时都校验产物文件内容。

## 计算定义

- 检索：分块 FP32 点积，每个 query 保留 `top_k`；不存完整分数矩阵。
  正例额外扫描全库计算精确 rank，因此 top_k 外的正例不截断为 1001。
  nDCG 使用 trec_eval 的线性 relevance gain，MRR/hit 使用 relevance > 0，均以
  `retrieval.k` 截断。分数同分时按 document ID 降序，与 trec_eval 一致。
- 分数诊断：保存最佳/平均正例分数、最高未标注候选分数和两者 margin。
  未标注候选按评测约定处理，但不能据此解释成已确认的语义负例。
- 固定对：各模型 top-`candidate_k` 的并集加全部正例。参考模型在这个集合里选出最高分
  正例和最高分未标注候选，之后所有模型比较同一对文档。输出 margin、边界距离
  `q·(d_pos-d_neg)/||d_pos-d_neg||` 和按 query 平均的正例 cosine。
  相同文档向量导致零分母时边界距离为空，不设成无穷大或零。
- 全局几何：query/document 分别计算 `log mean exp(-2||x-y||²)`，对固定随机非自身
  样本对取样；中心化总体协方差用 FP64 累积，输出完整谱和谱熵 effective rank。
  常量空间 effective rank 为 0；只有一个样本时 uniformity 为空。
- 邻域：所有 query 做 query–query 检索；固定抽样 document anchors 搜索整个文档库。
  排除自身，计算相对参考模型 top-k 的保留率，保存逐样本结果。
- 可选 `geometry.procrustes=true`：固定文档 anchors 的偶数位置拟合正交变换，奇数位置
  文档和全部 query 测试。Query/document 共用同一个变换，不做独立旋转或平移。
  保存拟合秩；若低于 embedding 维度，未约束方向上的位移不可过度解释。默认关闭。
- 扰动：复用真实 vMF 采样与维度相关 κ 反解，按 query 和强度设置可重复的种子。
  `alignment=1` 是不采样基线。默认每强度 32 次，随机固定最多 200 条 query，所有模型
  使用同一批 ID。报告 nDCG/MRR/hit、相对自身干净基线的下降和 nDCG 的 Monte Carlo SE。
  翻转率条件化于该模型原本严格正确的最高正例/最高未标注文档对，不同模型的 eligible
  集合可能不同；汇总同时报告 eligible query 数，不能当成完全匹配的因果对照。
- `perturb.scope=candidates` 只在共享候选中重排；`corpus` 对每个扰动 query 重搜全库。
  二者分开标记。全库模式仍读取共享候选，用于固定的正负文档对翻转诊断。
  该阶段只扰动输出 embedding，不重新编码文本、不执行训练时的分数校准。

## 读取结果与验证

报告的主表先按领域聚合再取宏平均，分数范围为 0–1。`paired_queries.csv` 按 query ID
配对，提供相对 E0 的检索、分数变化；有 geometry 时还包含固定对 margin/边界变化与
query 邻域保留率。`rank_bins.csv` 固定按 E0 正例 rank 的 1、2–10、11–100、101+ 分组。
`geometry.csv` 保存全局指标；`perturb.csv` 按模型、领域、强度汇总，保留 scope。
比较 RL 与 CL 可直接按 query_id 连接原始表，也可把 `reference` 设为 CL 使用新输出目录。

CPU 回归测试包含：块大小不影响检索/同分次序、截断外正例的精确 rank、多正例 graded
nDCG 对照 pytrec_eval、全局旋转不改变几何、固定对和候选并集、vMF 无噪声基线、缓存
篡改拒绝，以及本地微型模型实际编码。测试不下载模型，也不启动真实训练。

```bash
python -m unittest discover -s scripts/tests -p 'test_embedding_analysis.py' -v
```

真实模型正式使用前，用同精度、同协议的无扰动检索结果对照已有 MTEB 逐领域指标。
CPU 合成测试不等于已验证真实 checkpoint 的 MTEB 复现；精度和极近同分可能影响排名。
若用于论文，应记录训练对照协议、未标注相关文档问题和训练 seed；Monte Carlo SE
只刻画扰动采样误差，不代表训练 seed 之间的不确定性。
