# Embedding as Continuous Action: RL Training for Text Embedding Models

## Implementation Plan

---

## 1. Project Overview

### Core Idea

将 embedding 视为 continuous action，用 stochastic Gaussian policy + GRPO 直接优化 non-differentiable ranking metrics（如 nDCG@10），替代传统 contrastive learning 的 proxy loss（InfoNCE）。

### Key Design

```
Standard Embedding:  input x → LLM forward → h (deterministic embedding)
Ours (training):     input x → LLM forward → h → e ~ N(h, σ²I) → reward
Ours (inference):    input x → LLM forward → h (与标准 embedding 完全一致，zero overhead)
```

### 基于 E2Rank 的代码结构

项目代码基于 [E2Rank](https://github.com/Alibaba-NLP/E2Rank) 进行开发。E2Rank 的代码结构：

```
E2Rank/
├── src/
│   ├── train.py          # 训练入口 (Stage II: InfoNCE + RankNet)
│   ├── eval.py           # 评估入口 (BEIR/BRIGHT reranking)
│   ├── model.py          # 模型定义 (Qwen3-based embedding)
│   ├── data.py           # 数据加载 (query + pos + negatives)
│   └── ...
├── eval_mteb/            # MTEB 评估脚本
├── scripts/              # 训练脚本 (train_e2rank_0.6b.sh 等)
├── data/                 # 训练数据目录
└── requirements.txt
```

我们需要新增的核心文件：

```
E2Rank/
├── src/
│   ├── train_rl.py           # [新增] RL 训练入口 (Layer 1: pure embedding)
│   ├── train_rl_e2rank.py    # [新增] RL 训练入口 (Layer 2: dual GRPO for retrieval + reranking)
│   ├── grpo_trainer.py       # [新增] GRPO trainer (两个 Layer 共用核心逻辑)
│   ├── reward.py             # [新增] Reward 计算 (cosine, nDCG 等)
│   └── ...                   # 复用: model.py, data.py, eval.py
├── scripts/
│   ├── train_rl_0.6b.sh      # [新增] Layer 1 RL 训练脚本
│   └── train_rl_e2rank_0.6b.sh  # [新增] Layer 2 RL 训练脚本
└── configs/
    ├── rl_config.yaml         # [新增] Layer 1 RL 超参配置
    └── rl_e2rank_config.yaml  # [新增] Layer 2 RL 超参配置
```

---

## 2. 实现阶段

### Phase 1: 搭建 RL Training Pipeline (1-2 周)

**目标：** 用 differentiable reward (cosine similarity) 跑通整个 GRPO pipeline，验证代码正确性。

#### Step 1.1: Reward 函数模块 (`src/reward.py`)

```python
"""
实现多种 reward function:
1. contrastive_reward: sim(q, d+) - max(sim(q, d-))   [differentiable, 用于验证]
2. ndcg_reward: nDCG@K based on cosine scores          [non-differentiable, 核心]
3. mixed_reward: λ1 * contrastive + λ2 * ndcg          [最终版本]
"""
```

需要实现的函数：

- `compute_contrastive_reward(e_q, e_pos, e_neg) → [B]`
  - Cosine similarity based
  - 支持 in-batch negatives
- `compute_ndcg_reward(e_q, e_candidates, labels, K=10) → [B]`
  - 按 cosine similarity 排序
  - 计算 nDCG@K (non-differentiable)
  - 需要 ground truth relevance labels
- `compute_mrr_reward(e_q, e_candidates, labels) → [B]`
  - MRR 作为备选 reward

**注意：** nDCG reward 需要每个 query 对应多个 candidate documents（不只是 1 pos + n neg），才能有意义。需要检查 E2Rank 的 Stage II 训练数据格式是否满足（每个 query 有 1 pos + 15 neg，可以用 LLM label 的排序作为 graded relevance）。

#### Step 1.2: GRPO Trainer (`src/grpo_trainer.py`)

核心训练逻辑：

```python
class EmbeddingGRPOTrainer:
    """
    GRPO for embedding models.
    
    Key difference from standard GRPO:
    - No text generation / rollout
    - Action = embedding vector (continuous)
    - Gaussian noise for exploration
    - Embedding-level advantage, not token-level
    """
    
    def __init__(self, model, optimizer, config):
        self.model = model
        self.sigma = config.sigma          # 初始 noise scale (e.g., 0.05)
        self.G = config.group_size         # samples per query (e.g., 8)
        self.sigma_learnable = config.sigma_learnable  # 是否学习 sigma
    
    def train_step(self, batch):
        queries, positives, negatives, labels = batch
        
        # === Phase 1: Rollout (no grad) ===
        with torch.no_grad():
            h_q = self.model.encode(queries)       # [B, d]
            h_pos = self.model.encode(positives)   # [B, d]
            h_neg = self.model.encode(negatives)   # [B, n_neg, d]
        
        # Sample G perturbations per query
        epsilons = torch.randn(self.G, *h_q.shape)  # [G, B, d]
        rewards = []
        for g in range(self.G):
            e_q = h_q + self.sigma * epsilons[g]
            e_q = F.normalize(e_q, dim=-1)          # re-normalize after perturbation
            r = self.compute_reward(e_q, h_pos, h_neg, labels)
            rewards.append(r)
        
        rewards = torch.stack(rewards)              # [G, B]
        
        # === Phase 2: Compute advantages (GRPO) ===
        baseline = rewards.mean(dim=0)              # [B]
        advantages = rewards - baseline             # [G, B]
        # 可选: normalize advantages
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # === Phase 3: Policy gradient update (with grad) ===
        h_q = self.model.encode(queries)            # 重新 forward, 这次有 grad
        h_q_normalized = F.normalize(h_q, dim=-1)
        
        loss = 0.0
        for g in range(self.G):
            e_q = h_q_normalized + self.sigma * epsilons[g].detach()
            # log π(e|x) for isotropic Gaussian: -||e - μ||² / (2σ²) + const
            # 梯度只通过 μ (= h_q_normalized) 传播
            log_prob = -0.5 * ((e_q.detach() - h_q_normalized) ** 2).sum(-1) / (self.sigma ** 2)
            loss -= (advantages[g].detach() * log_prob).mean()
        
        loss = loss / self.G
        return loss
```

**实现要点：**

- `h_q` 需要两次 forward：一次 no grad (rollout)，一次 with grad (update)
- `epsilons` 在两次 forward 之间需要保持一致（detach 并复用）
- Re-normalization after perturbation 是必要的，否则 perturbed embedding 的 norm 会变化
- 考虑使用 gradient accumulation 来 fit larger effective batch size

#### Step 1.3: Training 入口 (`src/train_rl.py`)

```python
"""
Training pipeline:
1. 加载 E2Rank Stage I checkpoint (Embedding-Only model)
2. 加载训练数据 (复用 E2Rank 的 data.py)
3. 用 GRPO 做 Stage II RL fine-tuning
4. 评估 & 保存 checkpoint
"""
```

需要复用 E2Rank 的组件：
- `model.py`: 模型加载 (`AutoModel.from_pretrained`, `last_token_pool`)
- `data.py`: 数据加载 (query + pos + negatives format)
- `eval.py`: 评估 pipeline (BEIR, BRIGHT)

需要注意的适配：
- E2Rank Stage II 的 data format 是每个 query 有 1 pos + 15 negatives + LLM ranking label
- 我们需要用 LLM ranking label 来构造 graded relevance for nDCG reward
- 如果只用 binary relevance (pos=1, neg=0)，nDCG 退化为比较简单的 metric

#### Step 1.4: Pilot 验证

**实验配置：**
```yaml
# configs/rl_pilot.yaml
model: Alibaba-NLP/E2Rank-0.6B-Embedding-Only
data: data/train.jsonl  # E2Rank 的 Stage II 数据
sigma: 0.05
group_size: 8           # G
reward: contrastive     # 先用 differentiable reward 验证
lr: 1e-6
batch_size: 4           # per GPU
grad_accum: 8           # effective batch 32
max_steps: 500
eval_datasets: [scifact]
lora: true              # 先用 LoRA 省资源
lora_r: 64
```

**验证标准：**
- Training loss 能正常下降
- 在 SciFact 上的 nDCG@10 有改善（相比不做 Stage II）
- 和 E2Rank 的 Stage II (RankNet loss) 做同样步数的对比

---

### Phase 2: Non-differentiable Reward (1-2 周)

**目标：** 切换到 nDCG@10 reward，这是 RL 方法的核心价值所在。

#### Step 2.1: nDCG Reward 实现

```python
def compute_ndcg_reward(e_q, e_candidates, relevance_labels, K=10):
    """
    Args:
        e_q: [B, d] query embeddings (perturbed)
        e_candidates: [B, n_cand, d] candidate embeddings
        relevance_labels: [B, n_cand] graded relevance (e.g., 0,1,2,3)
        K: cutoff for nDCG
    Returns:
        ndcg: [B] nDCG@K scores
    """
    # Cosine similarity scores
    scores = torch.bmm(e_q.unsqueeze(1), e_candidates.transpose(1, 2)).squeeze(1)  # [B, n_cand]
    
    # Sort by predicted scores
    sorted_indices = scores.argsort(dim=-1, descending=True)[:, :K]
    
    # Gather relevance labels in predicted order
    sorted_relevance = relevance_labels.gather(1, sorted_indices)  # [B, K]
    
    # DCG
    positions = torch.arange(1, K + 1, device=scores.device).float()
    dcg = (2 ** sorted_relevance - 1) / torch.log2(positions + 1)
    dcg = dcg.sum(dim=-1)  # [B]
    
    # IDCG (ideal)
    ideal_sorted = relevance_labels.sort(dim=-1, descending=True).values[:, :K]
    idcg = (2 ** ideal_sorted - 1) / torch.log2(positions + 1)
    idcg = idcg.sum(dim=-1)  # [B]
    
    ndcg = dcg / (idcg + 1e-8)  # [B]
    return ndcg
```

#### Step 2.2: Relevance Label 构造

E2Rank 的训练数据由 Qwen3-32B 标注了 full ranking permutation。利用这个信息构造 graded relevance：

```python
# 方案 A: Binary relevance (简单)
# pos=1, neg=0

# 方案 B: Graded relevance from LLM ranking (推荐)
# rank 1 → relevance 3, rank 2-5 → relevance 2, rank 6-10 → relevance 1, rest → 0

# 方案 C: Soft relevance from original cosine similarity
# 用 Stage I model 的 cosine scores 作为 soft relevance
```

**推荐方案 B**：利用 E2Rank 已有的 LLM ranking labels，比 binary relevance 更 informative。

#### Step 2.3: 对比实验

在 BEIR 的一个 subset 上跑（比如 Covid, SciFact, NFCorpus）：

| Method | Training Signal | nDCG@10 |
|--------|----------------|---------|
| E2Rank Stage I only (no Stage II) | - | baseline |
| E2Rank Stage II (InfoNCE + RankNet) | supervised | E2Rank result |
| Ours: GRPO + cosine reward | RL (differentiable) | ? |
| Ours: GRPO + nDCG reward | RL (non-differentiable) | ? |
| Ours: GRPO + mixed reward | RL (mixed) | ? |

---

### Phase 3: Layer 1 — Pure Embedding Evaluation (2-3 周)

**目标：** 证明 RL training framework 的 general applicability。此阶段模型只作为 embedding model 使用，不涉及 reranking。

#### Step 3.1: MTEB Evaluation (General Embedding)

MTEB 是证明方法 generality 的核心 benchmark。复用 E2Rank 的 MTEB 评估脚本：

```bash
# MTEB (eng, v2) — 41 datasets, 7 task types, 用于 ablation
bash eval_mteb/scripts/run_mteb.sh checkpoints/rl-0.6b rl-0.6b "MTEB(eng, v2)"

# MTEB (eng, v1) — 56 datasets, 用于 final results
bash eval_mteb/scripts/run_mteb.sh checkpoints/rl-0.6b rl-0.6b "MTEB(eng, v1)"

# 汇总结果
python3 eval_mteb/summary.py results/mteb/rl-0.6b/rl-0.6b/no_version_available "MTEB(eng, v2)"
```

**对比表 (Layer 1 核心结果):**

| Method | Training | Retr. | Rerank. | Clust. | PairClass. | Class. | STS | Summ. | Avg. |
|--------|----------|-------|---------|--------|------------|--------|-----|-------|------|
| Base (Qwen3-0.6B, no training) | - | | | | | | | | |
| E2Rank Stage I (InfoNCE) | CL | | | | | | | | |
| GRACE-1.5B | RL (text gen) | | | | | | | | |
| **Ours: GRPO + cosine reward** | RL (emb action) | | | | | | | | |
| **Ours: GRPO + nDCG reward** | RL (emb action) | | | | | | | | |

**Key comparison:** 
- vs E2Rank Stage I → 同样的 base model，CL vs RL 的直接对比
- vs GRACE → 同样是 RL for embedding，但我们 zero inference overhead

#### Step 3.2: BRIGHT Retrieval Evaluation

BRIGHT 是 reasoning-intensive retrieval benchmark，特别适合展示 RL 对 complex query 的优势：

```bash
python src/eval_retrieval.py \
    --model checkpoints/rl-0.6b \
    --datasets bright \
    --metric ndcg@10
```

#### Step 3.3: Scale to 4B/8B

```bash
# 4B model
bash scripts/train_rl_4b.sh
bash eval_mteb/scripts/run_mteb.sh checkpoints/rl-4b rl-4b "MTEB(eng, v2)"

# 8B model (可能需要多卡)
bash scripts/train_rl_8b.sh
bash eval_mteb/scripts/run_mteb.sh checkpoints/rl-8b rl-8b "MTEB(eng, v2)"
```

#### Step 3.4: Baseline Reproduction

需要复现的 baselines：
- **Base model (no training)**: 直接用 Qwen3-0.6B/4B-Instruct 做 embedding
- **Standard CL**: E2Rank Stage I (已有 checkpoint，直接评测)
- **GRACE**: 需要在相同 backbone (Qwen3) 上复现，或引用论文数据并说明 backbone 差异
- **CL continued training**: 在 Stage I 基础上继续 InfoNCE 训练同样步数，作为 controlled baseline

---

### Phase 4: Layer 2 — E2Rank Integration (1-2 周)

**目标：** 证明 RL training 在 unified retrieval+reranking 场景的价值。

#### Step 4.1: Full GRPO Design for E2Rank Stage II

E2Rank 原始 Stage II 使用两个 supervised proxy loss：

```
原始 E2Rank Stage II:
L = L_InfoNCE(query emb vs doc emb)  +  λ * L_RankNet(listwise_prompt emb vs doc emb)
    ↑ retrieval: contrastive proxy         ↑ reranking: pairwise proxy
```

我们将两侧全部替换为 GRPO，形成统一的 RL 训练范式：

```
Ours:
L = L_GRPO(query emb vs doc emb, R=nDCG)  +  λ * L_GRPO(listwise_prompt emb vs doc emb, R=nDCG)
    ↑ retrieval: 直接优化 ranking metric         ↑ reranking: 直接优化 ranking metric
```

**为什么比 Hybrid (InfoNCE + GRPO) 更好：**
1. 统一的训练范式，method section 更 clean
2. 两侧都能直接优化 non-differentiable ranking metrics
3. E2Rank ablation 已证明 ranking signal 是核心价值 (Table 6: 去掉 RankNet 导致最大性能下降)

#### Step 4.2: 两侧 GRPO 的设计差异

虽然都用 GRPO，两侧在 reward 构造上有区别：

```python
def train_step_e2rank(self, batch):
    queries, listwise_prompts, documents, labels = batch
    
    # ---- Document embeddings: encode once, shared by both sides ----
    with torch.no_grad():
        h_docs = self.model.encode(documents)          # [B, n_doc, d]
    
    # ---- Query side GRPO (retrieval) ----
    # Input:      query text → h_q → perturb → e_q
    # Candidates: 1 pos + 15 neg (from training data)
    # Reward:     nDCG@10 (retrieval-oriented)
    query_loss = self.grpo_step(
        inputs=queries,
        candidates=h_docs,
        labels=labels,
        reward_fn=self.ndcg_reward,
        reward_K=10                 # retrieval 评 top-10
    )
    
    # ---- Listwise prompt side GRPO (reranking) ----
    # Input:      listwise prompt (query + top-K docs) → h_lp → perturb → e_lp
    # Candidates: same documents (embeddings 复用, 不需要重新 encode)
    # Reward:     nDCG@K on full candidate list (reranking 侧可以用更大 K)
    rerank_loss = self.grpo_step(
        inputs=listwise_prompts,
        candidates=h_docs,
        labels=labels,              # LLM ranking labels (graded relevance)
        reward_fn=self.ndcg_reward,
        reward_K=16                 # reranking 评 top-16 (= 1 pos + 15 neg)
    )
    
    loss = query_loss + self.lambda_rerank * rerank_loss
    return loss
```

**两侧的关键区别：**

| | Query 侧 (retrieval) | Listwise Prompt 侧 (reranking) |
|---|---|---|
| Input | query text | query + top-K docs (listwise prompt) |
| Action | query embedding | listwise prompt embedding |
| Candidates | 1 pos + 15 neg | 同上 (doc embeddings 复用) |
| Reward K | nDCG@10 | nDCG@16 (或 nDCG on full list) |
| Relevance labels | binary (pos=1, neg=0) | graded (from LLM ranking) |
| 作用 | 提升 retrieval 质量 | 提升 reranking listwise 排序 |

#### Step 4.3: 对比实验设计

**Ablation — 拆解两侧 GRPO 各自的贡献：**

| Method | Query 侧 | Reranking 侧 | DL19 | DL20 | BEIR | BRIGHT |
|--------|----------|-------------|------|------|------|--------|
| E2Rank (original) | InfoNCE | RankNet | 70.84 | 70.15 | 52.09 | 31.0 |
| GRPO-query only | **GRPO** | RankNet | ? | ? | ? | ? |
| GRPO-rerank only | InfoNCE | **GRPO** | ? | ? | ? | ? |
| **Full GRPO (ours)** | **GRPO** | **GRPO** | ? | ? | ? | ? |

这个 ablation 可以清楚地展示：
- GRPO-query only vs E2Rank → RL 对 retrieval 侧的贡献
- GRPO-rerank only vs E2Rank → RL 替代 RankNet 对 reranking 的贡献
- Full GRPO vs 两个 single side → 两侧是否有 synergy

#### Step 4.4: Reranking Evaluation (BEIR + TREC DL)

```bash
# BEIR reranking
python src/eval.py \
    --model checkpoints/rl-e2rank-0.6b \
    --rank-method listwise \
    --datasets covid nfcorpus touche dbpedia scifact signal news robust04 \
    --retriever bm25 \
    --topk 100 \
    --save-to "./results/rerank/rl_results.jsonl"

# TREC DL
python src/eval.py \
    --model checkpoints/rl-e2rank-0.6b \
    --rank-method listwise \
    --datasets dl19 dl20 \
    --retriever bm25 \
    --topk 100

# BRIGHT reranking
python src/eval.py \
    --model checkpoints/rl-e2rank-0.6b \
    --rank-method listwise \
    --datasets bright \
    --retriever reasonir \
    --topk 100
```

#### Step 4.5: MTEB Embedding Preservation Check

确认 reranking training 没有损害 embedding 质量（E2Rank 论文的一个 key finding 是 ranking training 反而提升了 MTEB）：

```bash
bash eval_mteb/scripts/run_mteb.sh checkpoints/rl-e2rank-0.6b rl-e2rank-0.6b "MTEB(eng, v2)"
```

---

### Phase 5: Analysis & Writing (2 周)

#### Step 5.1: Uncertainty Analysis

σ（perturbation scale）在训练过程中的变化，以及不同 query 的 σ 和 difficulty 的关系。

如果使用 input-dependent σ（后续扩展），可以分析：
- Easy query (明确的 factual query) vs Hard query (ambiguous/reasoning query) 的 σ 分布
- σ 和 retrieval performance 的 correlation

#### Step 5.2: RL vs CL Embedding Space 对比

- t-SNE / UMAP visualization of embedding space
- RL-trained vs CL-trained embeddings 的 anisotropy 对比
- Nearest neighbor analysis: 哪些 documents 被"移动"了？

#### Step 5.3: Reward Design Ablation

- Contrastive reward only
- nDCG reward only
- MRR reward only
- Mixed reward with different λ

#### Step 5.4: Hyperparameter Sensitivity

- σ: [0.01, 0.02, 0.05, 0.1, 0.2]
- G (group size): [2, 4, 8, 16]
- nDCG cutoff K: [5, 10, 20]

#### Step 5.5: Why RL > CL? Deep Analysis

这是 paper 的核心 analysis section，需要回答 reviewer 最可能问的问题：

- **Case study:** 找出 RL 比 CL 提升最大的 query，分析它们的特点（是否是 ambiguous / multi-intent / reasoning-intensive）
- **Reward landscape:** 在 embedding space 中可视化 nDCG reward vs InfoNCE loss 的 landscape，展示两者的 misalignment
- **Per-query improvement distribution:** RL 是均匀提升所有 query，还是在某些 query 上大幅提升、某些下降？

#### Step 5.6: Dual GRPO Synergy Analysis (Layer 2)

分析 query 侧和 listwise prompt 侧的 GRPO 是否有 synergy：

- 两侧 GRPO 各自让 embedding space 发生了什么变化？
- Query embedding 和 listwise prompt embedding 在 RL 训练后的关系变化
- 是否存在 conflict：query 侧优化 retrieval 会损害 reranking 侧？反之亦然？

---

## 3. 关键技术决策

### 3.1 Normalization after Perturbation

```python
# 方案 A: 先 perturb 再 normalize (推荐)
e = F.normalize(h + sigma * epsilon, dim=-1)

# 方案 B: 在 unnormalized space 加 noise
e = h + sigma * epsilon  # 不 normalize

# 方案 C: 在 angular space 加 noise
# 将 h 投影到超球面，在切平面上加 noise
```

推荐方案 A，因为 cosine similarity 是在 normalized space 上计算的。

### 3.2 Sigma 的设计

```python
# V1 (Minimal): Global scalar sigma
self.sigma = nn.Parameter(torch.tensor(0.05))  # 1 个参数

# V2 (Per-dimension): Diagonal sigma
self.log_sigma = nn.Parameter(torch.zeros(d))  # d 个参数

# V3 (Input-dependent): sigma head
self.sigma_head = nn.Linear(d, 1)  # 每个 input 有自己的 sigma

# V4 (Low-rank): 只在 k 维子空间 explore
self.A = nn.Parameter(torch.randn(d, k) * 0.01)  # d*k 个参数
# e = h + sigma * A @ epsilon,  epsilon ~ N(0, I_k)
```

**Pilot 阶段用 V1 (global scalar)**，如果 work 再尝试 V3。

### 3.3 和 InfoNCE 的关系

RL loss 和 InfoNCE loss 不是互斥的。根据实验阶段有不同的组合方式：

```python
# Layer 1: Pure Embedding (Phase 3)
# Stage I: 标准 InfoNCE training (已有 E2Rank checkpoint)
# Stage II: 纯 GRPO loss (用 RL 替代 continued CL)
loss = grpo_loss_query

# Layer 2: E2Rank Unified Retrieval+Reranking (Phase 4)
# Stage I: 标准 InfoNCE training (已有 E2Rank checkpoint)
# Stage II: 两侧 GRPO，完全替代原始的 InfoNCE + RankNet
loss = grpo_loss_query + lambda * grpo_loss_listwise_prompt
```

两个 Layer 的 Stage I 都复用 E2Rank 的 Embedding-Only checkpoint，差异在 Stage II。
Layer 2 的设计比原始 E2Rank 更 clean：原来是 InfoNCE + RankNet 两个不同的 supervised proxy，
现在统一为 GRPO + nDCG reward 一个 RL paradigm。

### 3.4 训练效率

每个 training step 的计算量：
- 标准 CL: 1 forward (query) + 1 forward (docs) + 1 backward
- 我们的 GRPO: 1 forward no-grad (query) + 1 forward no-grad (docs) + **G 次 reward 计算** + 1 forward with-grad (query) + 1 backward

关键是 **G 次 reward 计算不涉及额外的 forward pass**（只是 cosine + nDCG），非常快。
所以实际开销 ≈ 2x 标准 CL（多一次 no-grad forward），可接受。

---

## 4. 实验配置

### 4.1 Hardware

- Pilot (Phase 1-2): 单卡 A100 80G，Qwen3-0.6B + LoRA
- Full (Phase 3): 4-8 卡 A100，Qwen3-0.6B/4B/8B

### 4.2 Base Models

| Model | 用途 |
|-------|------|
| `Alibaba-NLP/E2Rank-0.6B-Embedding-Only` | Stage I checkpoint, RL fine-tune 起点 |
| `Alibaba-NLP/E2Rank-0.6B` | E2Rank full model, Stage II (RankNet) baseline |
| `Alibaba-NLP/E2Rank-4B-Embedding-Only` | 4B 版本 |
| `Alibaba-NLP/E2Rank-8B-Embedding-Only` | 8B 版本 |

### 4.3 Training Data

直接复用 E2Rank 的 Stage II 数据：
```bash
mkdir data
hf download Alibaba-NLP/E2Rank_ranking_datasets train.jsonl --local-dir ./data/ --repo-type dataset
```
约 87k 样本，每个 query 配 1 positive + 15 negatives + Qwen3-32B 的 ranking label。

### 4.4 Evaluation Benchmarks

**Layer 1: Pure Embedding (Phase 3)**

| Benchmark | Metrics | 用途 |
|-----------|---------|------|
| MTEB (eng, v2) | Various (7 task types) | General embedding, ablation 用 |
| MTEB (eng, v1) | Various (56 datasets) | General embedding, final results |
| BEIR retrieval (dense retrieval) | nDCG@10 | Retrieval quality |
| BRIGHT retrieval (dense retrieval) | nDCG@10 | Reasoning-intensive retrieval |

**Layer 2: E2Rank Integration (Phase 4)**

| Benchmark | Metrics | 用途 |
|-----------|---------|------|
| BEIR reranking (top-100) | nDCG@10 | General reranking |
| BRIGHT reranking (top-100) | nDCG@10 | Reasoning-intensive reranking |
| TREC DL19/DL20 | nDCG@10 | General reranking |
| End-to-end (retrieval + rerank) | nDCG@10 | Unified search |

---

## 5. 风险评估与应对

| 风险 | 可能性 | 影响 | 应对 |
|------|--------|------|------|
| RL 比 InfoNCE 没有 improvement | 中 | 高 | 关键在于 non-differentiable reward (nDCG)；如果 cosine reward 没有 improvement 是预期的 |
| 训练不稳定 | 中 | 中 | Stage I warm-up 缓解；减小 σ；增大 G；添加 KL penalty |
| 高维 variance 太大 | 低 | 中 | 用 low-rank perturbation (V4)；减小 σ |
| nDCG reward 太 sparse | 中 | 中 | 用 graded relevance (方案 B)；尝试 mixed reward |

---

## 6. Timeline

| Week | Phase | Task | Deliverable |
|------|-------|------|-------------|
| 1 | Phase 1 | 实现 reward.py + grpo_trainer.py | 核心代码 |
| 2 | Phase 1 | 实现 train_rl.py, pilot on SciFact (cosine reward) | Pilot 结果: training loss 下降 |
| 3 | Phase 2 | 切换 nDCG reward, BEIR subset 对比实验 | RL vs CL 初步对比 |
| 4 | Phase 3 | MTEB full evaluation (Layer 1 核心实验) | MTEB 结果表 |
| 5 | Phase 3 | BEIR/BRIGHT retrieval eval + scale to 4B | Layer 1 完整结果 |
| 6 | Phase 4 | E2Rank integration + reranking eval (Layer 2) | Layer 2 结果表 |
| 7 | Phase 5 | Baselines + ablation + analysis | Analysis figures |
| 8 | Phase 5 | Paper writing | Draft |