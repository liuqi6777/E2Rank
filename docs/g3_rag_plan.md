# G3-R2: RAG query-encoder experiments

Working log for the RAG (G3) query-encoder rounds on the frozen wiki18_100w
index. Round 1 (before this document) trained every arm for 200 steps and lost
to its own initialization; rounds 2–4 here diagnose that, repair what is
repairable, and establish what actually moves the number.

**Status as of 2026-09-20 (evening).** Best result: `G3-R2-RL-GradedNDCG-Anchor050`
(dynamic full-corpus RL, graded nDCG@10, E0-anchor coef 0.5, full LR) at
training-domain +0.0203, held-out **+0.0002**, **macro +0.0059** `answer_mrr@10`
— 3× the previous best (LRHalf +0.0018) with the in-domain gain intact. The
anchor penalty buys the round-3 in-domain gain at essentially zero held-out
cost, dominating the LR knob at every point of the dose-response (§6.4). The
plain-LR winner's +0.0010 macro replicates on seed 3407. Generation metrics
are filled for every arm: the CL family loses end-to-end (macro EM −0.015 to
−0.019, §6.2) and CL-Graded is a mathematical duplicate of CL-AnswerMasked,
while the RL family holds EM at E0 level or above (§6.4) — though EM's seed
noise (~±0.007) cannot resolve the anchor's retrieval-side advantage at n=1.
The CL+anchor ablation (§6.2) shows only ~14% of CL's held-out loss is drift:
the rest is the re-ranking objective itself. Single-seed caveat: the anchor
arms ran on seed 42 only; the round-3 seed noise band is ±0.0012 per column,
and +0.0059 is ~5× that band.

Contents: [1](#1-round-1-was-a-negative-result) background ·
[2](#2-root-cause-diagnosis-and-its-limits) diagnosis ·
[3](#3-round-2-design) round-2 design · [4](#4-round-3-design) round-3 design ·
[5](#5-execution-and-verification) execution ·
[6](#6-results) results · [7](#7-instrumentation-notes) instrumentation ·
[8](#8-further-work) further work.

## 1. Round 1 was a negative result

Every trained G3 arm finished at or below its own initialization on the fixed
7-dataset QA suite (51,713 queries, `src/eval_rag.py`, macro average over
datasets):

| run | answer_recall@5 | answer_mrr@10 | generator_em | generator_token_f1 |
|---|---:|---:|---:|---:|
| **G3-E0** (untrained Qwen3-Embedding-0.6B) | **0.4928** | **0.3976** | 0.3230 | 0.4076 |
| G3-CL (InfoNCE) | 0.4624 | 0.3747 | 0.3086 | 0.3941 |
| G3-RetRL-MRR | 0.4822 | 0.3873 | 0.3147 | 0.4006 |
| G3-RetRL-NDCG | 0.4885 | 0.3918 | 0.3256 | 0.4087 |

Held-out (the 5 non-training datasets) degrades further than the training
domain: E0 `answer_mrr@10` 0.3673 -> CL 0.3370. Training made the retriever
worse, and worse fastest where it had not seen data.

Sources: `data/rag/eval/{g3_e0,g3_cl_s42,g3_retrl_mrr_s42,g3_retrl_ndcg_s42}/summary.json`.

## 2. Root-cause diagnosis (and its limits)

The mined candidate manifest carries three independent judgments per passage.
Measured over the first 3,000 queries of
`data/rag/candidates/nq_hotpotqa_train.jsonl`:

| judgment | source | mean per query (of 1000 candidates) | mean inside frozen top-10 |
|---|---|---:|---:|
| `training_positive_mask` | DPR human positives / HotpotQA supporting facts | 1.01 | 0.64 |
| `evidence_group_ids` non-empty | multi-hop supporting facts | 1.01 | — |
| `answer_positive_mask` | passage text contains a golden answer | 56.82 | 2.03 |

36.4% of queries have **no** `training_positive_mask` passage anywhere in the
frozen top-10 (all queries do have at least one somewhere in the depth-1000
list, so InfoNCE never lacks a numerator).

`src/eval_rag.py:180-184` scores the third judgment and nothing else:

```python
"answer_recall_at_5": float(any(answer_mask[:5])),
"answer_mrr_at_10": _first_relevant_rank(answer_mask, 10),
```

Round-1 training supervised on the first judgment and treated everything else
as a negative. So on an average query the objective pushed **2.03 passages out
of the top-10 that the metric counts as correct**, against **0.64** it pushed
up — at temperature 0.03, where the softmax is sharp. The negative result is a
label/metric mismatch, not an optimization or budget failure.

This also explains the held-out asymmetry: NQ/HotpotQA have real qrels to
partially offset the damage, the other five datasets only ever get graded by
answer containment.

**Honest postscript (§6.2):** this diagnosis was directionally right and
quantitatively small — repairing the labels bought +0.002 macro. The failure
that actually mattered was a train/inference universe mismatch (§6.2), which
the label repair cannot touch.

## 3. Round-2 design

### 3.1 Relevance schemes (`src/rag/relevance.py`)

`rag_relevance_scheme` selects how the three judgments become supervision:

| scheme | positives | contrastive denominator | nDCG gains |
|---|---|---|---|
| `binary` | qrel | everything (round-1 behaviour) | qrel=1 |
| `answer_masked` | qrel | answer-bearing non-qrel passages **removed** | qrel=1 |
| `graded` | qrel ∪ evidence | answer-bearing non-positives removed | qrel=3, evidence=2, answer=1 |

`answer_masked` is the minimal repair: it stops demoting answer-bearing
passages without promoting them (answer containment is noisy — a golden answer
of "seven" matches a lot of Wikipedia). `graded` is the more aggressive one and
also supplies the reward G1-R2 found to be the only non-degenerate RL signal:
graded nDCG kept `degenerate_frac` at 0.3% where binary MRR drifted to 32%
(`docs/rollout_variance_discussion.md`).

The two masks are deliberately distinct. Under `graded` an answer-bearing
passage is a bad InfoNCE negative but a perfectly good gain-1 document for
nDCG, so it is dropped from `denominator_mask` and kept in `scoring_mask`.

### 3.2 CL-Strong negative pool (`src/rag/negatives.py`)

G1's pool-source ablation was worth about +1.0 nDCG going from device-local
representatives to a cross-device pool of every query's full candidate list.
RAG gets this more cheaply than G1: corpus vectors are frozen, so a pooled
candidate carries no gradient and a plain `all_gather` suffices instead of G1's
autograd-aware gather.

Note this is **not** G1's CL-Strong. G1's "strong" variant also let cross-query
documents receive real gradient (`baseline_detach_in_batch_documents: false`),
which is what made it strong. RAG cannot do that — the document side is a
frozen index. The RAG arm varies pool breadth only, so it should be expected to
recover less than G1's +4.9.

Every pooled ordinal is checked against the borrowing query's own judged set
and masked on a hit, so the pool cannot reintroduce the false negatives §2 is
about.

### 3.3 One-epoch budget

Round 1 ran 200 steps = 0.235 epoch, a budget set by `G3-AnsRL`'s generator
cost (128 x 32 = 4,096 generations per step) and shared by all four arms. No
answer-reward arm is in this round, so the retrieval arms are freed: `steps: -1`
selects `training_budget: one_epoch`, which is 807 steps over the 103,293-query
train split at global batch 128.

### 3.4 In-training probes

Two probes log during training; they are not interchangeable, and round 2
proved the first one unfit for model selection (§6.2).

**`src/rag/tuning_eval.py` (re-ranking probe).** Round 1 used `rag_split: full`
and had no in-training evaluation at all. Round 2 uses `rag_split: train`,
holding out the deterministic 5% tuning split, and logs answer recall/MRR on it
at every save point. It re-ranks each tuning query's stored depth-1000
candidate list and reads the stored `answer_positive_mask` — deliberately not a
live ANN search plus corpus text lookup, since corpus rows cost ~38 ms each on
JuiceFS and the offline variant would need over half an hour. Consequence: the
numbers are re-ranking metrics bounded by the frozen index's recall@1000; they
sit well above the headline eval.

**`src/rag/retrieval_probe.py` (retrieval probe).** Added after round 2
exposed the re-ranking probe's blind spot (§7): a real ANN search over a fixed
672-query slice of the evaluation suite, reporting in-domain (nq/hotpotqa) and
held-out scopes separately, with a step-0 baseline via `on_train_begin`.

## 4. Round-3 design

### 4.1 Dynamic full-corpus RL: the reward ablation

Round 2 established that the CL failure is a train/inference universe mismatch
(§6.2), so the RL sweep varies reward on the **dynamic** path — the only one
that optimizes the operation that actually runs at inference — and keeps one
static arm as the contrast.

| run | algorithm | reward | labels |
|---|---|---|---|
| G3-R2-RL-GradedNDCG | dynamic, score-function | nDCG@10 | graded 3/2/1 |
| G3-R2-RL-BinaryNDCG | dynamic, score-function | nDCG@10 | binary qrels |
| G3-R2-RL-MRR-Graded | dynamic, score-function | MRR@10 | graded relevant set |
| G3-R2-ShortRL-CP-Graded | static slate, conditional projection | nDCG@10 over slate 20 + shortlist 15 | graded 3/2/1 |

G1-R2's ordering on its own data was graded nDCG > binary nDCG > MRR, driven by
rollout degeneracy (binary MRR reached 32% `degenerate_frac`, graded nDCG 0.3%).
RAG's graded arm sat at ~10%, so the margin here should be smaller; these arms
measure it rather than assuming G1's answer transfers to a different corpus and
a different label source.

Signals watched: `reward/mean` climbing, `advantages/degenerate_frac` staying
low, and for the CP arm `projection/query_span_rank_mean` staying well under
1024 (saturation means the projector is the identity and CP has degenerated
into score-function).

### 4.2 Static-slate RL and conditional projection (`src/rag/shortlist_rl.py`)

The shipped RAG RL path searches the whole corpus per rollout, so it is stuck
on the score-function estimator: the reward depends on every passage that could
cross the top-k boundary, and projecting onto the returned top-k would delete
directions the reward responds to, making conditional projection *biased*
rather than lower-variance (`paper/METHOD_REDESIGN.md:133`, gated at
`src/config.py:122`). G1 measured that configuration gap at 17.52 (dynamic
query-only) vs 20.72 (static joint CP).

Ranking a fixed slate removes 4,096 ANN searches per step and makes the reward
a deterministic function of an enumerable set, which is exactly CP's
precondition.

**Slate width is a correctness parameter, not a tuning knob.** CP's variance
reduction is proportional to `rank(span)/D`. With D=1024, projecting onto a
depth-1000 candidate list saturates the span, the projector becomes the
identity, and CP silently *is* the score-function estimator. Slate 20 +
shortlist 15 + the mean spans ~36 columns, matching G1's recipe.
`projection/query_span_rank_mean` is logged so this stays observable;
`tests/test_rag_shortlist_rl.py` pins the degenerate case by asserting CP's
loss and gradient are identical to SF's at full rank.

Two arms, differing only in the estimator (only the CP one has run):

| run | estimator | slate | shortlist | alignment |
|---|---|---:|---:|---:|
| `G3-R2-ShortRL-CP-Graded` | conditional_projection | 20 | 15 cross-device | 0.70 |
| `G3-R2-ShortRL-SF-Graded` | score_function | 20 | 15 cross-device | 0.70 |

Alignment 0.70 and K=15/T=1 are inherited from G1's sweep
(`paper/G1_SHORTLIST_RESULTS.md`), not re-derived here. Note `frozen_doc_rescale`
has no analogue: it is inert whenever no document action is sampled
(`src/grpo.py:765-767`), and RAG's documents are always frozen means, so the
−1.01 CP ablation G1 measured cannot arise.

### 4.3 Round 4: the E0-anchor penalty (`src/rag/anchors.py`)

The winning round-3 arm still pays −0.0072 held-out for +0.0214 in-domain —
drift away from the frozen document space. The anchor penalty adds
`rag_anchor_coef * mean(1 - cos(e_theta(q), e_E0(q)))` to the loss. Anchors are
precomputed once with the initial weights into a shared fp16 memmap keyed by
dataset row order, with a metadata JSON pinning the manifest hash, model
revision, row count and width; a mismatched cache is rebuilt rather than
reused. Coefficients 0.1 and 0.5 give the dose-response, a half-LR arm is the
code-free control for "less movement overall", and a seed replication follows
the repo's three-seed convention.

**The penalty is the RLHF KL term, in this parameterization.** Both have the
shape `task_reward − coef · divergence(current, frozen reference)`: RLHF
penalizes `KL(π_θ ‖ π_ref)` against a frozen SFT reference, we penalize the
rotation of each query embedding against its frozen E0 vector. The reason both
regularizers exist is the same: the trainable objective is a proxy that is
only trustworthy in a neighborhood of the reference. The reward model is
calibrated on the reference policy's outputs, so straying far rewards hacking
its blind spots; our nDCG reward only grades the training queries' retrieval
through an index built in E0's language, so the reward cannot see the damage
the same update does to every other query's embedding. Both terms say: improve
locally, do not chase the proxy into regions where it lies.

The correspondence is exact, not analogical, in this setup. The exploration
policy is vMF around the current query embedding, with the concentration
resolved from the suite's `target_alignment: 0.80` (κ ≈ 2274 at d = 1024, via
`policy_math.kappa_for_alignment`) and held fixed. The KL between two vMF
distributions that share κ closes as

    KL(vMF(m_θ, κ) ‖ vMF(m_E0, κ)) = κ · A(κ) · (1 − cos θ) = 1819 · (1 − cos θ)

where A(κ) is the mean resultant length — which here *is* the target
alignment, 0.80, by construction. Since a vMF law is determined by its mean
direction, constraining the embedding's rotation angle *is* constraining the
action distribution's KL from the E0 reference policy: the anchor is the GRPO
KL-to-reference term written in closed form, and coef 0.5 corresponds to a
per-query KL coefficient of 0.5/1819 ≈ 2.7e−4. Writing it as `1 − cos` rather
than a literal KL buys boundedness ([0, 2]), no expectation estimation, and a
metric that measures exactly what normalized ANN search cares about —
direction.

Two disanalogies with RLHF are worth keeping straight. First, the failure
being prevented: RLHF's KL guards against reward-model over-optimization on
the *policy's own outputs*; the anchor guards against collateral damage
through parameter sharing — updates that improve the training queries drag
held-out queries' embeddings off the index manifold, and the anchor on
training queries bounds that drag only via the encoder's smoothness (which is
why the round-4 result is an empirical validation, not a foregone conclusion).
Second, the geometry: KL is an asymmetric, unbounded divergence on
distributions; the anchor is a bounded, symmetric rotation metric on points —
equivalent here only because the action distribution collapses onto its mean
direction.

## 5. Execution and verification

### 5.1 Suite and arms

Suite `configs/experiments/iclr2027/suite_g3_r2.yaml`, budget
`configs/experiments_g3_r2.yaml`, driver `scripts/run_g3_r2.py`.
Output root `checkpoints/iclr2027-g3-r2/`. Seed 42 unless noted, 8xH800,
LR 5e-6, global batch 128, one epoch, full fine-tune of the query encoder only.

Round-2 arms (label/pool factorial plus the first RL arm):

| run | objective | scheme | pool | purpose |
|---|---|---|---|---|
| `G3-R2-CL-Binary` | infonce | binary | — | control: round-1 recipe at a full epoch. If this alone fixes the regression, §2 is wrong. |
| `G3-R2-CL-AnswerMasked` | infonce | answer_masked | — | the minimal label repair |
| `G3-R2-CL-Graded` | infonce | graded | — | graded labels, evidence promoted |
| `G3-R2-CL-Strong-AnswerMasked` | infonce | answer_masked | cross-device, K15 | isolates pool breadth against AnswerMasked |
| `G3-R2-RL-GradedNDCG` | rl | graded | — | graded nDCG@10, G32, alignment 0.80 |

Round-4 arms:

| run | anchor coef | LR | seed |
|---|---:|---:|---:|
| `G3-R2-RL-GradedNDCG-Anchor010` | 0.1 | 5e-6 | 42 |
| `G3-R2-RL-GradedNDCG-Anchor050` | 0.5 | 5e-6 | 42 |
| `G3-R2-RL-GradedNDCG-LRHalf` | 0 | 2.5e-6 | 42 |
| `G3-R2-RL-GradedNDCG-Seed3407` | 0 | 5e-6 | 3407 |

### 5.2 QS execution

Job **294087** (`g3-r2-rag-0919`), queue `train_agent`, cluster 50
(alsh1-rdma-gpu-prod, H800 bare metal), resource pack 154 (8 GPU / 180C /
1900Gi), PytorchJob, single node, elastic (`is_overuse: 1`), `restart_num: 3`,
image `redaccel:0.12.1-gpu`. One trial per run via the entry script, which
smoke-tests three steps on 256 queries before committing to the epoch, then
trains, then runs the retrieval-only QA evaluation inline while the GPUs are
still allocated.

Operational notes worth keeping:

- **`--overuse` must be passed on the trial call, not only on the job call.**
  `qs training create --from-job-id ...` does not inherit the job's elastic
  setting; the first batch of trials came out dedicated and had to be
  recreated. Verify with
  `qs training get <trial> -o json -q | python3 -c "import json,sys; print(json.load(sys.stdin)['param']['enable_overuse'])"`.
- **The QS `trial_status` field lags badly.** It reported `Uncommit` for trials
  that had been training for hours; acting on it cost one 6-hour RL run
  (killed at step 600/806). Always corroborate with the filesystem (new files
  under the output root) before concluding anything about liveness, and never
  stop a trial whose output dir has been touched recently.
- **Entry scripts must pass the mode explicitly**:
  `source scripts/cluster_setup.sh install`. A bare `source` inherits the
  caller's positional parameters, and after the entry script's `shift` a
  trailing flag (or a RUN_ID) is read as cluster_setup's mode and aborts with
  its usage message. This bit three trials before being fixed.
- W&B runs offline unless `WANDB_API_KEY` is present in the trial environment;
  all metrics are also printed to the trial log as `[rag-tuning-eval] ...` /
  `[rag-retrieval-probe] ...` and written under each run's output dir.

Queue latency was 4–7 hours per batch (placement happens when a whole 8-GPU
node frees up); training itself is 20–40 min per arm plus ~25 min of
retrieval-only evaluation.

### 5.3 Pre-flight verification

The collectives are the expensive thing to get wrong: a shape mismatch in
`all_gather` does not raise, it deadlocks, and a deadlock costs the whole 8-GPU
slot. `tests/test_rag_distributed_cpu.py` runs the real code over gloo on CPU
across three ranks and checks that

- the cross-device pool has width `world_size x batch x pool_size`, contains
  foreign candidates, and never lets a query borrow from its own row;
- the cross-device InfoNCE forward/backward is finite;
- both shortlist-RL estimators run distributed;
- the tuning probe reduces to bit-identical metrics on every rank over a
  deliberately ragged 11-row split across 3 ranks.

Plus single-process suites: `tests/test_rag_relevance.py` (7),
`test_rag_supervised_forward.py` (4), `test_rag_shortlist_rl.py` (11),
`test_rag_retrieval_probe.py` (6), `test_rag_anchor.py` (7). Each QS trial
additionally smoke-tests three real optimizer steps on 256 queries before
committing to the epoch.

## 6. Results

All QA numbers are `src/eval_rag.py` on the 7-dataset suite, `answer_mrr@10`,
deltas against the untrained E0 reference **computed per scope** (E0's macro
0.3976, training-domain 0.4734 and held-out 0.3673 differ a lot; comparing any
scope against the macro number inflates its delta by ~0.076 — this exact
mistake was made once and corrected).

### 6.1 E0 reference line and headroom (measured, CPU only)

`scripts/rag_tuning_reference.py`, held-out tuning split, 5,471 queries (2,534
hotpotqa / 2,937 nq). "E0" is the frozen retrieval order, i.e. what
`rag/tuning_eval.py` reports at step 0; "oracle" is a perfect re-ranking of the
same depth-1000 pool.

| metric | hotpotqa | nq | macro |
|---|---:|---:|---:|
| E0 answer_recall@5 | 0.5825 | 0.7719 | 0.6772 |
| E0 answer_recall@20 | 0.6863 | 0.8863 | 0.7863 |
| **E0 answer_mrr@10** | 0.4723 | 0.6173 | **0.5448** |
| E0 qrel_mrr@10 | 0.6983 | 0.3825 | 0.5404 |
| oracle answer_recall@5 | 0.9811 | 0.9871 | 0.9841 |
| **oracle answer_mrr@10** | 0.9811 | 0.9871 | **0.9841** |
| answer-bearing candidates / query | 63.53 | 54.95 | 59.24 |
| qrel positives / query | 2.05 | 1.01 | 1.53 |

Two readings, both load-bearing.

**There is headroom.** Perfect re-ranking of the pool the model already sees
would take answer_mrr@10 from 0.5448 to 0.9841. Round 1's regression was not a
ceiling effect. Note these are re-ranking numbers over depth-1000 and sit well
above the full-corpus figures in §1; they are the scale the re-ranking probe
reports, not the scale `eval_rag.py` reports.

**The two label sets disagree, in opposite directions per source.** On NQ,
qrel_mrr@10 is 0.3825 while answer_mrr@10 is 0.6173; on HotpotQA the ordering
flips (0.6983 vs 0.4723). So maximizing the qrel ranking on NQ actively
reorders *away* from the answer-optimal ordering, and the two training sources
pull the shared encoder in opposite directions. This is independent
confirmation of §2 from the ranking side rather than the counting side.

### 6.2 Round 2 — contrastive training on a frozen candidate pool does not transfer

All five arms completed one epoch (806 steps) on 2026-09-19:

| run | training domain (nq+hotpotqa) | held-out (5 datasets) | macro | probe @800 |
|---|---:|---:|---:|---:|
| G3-E0 | 0.4734 (0.5913) | 0.3673 (0.4534) | 0.3976 (0.4928) | 0.5448 |
| CL-Binary | 0.4706 (−0.0028) | 0.3330 (**−0.0343**) | 0.3723 (−0.0253) | 0.5674 |
| CL-AnswerMasked | 0.4726 (−0.0008) | 0.3344 (**−0.0329**) | 0.3739 (−0.0237) | 0.5718 |
| CL-Graded | −0.0008 | −0.0329 | −0.0237 | 0.5718 |
| CL-Strong-AnswerMasked | 0.4729 (−0.0005) | 0.3344 (**−0.0329**) | 0.3740 (−0.0236) | 0.5716 |

**The §2 label diagnosis holds but is negligible.** `answer_masked` beats
`binary` on every column, so the false-negative argument is real — by +0.0020
macro. Graded labels land exactly on AnswerMasked (0.3739 vs 0.3739), so the
grading machinery matters for the RL reward, not for InfoNCE. The cross-device
pool adds +0.0001. Neither approaches the −0.024 gap.

**CL-Graded is a mathematical duplicate of CL-AnswerMasked, not a replicate.**
The two runs' 806-step loss curves are byte-identical. Cause, verified against
the data: `multi_positive_infonce_loss` consumes only `positive_mask` and
`denominator_mask`, never the graded 3/2/1 `labels`, and on this candidate
pool evidence passages are *always* qrel positives (0 of 20 000 sampled rows
have an evidence candidate outside the qrel set), so the graded view's masks
collapse onto answer_masked's exactly. Two consequences: the round-2 CL suite
effectively ran three arms, not four; and the evidence=2 tier of the graded
scheme is inert on this data *everywhere* — the RL graded reward is in practice
a two-tier 3/1 scheme, which is why it still differs from binary. The
graded-vs-binary effect in RL comes from the 3-vs-1 contrast, not from the
evidence tier.

**The generation half confirms the retrieval half (backfill 2026-09-20,
trial 2088954).** With vLLM co-located on GPUs 0–3 and the eval pinned to 4–7,
`generator_em`/`generator_token_f1` filled for all four CL arms: macro EM
−0.0147 to −0.0187 vs E0 (transmission ≈0.6× of the MRR loss), held-out EM
−0.017 to −0.021 vs in-domain −0.009 to −0.013 — same shape as the retrieval
side, so the CL regression is end-to-end, not a retrieval-metric artifact.
Arm differences on EM span 0.004 (noise); Strong-AnswerMasked is nominally
best. The graded duplicate is identical here too, as it must be.

**Training buys nothing, even in-domain.** On the two sources it trains on,
`answer_mrr@10` is flat (−0.0008) and `answer_recall@5` is *down* 0.0209. On
the five it does not see, both fall hard. Round 1 at 200 steps landed at
training 0.4689 / held-out 0.3370 / macro 0.3747; round 2 ran four times longer
with repaired labels and a wider negative pool and landed at 0.4726 / 0.3344 /
0.3739. Four levers, no movement. This is not a budget, label, or negative-pool
problem.

**The mechanism is a train/inference universe mismatch.** The probe rises
(0.5448 → 0.5718) while full-corpus retrieval does not: the model genuinely
gets better at *reordering E0's top-1000*, and that skill does not transfer to
ANN search over 21M passages. Training only ever sees E0's candidate list as
the universe, so nothing in the loss penalizes moving the query embedding
somewhere that pulls in different — and worse — documents from the other
~21M. Re-ranking within a frozen pool is simply not the operation that runs at
inference.

**The anchor ablation quantifies the split (2026-09-21, trial 2092426).**
`G3-R2-CL-AnswerMasked-Anchor050` — the round-2 recipe under the round-4
winning anchor dose — lands at in-domain +0.0039 / held-out −0.0284 / macro
−0.0192, against the unanchored arm's −0.0008 / −0.0329 / −0.0237. The anchor
recovers only ~14% of CL's held-out loss (and turns in-domain slightly
positive), so the regression is ~86% objective mismatch and only ~14% drift.
Set against the RL side, where the same penalty recovered ~100% of the
held-out loss (§6.4), the anchor acts as a decomposition probe: when the
training objective *is* the inference operation (dynamic RL: real ANN search),
held-out damage is pure drift and removable; when it is not (CL: re-ranking a
frozen pool), the damage is intrinsic to the objective and anchoring cannot
buy it back.

The generation half of the ablation (backfilled 2026-09-21, run locally on
the freed 8×L20Y machine after the cluster queue stalled) mirrors the split:
macro EM recovers from −0.0149 to −0.0123 — ~17% of the gap, against
retrieval's ~14% — with the in-domain EM recovering more (−0.0086 → −0.0045)
than held-out (−0.0174 → −0.0154). The anchor's small rescue concentrates
where the drift was; the bulk of the end-to-end regression belongs to the
objective.

Two consequences followed. First, the re-ranking probe is unfit for model
selection — held out in queries but in-domain in distribution, and measuring
re-ranking rather than retrieval, it reported a clean +0.027 win for runs that
were 0.024 below E0; the retrieval probe (§7) replaces it. Second, static-slate
RL (§4.2) was aimed the wrong way for this failure — it makes training *more*
of a re-ranking task — so the reward sweep moved to the dynamic path (§4.1).

Operational caveats of this round: single seed; `CL-Graded`'s evaluation
initially covered 2 of 7 datasets and `RL-GradedNDCG` stopped at step 600 of
806 because trials were terminated while the QS status API still reported
`Uncommit` (see §5.2). Both later completed — CL-Graded via eval-only trial
2081344, RL as trial 2078663 — and the completed numbers are in this table.

### 6.3 Round 3 — dynamic RL transfers; static does not

All four RL arms plus the static-slate arm completed one epoch on 2026-09-19
evening:

| run | reward / algorithm | training domain | held-out | macro | probe @806 |
|---|---|---:|---:|---:|---:|
| G3-E0 | — | 0.4734 | 0.3673 | 0.3976 | 0.5448 |
| CL-Strong (best CL) | infonce, answer_masked, cross-device pool | −0.0005 | −0.0329 | −0.0236 | 0.5716 |
| ShortRL-CP-Graded | nDCG@10 slate, conditional projection | **+0.0120** | −0.0318 | −0.0192 | 0.5756 |
| RL-MRR-Graded | dynamic, MRR@10 | +0.0088 | −0.0100 | −0.0046 | 0.5726 |
| RL-BinaryNDCG | dynamic, binary nDCG@10 | +0.0155 | −0.0119 | −0.0041 | 0.5766 |
| **RL-GradedNDCG** | **dynamic, graded nDCG@10** | **+0.0214** | **−0.0072** | **+0.0010** | **0.5825** |

**The transfer hypothesis is confirmed.** Training against the actual
inference-time operation changes the shape of the result: the dynamic arms hold
the held-out loss to −0.007…−0.012 where every arm trained on E0's frozen pool
(CL and the static slate, despite its +0.012 in-domain gain) loses −0.032.
`RL-GradedNDCG` is the first arm to reach E0 on the macro average, converting
+0.0214 in-domain into macro rather than losing it back.

**The G1 reward ordering reproduces on a different corpus and label source.**
Graded nDCG > binary nDCG > MRR on macro, and the mechanism travels with it:
`degenerate_frac` over training stayed at 0.09–0.20 for graded nDCG while
binary nDCG drifted 0.27→0.45 and graded-MRR 0.36→**0.63** — the same rollout
collapse G1 documented, proportionally smaller because most RAG groups are not
degenerate to begin with.

**Conditional projection ran correctly and did not save the static arm.**
`projection/query_span_rank_mean` held at 35.94 for all 806 steps — exactly the
designed 1 mean + 20 slate + 15 shortlist columns, far from D=1024 saturation,
so the estimator never silently degenerated into score-function. The static
slate still fails on held-out because the universe mismatch is about *what the
reward can see*, not about the estimator. CP is sound; the slate is the wrong
universe.

**Where the remaining loss lives.** Even the winning arm pays −0.0072 held-out
for +0.0214 in-domain. Round 4 attacks that residual drift directly (§4.3).

### 6.4 Round 4 — the anchor penalty buys the gain at zero held-out cost

All four round-4 arms completed one epoch plus the full retrieval-only
evaluation (LRHalf 2081555 and Seed3407 on 2026-09-19; Anchor010 2089180 and
Anchor050 2089181 on 2026-09-20 after the five-failure debug trail below):

| run | anchor | LR | seed | training domain | held-out | macro | probe @806 |
|---|---:|---:|---:|---:|---:|---:|---:|
| RL-GradedNDCG (round 3) | 0 | 5e-6 | 42 | +0.0214 | −0.0072 | +0.0010 | 0.5825 |
| **LRHalf** | 0 | 2.5e-6 | 42 | +0.0200 | −0.0055 | +0.0018 | 0.5807 |
| Seed3407 | 0 | 5e-6 | 3407 | +0.0225 | −0.0076 | +0.0010 | 0.5818 |
| Anchor010 | 0.1 | 5e-6 | 42 | +0.0210 | −0.0036 | +0.0034 | 0.5835 |
| **Anchor050** | 0.5 | 5e-6 | 42 | **+0.0203** | **+0.0002** | **+0.0059** | 0.5794 |

Four readings:

**The anchor dose-response is monotone and dominates the LR knob.** Held-out
cost shrinks monotonically with the coefficient (−0.0072 → −0.0036 → +0.0002)
while the in-domain gain barely moves (+0.0214 → +0.0210 → +0.0203). Every
anchor point beats the LR-half trade: Anchor010 keeps *more* in-domain gain
than LRHalf (+0.0210 vs +0.0200) at *less* held-out cost (−0.0036 vs −0.0055);
Anchor050 keeps essentially the full gain at essentially zero cost. The
exchange rate is ~7:1 favorable where the LR knob trades 1:1 — exactly the
sub-linear signature the drift account predicted for a penalty that suppresses
only the drift component of the update. In the trust-region language of
§4.3: the anchor constrains how far the *function output* moves (as the PPO
clip does), the LR constrains how far the *parameters* move (as a small step
size does) — and the damage here lives entirely in the output movement while
the gain does not, which is why the output-space constraint removes exactly
the bad part.

**Anchor050 is the study's best arm on every retrieval metric.** Macro
`answer_mrr@10` +0.0059 (3× LRHalf), macro `answer_recall@5` +0.0095 (vs
+0.0023 unanchored), and the held-out scope is *positive* on recall@5 (+0.0054
vs −0.0043 unanchored) — the held-out recovery is across metrics, not an MRR
artifact.

**No collapse at coef 0.5.** The pre-registered worry was that 0.5 pins the
encoder to E0 and forfeits the in-domain gain; it does not (+0.0203 intact),
so the dose ceiling, if any, lies above 0.5. What saturates instead is the
held-out cost: it is already ~0, so a higher coefficient could only spend
in-domain gain for nothing. The open direction is not more anchor but more
epoch/LR under the anchor, since the anchor removes the drift penalty that
motivated gentleness.

**The probe ordering inverts at the top.** Anchor050 has the *lowest* late
probe (0.5794) and the best macro; Anchor010 the highest probe (0.5835) and a
mid macro. The probe measures in-domain re-ranking movement, which the anchor
deliberately damps — final confirmation that it is a monitoring signal, not a
selection rule.

**The generation backfill (2026-09-21, trial 2092096) confirms no end-to-end
regression but cannot resolve the anchor's advantage.** EM/F1 are now filled
for all eight RL-family arms. The RL family holds macro EM at E0 level or
above (best Anchor010 +0.0038; worst of the dynamic arms MRR-Graded −0.0058)
with held-out EM ≈ 0 throughout — the round-3/4 retrieval gains and costs
transmit to the generator without damaging it. But the arm ordering does not
carry over: Anchor050's macro EM (+0.0017) is mid-pack despite its +0.0059
macro MRR lead, because EM's seed noise is far larger than retrieval's — the
s42/Seed3407 pair, one config at two seeds, differs by 0.0073 macro EM and
0.0098 held-out EM, an order of magnitude above the seed noise of the
retrieval columns (±0.0012). The ≈0.6× MRR→EM transmission measured on the CL
arms does not hold at this scale: s42's held-out MRR of −0.0072 coexists with
held-out EM of +0.0023. Consequence for the claims: the anchor's advantage is
established on retrieval metrics; EM at n=1 seeds can only certify "no
end-to-end damage", and resolving EM differences would need multi-seed
averaging. The static-slate arm (ShortRL-CP, EM −0.0181) regresses end-to-end
like the CL family — consistent with both training on objectives other than
the inference operation (§6.2's anchor ablation).

Seed-replication and LR-arm notes (from the 2026-09-19 partial results, kept
for the record): Seed3407 lands within 0.0012 of seed 42 on every column, so
the round-3 +0.0010 macro is real, and its `degenerate_frac` profile
replicates (max ~0.21). LRHalf's 1:1 trade was the pre-registered bar the
anchor arms had to beat; they clear it by a wide margin. The anchor arms ran
on seed 42 only — the natural replication is Seed 2026 at coef 0.5 (§8).

Both LR arms' tuning-probe curves rise monotonically and plateau late (0.5807
/ 0.5818), matching their round-3 sibling.

The two anchor arms and the generation backfill took five launch attempts
(2026-09-19/20) to reach the results above — each attempt failed fast in the
smoke phase (256 queries, 3 steps), which is what the smoke is for: no queue
slot was wasted on a full epoch. The full failure trail, kept for the record:

- **Anchor arms, attempts 1–2 (2081553/2081554, 2082629/2082630):**
  `RuntimeError: 'weight' must be 2-D`, then `AttributeError:
  'RAGAnchorPrecomputeCallback' object has no attribute 'on_init_end'`. First:
  the backbone is loaded under DeepSpeed's ZeRO-3 init context, so before the
  Trainer exists its parameters are partitioned and a plain forward fails —
  the precompute moved into a `RAGAnchorPrecomputeCallback` at
  `on_train_begin`, encoding through the wrapped model (whose forward gathers
  on the fly), before any optimizer step. Second: the `CallbackHandler` fires
  *every* event on every registered callback, starting with `on_init_end`
  inside `Trainer.__init__`, and the callback was a plain class with only the
  one method — it now subclasses `TrainerCallback` (as the probe and tuning
  callbacks already did), and `tests/test_rag_anchor.py` drives the real
  handler's `on_init_end`.
- **Anchor arms, attempt 3 (2088952/2088953):** `TypeError:
  RAGRLModel.forward() missing 1 required positional argument: 'query'`. The
  model the callback is handed is the engine wrapping `RAGRLModel`, whose
  forward takes `(query, reward_inputs)` for the RL loss. Fix: the precompute
  calls `encode_query` — the wrappers' own query path (pooling and
  normalization included), the same one the probe and tuning callbacks already
  used on the round-3/4 RL runs — with a direct-forward fallback for bare
  backbones.
- **Anchor arms, attempt 4 (2089107/2089108):** `NameError: name 'np' is not
  defined` in `CandidateManifestDataset.__getitem__` at step 0: the anchor
  path referenced numpy through an import that only existed inside
  `attach_anchors`. Every earlier test used a stub dataset (which imports
  numpy itself); the new regression test drives the *real* dataset and
  collator. The same pass fixed a not-yet-triggered race: both anchor arms
  resolved to one anchor cache path (same manifest, same variant), so a
  concurrently starting trial's rank 0 could truncate the file mid-write under
  the other's ranks; the cache variant now includes the run's output_dir, so
  each trial owns its file.
- **Generation backfill, attempts 1–2 (2081830, 2082633):** the
  `source cluster_setup.sh` positional-parameter bug of §5.2 (`$1` is a
  RUN_ID), then `Frozen index does not fit on cuda:0 ... only 2 GB is free` —
  the eval's retrieval phase spreads the frozen index over every visible CUDA
  device while vLLM already holds GPUs 0–3. Fix: `source ... install`, and the
  evaluation is pinned to the complementary GPUs (`CUDA_VISIBLE_DEVICES=4..7`
  with TP=4; a quarter of the 21M-passage index per free H800, ~25 GB with
  the budgeted headroom). The third attempt (2088954) completed all four CL
  arms.
- **Anchor memmap truncation (caught by test, before it ever hit GPU):**
  `np.memmap` mode `"w+"` truncates on every open, so each rank's open would
  have wiped the rows the other ranks had already written, leaving most anchors
  silently zero. `encode_anchor_rows` pre-sizes the file (rank 0, before the
  barrier) and opens it `"r+"`.

Attempt 5 (Anchor010 2089180, Anchor050 2089181, 2026-09-20) passed the smoke
end-to-end — precompute on 8 ranks, anchor penalty live in three optimizer
steps — and produced the results above.

## 7. Instrumentation notes

**`src/rag/retrieval_probe.py`** replaces the re-ranking probe for model
selection (§6.2). It runs a real ANN search over a fixed 672-query slice of the
evaluation suite (96 per dataset, deterministic hash sample) and reports
in-domain (nq/hotpotqa) and held-out scopes separately, plus a step-0 baseline
via `on_train_begin` so each run carries its own untrained reference.

Two bugs were caught before it ever ran:

- It originally drew from every manifest split, which includes `nq/train` and
  `hotpotqa/train` — the training data itself. Now restricted to the split
  `EVALUATION_SUITE` declares.
- `index.search` is collective over the sharded index, so an uneven query split
  across ranks would **hang** rather than fail (10 queries over 4 ranks splits
  3/3/2/2). Ranks are padded to equal length and the padding is searched but
  not scored. `tests/test_rag_retrieval_probe.py` asserts all ranks issue
  identical search counts.

It is deliberately **off** for trials launched before it was written: newly
written in-training code should not run on jobs that have already queued hours.
It turns on for future batches, where the entry script's smoke test exercises
it via `on_train_begin` before the real epoch begins.

**Generation backfill** (`scripts/qs_g3_generation_eval.sh`) serves
Qwen2.5-7B-Instruct with vLLM inside the trial and re-runs `eval_rag.py` with
generation enabled, filling `generator_em` / `generator_token_f1` for the
already-trained arms. vLLM is co-located rather than run as its own service
because the Alibaba training clusters only offer a whole-node 8-GPU pack; the
1/2/4-GPU packs live on cluster 1067, a different cloud, so an endpoint there
is not reachable from a training pod. `eval_rag.py` releases the index GPUs
between its retrieval and generation phases, so the two do not contend.

## 8. Further work

Ordered by expected value against the current best (Anchor050, macro +0.0059):

1. **Push in-domain gain under the anchor.** The anchor removes the drift
   penalty that motivated the LR cut, and coef 0.5 already runs at full LR —
   yet the held-out cost has saturated at ~0 while the in-domain gain still
   matches the unanchored arm. The unexplored direction is buying *more*
   in-domain: two epochs at coef 0.5, or a higher LR under the anchor. The
   dose-response says a higher coefficient is the wrong knob (nothing left to
   save on held-out); more optimization under the same anchor is the right one.
2. **Seed 2026 at coef 0.5** completes the winner's seed replication. The
   +0.0059 macro is ~5× the round-3 seed noise band (±0.0012 per column), so
   this is confirmation, not exploration.
3. **Answer-F1 RL under the anchor**: the answer-F1 reward
   (`rag_retrieval_reward: answer_f1`) is the remaining untried reward family,
   and round 3 suggests reward shape matters more than anything else tried.
   The vLLM co-location path is now proven (§6.2 backfill and the 2026-09-21
   RL backfill), so the in-training generator endpoint is an engineering step,
   not a research risk. Combining it with coef 0.5 tests whether the anchor's
   zero-cost property survives a different reward.
4. **Refresh the candidate pool during training (ANCE-style).** Round 2's
   mechanism suggests its own fix: the pool is mined once with E0, so every
   negative a query ever sees is already an E0 near-neighbour, and the model
   never learns to suppress passages that become near-neighbours only after its
   own embedding moves — exactly what ANN search returns at inference.
   Re-mining is cheap here (one search per training query, vs the 4,096/step
   the dynamic RL arm already pays). The dynamic RL arm already has this
   property, which round 3 confirmed transfers; refreshed-pool CL is the
   natural follow-up if a cheaper-than-RL method is wanted. The CL+anchor
   ablation sharpens the bar: 86% of refreshed-pool CL's expected held-out
   loss is the objective, not drift, so the pool must actually be refreshed
   for CL to compete — anchoring alone cannot save it.
5. **ShortRL-SF contrast** is implemented and preflighted but low priority: the
   static slate failed on universe grounds (and end-to-end: EM −0.0181), so
   the estimator contrast on it is no longer informative.

Done along the way: the generation backfill now covers every trained arm,
`CL-AnswerMasked-Anchor050` included (filled 2026-09-21 by running the
co-located vLLM eval on the locally freed 8×L20Y machine after the cluster
queue stalled; its EM confirms the §6.2 drift/objective split end-to-end).
