# Experiment Plan: Reward-Programmable Post-Training for Text Embeddings

This document is the source of truth for the revised paper experiments. It
replaces the earlier plan organized around training from a base LLM and the
BGE-M3 binary mining corpus.

**Revision: 2026-09-10.** The paper now studies RL as a post-training method for
an already contrastively trained embedding model. RL is not positioned as a
replacement for contrastive pre-training. A fixed-corpus RAG experiment is
included as a downstream query-only specialization.

---

## 0. Claims and boundaries

### 0.1 Primary claim

Given a contrastively trained embedding checkpoint, embedding-space policy
optimization is a reward-programmable post-training method. Under the same
initialization, post-training data, candidate slates, and optimization budget,
compare:

1. no post-training;
2. continued contrastive learning (CL->CL);
3. rank-based supervised post-training (CL->RankNet); and
4. exact-reward policy optimization (CL->RL).

The decisive comparison is **CL->RL vs. both CL->CL and CL->RankNet**, not RL
from a base LLM vs. CL from a base LLM.

### 0.2 Secondary claims

- The same post-training interface accepts rewards with different semantics
  without changing the encoder or its inference interface.
- Exact non-differentiable ranking rewards provide the clearest reason to use a
  policy-gradient estimator.
- With a differentiable InfoNCE reward, RL should approach the direction of
  direct backpropagation at small exploration; this is a control rather than the
  main result.
- Sampling is training-only. Deployment uses the deterministic policy mean and
  adds no inference latency.
- With a frozen document encoder and corpus index, the same interface can
  post-train only the query encoder for retrieval or answer-aware RAG utility.

### 0.3 Claims this revision does not make

- RL replaces contrastive pre-training.
- RL is the best way to create embedding geometry from a base LLM.
- The method is state of the art solely by comparison with published models
  trained on different data and budgets.
- nDCG and MRR alone demonstrate unrestricted reward generality.
- That fixed-corpus RAG is validated before the benchmark, frozen generator,
  reward, and planned query-only runs are complete.

---

## 1. Experimental object

### 1.1 Primary initialization

Use the public contrastively trained checkpoint:

```text
E0 = Qwen/Qwen3-Embedding-0.6B
```

Every main post-training row starts from the exact same `E0` weights. A newly
trained Stage 1 is not on the critical path. Evaluate `E0` once as the
no-post-training reference.

A second public embedding checkpoint is an extension, not a blocker. Select it
only after checking:

- compatibility with the repository's pooling and prompting convention;
- no known overlap with the post-training evaluation split;
- a compatible license; and
- enough architectural difference to test transfer rather than only scale.

`Alibaba-NLP/E2Rank-0.6B-Embedding-Only` may be a sanity check, but it must not
be the sole transfer result if it was trained on the same ranking corpus.

### 1.2 Primary post-training data

Use the original E2Rank listwise corpus:

```text
data/train.jsonl
{query, document, ranking, [source]}
```

`ranking` is the 1-indexed teacher permutation of the complete candidate list.
Preserve the full list and permutation.

All post-training objectives consume the same records and slates:

- **InfoNCE:** rank-1 document is positive; the remainder of its own slate and
  the other samples' rank-1 documents are negatives.
- **RankNet:** every strictly ordered pair in the teacher permutation is a
  supervised comparison; the other samples' rank-1 documents are appended with
  a shared zero label below the complete own-query permutation.
- **RL-nDCG:** convert the same teacher permutation into declared relevance
  gains, append the other samples' rank-1 documents as zero-relevance frozen
  candidates, and use exact nDCG of the sampled score ordering as reward.

The methods therefore share the supervision source and candidate pool while
transforming them through different objectives. Cross-query document embeddings
are detached in the supervised controls, matching the frozen in-batch candidates
used to compute the RL reward; each document still receives gradients through
its own sample's slate.

BGE-M3 remains supported as a secondary data-regime check. It is not the main
corpus because its binary single-positive supervision produces coarse ranking
rewards and cannot provide a strong full-ranking supervised control.

An augmented listwise corpus is tracked separately as a scale-up extension:

```text
data/train_v2.jsonl
```

`train_v2` merges the original E2Rank records with ReasonRank RL records converted
to the same 16-document schema. For each ReasonRank record, retain the first 16
candidates in the original retrieval order and filter the teacher permutation to
those candidates; do not use `relevant_docids`. Repeat the C1/C3 continued-CL
versus RL comparison on this corpus so both methods remain directly comparable
within the augmented-data regime. Cross-corpus comparisons still process
different numbers of examples per epoch, so report the extra compute explicitly.

### 1.3 Default action structure

The primary RL runs use two action components:

```text
[[query], [positive, negative]]
```

The second component is the complete document slate, so the rollout evaluates a
`G x G` query-by-slate reward matrix. Positive and negative documents share the
same rollout axis; the main method does not split them into a three-factor
`[[query], [positive], [negative]]` policy.

### 1.4 Fixed data split

Create one train/development split and reuse it for every method. The training
seed must not change split membership. Store a split manifest containing:

- dataset revision/hash;
- split seed;
- per-source train/dev counts; and
- duplicate/leakage checks.

The current split is coupled to the training seed. Decoupling it is blocking
work before final runs.

---

## 2. Fair-comparison protocol

### 2.1 What is held fixed

For CL->CL, CL->RankNet, and CL->RL, hold fixed:

- initial model weights;
- the full backbone as the trainable parameter set;
- train/dev records and candidate slates;
- own-slate plus cross-query-positive candidate construction;
- trainable parameter set and query/document gradient participation;
- training seed and data order;
- examples and tokens processed;
- batch size and gradient accumulation;
- maximum query/document lengths;
- optimizer family and precision; and
- checkpoint/evaluation schedule.

The main comparison is example/token matched. Also report wall-clock time, peak
memory, and GPU-hours so the policy-estimator cost is explicit.

### 2.2 Hyperparameter selection

Do not select settings on MTEB. Use the fixed E2Rank development split.

Give each objective the same small tuning budget. At minimum, test a shared
three-point learning-rate grid. RL additionally receives a small pre-declared
exploration grid; RankNet and InfoNCE use the same temperature grid. Freeze the
selected settings before final evaluation.

The primary RL reward is **graded in-batch nDCG**, not an nDCG+InfoNCE mixture.
Here `in-batch` means that other samples' rank-1 documents are appended as
zero-relevance frozen candidates; it does not append their complete slates. The
single reward keeps attribution clean: an improvement cannot be credited to an
embedded contrastive term. Mean-resultant-length rescaling is part of this
default because sampled own-slate documents and frozen in-batch candidates must
be score-calibrated. Reward mixtures are optional appendix experiments.

### 2.3 Seed policy

Use **one seed everywhere**. The main methods use the same seed and therefore
the same initialization, data order, and split manifest.

- Report measured differences without error bars.
- State explicitly that seed variance is not estimated.
- Add two seeds only for a specific decisive comparison if its observed gap is
  small enough that the paper's conclusion depends on run-to-run variance.
- Do not replicate broad ablation sweeps.

### 2.4 Evaluation

**Model-selection metrics on the fixed E2Rank dev split**

- nDCG@10;
- MRR@10;
- pairwise ordering accuracy; and
- rank-1 accuracy/Recall@1.

**Final general embedding evaluation**

- MTEB English v2 overall;
- Retrieval;
- Reranking;
- mean of non-retrieval categories; and
- full category breakdown in the appendix.

Run MTEB only on selected checkpoints. Do not use it to tune learning rate,
exploration, reward mixture, or stopping time.

**Training diagnostics**

- reward mean and within-group standard deviation;
- degenerate-group fraction;
- advantage standard deviation by action component;
- sigma/kappa and KL when enabled;
- wall-clock, peak memory, and GPU-hours.

**Fixed-corpus RAG evaluation**

- Recall@5 and Recall@20;
- MRR;
- end-to-end answer exact match and token F1 under one frozen generator; and
- corpus re-encoding/index-rebuild cost, which is zero for every compared row.

---

## 3. Paper tables

### Table 1 — Main post-training comparison

Primary checkpoint: `Qwen3-Embedding-0.6B`. All training rows use one shared
seed and the same post-training corpus and budget.

| Run | Initialization | Post-training objective | Role |
|---|---|---|---|
| C0 | E0 | none | initialization reference |
| C1 | E0 | InfoNCE with in-batch negatives | continued-training control |
| C2 | E0 | RankNet with in-batch zero-label candidates | rank-based supervised control |
| C3 | E0 | exact graded in-batch nDCG@10 | ours |

Report dev nDCG@10, MRR@10, MTEB Retrieval, MTEB non-Retrieval, MTEB overall,
GPU-hours, and peak memory. Full MTEB categories may move to the appendix.

Interpretation rules:

- C3 > C1 and C2 on ranking metrics supports the primary claim.
- C3 > C1 but not C2 means exact-reward RL did not beat a strong rank-based
  supervised objective; weaken the headline.
- C3 improves dev nDCG but degrades non-Retrieval MTEB: report objective
  specialization rather than a general embedding improvement.
- C3 only beats C0: extra post-training helps, but the RL-specific claim is not
  established.

### Table 2 — Reward programmability

All rows start from `E0`, use the same data and RL implementation, and differ
only in reward.

| Run | RL reward | Purpose |
|---|---|---|
| R1 | graded in-batch nDCG@10 | primary exact reward; reuse C3 |
| R2 | in-batch MRR@10 | top-heavy ranking utility |
| R3 | InfoNCE with in-batch negatives | differentiable-reward control |
| R4 | in-batch nDCG + InfoNCE, normalized advantages | optional mixture |

Report dev nDCG@10, MRR@10, pairwise accuracy, and MTEB Retrieval/overall. The
important evidence is reward-specific movement, not only the best average row.

Show R3 adjacent to direct InfoNCE C1. They must use the same own-slate plus
cross-query-positive pool, temperature, score normalization, positive
definition, and training budget; the estimator is the only intended difference.
A near-zero gap supports the theoretical boundary that policy optimization is
unnecessary for a directly differentiable reward. A large gap requires
revisiting that argument.

MRR is no longer dismissed as an algebraic duplicate: under restored graded
full-ranking supervision it differs from binary single-positive nDCG.

### Table 3 — Fixed-corpus RAG

Encode one declared QA corpus once with the document side of `E0`. Freeze the
document encoder, document vectors, ANN index, generator, prompt, context budget,
and decoding settings. Initialize a separate trainable query encoder from `E0`.

| Query encoder | Post-training objective/reward | Index rebuilt? | Role |
|---|---|---:|---|
| E0 | none | no | initialization reference |
| E0 + query CL | InfoNCE | no | supervised continuation |
| E0 + query RankNet | pairwise ranking | no | rank-based supervised control |
| E0 + query RL | exact retrieval reward | no | query-only RL test |
| E0 + query RL | deterministic answer-aware reward | no | stronger programmability extension |

Use the same train/dev/test queries and fully trainable query encoder for every
trained row. Report retrieval Recall@5/20 and MRR, plus end-to-end EM/F1 from the
frozen generator. If the answer-aware row is not run, describe the experiment as
retrieval adaptation for RAG rather than direct RAG-reward optimization.

Before implementing or running this table, freeze in a separate RAG manifest:

- benchmark and corpus revisions;
- document chunking and corpus encoder prompt;
- ANN implementation and search parameters;
- generator checkpoint, prompt, retrieval depth, context budget, and decoding;
- retrieval and answer-aware reward definitions; and
- leakage checks and fixed split identifiers.

### Table 4 — Initialization transfer (extension)

| Initialization | Original | + continued CL | + RL-nDCG |
|---|---:|---:|---:|
| primary E0 | C0 | C1 | C3 |
| second model family | T0 | T1 | T2 |

Run this only after Table 1 works. Do not replace the RankNet comparison with a
scale sweep.

### Table 5 — Focused ablations (appendix)

Every row except the from-scratch boundary starts from `E0` and changes one C3
setting.

| Run | Change | Question |
|---|---|---|
| A0 | base LLM, no training | raw initialization reference |
| A1 | base LLM -> InfoNCE | pure CL boundary |
| A2 | base LLM -> RL-nDCG | pure RL boundary |
| A3 | query-only `[query]` | is document-slate exploration useful? |
| A4 | slate-only `[positive, negative]` | is query exploration useful? |
| A5 | projected Gaussian | does exact vMF sampling matter? |
| A6 | default in-batch nDCG, no rescaling | does calibration prevent collapse? |
| A7 | own-slate nDCG | what do in-batch candidates add? |
| A8 | smaller group size | estimator quality/cost trade-off |

The default query-by-document-slate run with rescaling is C3 and is reused as
the control for A3--A7. A6 differs from C3 only by disabling rescaling; A7
differs only by removing the frozen in-batch candidates. Add more kappa,
normalization, KL, group-size, or rollout rows only if a main-run diagnostic
reveals a concrete unresolved failure. Do not add a positive/negative-separated
`G^3` rollout unless a later result creates a specific need for it.

### Appendix data-regime check

Optionally repeat C1/C2/C3 on BGE-M3 binary slates after the main result is
stable. This tests whether conclusions survive weaker supervision; it does not
select the main method. Report reward resolution and degenerate-group fraction.

### Augmented-data CL-vs-RL replication

Repeat the continued-CL versus RL comparison with the same initialization, method
configurations, seed, and one-epoch schedule, changing only the listwise corpus:

| Run | Initialization | Post-training data | Objective | Role |
|---|---|---|---|---|
| D0 | E0 | none | none | initialization reference; reuse C0 |
| D1 | E0 | `data/train_v2.jsonl` | InfoNCE with in-batch negatives | augmented-data continued-training control |
| D2 | E0 | `data/train_v2.jsonl` | graded in-batch nDCG@10 | augmented-data RL |

D1--D2 consume the same augmented records and are a fair method comparison. Their
cross-corpus counterparts C1 and C3 process fewer examples and tokens
per epoch, so C-vs-D differences measure the end-to-end value of the larger data
recipe rather than a budget-matched causal data ablation.

---

## 4. Execution phases

### Phase 0 — Blocking engineering and data audit

- [x] Support E2Rank listwise and BGE-M3 records in one loader.
- [x] Preserve per-record source batching for mixed listwise files.
- [x] Implement configurable InfoNCE/RankNet supervised objectives.
- [x] Preserve the full teacher permutation for RankNet.
- [x] Add detached cross-query positives to the InfoNCE and RankNet controls.
- [ ] Download and fingerprint `data/train.jsonl`.
- [x] Confirm the public dataset uses fixed 16-document slates.
- [ ] Audit source distributions after downloading the exact training artifact.
- [ ] Create a fixed split manifest independent of training seed.
- [ ] Restore deterministic dev ranking evaluation.
- [x] Add a runner that initializes directly from `E0` without the old Stage-1
      merge dependency.
- [x] Add matched in-batch listwise reward configs for nDCG@10, MRR@10, and
      InfoNCE.
- [x] Make every new command pass `scripts/validate_experiments.py`.

### Phase 1 — Short smoke runs

Run 100--200 steps:

```text
S1  E0 -> InfoNCE
S2  E0 -> RankNet
S3  E0 -> RL-in-batch-nDCG
```

Completion criteria:

- all runs load the same train/dev manifest;
- all losses and reward diagnostics are finite;
- RankNet pairwise accuracy improves over initialization;
- RL has non-zero reward spread and non-degenerate advantages;
- checkpoints reload into the MTEB evaluator; and
- no run silently initializes from a base LLM.

### Phase 2 — Equal-budget calibration

Use the fixed dev split and one calibration seed. Give C1/C2/C3 the same number
of trials. Select settings before running MTEB.

### Phase 3 — Main comparison

Run C1, C2, and C3 once with the shared seed. Evaluate C0 once. This phase
decides whether the primary empirical claim is supported. If the decisive gap
is marginal, replicate only that comparison before drawing a conclusion.

### Phase 4 — Reward study

Run R2 and R3. Run R4 only if primary in-batch nDCG exhibits meaningful
degenerate groups or if the mixture becomes part of the final recipe.

### Phase 5 — Focused ablations

Run A2, A3, A5, A6, and A7 first. A0/A1 may reuse existing checkpoints only
when their data and budgets match. A4/A8 are lower priority.

### Phase 6 — Fixed-corpus RAG

Freeze the RAG manifest, encode and index the corpus once, then run the
query-only InfoNCE, RankNet, and exact-retrieval-reward rows. Evaluate the
unmodified query encoder once. Add the answer-aware RL row only after the frozen
generator path is deterministic and its training cost is measured.

### Phase 7 — Transfer and secondary data

Run D1--D2 after the primary C1--C3 recipe is stable. Select a second qualified
embedding initialization separately. Only then consider BGE-M3 replication or
larger models.

---

## 5. Run budget and cut order

Minimum empirical closure:

- 3 main post-training runs: in-batch InfoNCE, in-batch RankNet, and RL with
  in-batch nDCG;
- 1 initialization evaluation;
- 2 reward runs: MRR and InfoNCE reward;
- 5 high-value ablation runs: pure RL, query-only, Gaussian, no rescaling, and
  own-slate nDCG. C3 is reused as the matched control for the last two.

The embedding-model study is **10 training runs plus one evaluation-only row**,
excluding short calibration trials. The minimum RAG addition is three
query-only training runs plus evaluation of the unmodified query encoder; the
answer-aware reward is one optional additional run. Optional mixture, second
initialization, the two-run `train_v2` CL-vs-RL replication, BGE-M3 replication, and
additional estimator sweeps follow afterward.

Cut in this order when compute is limited:

1. extra kappa/group-size/normalization sweeps;
2. BGE-M3 replication;
3. reward mixture R4;
4. the `train_v2` CL-vs-RL replication D1--D2;
5. second-initialization continued-CL control T1;
6. second initialization entirely;
7. the answer-aware RAG extension, while retaining the three-row fixed-index
   retrieval comparison.

Do not cut RankNet or continued CL from the decisive comparison before cutting
appendix breadth.

---

## 6. Configuration and script migration

The old scripts encode the superseded dependency:

```text
base LLM -> locally trained Stage 1 -> Stage 2
```

Do not use them unchanged for the new main table. Replace the critical path with:

```text
scripts/experiments/posttrain_smoke.sh
scripts/experiments/posttrain_core.sh
scripts/experiments/posttrain_rewards.sh
scripts/experiments/posttrain_ablations.sh
scripts/experiments/posttrain_data_v2.sh
scripts/experiments/posttrain_transfer.sh
scripts/experiments/posttrain_eval.sh
```

Each runner should accept:

```text
INIT_MODEL_CONFIG
POSTTRAIN_DATASET
SEED
CKPT_ROOT
DRY_RUN
```

Required configurations:

```text
configs/dataset/e2rank_listwise.yaml
configs/dataset/e2rank_listwise_v2.yaml
configs/baseline/default.yaml
configs/baseline/ranknet.yaml
configs/baseline/infonce_in_batch.yaml
configs/baseline/ranknet_in_batch.yaml
configs/reward/ndcg_listwise.yaml
configs/reward/ndcg_listwise_in_batch.yaml
configs/reward/mrr_listwise.yaml
configs/reward/infonce_listwise.yaml
```

The old `stage1.sh` and `phaseB_recipe.sh` remain useful only for from-scratch
ablations until renamed. The revised runners suffix in-batch run IDs with `-ib`
so an older own-slate checkpoint cannot be silently reused. Do not reuse legacy
run IDs for public-checkpoint runs.

---

## 7. Paper restructuring implied by this plan

After Phase 1 passes:

- Motivation: RL complements contrastive pre-training as reward-directed
  post-training.
- Setup: initialize from an existing embedding checkpoint, not a base LLM.
- Main table: no post-training, continued CL, RankNet, and RL.
- Reward section: exact graded in-batch nDCG is primary; differentiable rewards
  are controls; own-slate nDCG isolates the value of the expanded candidate
  pool; mixtures are optional.
- Pure CL and pure RL: appendix boundary experiments.
- BGE-M3 binary reward-resolution analysis: secondary data regime rather than
  the organizing principle.
- Scale sweeps: lower priority than strong post-training baselines and transfer.
- RAG: present as fixed-corpus query-only adaptation; claim direct RAG-reward
  optimization only if the answer-aware reward row is actually run.
