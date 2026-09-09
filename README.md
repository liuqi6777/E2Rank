# RL Training and Evaluation for Embedding

This repository keeps the RL training pipeline and evaluation scripts for embedding models.

Current scope:

- GRPO-based RL training in [`src/train.py`](src/train.py)
- Config-driven experiments in [`configs/`](configs)
- Shell launcher in [`scripts/run.sh`](scripts/run.sh)
- MTEB/BEIR-style evaluation in [`eval_mteb/`](eval_mteb)

## Environment Setup

This project uses `uv` and targets Python `3.10` (`.python-version` is already included). First install uv by:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then run:

```bash
uv sync
source .venv/bin/activate
```

## Data and Model Preparation

Training accepts both the original E2Rank listwise JSONL format and BGE-M3 mining
records. The formats can coexist; the loader detects each record schema.

For the listwise post-training data, download `data/train.jsonl` with:

```bash
mkdir -p data
hf download \
  Alibaba-NLP/E2Rank_ranking_datasets \
  train.jsonl \
  --local-dir ./data \
  --repo-type dataset
```

Each listwise sample contains the fields used by
[`src/embedding_data.py`](src/embedding_data.py):

- `query`
- `document`
- `ranking`
- `source` (optional, used to choose task prompts)

ReasonRank RL data can be converted to the same fixed 16-document format. The
converter retains the first 16 candidates in the original retrieval order, filters the
teacher permutation to those candidates, and intentionally ignores `relevant_docids`:

```bash
mkdir -p data/reasonrank
hf download \
  liuwenhan/reasonrank_data_rl \
  train.parquet \
  --local-dir data/reasonrank \
  --repo-type dataset

uv run python scripts/convert_reasonrank.py \
  --input data/reasonrank/train.parquet \
  --output data/reasonrank_train_slate16.jsonl
```

Rows with fewer than 16 candidates are skipped. The resulting JSONL uses the existing
`{query, document, ranking, source}` schema and can be concatenated with the E2Rank
listwise JSONL without changing the training code.

The merged training artifact is `data/train_v2.jsonl`. It contains the original
E2Rank listwise records plus the converted ReasonRank records; every sample keeps a
16-document slate and derives supervision only from its teacher permutation. Use
`configs/dataset/e2rank_listwise_v2.yaml` to load it. The matched continued-CL and RL
experiments are launched together with:

```bash
bash scripts/experiments/posttrain_data_v2.sh
```

This produces D1 (continued InfoNCE with in-batch negatives) and D2 (RL with graded
in-batch nDCG@10), both initialized from the same embedding checkpoint and trained on
the same merged data and budget.

BGE-M3 records use `query`, `pos`, `neg`, and optional `pos_scores` /
`neg_scores`; they are converted to fixed-size slates according to the dataset
config. Use `configs/dataset/e2rank_listwise.yaml` for the original listwise
data and `configs/dataset/default.yaml` for BGE-M3.

## Training

The recommended entrypoint is the shell wrapper plus an explicit config path:

```bash
bash ./scripts/run.sh configs/exp/template.yaml
```

You can run training without any `configs/exp/*.yaml` and compose the config directly from the shell:

```bash
bash ./scripts/run.sh \
  --base-train configs/train/default.yaml \
  --base-dataset configs/dataset/default.yaml \
  --base-model configs/model/e2rank_0.6b_embedding_only.yaml \
  --base-grpo configs/grpo/default.yaml \
  --base-reward configs/reward/contrastive_in_batch.yaml \
  --base-eval configs/eval/default.yaml
```

If you omit `run_name` and `output_dir`, they are auto-generated. For example, the command above becomes:

- `run_name=e2rank_0.6b_embedding_only__contrastive_in_batch`
- `output_dir=checkpoints/<run_name>`

Supported base config flags:

- `--base-train`
- `--base-dataset`
- `--base-model`
- `--base-baseline` (supervised launcher only)
- `--base-grpo`
- `--base-reward`
- `--base-eval`

Supervised post-training supports InfoNCE and RankNet. For example, RankNet on
the original teacher ranking is launched with:

```bash
bash ./scripts/run_baseline.sh \
  --base-train configs/train/stage2.yaml \
  --base-dataset configs/dataset/e2rank_listwise.yaml \
  --base-model configs/model/qwen3_embedding_0.6b.yaml \
  --base-baseline configs/baseline/ranknet.yaml \
  --base-eval configs/eval/default.yaml
```

The revised paper controls use `configs/baseline/infonce_in_batch.yaml` and
`configs/baseline/ranknet_in_batch.yaml`. They append the other samples' rank-1
documents as detached cross-query candidates, matching the candidate pool of
the primary in-batch nDCG reward.

The revised paper experiments initialize directly from an existing embedding
checkpoint and are grouped by purpose:

```bash
bash scripts/experiments/posttrain_smoke.sh
bash scripts/experiments/posttrain_core.sh
bash scripts/experiments/posttrain_rewards.sh
bash scripts/experiments/posttrain_ablations.sh
bash scripts/experiments/posttrain_data_v2.sh
bash scripts/experiments/posttrain_eval.sh full
# Optional, after selecting an independent second embedding initialization:
TRANSFER_MODEL_CONFIG=... TRANSFER_MODEL_ID=... \
  bash scripts/experiments/posttrain_transfer.sh
```

All use `Qwen/Qwen3-Embedding-0.6B` and seed 42 by default. The standard runners use
the original E2Rank listwise data; `posttrain_data_v2.sh` uses the merged `train_v2`
corpus. Override their settings with `INIT_MODEL_CONFIG`, `INIT_MODEL_ID`,
`POSTTRAIN_DATASET`, `POSTTRAIN_TRAIN_CONFIG`, `POSTTRAIN_GRPO_CONFIG`, `SEED`,
and `CKPT_ROOT`. Post-training runs use `configs/eval/mteb.yaml` by default, so
the `MTEB(eng, v1, subset)` benchmark runs on the initial weights and every
saved checkpoint. Set `POSTTRAIN_EVAL_CONFIG=configs/eval/default.yaml` to
disable it or point `POSTTRAIN_EVAL_CONFIG` at another eval preset. Set
`DRY_RUN=1` to inspect every command without launching training.
`posttrain_eval.sh` evaluates the initialization and every checkpoint under
`CKPT_ROOT`; set `EVAL_INITIALIZATION=0` or override `CKPT_GLOB` when needed.
The older `stage1.sh` / `phase*.sh` scripts are retained for the superseded
from-scratch experiment path and its ablations.

### Using a non-Qwen embedding checkpoint

The model config defines the checkpoint's complete dense-embedding protocol;
GRPO itself only receives normalized vectors. Supported pooling methods are
`last`, `mean`, and `cls`. For example:

```yaml
model_name_or_path: intfloat/multilingual-e5-large
pooling_method: mean
padding_side: right
append_token: none        # none, eos, or pad
query_prompt_template: "query: {query}"
document_prompt_template: "passage: {document}"
embedding_max_length: 512
lora_target_modules: [query, key, value]
deepspeed: ./scripts/zero3.json
```

Ready-to-use retrieval-training examples are provided in
`configs/model/bge_m3.yaml` and `configs/model/multilingual_e5_large.yaml`.
They can be passed anywhere a model base config is accepted:

```bash
bash ./scripts/run.sh \
  --base-train configs/train/posttrain.yaml \
  --base-dataset configs/dataset/e2rank_listwise.yaml \
  --base-model configs/model/bge_m3.yaml \
  --base-grpo configs/grpo/posttrain.yaml \
  --base-reward configs/reward/ndcg_listwise_in_batch.yaml \
  --base-eval configs/eval/mteb.yaml
```

`query_prompt_template` may use `{query}` (or `{text}`) and
`{task_description}`. `document_prompt_template` may use `{document}` (or
`{text}`). The same protocol is automatically passed to in-training MTEB
evaluation. When adding another architecture, set `lora_target_modules` to
names that actually occur in that model; full fine-tuning does not require this
field. Checkpoints that depend on extra SentenceTransformers projection modules
are not represented by `AutoModel` alone and need a dedicated adapter before
they can be trained faithfully.

Regular training arguments can still be appended after that:

```bash
bash ./scripts/run.sh \
  --base-train configs/train/default.yaml \
  --base-dataset configs/dataset/default.yaml \
  --base-model configs/model/qwen3_embedding_0.6b.yaml \
  --base-grpo configs/grpo/default.yaml \
  --base-reward configs/reward/mrr.yaml \
  --base-eval configs/eval/default.yaml \
  --learning_rate 5e-5 \
  --sigma 0.03
```

You can still override either one manually:

```bash
bash ./scripts/run.sh \
  --base-train configs/train/default.yaml \
  --base-dataset configs/dataset/default.yaml \
  --base-model configs/model/qwen3_embedding_0.6b.yaml \
  --base-grpo configs/grpo/default.yaml \
  --base-reward configs/reward/mrr.yaml \
  --base-eval configs/eval/default.yaml \
  --run_name qwen3-embed-mrr
```

In that case `output_dir` falls back to `checkpoints/qwen3-embed-mrr`.

If you still want to keep a top-level experiment template, the old form also works:

```bash
bash ./scripts/run.sh configs/exp/template.yaml
```

For simple sweeps over `train/dataset/model/grpo/reward/eval` entries, use [`scripts/run_grid.py`](scripts/run_grid.py):

```bash
uv run python scripts/run_grid.py \
  --set-base train=configs/train/default.yaml \
  --set-base dataset=configs/dataset/default.yaml \
  --set-base model=configs/model/qwen3_0.6b.yaml,configs/model/e2rank_0.6b_embedding_only.yaml \
  --set-base grpo=configs/grpo/default.yaml \
  --set-base reward=configs/reward/ndcg.yaml,configs/reward/contrastive_in_batch.yaml \
  --set-base eval=configs/eval/default.yaml,configs/eval/mteb_retrieval.yaml \
  --run-name-prefix sweep \
  --output-root checkpoints/sweep \
  --dry-run
```

The example top-level config is `configs/exp/template.yaml`.

The config layout is:

```text
configs/
  train/
  dataset/
  model/
  grpo/
  reward/
  eval/
  exp/
```

Example inheritance:

```yaml
_base_:
  - ../train/default.yaml
  - ../dataset/default.yaml
  - ../model/e2rank_0.6b_embedding_only.yaml
  - ../grpo/default.yaml
  - ../reward/ndcg.yaml
  - ../eval/default.yaml

output_dir: checkpoints/E2Rank-Full-GRPO-0.6B
run_name: E2Rank-Full-GRPO-0.6B
```

Important data fields exposed by [`src/config.py`](src/config.py):

- `data_path`
- `per_dataset_max_samples` (`null` keeps all samples)
- `q_max_len`
- `d_max_len`
- `relevance_scheme`

Important RL-related fields exposed by [`src/config.py`](src/config.py):

- `action_components`
- `group_size`
- `sigma`
- `sigma_learnable`
- `reward_type`
- `reward_ndcg_k`
- `reward_rbo_p`
- `ndcg_in_batch_include_negatives`
- `contrastive_use_in_batch_negatives`
- `contrastive_temperature`
- `advantage_norm`

Common GRPO component settings:

- Query-only: `action_components: [[query]]`
- Default query-by-slate product: `action_components: [[query], [positive, negative]]`
- Old factorized: `action_components: [[query], [positive], [negative]]`
- Doc-only joint: `action_components: [[positive, negative]]`
- Pos-only: `action_components: [[positive]]`
- Neg-only: `action_components: [[negative]]`

Supported reward presets in [`configs/reward/`](configs/reward):

- `ndcg.yaml`
- `ndcg_in_batch.yaml`
- `contrastive_in_batch.yaml`
- `contrastive_no_in_batch.yaml`
- `infonce_in_batch.yaml`
- `infonce_no_in_batch.yaml`
- `mrr.yaml`
- `top_weighted_pairwise_listwise.yaml`
- `rbo_listwise.yaml`

Reward semantics:

- `ndcg`: per-query slate nDCG only, kept as the backward-compatible no-in-batch option
- `ndcg_in_batch`: append positives from other samples in the batch as extra zero-relevance candidates; set `ndcg_in_batch_include_negatives: true` to append all candidates from other samples
- `contrastive`: `s+ - tau * logsumexp(s- / tau)` over negatives only
- `infonce`: `s+ - tau * logsumexp([s+, s-] / tau)` over the full partition
- `mrr`: under `graded` relevance, only labels with `relevance >= 2` count as relevant; under `binary`, the threshold remains `relevance > 0`
- `mrr_in_batch`: the same MRR definition after appending in-batch candidates, parallel to `ndcg_in_batch`
- `top_weighted_pairwise`: own-slate agreement on every teacher-preferred pair whose better document lies in the teacher top-k; each pair is weighted by that better rank's logarithmic discount
- `rbo`: normalized truncated rank-biased overlap between the model and teacher permutations; `reward_rbo_p` controls how quickly prefix weights decay

The two permutation-native rewards consume `rank_labels` and intentionally operate on the
record's own slate. Cross-query documents are not included because the data provides no teacher
order for them.

Example:

```yaml
reward_type: contrastive
contrastive_temperature: 0.03
```

MTEB eval during regular GRPO training is disabled by default through [`configs/eval/default.yaml`](configs/eval/default.yaml). To evaluate the initial weights and every saved checkpoint with the existing `MTEB(eng, v1, subset)` benchmark preset, use:

```bash
bash ./scripts/run.sh configs/exp/template.yaml \
  --base-eval configs/eval/mteb.yaml
```

The preset sets:

```yaml
mteb_eval_benchmark: "MTEB(eng, v1, subset)"
mteb_eval_langs: eng
mteb_eval_batch_size: 16
```

By default the training config enables:

- LoRA
- DeepSpeed ZeRO-3
- gradient checkpointing
- Weights & Biases reporting

To enable the W&B login:

```bash
wandb login
```

## Fixed-corpus FlashRAG experiments

The RAG pipeline uses only the datasets and `wiki18_100w` corpus from
[`RUC-NLPIR/FlashRAG_datasets`](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets).
It does not download Search-R1 parquet files, checkpoints, trajectories,
retrievals, generations, or reported results. The index manifest pins and hashes
the corpus and 64 FP16 Qwen3 document-vector shards; training only instantiates a
query encoder and query-side LoRA. The downloadable FlashRAG E5 index is
intentionally unsupported because it is not in the Qwen3 embedding space.

Data preparation invokes the external `hf download` CLI and uses Python's
standard library to resolve the dataset revision. It does not add a direct
`huggingface_hub` dependency to this project. Ensure `hf` is available on the
host before running the preparation command.

Install a CUDA-compatible exact FAISS build on the four-GPU training host. Run
the frozen `Qwen/Qwen2.5-7B-Instruct` vLLM server in a separate GPU allocation
for `answer_f1` training or final generation:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --dtype bfloat16 --host 127.0.0.1 --port 8000
```

Prepare and hash the complete inputs, encode `contents` exactly once, and mine
the immutable E0 candidate pool:

```bash
scripts/rag_pipeline.sh prepare --revision <FLASHRAG_COMMIT_SHA>
scripts/rag_pipeline.sh encode \
  --revision <QWEN3_EMBEDDING_COMMIT_SHA> --num-shards 64 --max-length 512
scripts/rag_pipeline.sh candidates \
  --model-revision <QWEN3_EMBEDDING_COMMIT_SHA> --depth 1000
```

The prepare step validates the expected 169,615 raw training rows and 51,713
evaluation rows and audits normalized train/evaluation question overlap. NQ
queries with no answer-containing E0 top-1000 passage and HotpotQA queries with
unmappable supporting facts are excluded identically from every training method;
coverage is recorded next to the candidate JSONL.

Tune on the deterministic source-wise 95/5 split using the LR grid
`5e-6,1e-5,2e-5` and temperature grid `0.02,0.03,0.05`. Example tuning runs are:

```bash
scripts/rag_pipeline.sh train configs/rag/infonce.yaml --rag_split train --learning_rate 1e-5 --rag_temperature 0.03
scripts/rag_pipeline.sh train configs/rag/ranknet.yaml --rag_split train --learning_rate 1e-5 --rag_temperature 0.03
scripts/rag_pipeline.sh train configs/rag/rl_source_aware.yaml --rag_split train --learning_rate 1e-5
scripts/rag_pipeline.sh train configs/rag/rl_answer_mrr.yaml --rag_split train --learning_rate 1e-5
scripts/rag_pipeline.sh train configs/rag/rl_answer_f1.yaml --rag_split train --learning_rate 1e-5
```

Score any tuning checkpoint without touching the seven official evaluation
splits:

```bash
scripts/rag_pipeline.sh tune-eval \
  --checkpoint checkpoints/rag-rl-source-aware-mrr/checkpoint-1000 \
  --output results/tuning/source-aware-step1000.json
```

After selecting hyperparameters and a 95%-split step budget `S`, start a new
output directory from E0, set `rag_split=full`, and use
`max_steps=ceil(S/0.95)`. For example, `S=10000` becomes 10527:

```bash
scripts/rag_pipeline.sh train configs/rag/rl_source_aware.yaml \
  --rag_split full --max_steps 10527 --learning_rate 1e-5 \
  --run_name rag-rl-source-aware-final \
  --output_dir checkpoints/rag-rl-source-aware-final
```

Evaluate E0 by omitting `--checkpoint`, or evaluate a project-produced LoRA
adapter by providing it. The command reruns retrieval and generation and writes
per-dataset retrieval/generation JSONL plus `summary.json`:

```bash
scripts/rag_pipeline.sh eval \
  --checkpoint checkpoints/rag-rl-source-aware-mrr \
  --output-dir results/rag-rl-source-aware-mrr \
  --generator-endpoint http://127.0.0.1:8000
```

For an inexpensive retrieval smoke test, add `--retrieval-only`. The final run
reports Recall@5/20, MRR@20, frozen-generator EM/F1, the seven-dataset macro,
training-domain and held-out averages, plus multi-hop evidence coverage and
runtime telemetry. `configs/grpo/query_only.yaml` remains the older slate
ablation and is not an entrypoint for these fixed-corpus experiments.

## Evaluation

The wrapper script runs retrieval evaluation with the settings currently baked into [`eval_mteb/scripts/run_mteb.sh`](eval_mteb/scripts/run_mteb.sh), including:

- `--benchmark MTEB(eng, v1)`
- `--langs eng`
- `--batch_size 16`

Run evaluation with:

```bash
bash eval_mteb/scripts/run_mteb.sh \
  checkpoints/E2Rank-Full-GRPO-0.6B \
  exp/E2Rank-Full-GRPO-0.6B
```

Results are written under `results/mteb/<model_name>/`.

To summarize scores:

```bash
python eval_mteb/summary.py \
  results/mteb/E2Rank-Full-GRPO-0.6B/E2Rank-Full-GRPO-0.6B/no_version_available \
  "MTEB(eng, v2)"
```

If you need custom evaluation arguments, call the Python entrypoint directly:

```bash
python eval_mteb/run_mteb.py --help
```
