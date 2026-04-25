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

Training expects a JSONL file at `data/train.jsonl`.

You can download the prepared dataset with:

```bash
mkdir -p data
hf download \
  Alibaba-NLP/E2Rank_ranking_datasets \
  train.jsonl \
  --local-dir ./data \
  --repo-type dataset
```

Each sample should contain the fields used by [`src/ranking_data.py`](src/ranking_data.py):

- `query`
- `document`
- `ranking`
- `source` (optional, used to choose task prompts)

## Training

The recommended entrypoint is the shell wrapper plus an explicit config path:

```bash
bash ./scripts/run.sh configs/exp/template.yaml
```

You can run training without any `configs/exp/*.yaml` and compose the config directly from the shell:

```bash
bash ./scripts/run.sh \
  --base-train configs/train/default.yaml \
  --base-model configs/model/e2rank_0.6b_embedding_only.yaml \
  --base-grpo configs/grpo/default.yaml \
  --base-reward configs/reward/contrastive_in_batch.yaml \
  --base-eval configs/eval/default.yaml \
  --data_path data/train.jsonl
```

If you omit `run_name` and `output_dir`, they are auto-generated. For example, the command above becomes:

- `run_name=e2rank_0.6b_embedding_only__contrastive_in_batch`
- `output_dir=checkpoints/<run_name>`

Supported base config flags:

- `--base-train`
- `--base-model`
- `--base-grpo`
- `--base-reward`
- `--base-eval`

Regular training arguments can still be appended after that:

```bash
bash ./scripts/run.sh \
  --base-train configs/train/default.yaml \
  --base-model configs/model/qwen3_embedding_0.6b.yaml \
  --base-grpo configs/grpo/default.yaml \
  --base-reward configs/reward/mrr.yaml \
  --base-eval configs/eval/default.yaml \
  --data_path data/train.jsonl \
  --learning_rate 5e-5 \
  --sigma 0.03
```

You can still override either one manually:

```bash
bash ./scripts/run.sh \
  --base-train configs/train/default.yaml \
  --base-model configs/model/qwen3_embedding_0.6b.yaml \
  --base-grpo configs/grpo/default.yaml \
  --base-reward configs/reward/mrr.yaml \
  --base-eval configs/eval/default.yaml \
  --data_path data/train.jsonl \
  --run_name qwen3-embed-mrr
```

In that case `output_dir` falls back to `checkpoints/qwen3-embed-mrr`.

If you still want to keep a top-level experiment template, the old form also works:

```bash
bash ./scripts/run.sh configs/exp/template.yaml
```

For simple sweeps over `train/model/grpo/reward/eval` entries, use [`scripts/run_grid.py`](scripts/run_grid.py):

```bash
uv run python scripts/run_grid.py \
  --set-base train=configs/train/default.yaml \
  --set-base model=configs/model/qwen3_0.6b.yaml,configs/model/e2rank_0.6b_embedding_only.yaml \
  --set-base grpo=configs/grpo/default.yaml \
  --set-base reward=configs/reward/ndcg.yaml,configs/reward/contrastive_in_batch.yaml \
  --set-base eval=configs/eval/default.yaml,configs/eval/mteb_retrieval.yaml \
  --run-name-prefix sweep \
  --output-root checkpoints/sweep \
  --dry-run \
  -- \
  --data_path data/train.jsonl
```

The example top-level config is `configs/exp/template.yaml`.

The config layout is:

```text
configs/
  train/
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
  - ../model/e2rank_0.6b_embedding_only.yaml
  - ../grpo/default.yaml
  - ../reward/ndcg.yaml
  - ../eval/default.yaml

data_path: data/train.jsonl
output_dir: checkpoints/E2Rank-Full-GRPO-0.6B
run_name: E2Rank-Full-GRPO-0.6B
```

Important RL-related fields exposed by [`src/config.py`](src/config.py):

- `grpo_mode`
- `group_size`
- `sigma`
- `sigma_learnable`
- `perturb_negatives`
- `reward_type`
- `reward_ndcg_k`
- `ndcg_in_batch_include_negatives`
- `contrastive_use_in_batch_negatives`
- `contrastive_temperature`
- `advantage_norm`
- `relevance_scheme`

Supported reward presets in [`configs/reward/`](configs/reward):

- `ndcg.yaml`
- `ndcg_in_batch.yaml`
- `contrastive_in_batch.yaml`
- `contrastive_no_in_batch.yaml`
- `infonce_in_batch.yaml`
- `infonce_no_in_batch.yaml`
- `mrr.yaml`

Contrastive-style rewards now follow the paper appendix:

- `ndcg`: per-query slate nDCG only, kept as the backward-compatible no-in-batch option
- `ndcg_in_batch`: append positives from other samples in the batch as extra zero-relevance candidates; set `ndcg_in_batch_include_negatives: true` to append all candidates from other samples
- `contrastive`: `s+ - tau * logsumexp(s- / tau)` over negatives only
- `infonce`: `s+ - tau * logsumexp([s+, s-] / tau)` over the full partition
- `mrr`: under `graded` relevance, only labels with `relevance >= 2` count as relevant; under `binary`, the threshold remains `relevance > 0`

Example:

```yaml
reward_type: contrastive
contrastive_temperature: 0.03
```

MTEB eval during GRPO training is disabled by default through [`configs/eval/default.yaml`](configs/eval/default.yaml). To evaluate every saved checkpoint with the existing `MTEB(eng, v1, subset)` benchmark preset, use:

```bash
bash ./scripts/run.sh configs/exp/template.yaml \
  --base-eval configs/eval/mteb_retrieval.yaml
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
