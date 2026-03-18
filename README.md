# RL Training and Evaluation for Embedding

This repository keeps the RL training pipeline and evaluation scripts for embedding models.

Current scope:

- GRPO-based RL training in [`src/train.py`](src/train.py)
- Config-driven experiments in [`configs/`](configs)
- Shell launcher in [`scripts/run.sh`](scripts/train_rl_0.6b.sh)
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
bash ./scripts/run.sh configs/exp/train_rl_0.6b_ndcg.yaml
```

The top-level config currently resolves to `configs/exp/train_rl_0.6b_ndcg.yaml`.

The config layout is:

```text
configs/
  base/
  model/
  reward/
  exp/
```

Example inheritance:

```yaml
_base_:
  - ../base/train.yaml
  - ../model/e2rank_0.6b_embedding_only.yaml
  - ../reward/ndcg.yaml

data_path: data/train.jsonl
output_dir: checkpoints/E2Rank-Full-GRPO-0.6B
run_name: E2Rank-Full-GRPO-0.6B
```

Important RL-related fields exposed by [`src/config.py`](src/config.py):

- `rl_mode`: `query_only`, `listwise_only`, or `dual`
- `group_size`
- `sigma`
- `sigma_learnable`
- `query_reward_type`
- `listwise_reward_type`
- `query_reward_ndcg_k`
- `listwise_reward_ndcg_k`
- `listwise_loss_weight`
- `advantage_norm`
- `query_relevance_scheme`
- `listwise_relevance_scheme`

Supported reward presets in [`configs/reward/`](configs/reward):

- `ndcg.yaml`
- `mixed.yaml`
- `contrastive.yaml`
- `mrr.yaml`

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
