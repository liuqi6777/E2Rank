# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

GRPO-based reinforcement-learning training for embedding models. Treats an embedding as a stochastic action on the unit hypersphere (von Mises–Fisher policy, sampled exactly with Wood's algorithm) and reuses an HF Trainer for the optimization loop. Ships with several reward presets (nDCG / contrastive / InfoNCE / MRR, singly or additively combined), a layered YAML config system, and an in-training MTEB/BEIR eval callback. The one supervised counterpart is InfoNCE (`src/train_baseline.py`), sharing the same data and trainer plumbing; it serves as Stage 1, as the compute-matched CL→CL control, and as the backpropagated twin of the InfoNCE-as-reward row.

Python 3.10, managed with `uv` (`.python-version` pinned). Single-source-of-truth deps in `pyproject.toml`; `uv sync` installs them.

## Common commands

```bash
# env
uv sync && source .venv/bin/activate

# GRPO training (preferred entrypoint — wraps torchrun)
bash ./scripts/run.sh configs/exp/template.yaml

# Or compose configs slot-by-slot without a top-level YAML
bash ./scripts/run.sh \
  --base-train  configs/train/stage2.yaml \
  --base-dataset configs/dataset/stage2.yaml \
  --base-model  configs/model/qwen3_0.6b.yaml \
  --base-grpo   configs/grpo/default.yaml \
  --base-reward configs/reward/default_mixture.yaml \
  --base-eval   configs/eval/default.yaml \
  --learning_rate 5e-5            # any HfTrainingArguments / dataclass field can be appended

# Supervised (InfoNCE) training — Stage 1 and the CL->CL control. Same slots minus grpo/reward.
bash ./scripts/run_baseline.sh --base-train configs/train/stage1.yaml ...

# The paper's runs: one script per phase, driven by scripts/experiments/_common.sh.
# Stage 1 lives in its own script and the phase scripts REQUIRE it rather than training it.
bash scripts/experiments/stage1.sh              # once per scale; everything branches from it
bash scripts/experiments/phase1_pilot.sh        # gates the rest on training diagnostics
DRY_RUN=1 bash scripts/experiments/phaseB_recipe.sh    # inspect before committing GPUs
uv run python scripts/validate_experiments.py   # parse every command in every script

# Post-hoc MTEB evaluation
bash eval_mteb/scripts/run_mteb.sh checkpoints/<run_name> <exp_label>
python eval_mteb/summary.py results/mteb/<model_name>/<exp_label>/no_version_available "MTEB(eng, v2)"
```

Distributed defaults: `NNODES=1`, `NPROC_PER_NODE=8` (override via env). `WANDB_PROJECT` defaults to `E2Rank-RL`; `wandb login` needed once.

## Architecture

### Training entrypoint (`src/train.py`, `src/train_baseline.py`)
`train.py` owns the shared setup helpers — `split_launcher_args` / `parse_arguments` (YAML, `--base-<slot>` flags, or plain CLI, then `run_name`/`output_dir` resolution), `guard_output_dir`, `setup_logging`, `load_backbone_and_tokenizer` (AutoConfig + AutoModel + AutoTokenizer + optional PEFT/LoRA), `apply_gradient_checkpointing`, `build_embedding_data`, `save_run_artifacts`, `shutdown_distributed` — and `train_baseline.py` imports them. Each `main()` is then only: parse → load backbone → wrap in `GRPOModel`/`BaselineModel` → build the trainer → `train()` → save. `save_run_artifacts` routes through `save_model_for_trainer`, which sends DeepSpeed ZeRO-3 saves to `trainer.save_model` and gathers state on CPU otherwise.

### Config system (`src/utils.py`)
Configs compose via a `_base_` list at the YAML top level. Inheritance is resolved recursively; later entries (and CLI args) win. Slot names in `BASE_CONFIG_SLOTS = ("train", "dataset", "model", "grpo", "reward", "eval")` map to `--base-<slot>` launcher flags — the **parent directory name** of each `_base_` entry decides which slot it occupies (so move a file between slot dirs only with intent). `run_name`/`output_dir` are auto-derived from non-default slot stems when omitted; `checkpoints/<run_name>` is the default output.

### GRPO core (`src/grpo.py`)
`GRPO.forward` samples `group_size` vMF actions per **active action component** from *detached* rollout means (exact Wood rejection sampler, `kappa = 1/sigma^2`), scores them into a reward table, converts rewards to per-component advantages, and returns `-(advantage.detach() * log_prob).mean()` summed over components. Gradients reach the encoder only through the log-prob `kappa * h^T e` against the live policy embeddings; the vMF normalizer is omitted because it cancels under group-centered advantages. Key design points:

- **Action components** (`RLArguments.action_components`) are groups like `[[query]]`, `[[query], [positive, negative]]`, `[[positive]]`. Each unique role appears in at most one group; `query` must be solo. The joint `[positive, negative]` group encodes positive+negative documents together so one sigma noise term covers them.
- When a component is *not* sampled, its rollout embedding is reused as a detached score input; only sampled components contribute to log-probs.
- Advantages are computed per-component by **marginalizing rewards over the other sampled components' axes**, then normalized within the group per `advantage_norm` mode: `per_component` (each component's group → unit std; legacy `true`), `shared` (all components divided by the per-sample std of the raw reward tensor, preserving relative effect sizes), or `none` (centering only; legacy `false`). Mean/std are promoted to fp32 to dodge bf16 round-off bias (`_compute_advantages`). Degenerate groups (reward spread ≈ 0, threshold scaled by reward magnitude) are zeroed instead of dividing.
- Score tables are built in **fp32** even under bf16 training (`_compute_score_table`) — bf16 cosine round-off reorders the slate, and since `topk` breaks ties toward index 0 where the collator puts the gold positive, it biases the reward upward rather than just adding noise.
- In-batch candidates from other samples are scored with **detached mean embeddings** by default (`in_batch_use_sampled_documents=False`); the legacy sampled-embedding behavior leaks other samples' perturbations into each sample's advantages via the shared group index.
- Whenever any document component is sampled, **every** frozen-document score table — unsampled slate components as well as in-batch cross tables — is rescaled by `A_d(kappa)` (`_frozen_doc_scale`) so frozen and sampled candidates land on the same expected score scale. Dropping the rescale makes frozen candidates outrank every sampled document; the signature is `reward/std → 0` with `degenerate_frac → 1`.
- Optional KL term (`kl_coef > 0`): vMF KL `kappa * A_d(kappa) * (1 - mu_pol . mu_ref)` between policy μ and the **LoRA-disabled base model** μ (so the reference is the un-adapted policy, not a frozen copy). Requires a PEFT/LoRA model exposing `disable_adapter()`.
- `sigma` is either a fixed buffer or `log_sigma` nn.Parameter (`sigma_learnable=True`), clamped to `[sigma_min, sigma_max]` to prevent exploration collapse; `kappa` in `RLArguments` overrides `sigma` directly (`sigma = 1/sqrt(kappa)`).

### Rewards (`src/rewards.py`)
`compute_reward_from_scores` dispatches on `reward_type`:
- `ndcg` / `ndcg_in_batch` (optionally including all in-batch candidates as zero-relevance distractors when `ndcg_in_batch_include_negatives`),
- `contrastive` = `s+ − τ·logsumexp(s−/τ)` (no positive in the partition),
- `infonce` = `s+ − τ·logsumexp([s+, s−]/τ)`,
- `mrr` (binary uses `>0`; graded uses `≥2`).
The `in_batch_positive_scores` / `in_batch_candidate_scores` inputs are pre-expanded by `GRPO._compute_score_table` with the `cross=True` einsum path; `_mask_cross_batch_diagonal` drops the same-sample diagonal *before* expansion, so the reward never sees it and the mask never materializes at the expanded size.

**Additive combination.** `reward_terms` (a list of `RewardTerm`s, normalized by `normalize_reward_terms`) replaces the single `reward_type` when set; each term carries its own `weight`, `k`, `temperature` and in-batch flags, and unset per-term fields inherit the run-level defaults so a one-term spec is byte-identical to the legacy path. YAML uses a list of mappings, CLI uses `"ndcg:1.0,k=16;contrastive:0.5,in_batch_negatives=true"`. `_compute_component_loss` builds the cross-batch score tables **once** from the union of what the terms request (`reward_terms_need_in_batch_positives` / `..._candidates`) and hands the same tables to every term.

`reward_combine` decides where the addition happens, and the two are *not* equivalent because advantages are divided by a group std:
- `sum` — `R = Σ wᵢRᵢ`, one advantage over the combined reward. Each term reaches the gradient in proportion to `wᵢ × its own within-group std`, so with a bounded nDCG (std ~1e-1) against a temperature-scaled contrastive margin the weights do **not** describe the mixing ratio. Log `reward/<term>/group_std` to see the ratio actually obtained.
- `normalized_sum` — each term is group-standardized into its own advantage, then `A = Σ wᵢAᵢ`. The surrogate is linear in `A`, so this is still a reward combination, just a scale-free one where `wᵢ` is the mixing ratio. Required in practice whenever `BOUNDED_REWARD_TYPES` meets `UNBOUNDED_REWARD_TYPES`; `RLArguments.__post_init__` warns when that mix runs under `sum`. Incompatible with `advantage_baseline='ema'` for >1 term (one global scalar cannot baseline several reward scales).

With more than one term, per-term stats are emitted as `reward/<term>/{mean,std,min,max,group_std}` and flow to W&B through `GRPOModelOutput.reward_terms`. Independently of term count, every (advantage input × sampled component) pair also emits `reward/<name>/<component>/{group_std,degenerate_frac}` — the marginalized per-axis signal each component's advantage was actually computed from (`<name>` is the term, or `combined` under raw `sum` with several terms; `<component>` is `query`/`documents`/`positive`/`negative`). The aggregate `advantages/*` stats concatenate axes, so these per-axis keys are the only way to see which component carries the gradient — the observable behind the §0.9 component×reward interaction test in `paper/EXPERIMENT_PLAN.md`.

### Trainer wrapper (`src/grpo_trainer.py`)
`RankingTrainerMixin` (also used by `BaselineTrainer`, which imports it from here) carries the plumbing both trainers need:
1. it accumulates the scalars named in `train_metric_names` across micro-batches, reduces them across ranks on `log()`, and renames keys via `train_metric_log_names` + `base_log_name_map` into a `train/…`, `reward/…`, `advantages/…` schema for W&B. `train_metric_names` is a static whitelist; config-driven metrics bypass it through the `_extra_train_metrics` hook, which `GRPOTrainer` fills from the model output's `reward_terms` dict — those arrive pre-namespaced and in a rank-invariant order because the cross-rank reduce walks the accumulator dict entry by entry;
2. it overrides `_get_train_sampler` with `SingleSourceBatchSampler`, without which the stock `RandomSampler` silently destroys `RankingDataset`'s per-source batching (see Data below); and
3. it overrides `_save` (via `save_wrapped_backbone`) to strip the `"model."` prefix introduced by the `GRPOModel`/`BaselineModel` wrapper. Note this writes a **PEFT adapter** whenever `lora_enabled` — the default — not a full model; only non-LoRA runs reload with plain `AutoModel.from_pretrained`.

`GRPOTrainer` adds only the reward-term hook and the sigma bookkeeping; `BaselineTrainer` adds only its `@k`-suffixed log names.

A learnable `sigma` lives outside those `model.*` keys, so `GRPOTrainer._save` writes it to `grpo_state.json` and `restore_grpo_state` reloads it in `train.py` before the trainer is built (DeepSpeed partitions the parameter after that point).

### Data (`src/embedding_data.py`)
`EmbeddingDataset` reads JSONL with fields `query`, `document` (list of texts), `ranking` (1-indexed permutation), and optional `source`. Source picks a task-specific instruction prompt from `TASK_PROMPTS` and formats `Instruct: <task>\nQuery:<query>`. Samples are pre-batched **per source** so every batch contains samples from a single task (the trailing partial batch per source is dropped); batches are then shuffled. `per_dataset_max_samples` caps each source independently.

That layout survives to the GPU only because both trainers install `SingleSourceBatchSampler`, which shuffles whole batch-sized **blocks** rather than individual samples. Keep the dataset's `batch_size` equal to the dataloader's, or batches go mixed-source again (logged as a warning).

`RankingDataCollator` left-pads with `tokenizer.pad_token` *appended to the text* (Qwen-style EOS-as-pad), reorders documents to put the gold positive first, and builds graded/binary relevance labels via `build_relevance_labels` (graded: 3 for rank 1, 2 for ranks 2–5, 1 for ranks 6–10, 0 elsewhere).

It emits the slate **split** into `positive_document` (`[batch, ...]`) and `negative_document` (`[batch * (slate-1), ...]`, sample-major). Anything that encodes a slate must put it back together with `build_slate_inputs`, which interleaves them into `(sample 0 positive, sample 0 negatives, sample 1 positive, ...)` so the downstream `reshape(batch, slate, -1)` lines each sample's own candidates up with its own `relevance_labels`. A plain `cat((positive, negative), dim=0)` looks equivalent and is not — it lays out all positives first, so `reshape` deals other samples' positives into sample 0's slate. That was a live bug in `BaselineModel.forward` and `score_slate_deterministically`.

### MTEB-during-training (`src/mteb_eval_callback.py`)
Runs on every `on_save`. In distributed mode every rank loads its own eval-model copy onto `cuda:<local_rank>`, then encode calls are sharded across ranks via a dedicated **gloo** sub-group. Loading happens inside `_disable_deepspeed_zero3`, which detaches the global `HfDeepSpeedConfig` so `from_pretrained` does not partition the eval model with `zero.Init` — without it the weights come back as rank-local shards and encoding breaks. Metrics are logged through the trainer under `eval_mteb/<task>/main_score`.

### LoRA + DeepSpeed gotcha (`src/utils.py:resolve_gradient_checkpointing_kwargs`)
With ZeRO-3 + LoRA + gradient checkpointing, `use_reentrant=False` triggers `torch.utils.checkpoint.CheckpointError` on empty ZeRO-3 parameter shards. The util forces `use_reentrant=True` in that combination — keep this in mind if changing checkpointing settings.

### Eval CLI (`eval_mteb/run_mteb.py`)
Standalone MTEB runner consumed both by `eval_mteb/scripts/run_mteb.sh` (post-hoc) and by `MTEBEvalCallback` (in-training). `EvalArguments` and `get_model`/`get_tasks`/`run_eval` are the public surface.

## Notes for editing

- The repo is launched almost exclusively through `bash scripts/run.sh ...`; calling `python src/train.py` directly bypasses `torchrun` and the env vars (`FORCE_TORCHRUN`, `NPROC_PER_NODE`, `WANDB_PROJECT`).
- `GRPOModel.forward` re-encodes documents with `torch.no_grad()` when **no** document role is sampled; if you add a new sampled role, mirror the existing `sample_positive`/`sample_negative` branches in both the encode block and the reference-policy KL block.
- New reward types must be registered in `SUPPORTED_REWARD_TYPES` (and in `BOUNDED_REWARD_TYPES` / `UNBOUNDED_REWARD_TYPES` so the scale-mixing warning stays accurate) and handled in both `compute_reward_from_scores` and the `reward_terms_need_in_batch_*` predicates that drive in-batch score-table construction.
- The dataset assumes `len(documents) == max(ranking)`; downstream tensor shapes will silently mismatch if that invariant is broken.
