"""GRPO trainer plus the trainer plumbing shared with ``baselines.trainer.BaselineTrainer``.

Both trainers wrap the backbone in a ``nn.Module`` that returns a ``ModelOutput`` with
extra per-step scalars, need those scalars averaged across micro-batches *and* ranks
before they reach W&B, need ``EmbeddingDataset``'s per-source batching preserved, and need
the wrapper's ``"model."`` state-dict prefix stripped on save. That lives in
``EmbeddingTrainerMixin`` so each trainer only declares what is actually different.
"""

import json
import logging
import math
import os

import torch
from transformers import Trainer as HFTrainer

from embedding_data import EmbeddingDataset, SingleSourceBatchSampler


logger = logging.getLogger(__name__)


GRPO_STATE_FILENAME = "grpo_state.json"


def restore_grpo_state(model, checkpoint_dir: str | None) -> None:
    """Restore the learnable exploration scale saved next to a checkpoint.

    Call before the trainer is built, i.e. before DeepSpeed partitions the parameter.
    """
    if not checkpoint_dir or (
        not model.grpo.sigma_learnable and model.grpo.advantage_baseline != "ema"
    ):
        return
    state_path = os.path.join(checkpoint_dir, GRPO_STATE_FILENAME)
    if not os.path.exists(state_path):
        logger.warning("No %s in %s; sigma restarts from its configured init.", GRPO_STATE_FILENAME, checkpoint_dir)
        return

    with open(state_path, "r", encoding="utf-8") as fp:
        state = json.load(fp)
    with torch.no_grad():
        if model.grpo.sigma_learnable and state.get("sigma") is not None:
            sigma = float(state["sigma"])
            model.grpo.log_sigma.fill_(math.log(sigma))
            logger.info("Restored learnable sigma=%s from %s", sigma, state_path)
        if model.grpo.advantage_baseline == "ema" and "reward_baseline" in state:
            model.grpo.reward_baseline.fill_(float(state["reward_baseline"]))
            model.grpo.reward_baseline_initialized.fill_(
                bool(state.get("reward_baseline_initialized", True))
            )
            logger.info("Restored EMA reward baseline from %s", state_path)


def build_single_source_sampler(trainer: HFTrainer, train_dataset):
    """Return a block-preserving sampler for ``EmbeddingDataset``, else ``None``.

    The Trainer's default ``RandomSampler`` shuffles at the sample level, which
    destroys the per-source pre-batching that ``EmbeddingDataset`` builds (and that
    in-batch negatives depend on).
    """
    if not isinstance(train_dataset, EmbeddingDataset):
        return None

    dataloader_batch_size = getattr(trainer, "_train_batch_size", None) or trainer.args.train_batch_size
    if train_dataset.batch_size != dataloader_batch_size:
        logger.warning(
            "EmbeddingDataset was pre-batched with batch_size=%s but the dataloader uses %s; "
            "batches will span multiple sources. Rebuild the dataset with the dataloader batch size.",
            train_dataset.batch_size,
            dataloader_batch_size,
        )
    return SingleSourceBatchSampler(
        dataset=train_dataset,
        batch_size=train_dataset.batch_size,
        seed=trainer.args.seed,
    )


def save_wrapped_backbone(trainer: HFTrainer, output_dir=None, state_dict=None) -> str:
    """Save the wrapped backbone, stripping the ``"model."`` prefix added by the wrapper.

    ``Trainer.save_model`` calls ``_save(output_dir)`` without a state dict on the
    plain (non-DeepSpeed, non-FSDP) path, so fall back to the wrapper's own state
    dict instead of dereferencing ``None``.
    """
    output_dir = output_dir if output_dir is not None else trainer.args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving model checkpoint to {output_dir}")

    if state_dict is None:
        state_dict = trainer.model.state_dict()

    model_to_save = trainer.deepspeed.model if trainer.is_deepspeed_enabled else trainer.model.model
    model_to_save.save_pretrained(
        output_dir,
        safe_serialization=trainer.args.save_safetensors,
        state_dict={
            key.removeprefix("model."): value
            for key, value in state_dict.items()
            if key.startswith("model.")
        },
    )

    processing_class = getattr(trainer, "processing_class", None)
    if processing_class is not None and trainer.is_world_process_zero():
        processing_class.save_pretrained(
            output_dir,
            safe_serialization=trainer.args.save_safetensors,
        )
    return output_dir


class EmbeddingTrainerMixin:
    """Accumulate model-emitted scalars per step, reduce across ranks, rename for W&B.

    Subclasses declare ``train_metric_names`` (which fields of the model output to track)
    and ``train_metric_log_names`` (how each is spelled in the logs; may be a property).
    Metrics whose key set is config-driven rather than static go through
    ``_extra_train_metrics``, which must return the same keys in the same order on every
    rank -- ``_consume_train_metrics`` reduces the accumulator dict entry by entry.
    """

    base_log_name_map = {
        "loss": "train/loss",
        "learning_rate": "train/lr",
        "grad_norm": "train/grad_norm",
        "epoch": "train/epoch",
    }
    train_metric_names: tuple[str, ...] = ()
    train_metric_log_names: dict[str, str] = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_metric_sums: dict[str, torch.Tensor] = {}
        self._train_metric_updates = 0

    @classmethod
    def _rename_log_keys(cls, logs: dict[str, float]) -> dict[str, float]:
        return {cls.base_log_name_map.get(key, key): value for key, value in logs.items()}

    @staticmethod
    def _output_field(outputs, name):
        if isinstance(outputs, dict):
            return outputs.get(name)
        return getattr(outputs, name, None)

    def _extra_train_metrics(self, outputs) -> dict:
        """Pre-namespaced, rank-invariant scalars that the static whitelist cannot cover."""
        return {}

    def _accumulate_train_metrics(self, outputs) -> None:
        extra_metrics = self._extra_train_metrics(outputs)
        for metric_name in (*self.train_metric_names, *extra_metrics):
            if metric_name in extra_metrics:
                metric_value = extra_metrics[metric_name]
            else:
                metric_value = self._output_field(outputs, metric_name)
            if metric_value is None:
                continue
            if not isinstance(metric_value, torch.Tensor):
                metric_value = torch.tensor(metric_value, device=self.args.device, dtype=torch.float32)
            metric_value = metric_value.detach()
            if metric_value.numel() != 1:
                metric_value = metric_value.mean()
            metric_value = metric_value.to(device=self.args.device, dtype=torch.float32)
            self._train_metric_sums[metric_name] = self._train_metric_sums.get(
                metric_name,
                torch.zeros((), device=self.args.device, dtype=torch.float32),
            ) + metric_value
        self._train_metric_updates += 1

    def _consume_train_metrics(self) -> dict[str, float]:
        if self._train_metric_updates == 0:
            return {}

        logs = {}
        metric_count = torch.tensor(
            float(self._train_metric_updates),
            device=self.args.device,
            dtype=torch.float32,
        )
        total_metric_count = self._nested_gather(metric_count).sum().item()

        for metric_name, metric_sum in self._train_metric_sums.items():
            total_metric_sum = self._nested_gather(metric_sum).sum().item()
            log_name = self.train_metric_log_names.get(metric_name, metric_name)
            logs[log_name] = round(total_metric_sum / max(total_metric_count, 1.0), 6)

        self._train_metric_sums = {}
        self._train_metric_updates = 0
        return logs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )
        if model.training:
            self._accumulate_train_metrics(outputs)
        return (loss, outputs) if return_outputs else loss

    def _peak_memory_gib(self) -> float | None:
        """Largest allocation high-water mark across ranks, in GiB.

        Deliberately never reset: the reported number is the peak over the whole run, which is
        what decides whether a configuration fits, and it is the axis the rollout ablations
        trade against (the reward tensor grows as G^C, and with in-batch candidates its last
        dimension grows as batch x slate). Max-reduced rather than averaged because one rank
        OOMing is what OOM means. Allocated, not reserved, so it does not move with the caching
        allocator's fragmentation.
        """
        if not torch.cuda.is_available():
            return None
        peak = torch.tensor(
            float(torch.cuda.max_memory_allocated()),
            device=self.args.device,
            dtype=torch.float64,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(peak, op=torch.distributed.ReduceOp.MAX)
        return peak.item() / 1024**3

    def log(self, logs, start_time=None):
        if "loss" in logs:
            logs = {**logs, **self._consume_train_metrics()}
            peak_memory = self._peak_memory_gib()
            if peak_memory is not None:
                logs["train/peak_mem_gib"] = round(peak_memory, 3)
        super().log(self._rename_log_keys(logs), start_time=start_time)

    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        sampler = build_single_source_sampler(self, dataset)
        if sampler is not None:
            return sampler
        return super()._get_train_sampler(train_dataset)

    def _save(self, output_dir=None, state_dict=None) -> str:
        return save_wrapped_backbone(self, output_dir=output_dir, state_dict=state_dict)


class GRPOTrainer(EmbeddingTrainerMixin, HFTrainer):
    train_metric_log_names = {
        "reward": "reward",
        "reward_mean": "reward/mean",
        "reward_std": "reward/std",
        "reward_min": "reward/min",
        "reward_max": "reward/max",
        "advantages_mean": "advantages/mean",
        "advantages_std": "advantages/std",
        "advantages_min": "advantages/min",
        "advantages_max": "advantages/max",
        "advantages_degenerate_frac": "advantages/degenerate_frac",
        "sigma": "sigma",
        "kl": "kl",
    }
    train_metric_names = tuple(train_metric_log_names)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Sigma observed during the forward pass. Reading grpo.log_sigma directly at save
        # time is unsafe under ZeRO-3 (the parameter is a rank-local shard outside a
        # gather context, and _save runs on one rank only, so we cannot collect it there).
        self._last_sigma: float | None = None

    def _extra_train_metrics(self, outputs) -> dict:
        # Per-reward-term scalars are config-driven, so the static whitelist cannot cover
        # them. They arrive pre-namespaced and in a rank-invariant order, which
        # _consume_train_metrics needs because it reduces the accumulator entry by entry.
        return self._output_field(outputs, "reward_terms") or {}

    def _accumulate_train_metrics(self, outputs) -> None:
        sigma = self._output_field(outputs, "sigma")
        if sigma is not None:
            self._last_sigma = float(sigma)
        super()._accumulate_train_metrics(outputs)

    def _save(self, output_dir=None, state_dict=None) -> str:
        output_dir = super()._save(output_dir=output_dir, state_dict=state_dict)

        # The backbone save above only covers `model.*`; the learnable exploration scale
        # lives on the GRPO head and would otherwise silently reset to its init on resume.
        if self.is_world_process_zero() and (
            (self.model.grpo.sigma_learnable and self._last_sigma is not None)
            or self.model.grpo.advantage_baseline == "ema"
        ):
            state = {}
            if self.model.grpo.sigma_learnable and self._last_sigma is not None:
                state["sigma"] = self._last_sigma
            if self.model.grpo.advantage_baseline == "ema":
                state["reward_baseline"] = float(self.model.grpo.reward_baseline)
                state["reward_baseline_initialized"] = bool(
                    self.model.grpo.reward_baseline_initialized
                )
            with open(os.path.join(output_dir, GRPO_STATE_FILENAME), "w", encoding="utf-8") as fp:
                json.dump(state, fp)
        return output_dir
