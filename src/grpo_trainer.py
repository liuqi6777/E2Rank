import os

import torch
from transformers import Trainer as HFTrainer


class GRPOTrainer(HFTrainer):
    base_log_name_map = {
        "loss": "train/loss",
        "learning_rate": "train/lr",
        "grad_norm": "train/grad_norm",
        "epoch": "train/epoch",
    }
    train_metric_names = (
        "reward",
        "reward_mean",
        "reward_std",
        "reward_min",
        "reward_max",
        "advantages_mean",
        "advantages_std",
        "advantages_min",
        "advantages_max",
        "sigma",
        "kl",
    )
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
        "sigma": "sigma",
        "kl": "kl",
    }

    @classmethod
    def _rename_log_keys(cls, logs: dict[str, float]) -> dict[str, float]:
        renamed_logs = {}
        for key, value in logs.items():
            renamed_key = cls.base_log_name_map.get(key, key)
            renamed_logs[renamed_key] = value
        return renamed_logs

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_metric_sums: dict[str, torch.Tensor] = {}
        self._train_metric_updates = 0

    def _accumulate_train_metrics(self, outputs) -> None:
        for metric_name in self.train_metric_names:
            if isinstance(outputs, dict):
                metric_value = outputs.get(metric_name)
            else:
                metric_value = getattr(outputs, metric_name, None)
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

    def log(self, logs, start_time=None):
        if "loss" in logs:
            logs = {**logs, **self._consume_train_metrics()}
        super().log(self._rename_log_keys(logs), start_time=start_time)

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving model checkpoint to {output_dir}")

        model_to_save = self.deepspeed.model if self.is_deepspeed_enabled else self.model.model
        model_to_save.save_pretrained(
            output_dir,
            safe_serialization=self.args.save_safetensors,
            state_dict={
                key.removeprefix("model."): value
                for key, value in state_dict.items()
                if key.startswith("model.")
            },
        )

        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(
                output_dir,
                safe_serialization=self.args.save_safetensors,
            )
