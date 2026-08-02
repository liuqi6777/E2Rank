import torch
from transformers import Trainer as HFTrainer

from grpo_trainer import build_single_source_sampler, save_wrapped_backbone
from ranking_eval import RankingEvalMixin


class BaselineTrainer(RankingEvalMixin, HFTrainer):
    base_log_name_map = {
        "loss": "train/loss",
        "learning_rate": "train/lr",
        "grad_norm": "train/grad_norm",
        "epoch": "train/epoch",
    }
    train_metric_names = ("ndcg", "mrr")

    @classmethod
    def _rename_log_keys(cls, logs: dict[str, float]) -> dict[str, float]:
        return {cls.base_log_name_map.get(key, key): value for key, value in logs.items()}

    def __init__(self, *args, metric_k: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric_k = metric_k
        self._train_metric_sums: dict[str, torch.Tensor] = {}
        self._train_metric_updates = 0

    @property
    def train_metric_log_names(self) -> dict[str, str]:
        suffix = f"@{self.metric_k}" if self.metric_k is not None else ""
        return {
            "ndcg": f"train/ndcg{suffix}",
            "mrr": f"train/mrr{suffix}",
        }

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

    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        sampler = build_single_source_sampler(self, dataset)
        if sampler is not None:
            return sampler
        return super()._get_train_sampler(train_dataset)

    def _save(self, output_dir=None, state_dict=None):
        save_wrapped_backbone(self, output_dir=output_dir, state_dict=state_dict)
