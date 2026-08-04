from transformers import Trainer as HFTrainer

from grpo_trainer import RankingTrainerMixin


class BaselineTrainer(RankingTrainerMixin, HFTrainer):
    train_metric_names = ("ndcg", "mrr")

    def __init__(self, *args, metric_k: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric_k = metric_k

    @property
    def train_metric_log_names(self) -> dict[str, str]:
        suffix = f"@{self.metric_k}" if self.metric_k is not None else ""
        return {
            "ndcg": f"train/ndcg{suffix}",
            "mrr": f"train/mrr{suffix}",
        }
