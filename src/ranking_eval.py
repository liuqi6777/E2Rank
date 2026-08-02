"""Deterministic development-set evaluation shared by the GRPO and baseline trainers.

Model selection and hyperparameter tuning must not touch MTEB: tuning a surrogate's
smoothing parameter or a learning rate on the benchmark the paper reports is the
easiest way to invalidate every number in it. This module scores a held-out split
with the *deployed* quantity -- the deterministic mean embedding, no sampling -- so
dev nDCG measures what inference will actually do.
"""

from __future__ import annotations

from typing import Any

import torch


EVAL_METRIC_NAMES = ("ndcg", "mrr")


def score_slate_deterministically(
    encode,
    query: dict[str, torch.Tensor],
    positive_document: dict[str, torch.Tensor],
    negative_document: dict[str, torch.Tensor],
    slate_length: int,
) -> torch.Tensor:
    """Cosine scores of a query against its slate, from mean embeddings only."""
    batch_size = slate_length and positive_document["input_ids"].size(0)
    query_embeddings = encode(query)
    document_inputs = {
        key: torch.cat((positive_document[key], negative_document[key]), dim=0)
        for key in positive_document
    }
    document_embeddings = encode(document_inputs).reshape(batch_size, slate_length, -1)
    return torch.matmul(document_embeddings, query_embeddings.unsqueeze(-1)).squeeze(-1)


def ranking_eval_metrics(
    scores: torch.Tensor,
    relevance_labels: torch.Tensor,
    k: int | None,
) -> dict[str, torch.Tensor]:
    # Imported lazily: `baselines/__init__` pulls in BaselineTrainer, which imports this
    # module, so a top-level import here would be circular.
    from baselines.metrics import compute_mrr_at_k, compute_ndcg_at_k

    return {
        "ndcg": compute_ndcg_at_k(scores=scores, relevance_labels=relevance_labels, k=k).mean(),
        "mrr": compute_mrr_at_k(scores=scores, relevance_labels=relevance_labels, k=k).mean(),
    }


def ranking_compute_metrics(eval_prediction) -> dict[str, float]:
    """Average the per-batch (ndcg, mrr) rows the mixin emits as `predictions`."""
    predictions = eval_prediction.predictions
    if isinstance(predictions, (tuple, list)):
        predictions = predictions[0]
    predictions = torch.as_tensor(predictions).reshape(-1, len(EVAL_METRIC_NAMES)).float()
    return {
        name: round(float(predictions[:, index].mean()), 6)
        for index, name in enumerate(EVAL_METRIC_NAMES)
    }


class RankingEvalMixin:
    """Trainer mixin turning a model's per-batch ranking metrics into eval metrics.

    The wrapped model is expected to expose ``eval_ranking_metrics(**inputs)`` returning
    a dict with the keys in ``EVAL_METRIC_NAMES``. We route through ``prediction_step``
    rather than ``compute_loss`` so that the eval path never samples: for GRPO the
    training loss is a policy-gradient surrogate whose value is not comparable across
    configurations, and is therefore useless for model selection.
    """

    def prediction_step(
        self,
        model,
        inputs: dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ):
        inputs = self._prepare_inputs(inputs)
        unwrapped = model
        while hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
        if not hasattr(unwrapped, "eval_ranking_metrics"):
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

        with torch.no_grad():
            metrics = unwrapped.eval_ranking_metrics(**inputs)

        row = torch.stack(
            [metrics[name].detach().reshape(()).float() for name in EVAL_METRIC_NAMES]
        ).reshape(1, -1)
        # No scalar loss is reported: see the class docstring.
        return (None, row, None)
