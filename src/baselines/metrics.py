from __future__ import annotations

import torch

from baselines.losses import _compute_discounts, _compute_gains, _compute_idcg, _resolve_cutoff
from rewards import resolve_relevant_mask


def compute_ndcg_at_k(
    scores: torch.Tensor,
    relevance_labels: torch.Tensor,
    k: int | None = None,
) -> torch.Tensor:
    cutoff = _resolve_cutoff(k, scores.size(1))
    if cutoff <= 0:
        return torch.zeros(scores.size(0), device=scores.device, dtype=scores.dtype)

    ranked_indices = scores.topk(k=cutoff, dim=-1).indices
    ranked_relevance = relevance_labels.gather(dim=1, index=ranked_indices)
    discounts = _compute_discounts(
        cutoff,
        device=scores.device,
        dtype=scores.dtype,
    )
    dcg = (_compute_gains(ranked_relevance) * discounts.unsqueeze(0)).sum(dim=-1)
    idcg = _compute_idcg(relevance_labels, cutoff)
    return torch.where(idcg > 0, dcg / idcg.clamp_min(1e-8), torch.zeros_like(dcg))


def compute_mrr_at_k(
    scores: torch.Tensor,
    relevance_labels: torch.Tensor,
    k: int | None = None,
    relevance_scheme: str | None = None,
) -> torch.Tensor:
    cutoff = _resolve_cutoff(k, scores.size(1))
    if cutoff <= 0:
        return torch.zeros(scores.size(0), device=scores.device, dtype=scores.dtype)

    ranked_indices = scores.topk(k=cutoff, dim=-1).indices
    ranked_relevance = relevance_labels.gather(dim=1, index=ranked_indices)
    relevant_mask = resolve_relevant_mask(
        ranked_relevance=ranked_relevance,
        relevance_labels=relevance_labels,
        relevance_scheme=relevance_scheme,
    )
    reciprocal_ranks = relevant_mask.to(scores.dtype) / torch.arange(
        1,
        cutoff + 1,
        device=scores.device,
        dtype=scores.dtype,
    ).unsqueeze(0)
    return reciprocal_ranks.max(dim=-1).values
