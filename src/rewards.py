from __future__ import annotations

import torch


def build_relevance_labels(
    ranking: torch.Tensor,
    scheme: str = "graded",
) -> torch.Tensor:
    if ranking is None:
        raise ValueError("ranking is required to build relevance labels")
    if ranking.dim() != 2:
        raise ValueError(f"ranking must be a 2D tensor, got shape {tuple(ranking.shape)}")
    if scheme not in {"graded", "binary"}:
        raise ValueError(f"Unsupported relevance scheme: {scheme}")

    batch_size, slate_length = ranking.shape
    relevance = torch.zeros(batch_size, slate_length, device=ranking.device, dtype=torch.float32)

    rank_scores = torch.zeros(slate_length, device=ranking.device, dtype=torch.float32)
    if slate_length > 0:
        rank_scores[0] = 3.0 if scheme == "graded" else 1.0
    if scheme == "graded":
        if slate_length > 1:
            rank_scores[1:min(5, slate_length)] = 2.0
        if slate_length > 5:
            rank_scores[5:min(10, slate_length)] = 1.0

    relevance.scatter_(
        dim=1,
        index=ranking,
        src=rank_scores.unsqueeze(0).expand(batch_size, -1),
    )
    return relevance


def compute_ndcg_reward(
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    relevance_labels: torch.Tensor,
    k: int = 10,
) -> torch.Tensor:
    if candidate_embeddings.dim() != 3:
        raise ValueError(
            f"candidate_embeddings must be 3D, got shape {tuple(candidate_embeddings.shape)}"
        )
    if relevance_labels.shape != candidate_embeddings.shape[:2]:
        raise ValueError(
            "relevance_labels shape must match candidate_embeddings[:2], "
            f"got labels={tuple(relevance_labels.shape)} candidates={tuple(candidate_embeddings.shape)}"
        )

    cutoff = min(k, candidate_embeddings.size(1))
    if cutoff <= 0:
        return torch.zeros(query_embeddings.size(0), device=query_embeddings.device, dtype=query_embeddings.dtype)

    scores = torch.matmul(candidate_embeddings, query_embeddings.unsqueeze(-1)).squeeze(-1)
    topk_indices = scores.topk(k=cutoff, dim=-1).indices
    topk_relevance = relevance_labels.gather(dim=1, index=topk_indices)

    discounts = 1.0 / torch.log2(
        torch.arange(2, cutoff + 2, device=query_embeddings.device, dtype=query_embeddings.dtype)
    )
    dcg = (((2.0 ** topk_relevance) - 1.0) * discounts.unsqueeze(0)).sum(dim=-1)

    ideal_relevance = relevance_labels.topk(k=cutoff, dim=-1).values
    idcg = (((2.0 ** ideal_relevance) - 1.0) * discounts.unsqueeze(0)).sum(dim=-1)
    return torch.where(idcg > 0, dcg / idcg, torch.zeros_like(dcg))
