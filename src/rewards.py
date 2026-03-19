from __future__ import annotations

import torch
import torch.nn.functional as F


SUPPORTED_REWARD_TYPES = {"ndcg", "contrastive", "infonce", "mrr", "mixed"}


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


def _normalize_inputs(
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if query_embeddings.dim() != 2:
        raise ValueError(
            f"query_embeddings must be 2D, got shape {tuple(query_embeddings.shape)}"
        )
    if candidate_embeddings.dim() != 3:
        raise ValueError(
            f"candidate_embeddings must be 3D, got shape {tuple(candidate_embeddings.shape)}"
        )
    if query_embeddings.shape[0] != candidate_embeddings.shape[0]:
        raise ValueError(
            "query_embeddings and candidate_embeddings batch size must match, "
            f"got queries={tuple(query_embeddings.shape)} candidates={tuple(candidate_embeddings.shape)}"
        )
    if query_embeddings.shape[-1] != candidate_embeddings.shape[-1]:
        raise ValueError(
            "query_embeddings and candidate_embeddings embedding dim must match, "
            f"got queries={tuple(query_embeddings.shape)} candidates={tuple(candidate_embeddings.shape)}"
        )
    return (
        F.normalize(query_embeddings, dim=-1),
        F.normalize(candidate_embeddings, dim=-1),
    )


def _validate_relevance_labels(
    relevance_labels: torch.Tensor,
    candidate_embeddings: torch.Tensor,
) -> None:
    if relevance_labels.shape != candidate_embeddings.shape[:2]:
        raise ValueError(
            "relevance_labels shape must match candidate_embeddings[:2], "
            f"got labels={tuple(relevance_labels.shape)} candidates={tuple(candidate_embeddings.shape)}"
        )


def _compute_scores(
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
) -> torch.Tensor:
    normalized_queries, normalized_candidates = _normalize_inputs(
        query_embeddings=query_embeddings,
        candidate_embeddings=candidate_embeddings,
    )
    return torch.matmul(normalized_candidates, normalized_queries.unsqueeze(-1)).squeeze(-1)


def _temperature_scaled_logsumexp(
    values: torch.Tensor,
    temperature: float,
    empty_value: float = 0.0,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    temperature_tensor = torch.as_tensor(
        float(temperature),
        device=values.device,
        dtype=values.dtype,
    )
    finite_mask = torch.isfinite(values)
    scaled_values = torch.where(
        finite_mask,
        values / temperature_tensor,
        torch.full_like(values, float("-inf")),
    )
    aggregated_values = temperature_tensor * torch.logsumexp(scaled_values, dim=-1)
    default_values = torch.full_like(aggregated_values, fill_value=empty_value)
    return torch.where(finite_mask.any(dim=-1), aggregated_values, default_values)


def _gather_positive_and_negative_scores(
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    relevance_labels: torch.Tensor,
    use_in_batch_negatives: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = _compute_scores(
        query_embeddings=query_embeddings,
        candidate_embeddings=candidate_embeddings,
    )

    positive_indices = relevance_labels.argmax(dim=-1, keepdim=True)
    positive_scores = scores.gather(dim=1, index=positive_indices).squeeze(1)

    negative_scores = scores.masked_fill(
        F.one_hot(positive_indices.squeeze(-1), num_classes=scores.size(1)).bool(),
        float("-inf"),
    )
    all_negative_scores = [negative_scores]

    if use_in_batch_negatives and query_embeddings.size(0) > 1:
        _, normalized_candidates = _normalize_inputs(
            query_embeddings=query_embeddings,
            candidate_embeddings=candidate_embeddings,
        )
        positive_embeddings = normalized_candidates.gather(
            dim=1,
            index=positive_indices.unsqueeze(-1).expand(-1, 1, normalized_candidates.size(-1)),
        ).squeeze(1)
        normalized_queries = F.normalize(query_embeddings, dim=-1)
        in_batch_scores = torch.matmul(normalized_queries, positive_embeddings.transpose(0, 1))
        in_batch_scores.fill_diagonal_(float("-inf"))
        all_negative_scores.append(in_batch_scores)

    return positive_scores, torch.cat(all_negative_scores, dim=1)


def compute_contrastive_reward(
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    relevance_labels: torch.Tensor,
    use_in_batch_negatives: bool = False,
    temperature: float = 0.03,
) -> torch.Tensor:
    _validate_relevance_labels(relevance_labels, candidate_embeddings)
    positive_scores, negative_scores = _gather_positive_and_negative_scores(
        query_embeddings=query_embeddings,
        candidate_embeddings=candidate_embeddings,
        relevance_labels=relevance_labels,
        use_in_batch_negatives=use_in_batch_negatives,
    )
    return positive_scores - _temperature_scaled_logsumexp(
        negative_scores,
        temperature=temperature,
    )


def compute_infonce_reward(
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    relevance_labels: torch.Tensor,
    use_in_batch_negatives: bool = False,
    temperature: float = 0.03,
) -> torch.Tensor:
    _validate_relevance_labels(relevance_labels, candidate_embeddings)
    positive_scores, negative_scores = _gather_positive_and_negative_scores(
        query_embeddings=query_embeddings,
        candidate_embeddings=candidate_embeddings,
        relevance_labels=relevance_labels,
        use_in_batch_negatives=use_in_batch_negatives,
    )
    partition_scores = torch.cat((positive_scores.unsqueeze(1), negative_scores), dim=1)
    return positive_scores - _temperature_scaled_logsumexp(
        partition_scores,
        temperature=temperature,
    )


def compute_ndcg_reward(
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    relevance_labels: torch.Tensor,
    k: int = 10,
) -> torch.Tensor:
    _validate_relevance_labels(relevance_labels, candidate_embeddings)

    cutoff = min(k, candidate_embeddings.size(1))
    if cutoff <= 0:
        return torch.zeros(query_embeddings.size(0), device=query_embeddings.device, dtype=query_embeddings.dtype)

    scores = _compute_scores(
        query_embeddings=query_embeddings,
        candidate_embeddings=candidate_embeddings,
    )
    topk_indices = scores.topk(k=cutoff, dim=-1).indices
    topk_relevance = relevance_labels.gather(dim=1, index=topk_indices)

    discounts = 1.0 / torch.log2(
        torch.arange(2, cutoff + 2, device=query_embeddings.device, dtype=query_embeddings.dtype)
    )
    dcg = (((2.0 ** topk_relevance) - 1.0) * discounts.unsqueeze(0)).sum(dim=-1)

    ideal_relevance = relevance_labels.topk(k=cutoff, dim=-1).values
    idcg = (((2.0 ** ideal_relevance) - 1.0) * discounts.unsqueeze(0)).sum(dim=-1)
    return torch.where(idcg > 0, dcg / idcg, torch.zeros_like(dcg))


def compute_mrr_reward(
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    relevance_labels: torch.Tensor,
    k: int | None = None,
) -> torch.Tensor:
    _validate_relevance_labels(relevance_labels, candidate_embeddings)

    cutoff = candidate_embeddings.size(1) if k is None else min(k, candidate_embeddings.size(1))
    if cutoff <= 0:
        return torch.zeros(query_embeddings.size(0), device=query_embeddings.device, dtype=query_embeddings.dtype)

    scores = _compute_scores(
        query_embeddings=query_embeddings,
        candidate_embeddings=candidate_embeddings,
    )
    ranked_indices = scores.topk(k=cutoff, dim=-1).indices
    ranked_relevance = relevance_labels.gather(dim=1, index=ranked_indices)
    relevant_mask = ranked_relevance > 0

    reciprocal_ranks = relevant_mask.to(query_embeddings.dtype) / torch.arange(
        1,
        cutoff + 1,
        device=query_embeddings.device,
        dtype=query_embeddings.dtype,
    ).unsqueeze(0)
    return reciprocal_ranks.max(dim=-1).values


def compute_mixed_reward(
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    relevance_labels: torch.Tensor,
    k: int = 10,
    contrastive_weight: float = 1.0,
    ndcg_weight: float = 1.0,
    contrastive_use_in_batch_negatives: bool = False,
    contrastive_temperature: float = 0.03,
) -> torch.Tensor:
    if contrastive_weight == 0 and ndcg_weight == 0:
        raise ValueError("At least one mixed reward weight must be non-zero")

    reward = torch.zeros(
        query_embeddings.size(0),
        device=query_embeddings.device,
        dtype=query_embeddings.dtype,
    )
    if contrastive_weight != 0:
        reward = reward + contrastive_weight * compute_contrastive_reward(
            query_embeddings=query_embeddings,
            candidate_embeddings=candidate_embeddings,
            relevance_labels=relevance_labels,
            use_in_batch_negatives=contrastive_use_in_batch_negatives,
            temperature=contrastive_temperature,
        )
    if ndcg_weight != 0:
        reward = reward + ndcg_weight * compute_ndcg_reward(
            query_embeddings=query_embeddings,
            candidate_embeddings=candidate_embeddings,
            relevance_labels=relevance_labels,
            k=k,
        )
    return reward


def compute_reward(
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    relevance_labels: torch.Tensor,
    reward_type: str = "ndcg",
    k: int = 10,
    mixed_contrastive_weight: float = 1.0,
    mixed_ndcg_weight: float = 1.0,
    contrastive_use_in_batch_negatives: bool = False,
    contrastive_temperature: float = 0.03,
) -> torch.Tensor:
    reward_type = reward_type.lower()
    if reward_type not in SUPPORTED_REWARD_TYPES:
        raise ValueError(
            f"Unsupported reward type: {reward_type}. Supported types: {sorted(SUPPORTED_REWARD_TYPES)}"
        )

    if reward_type == "ndcg":
        return compute_ndcg_reward(
            query_embeddings=query_embeddings,
            candidate_embeddings=candidate_embeddings,
            relevance_labels=relevance_labels,
            k=k,
        )
    if reward_type == "contrastive":
        return compute_contrastive_reward(
            query_embeddings=query_embeddings,
            candidate_embeddings=candidate_embeddings,
            relevance_labels=relevance_labels,
            use_in_batch_negatives=contrastive_use_in_batch_negatives,
            temperature=contrastive_temperature,
        )
    if reward_type == "infonce":
        return compute_infonce_reward(
            query_embeddings=query_embeddings,
            candidate_embeddings=candidate_embeddings,
            relevance_labels=relevance_labels,
            use_in_batch_negatives=contrastive_use_in_batch_negatives,
            temperature=contrastive_temperature,
        )
    if reward_type == "mrr":
        return compute_mrr_reward(
            query_embeddings=query_embeddings,
            candidate_embeddings=candidate_embeddings,
            relevance_labels=relevance_labels,
            k=k,
        )
    return compute_mixed_reward(
        query_embeddings=query_embeddings,
        candidate_embeddings=candidate_embeddings,
        relevance_labels=relevance_labels,
        k=k,
        contrastive_weight=mixed_contrastive_weight,
        ndcg_weight=mixed_ndcg_weight,
        contrastive_use_in_batch_negatives=contrastive_use_in_batch_negatives,
        contrastive_temperature=contrastive_temperature,
    )
