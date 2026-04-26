from __future__ import annotations

import torch
import torch.nn.functional as F


SUPPORTED_REWARD_TYPES = {"ndcg", "ndcg_in_batch", "contrastive", "infonce", "mrr"}


def _resolve_relevant_mask(
    ranked_relevance: torch.Tensor,
    relevance_labels: torch.Tensor,
    relevance_scheme: str | None = None,
) -> torch.Tensor:
    if relevance_scheme is not None and relevance_scheme not in {"graded", "binary"}:
        raise ValueError(f"Unsupported relevance scheme: {relevance_scheme}")

    use_graded_threshold = (
        relevance_scheme == "graded"
        if relevance_scheme is not None
        else bool((relevance_labels > 1).any().item())
    )
    return ranked_relevance >= 2.0 if use_graded_threshold else ranked_relevance > 0.0


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


def _append_in_batch_positive_negatives(
    candidate_embeddings: torch.Tensor,
    relevance_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, _, embedding_dim = candidate_embeddings.shape
    if batch_size <= 1:
        return candidate_embeddings, relevance_labels

    positive_indices = relevance_labels.argmax(dim=-1)
    positive_embeddings = candidate_embeddings[
        torch.arange(batch_size, device=candidate_embeddings.device),
        positive_indices,
    ]
    expanded_positive_embeddings = positive_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
    cross_batch_mask = ~torch.eye(batch_size, device=candidate_embeddings.device, dtype=torch.bool)
    in_batch_negative_embeddings = expanded_positive_embeddings[cross_batch_mask].reshape(
        batch_size,
        batch_size - 1,
        embedding_dim,
    )
    in_batch_negative_labels = torch.zeros(
        batch_size,
        batch_size - 1,
        device=relevance_labels.device,
        dtype=relevance_labels.dtype,
    )
    return (
        torch.cat((candidate_embeddings, in_batch_negative_embeddings), dim=1),
        torch.cat((relevance_labels, in_batch_negative_labels), dim=1),
    )


def _append_in_batch_candidate_negatives(
    candidate_embeddings: torch.Tensor,
    relevance_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, slate_length, embedding_dim = candidate_embeddings.shape
    if batch_size <= 1:
        return candidate_embeddings, relevance_labels

    expanded_candidates = candidate_embeddings.unsqueeze(0).expand(batch_size, -1, -1, -1)
    cross_batch_mask = ~torch.eye(batch_size, device=candidate_embeddings.device, dtype=torch.bool)
    in_batch_negative_embeddings = expanded_candidates[cross_batch_mask].reshape(
        batch_size,
        (batch_size - 1) * slate_length,
        embedding_dim,
    )
    in_batch_negative_labels = torch.zeros(
        batch_size,
        (batch_size - 1) * slate_length,
        device=relevance_labels.device,
        dtype=relevance_labels.dtype,
    )
    return (
        torch.cat((candidate_embeddings, in_batch_negative_embeddings), dim=1),
        torch.cat((relevance_labels, in_batch_negative_labels), dim=1),
    )


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
    use_in_batch_negatives: bool = False,
    include_in_batch_negatives: bool = False,
) -> torch.Tensor:
    _validate_relevance_labels(relevance_labels, candidate_embeddings)
    if use_in_batch_negatives:
        append_fn = (
            _append_in_batch_candidate_negatives
            if include_in_batch_negatives
            else _append_in_batch_positive_negatives
        )
        candidate_embeddings, relevance_labels = append_fn(
            candidate_embeddings=candidate_embeddings,
            relevance_labels=relevance_labels,
        )

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
    relevance_scheme: str | None = None,
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
    relevant_mask = _resolve_relevant_mask(
        ranked_relevance=ranked_relevance,
        relevance_labels=relevance_labels,
        relevance_scheme=relevance_scheme,
    )

    reciprocal_ranks = relevant_mask.to(query_embeddings.dtype) / torch.arange(
        1,
        cutoff + 1,
        device=query_embeddings.device,
        dtype=query_embeddings.dtype,
    ).unsqueeze(0)
    return reciprocal_ranks.max(dim=-1).values


def compute_reward(
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    relevance_labels: torch.Tensor,
    reward_type: str = "ndcg",
    k: int = 10,
    ndcg_in_batch_include_negatives: bool = False,
    contrastive_use_in_batch_negatives: bool = False,
    contrastive_temperature: float = 0.03,
    relevance_scheme: str | None = None,
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
    if reward_type == "ndcg_in_batch":
        return compute_ndcg_reward(
            query_embeddings=query_embeddings,
            candidate_embeddings=candidate_embeddings,
            relevance_labels=relevance_labels,
            k=k,
            use_in_batch_negatives=True,
            include_in_batch_negatives=ndcg_in_batch_include_negatives,
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
            relevance_scheme=relevance_scheme,
        )
    raise AssertionError(f"Unhandled reward type: {reward_type}")


def _reward_uses_in_batch_negatives(
    reward_type: str,
    contrastive_use_in_batch_negatives: bool = False,
) -> bool:
    reward_type = reward_type.lower()
    if reward_type == "ndcg_in_batch":
        return True
    return reward_type in {"contrastive", "infonce"} and contrastive_use_in_batch_negatives


def _expand_rollout_candidates(
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
) -> torch.Tensor:
    batch_size = query_embeddings.size(0)
    rollout_shape = query_embeddings.shape[1:-1]
    embedding_dim = query_embeddings.size(-1)

    if candidate_embeddings.dim() == 3:
        expected_shape = (batch_size, candidate_embeddings.size(1), embedding_dim)
        if tuple(candidate_embeddings.shape) != expected_shape:
            raise ValueError(
                "candidate_embeddings must be [batch, slate, dim] when no rollout dims are provided, "
                f"got {tuple(candidate_embeddings.shape)} and expected {expected_shape}"
            )
        return candidate_embeddings.reshape(
            batch_size,
            *((1,) * len(rollout_shape)),
            candidate_embeddings.size(1),
            embedding_dim,
        ).expand(
            batch_size,
            *rollout_shape,
            candidate_embeddings.size(1),
            embedding_dim,
        )

    expected_dim = len(rollout_shape) + 3
    if candidate_embeddings.dim() != expected_dim:
        raise ValueError(
            "candidate_embeddings must be [batch, *rollout, slate, dim] "
            f"or [batch, slate, dim], got shape {tuple(candidate_embeddings.shape)}"
        )
    if candidate_embeddings.shape[0] != batch_size:
        raise ValueError(
            "query_embeddings and candidate_embeddings batch size must match, "
            f"got queries={tuple(query_embeddings.shape)} candidates={tuple(candidate_embeddings.shape)}"
        )
    if candidate_embeddings.shape[1:-2] != rollout_shape:
        raise ValueError(
            "candidate_embeddings rollout shape must match query_embeddings rollout shape, "
            f"got queries={tuple(query_embeddings.shape)} candidates={tuple(candidate_embeddings.shape)}"
        )
    if candidate_embeddings.shape[-1] != embedding_dim:
        raise ValueError(
            "query_embeddings and candidate_embeddings embedding dim must match, "
            f"got queries={tuple(query_embeddings.shape)} candidates={tuple(candidate_embeddings.shape)}"
        )
    return candidate_embeddings


def _compute_scores_rollout(
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
) -> torch.Tensor:
    normalized_queries = F.normalize(query_embeddings, dim=-1)
    normalized_candidates = F.normalize(candidate_embeddings, dim=-1)
    return torch.einsum("brkd,brd->brk", normalized_candidates, normalized_queries)


def _append_in_batch_positive_negatives_rollout(
    candidate_embeddings: torch.Tensor,
    relevance_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, rollout_count, _, embedding_dim = candidate_embeddings.shape
    if batch_size <= 1:
        return candidate_embeddings, relevance_labels

    positive_indices = relevance_labels.argmax(dim=-1)
    arange_b = torch.arange(batch_size, device=candidate_embeddings.device)
    positive_embeddings = candidate_embeddings[arange_b, :, positive_indices]
    expanded_positive_embeddings = positive_embeddings.unsqueeze(0).expand(
        batch_size, batch_size, rollout_count, embedding_dim
    )
    cross_batch_mask = ~torch.eye(batch_size, device=candidate_embeddings.device, dtype=torch.bool)
    in_batch_negative_embeddings = expanded_positive_embeddings[cross_batch_mask].reshape(
        batch_size, batch_size - 1, rollout_count, embedding_dim
    ).permute(0, 2, 1, 3).contiguous()
    in_batch_negative_labels = torch.zeros(
        batch_size,
        batch_size - 1,
        device=relevance_labels.device,
        dtype=relevance_labels.dtype,
    )
    return (
        torch.cat((candidate_embeddings, in_batch_negative_embeddings), dim=2),
        torch.cat((relevance_labels, in_batch_negative_labels), dim=1),
    )


def _append_in_batch_candidate_negatives_rollout(
    candidate_embeddings: torch.Tensor,
    relevance_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, rollout_count, slate_length, embedding_dim = candidate_embeddings.shape
    if batch_size <= 1:
        return candidate_embeddings, relevance_labels

    expanded_candidates = candidate_embeddings.unsqueeze(0).expand(
        batch_size, batch_size, rollout_count, slate_length, embedding_dim
    )
    cross_batch_mask = ~torch.eye(batch_size, device=candidate_embeddings.device, dtype=torch.bool)
    in_batch_negative_embeddings = expanded_candidates[cross_batch_mask].reshape(
        batch_size, batch_size - 1, rollout_count, slate_length, embedding_dim
    ).permute(0, 2, 1, 3, 4).reshape(
        batch_size, rollout_count, (batch_size - 1) * slate_length, embedding_dim
    )
    in_batch_negative_labels = torch.zeros(
        batch_size,
        (batch_size - 1) * slate_length,
        device=relevance_labels.device,
        dtype=relevance_labels.dtype,
    )
    return (
        torch.cat((candidate_embeddings, in_batch_negative_embeddings), dim=2),
        torch.cat((relevance_labels, in_batch_negative_labels), dim=1),
    )


def _compute_ndcg_reward_rollout(
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    relevance_labels: torch.Tensor,
    k: int,
) -> torch.Tensor:
    batch_size, rollout_count, slate_length, _ = candidate_embeddings.shape
    cutoff = min(k, slate_length)
    if cutoff <= 0:
        return torch.zeros(
            batch_size, rollout_count, device=query_embeddings.device, dtype=query_embeddings.dtype
        )

    scores = _compute_scores_rollout(query_embeddings, candidate_embeddings)
    topk_indices = scores.topk(k=cutoff, dim=-1).indices
    expanded_labels = relevance_labels.unsqueeze(1).expand(batch_size, rollout_count, slate_length)
    topk_relevance = expanded_labels.gather(dim=-1, index=topk_indices)

    discounts = 1.0 / torch.log2(
        torch.arange(2, cutoff + 2, device=query_embeddings.device, dtype=query_embeddings.dtype)
    )
    dcg = (((2.0 ** topk_relevance) - 1.0) * discounts).sum(dim=-1)

    ideal_relevance = expanded_labels.topk(k=cutoff, dim=-1).values
    idcg = (((2.0 ** ideal_relevance) - 1.0) * discounts).sum(dim=-1)
    return torch.where(idcg > 0, dcg / idcg, torch.zeros_like(dcg))


def _gather_positive_and_negative_scores_rollout(
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    relevance_labels: torch.Tensor,
    use_in_batch_negatives: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, _, slate_length, _ = candidate_embeddings.shape
    scores = _compute_scores_rollout(query_embeddings, candidate_embeddings)

    positive_indices = relevance_labels.argmax(dim=-1)
    arange_b = torch.arange(batch_size, device=candidate_embeddings.device)
    positive_scores = scores[arange_b, :, positive_indices]

    positive_one_hot = F.one_hot(positive_indices, num_classes=slate_length).bool()
    negative_scores = scores.masked_fill(positive_one_hot.unsqueeze(1), float("-inf"))
    all_negative_scores = [negative_scores]

    if use_in_batch_negatives and batch_size > 1:
        normalized_candidates = F.normalize(candidate_embeddings, dim=-1)
        positive_embeddings = normalized_candidates[arange_b, :, positive_indices]
        normalized_queries = F.normalize(query_embeddings, dim=-1)
        in_batch_scores = torch.einsum("brd,crd->brc", normalized_queries, positive_embeddings)
        eye_mask = torch.eye(batch_size, device=candidate_embeddings.device, dtype=torch.bool)
        in_batch_scores = in_batch_scores.masked_fill(eye_mask.unsqueeze(1), float("-inf"))
        all_negative_scores.append(in_batch_scores)

    return positive_scores, torch.cat(all_negative_scores, dim=-1)


def _compute_contrastive_reward_rollout(
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    relevance_labels: torch.Tensor,
    use_in_batch_negatives: bool,
    temperature: float,
) -> torch.Tensor:
    positive_scores, negative_scores = _gather_positive_and_negative_scores_rollout(
        query_embeddings=query_embeddings,
        candidate_embeddings=candidate_embeddings,
        relevance_labels=relevance_labels,
        use_in_batch_negatives=use_in_batch_negatives,
    )
    return positive_scores - _temperature_scaled_logsumexp(
        negative_scores,
        temperature=temperature,
    )


def _compute_infonce_reward_rollout(
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    relevance_labels: torch.Tensor,
    use_in_batch_negatives: bool,
    temperature: float,
) -> torch.Tensor:
    positive_scores, negative_scores = _gather_positive_and_negative_scores_rollout(
        query_embeddings=query_embeddings,
        candidate_embeddings=candidate_embeddings,
        relevance_labels=relevance_labels,
        use_in_batch_negatives=use_in_batch_negatives,
    )
    partition_scores = torch.cat((positive_scores.unsqueeze(-1), negative_scores), dim=-1)
    return positive_scores - _temperature_scaled_logsumexp(
        partition_scores,
        temperature=temperature,
    )


def compute_rollout_reward(
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    relevance_labels: torch.Tensor,
    reward_type: str = "ndcg",
    k: int = 10,
    ndcg_in_batch_include_negatives: bool = False,
    contrastive_use_in_batch_negatives: bool = False,
    contrastive_temperature: float = 0.03,
    relevance_scheme: str | None = None,
) -> torch.Tensor:
    if query_embeddings.dim() < 2:
        raise ValueError(
            f"query_embeddings must be [batch, *rollout, dim], got shape {tuple(query_embeddings.shape)}"
        )
    if relevance_labels.dim() != 2:
        raise ValueError(
            f"relevance_labels must be [batch, slate], got shape {tuple(relevance_labels.shape)}"
        )
    if query_embeddings.size(0) != relevance_labels.size(0):
        raise ValueError(
            "query_embeddings and relevance_labels batch size must match, "
            f"got queries={tuple(query_embeddings.shape)} labels={tuple(relevance_labels.shape)}"
        )

    rollout_shape = query_embeddings.shape[1:-1]
    if not rollout_shape:
        return compute_reward(
            query_embeddings=query_embeddings,
            candidate_embeddings=candidate_embeddings,
            relevance_labels=relevance_labels,
            reward_type=reward_type,
            k=k,
            ndcg_in_batch_include_negatives=ndcg_in_batch_include_negatives,
            contrastive_use_in_batch_negatives=contrastive_use_in_batch_negatives,
            contrastive_temperature=contrastive_temperature,
            relevance_scheme=relevance_scheme,
        )

    candidate_embeddings = _expand_rollout_candidates(
        query_embeddings=query_embeddings,
        candidate_embeddings=candidate_embeddings,
    )
    if relevance_labels.shape != candidate_embeddings.shape[:1] + candidate_embeddings.shape[-2:-1]:
        raise ValueError(
            "relevance_labels shape must match [batch, slate], "
            f"got labels={tuple(relevance_labels.shape)} candidates={tuple(candidate_embeddings.shape)}"
        )

    batch_size = query_embeddings.size(0)
    rollout_count = 1
    for rollout_dim in rollout_shape:
        rollout_count *= rollout_dim

    if _reward_uses_in_batch_negatives(
        reward_type=reward_type,
        contrastive_use_in_batch_negatives=contrastive_use_in_batch_negatives,
    ):
        query_by_rollout = query_embeddings.reshape(batch_size, rollout_count, query_embeddings.size(-1))
        candidate_by_rollout = candidate_embeddings.reshape(
            batch_size,
            rollout_count,
            candidate_embeddings.size(-2),
            candidate_embeddings.size(-1),
        )
        if reward_type == "ndcg_in_batch":
            append_fn = (
                _append_in_batch_candidate_negatives_rollout
                if ndcg_in_batch_include_negatives
                else _append_in_batch_positive_negatives_rollout
            )
            augmented_candidates, augmented_labels = append_fn(
                candidate_embeddings=candidate_by_rollout,
                relevance_labels=relevance_labels,
            )
            rewards = _compute_ndcg_reward_rollout(
                query_embeddings=query_by_rollout,
                candidate_embeddings=augmented_candidates,
                relevance_labels=augmented_labels,
                k=k,
            )
        elif reward_type == "contrastive":
            rewards = _compute_contrastive_reward_rollout(
                query_embeddings=query_by_rollout,
                candidate_embeddings=candidate_by_rollout,
                relevance_labels=relevance_labels,
                use_in_batch_negatives=contrastive_use_in_batch_negatives,
                temperature=contrastive_temperature,
            )
        elif reward_type == "infonce":
            rewards = _compute_infonce_reward_rollout(
                query_embeddings=query_by_rollout,
                candidate_embeddings=candidate_by_rollout,
                relevance_labels=relevance_labels,
                use_in_batch_negatives=contrastive_use_in_batch_negatives,
                temperature=contrastive_temperature,
            )
        else:
            raise AssertionError(
                f"Unhandled in-batch reward type in vectorized path: {reward_type}"
            )
        return rewards.reshape(batch_size, *rollout_shape)

    expanded_labels = relevance_labels.reshape(
        batch_size,
        *((1,) * len(rollout_shape)),
        relevance_labels.size(-1),
    ).expand(batch_size, *rollout_shape, relevance_labels.size(-1))
    flat_query_embeddings = query_embeddings.reshape(batch_size * rollout_count, query_embeddings.size(-1))
    flat_candidate_embeddings = candidate_embeddings.reshape(
        batch_size * rollout_count,
        candidate_embeddings.size(-2),
        candidate_embeddings.size(-1),
    )
    flat_relevance_labels = expanded_labels.reshape(batch_size * rollout_count, relevance_labels.size(-1))
    return compute_reward(
        query_embeddings=flat_query_embeddings,
        candidate_embeddings=flat_candidate_embeddings,
        relevance_labels=flat_relevance_labels,
        reward_type=reward_type,
        k=k,
        ndcg_in_batch_include_negatives=ndcg_in_batch_include_negatives,
        contrastive_use_in_batch_negatives=contrastive_use_in_batch_negatives,
        contrastive_temperature=contrastive_temperature,
        relevance_scheme=relevance_scheme,
    ).reshape(batch_size, *rollout_shape)
