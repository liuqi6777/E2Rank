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


def compute_reward_from_scores(
    scores: torch.Tensor,
    relevance_labels: torch.Tensor,
    reward_type: str = "ndcg",
    k: int | None = 10,
    ndcg_in_batch_include_negatives: bool = False,
    contrastive_use_in_batch_negatives: bool = False,
    contrastive_temperature: float = 0.03,
    relevance_scheme: str | None = None,
    in_batch_positive_scores: torch.Tensor | None = None,
    in_batch_candidate_scores: torch.Tensor | None = None,
) -> torch.Tensor:
    squeeze_rollout_dim = False
    if scores.dim() == 2:
        scores = scores.unsqueeze(1)
        squeeze_rollout_dim = True
        if in_batch_positive_scores is not None:
            in_batch_positive_scores = in_batch_positive_scores.unsqueeze(1)
        if in_batch_candidate_scores is not None:
            in_batch_candidate_scores = in_batch_candidate_scores.unsqueeze(1)

    reward_type = reward_type.lower()
    if reward_type not in SUPPORTED_REWARD_TYPES:
        raise ValueError(
            f"Unsupported reward type: {reward_type}. Supported types: {sorted(SUPPORTED_REWARD_TYPES)}"
        )
    if scores.dim() < 3:
        raise ValueError(
            "scores must be [batch, slate] or [batch, *rollout, slate], "
            f"got shape {tuple(scores.shape)}"
        )
    if relevance_labels.dim() != 2:
        raise ValueError(
            f"relevance_labels must be [batch, slate], got shape {tuple(relevance_labels.shape)}"
        )
    if scores.shape[0] != relevance_labels.shape[0] or scores.shape[-1] != relevance_labels.shape[1]:
        raise ValueError(
            "relevance_labels shape must match scores [batch, slate], "
            f"got scores={tuple(scores.shape)} labels={tuple(relevance_labels.shape)}"
        )

    batch_size = scores.size(0)
    rollout_shape = scores.shape[1:-1]
    slate_length = scores.size(-1)

    def restore_reward_shape(reward: torch.Tensor) -> torch.Tensor:
        return reward.squeeze(1) if squeeze_rollout_dim else reward

    rollout_count = 1
    for rollout_dim in rollout_shape:
        rollout_count *= rollout_dim

    scores = scores.reshape(batch_size, rollout_count, slate_length)
    if in_batch_positive_scores is not None:
        in_batch_positive_scores = in_batch_positive_scores.reshape(batch_size, rollout_count, -1)
    if in_batch_candidate_scores is not None:
        in_batch_candidate_scores = in_batch_candidate_scores.reshape(batch_size, rollout_count, -1)

    expanded_labels = relevance_labels.unsqueeze(1).expand(batch_size, rollout_count, slate_length)
    if reward_type in {"ndcg", "ndcg_in_batch"}:
        ranking_scores = scores
        ranking_labels = expanded_labels
        if reward_type == "ndcg_in_batch":
            extra_scores = (
                in_batch_candidate_scores
                if ndcg_in_batch_include_negatives
                else in_batch_positive_scores
            )
            if extra_scores is not None:
                ranking_scores = torch.cat((ranking_scores, extra_scores), dim=-1)
                ranking_labels = torch.cat((ranking_labels, torch.zeros_like(extra_scores)), dim=-1)

        cutoff = ranking_scores.size(-1) if k is None else min(k, ranking_scores.size(-1))
        if cutoff <= 0:
            reward = torch.zeros(batch_size, *rollout_shape, device=scores.device, dtype=scores.dtype)
            return restore_reward_shape(reward)

        topk_indices = ranking_scores.topk(k=cutoff, dim=-1).indices
        topk_relevance = ranking_labels.gather(dim=-1, index=topk_indices)
        discounts = 1.0 / torch.log2(torch.arange(2, cutoff + 2, device=scores.device, dtype=scores.dtype))
        dcg = (((2.0 ** topk_relevance) - 1.0) * discounts).sum(dim=-1)

        ideal_relevance = ranking_labels.topk(k=cutoff, dim=-1).values
        idcg = (((2.0 ** ideal_relevance) - 1.0) * discounts).sum(dim=-1)
        reward = torch.where(idcg > 0, dcg / idcg, torch.zeros_like(dcg)).reshape(batch_size, *rollout_shape)
        return restore_reward_shape(reward)

    if reward_type == "mrr":
        cutoff = slate_length if k is None else min(k, slate_length)
        if cutoff <= 0:
            reward = torch.zeros(batch_size, *rollout_shape, device=scores.device, dtype=scores.dtype)
            return restore_reward_shape(reward)
        ranked_indices = scores.topk(k=cutoff, dim=-1).indices
        ranked_relevance = expanded_labels.gather(dim=-1, index=ranked_indices)
        relevant_mask = _resolve_relevant_mask(
            ranked_relevance=ranked_relevance,
            relevance_labels=relevance_labels,
            relevance_scheme=relevance_scheme,
        )
        reciprocal_ranks = relevant_mask.to(scores.dtype) / torch.arange(
            1,
            cutoff + 1,
            device=scores.device,
            dtype=scores.dtype,
        )
        reward = reciprocal_ranks.max(dim=-1).values.reshape(batch_size, *rollout_shape)
        return restore_reward_shape(reward)

    positive_indices = relevance_labels.argmax(dim=-1)
    arange_b = torch.arange(batch_size, device=scores.device)
    positive_scores = scores[arange_b, :, positive_indices]
    negative_scores = scores.masked_fill(
        F.one_hot(positive_indices, num_classes=slate_length).bool().unsqueeze(1),
        float("-inf"),
    )
    if contrastive_use_in_batch_negatives and in_batch_positive_scores is not None:
        negative_scores = torch.cat((negative_scores, in_batch_positive_scores), dim=-1)

    if reward_type == "contrastive":
        rewards = positive_scores - _temperature_scaled_logsumexp(
            negative_scores,
            temperature=contrastive_temperature,
        )
        reward = rewards.reshape(batch_size, *rollout_shape)
        return restore_reward_shape(reward)
    if reward_type == "infonce":
        partition_scores = torch.cat((positive_scores.unsqueeze(-1), negative_scores), dim=-1)
        rewards = positive_scores - _temperature_scaled_logsumexp(
            partition_scores,
            temperature=contrastive_temperature,
        )
        reward = rewards.reshape(batch_size, *rollout_shape)
        return restore_reward_shape(reward)
    raise AssertionError(f"Unhandled reward type: {reward_type}")
