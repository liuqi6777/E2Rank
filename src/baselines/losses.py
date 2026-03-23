from __future__ import annotations

import math

import torch
import torch.nn.functional as F


SUPPORTED_BASELINE_TYPES = {"ranknet", "lambdaloss", "neuralndcg", "approxndcg", "softrank", "infonce"}
EPS = 1e-8


def _validate_scores_and_labels(
    scores: torch.Tensor,
    relevance_labels: torch.Tensor,
) -> None:
    if scores.dim() != 2:
        raise ValueError(f"scores must be a 2D tensor, got shape {tuple(scores.shape)}")
    if relevance_labels.dim() != 2:
        raise ValueError(
            f"relevance_labels must be a 2D tensor, got shape {tuple(relevance_labels.shape)}"
        )
    if scores.shape != relevance_labels.shape:
        raise ValueError(
            "scores and relevance_labels must have matching shape, "
            f"got scores={tuple(scores.shape)} labels={tuple(relevance_labels.shape)}"
        )


def _resolve_cutoff(k: int | None, slate_length: int) -> int:
    if slate_length <= 0:
        return 0
    if k is None:
        return slate_length
    return max(0, min(int(k), slate_length))


def _compute_discounts(
    length: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if length <= 0:
        return torch.zeros(0, device=device, dtype=dtype)
    return 1.0 / torch.log2(torch.arange(2, length + 2, device=device, dtype=dtype))


def _compute_gains(relevance_labels: torch.Tensor) -> torch.Tensor:
    return torch.pow(2.0, relevance_labels) - 1.0


def _compute_idcg(
    relevance_labels: torch.Tensor,
    k: int | None,
) -> torch.Tensor:
    cutoff = _resolve_cutoff(k, relevance_labels.size(1))
    if cutoff <= 0:
        return torch.zeros(relevance_labels.size(0), device=relevance_labels.device, dtype=relevance_labels.dtype)

    ideal_relevance = relevance_labels.topk(k=cutoff, dim=-1).values
    discounts = _compute_discounts(
        cutoff,
        device=relevance_labels.device,
        dtype=relevance_labels.dtype,
    )
    return (_compute_gains(ideal_relevance) * discounts.unsqueeze(0)).sum(dim=-1)


def _pairwise_differences(values: torch.Tensor) -> torch.Tensor:
    return values.unsqueeze(2) - values.unsqueeze(1)


def _reduce_masked_average(
    values: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    masked_values = torch.where(mask, values, torch.zeros_like(values))
    if weights is None:
        numerator = masked_values.sum(dim=(1, 2))
        denominator = mask.to(values.dtype).sum(dim=(1, 2)).clamp_min(1.0)
    else:
        masked_weights = torch.where(mask, weights, torch.zeros_like(weights))
        numerator = (masked_values * masked_weights).sum(dim=(1, 2))
        denominator = masked_weights.sum(dim=(1, 2)).clamp_min(1.0)

    reduced = numerator / denominator
    has_valid_entries = mask.any(dim=(1, 2))
    return torch.where(has_valid_entries, reduced, torch.zeros_like(reduced))


def _normal_cdf(values: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(values / math.sqrt(2.0)))


def compute_ranknet_loss(
    scores: torch.Tensor,
    relevance_labels: torch.Tensor,
    sigma: float = 1.0,
) -> torch.Tensor:
    _validate_scores_and_labels(scores, relevance_labels)
    if sigma <= 0:
        raise ValueError(f"ranknet sigma must be positive, got {sigma}")

    score_diff = _pairwise_differences(scores)
    label_diff = _pairwise_differences(relevance_labels)
    valid_pairs = label_diff > 0
    pair_loss = F.softplus(-float(sigma) * score_diff)
    return _reduce_masked_average(pair_loss, valid_pairs)


def compute_lambdaloss_loss(
    scores: torch.Tensor,
    relevance_labels: torch.Tensor,
    k: int | None = None,
    sigma: float = 1.0,
) -> torch.Tensor:
    _validate_scores_and_labels(scores, relevance_labels)
    if sigma <= 0:
        raise ValueError(f"lambdaloss sigma must be positive, got {sigma}")

    cutoff = _resolve_cutoff(k, scores.size(1))
    if cutoff <= 0:
        return torch.zeros(scores.size(0), device=scores.device, dtype=scores.dtype)

    score_diff = _pairwise_differences(scores)
    label_diff = _pairwise_differences(relevance_labels)
    valid_pairs = label_diff > 0
    pair_loss = F.softplus(-float(sigma) * score_diff)

    ranking_indices = scores.argsort(dim=-1, descending=True)
    positions = torch.empty_like(ranking_indices)
    positions.scatter_(
        dim=1,
        index=ranking_indices,
        src=torch.arange(scores.size(1), device=scores.device).unsqueeze(0).expand_as(ranking_indices),
    )

    position_discounts = torch.zeros(scores.size(1), device=scores.device, dtype=scores.dtype)
    position_discounts[:cutoff] = _compute_discounts(
        cutoff,
        device=scores.device,
        dtype=scores.dtype,
    )
    item_discounts = position_discounts[positions]
    gains = _compute_gains(relevance_labels)
    delta_ndcg = torch.abs(
        (gains.unsqueeze(2) - gains.unsqueeze(1))
        * (item_discounts.unsqueeze(2) - item_discounts.unsqueeze(1))
    )

    idcg = _compute_idcg(relevance_labels, cutoff)
    normalized_delta = torch.where(
        idcg.view(-1, 1, 1) > 0,
        delta_ndcg / idcg.clamp_min(EPS).view(-1, 1, 1),
        torch.zeros_like(delta_ndcg),
    )
    return _reduce_masked_average(pair_loss, valid_pairs, weights=normalized_delta)


def compute_approxndcg_loss(
    scores: torch.Tensor,
    relevance_labels: torch.Tensor,
    k: int | None = None,
    alpha: float = 10.0,
) -> torch.Tensor:
    _validate_scores_and_labels(scores, relevance_labels)
    if alpha <= 0:
        raise ValueError(f"approxndcg alpha must be positive, got {alpha}")

    cutoff = _resolve_cutoff(k, scores.size(1))
    if cutoff <= 0:
        return torch.zeros(scores.size(0), device=scores.device, dtype=scores.dtype)

    pairwise_score_gap = scores.unsqueeze(1) - scores.unsqueeze(2)
    pairwise_prob = torch.sigmoid(float(alpha) * pairwise_score_gap)
    pairwise_prob = pairwise_prob * (~torch.eye(scores.size(1), device=scores.device, dtype=torch.bool)).unsqueeze(0)
    approx_rank = 1.0 + pairwise_prob.sum(dim=-1)

    cutoff_mask = torch.sigmoid(float(alpha) * (cutoff + 0.5 - approx_rank))
    dcg = (
        _compute_gains(relevance_labels)
        * cutoff_mask
        * (1.0 / torch.log2(1.0 + approx_rank).clamp_min(EPS))
    ).sum(dim=-1)
    idcg = _compute_idcg(relevance_labels, cutoff)
    ndcg = torch.where(idcg > 0, dcg / idcg.clamp_min(EPS), torch.zeros_like(dcg))
    return -ndcg


def _neural_sort(
    scores: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    batch_size, slate_length = scores.shape
    one = torch.ones((slate_length, 1), device=scores.device, dtype=scores.dtype)
    score_column = scores.unsqueeze(-1)
    pairwise_distance = torch.abs(score_column - score_column.transpose(1, 2))
    row_sum_matrix = torch.matmul(pairwise_distance, torch.matmul(one, one.transpose(0, 1)))

    scaling = slate_length + 1 - 2 * (torch.arange(slate_length, device=scores.device, dtype=scores.dtype) + 1)
    scaled_scores = score_column * scaling.view(1, 1, -1)
    permutation_logits = (scaled_scores - row_sum_matrix).transpose(1, 2)
    return torch.softmax(permutation_logits / float(temperature), dim=-1)


def compute_neuralndcg_loss(
    scores: torch.Tensor,
    relevance_labels: torch.Tensor,
    k: int | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    _validate_scores_and_labels(scores, relevance_labels)
    if temperature <= 0:
        raise ValueError(f"neuralndcg temperature must be positive, got {temperature}")

    cutoff = _resolve_cutoff(k, scores.size(1))
    if cutoff <= 0:
        return torch.zeros(scores.size(0), device=scores.device, dtype=scores.dtype)

    soft_permutation = _neural_sort(scores, temperature=float(temperature))
    soft_sorted_gains = torch.matmul(
        soft_permutation,
        _compute_gains(relevance_labels).unsqueeze(-1),
    ).squeeze(-1)
    discounts = _compute_discounts(
        cutoff,
        device=scores.device,
        dtype=scores.dtype,
    )
    dcg = (soft_sorted_gains[:, :cutoff] * discounts.unsqueeze(0)).sum(dim=-1)
    idcg = _compute_idcg(relevance_labels, cutoff)
    ndcg = torch.where(idcg > 0, dcg / idcg.clamp_min(EPS), torch.zeros_like(dcg))
    return -ndcg


def compute_softrank_loss(
    scores: torch.Tensor,
    relevance_labels: torch.Tensor,
    k: int | None = None,
    sigma: float = 1.0,
) -> torch.Tensor:
    _validate_scores_and_labels(scores, relevance_labels)
    if sigma <= 0:
        raise ValueError(f"softrank sigma must be positive, got {sigma}")

    cutoff = _resolve_cutoff(k, scores.size(1))
    if cutoff <= 0:
        return torch.zeros(scores.size(0), device=scores.device, dtype=scores.dtype)

    pairwise_score_gap = scores.unsqueeze(1) - scores.unsqueeze(2)
    pairwise_prob = _normal_cdf(pairwise_score_gap / (math.sqrt(2.0) * float(sigma)))
    pairwise_prob = pairwise_prob * (~torch.eye(scores.size(1), device=scores.device, dtype=torch.bool)).unsqueeze(0)
    expected_rank = 1.0 + pairwise_prob.sum(dim=-1)

    cutoff_mask = _normal_cdf((cutoff + 0.5 - expected_rank) / float(sigma))
    dcg = (
        _compute_gains(relevance_labels)
        * cutoff_mask
        * (1.0 / torch.log2(1.0 + expected_rank).clamp_min(EPS))
    ).sum(dim=-1)
    idcg = _compute_idcg(relevance_labels, cutoff)
    ndcg = torch.where(idcg > 0, dcg / idcg.clamp_min(EPS), torch.zeros_like(dcg))
    return -ndcg


def compute_infonce_loss(
    scores: torch.Tensor,
    relevance_labels: torch.Tensor,
    temperature: float = 0.03,
) -> torch.Tensor:
    _validate_scores_and_labels(scores, relevance_labels)
    if temperature <= 0:
        raise ValueError(f"infonce temperature must be positive, got {temperature}")

    positive_exists = relevance_labels.max(dim=-1).values > 0
    positive_indices = relevance_labels.argmax(dim=-1, keepdim=True)
    scaled_scores = scores / float(temperature)
    positive_scores = scaled_scores.gather(dim=1, index=positive_indices).squeeze(1)
    partition = torch.logsumexp(scaled_scores, dim=-1)
    loss = partition - positive_scores
    return torch.where(positive_exists, loss, torch.zeros_like(loss))


def compute_baseline_loss(
    scores: torch.Tensor,
    relevance_labels: torch.Tensor,
    baseline_type: str,
    ndcg_k: int | None = None,
    ranknet_sigma: float = 1.0,
    lambdaloss_sigma: float = 1.0,
    approxndcg_alpha: float = 10.0,
    neuralndcg_temperature: float = 1.0,
    softrank_sigma: float = 1.0,
    infonce_temperature: float = 0.03,
) -> torch.Tensor:
    baseline_type = baseline_type.lower()
    if baseline_type not in SUPPORTED_BASELINE_TYPES:
        raise ValueError(
            f"Unsupported baseline_type: {baseline_type}. Supported types: {sorted(SUPPORTED_BASELINE_TYPES)}"
        )

    if baseline_type == "ranknet":
        return compute_ranknet_loss(
            scores=scores,
            relevance_labels=relevance_labels,
            sigma=ranknet_sigma,
        )
    if baseline_type == "lambdaloss":
        return compute_lambdaloss_loss(
            scores=scores,
            relevance_labels=relevance_labels,
            k=ndcg_k,
            sigma=lambdaloss_sigma,
        )
    if baseline_type == "approxndcg":
        return compute_approxndcg_loss(
            scores=scores,
            relevance_labels=relevance_labels,
            k=ndcg_k,
            alpha=approxndcg_alpha,
        )
    if baseline_type == "neuralndcg":
        return compute_neuralndcg_loss(
            scores=scores,
            relevance_labels=relevance_labels,
            k=ndcg_k,
            temperature=neuralndcg_temperature,
        )
    if baseline_type == "infonce":
        return compute_infonce_loss(
            scores=scores,
            relevance_labels=relevance_labels,
            temperature=infonce_temperature,
        )
    return compute_softrank_loss(
        scores=scores,
        relevance_labels=relevance_labels,
        k=ndcg_k,
        sigma=softrank_sigma,
    )
