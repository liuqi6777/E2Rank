"""Action-independent, stratified shortlists for fixed cross-query RL pools."""

import math

import torch

from rewards import compute_reward_terms_over_fixed_pool


SHORTLIST_VERSION = 1


def validate_shortlist_sampling(count, size, hard_count, hard_pool_size):
    for name, value, minimum in (("count", count, 0), ("size", size, 0),
                                 ("hard_count", hard_count, 0), ("hard_pool_size", hard_pool_size, 0)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"reward_shortlist_{name} must be an integer >= {minimum}")
    if hard_count > size or hard_pool_size < hard_count:
        raise ValueError("Shortlist hard_count must not exceed size or hard_pool_size")


def validate_shortlist_objectives(count, reward_terms, binary_weight, pairwise_coef=0.,
                                 gradient_estimator="conditional_projection"):
    if not math.isfinite(binary_weight) or not 0 <= binary_weight <= 1:
        raise ValueError("reward_shortlist_binary_weight must be finite and in [0, 1]")
    if binary_weight and (not count or len(reward_terms) != 1
                          or reward_terms[0].type != "ndcg_in_batch"
                          or reward_terms[0].weight != 1
                          or reward_terms[0].name == "binary_ndcg"):
        raise ValueError("Binary shortlist mixing requires shortlists and one unit-weight nDCG term; "
                         "binary_ndcg is reserved for the original-positive reward")
    if not math.isfinite(pairwise_coef) or pairwise_coef < 0:
        raise ValueError("reward_shortlist_pairwise_coef must be finite and non-negative")
    if pairwise_coef and (not count or gradient_estimator != "conditional_projection"):
        raise ValueError("Pairwise shortlist rewards require shortlists and conditional_projection")


def shortlist_positive_mask(positive_mask, valid):
    """Use annotated identities, never a threshold on teacher grades."""
    if (positive_mask is None or positive_mask.shape != valid.shape
            or positive_mask.dtype != torch.bool):
        raise ValueError("Shortlist binary objectives require boolean positive_mask matching candidates")
    positives = positive_mask & valid
    if not positives.any(-1).all():
        raise ValueError("Shortlist binary objectives require a valid annotated positive per query")
    return positives


@torch.no_grad()
def shortlist_rewards(reward_terms, *, scores, labels, valid, rank_labels, cross_scores,
                      binary_weight=0., positive_mask=None):
    """Return the mixed reward and unweighted term diagnostics on identical actions/pools."""
    terms = compute_reward_terms_over_fixed_pool(
        reward_terms, scores=scores, relevance_labels=labels, candidate_mask=valid,
        rank_labels=rank_labels, cross_scores=cross_scores,
    )
    reward = sum(term.weight * terms[term.name].float() for term in reward_terms)
    if binary_weight:
        positives = shortlist_positive_mask(positive_mask, valid)
        binary = compute_reward_terms_over_fixed_pool(
            reward_terms, scores=scores, relevance_labels=positives.float(), candidate_mask=valid,
            rank_labels=None, cross_scores=cross_scores,
        )[reward_terms[0].name].float()
        terms["binary_ndcg"] = binary
        reward = (1 - binary_weight) * reward + binary_weight * binary
    return reward, terms


def shortlist_contract(head):
    if not getattr(head, "reward_shortlist_count", 0):
        return None
    contract = dict(version=SHORTLIST_VERSION, **{
        key: getattr(head, f"reward_shortlist_{key}")
        for key in ("count", "size", "hard_count", "hard_pool_size")
    })
    # Preserve the exact v1 contract for existing cross-device/all-candidate runs.
    source = getattr(head, "reward_shortlist_pool_source", "cross_device_all")
    if source != "cross_device_all":
        contract["pool_source"] = source
    if getattr(head, "reward_shortlist_binary_weight", 0):
        contract["binary_reward"] = dict(
            weight=head.reward_shortlist_binary_weight, labels="positive_mask", version=1,
        )
    if getattr(head, "reward_shortlist_pairwise_coef", 0):
        contract["pairwise_reward"] = dict(
            coefficient=head.reward_shortlist_pairwise_coef, labels="positive_mask", version=1,
            pairs="all_own_positives_x_own_and_selected_negatives", reduction="per_query_mean",
            estimator="per_pair_conditional_projection", ties=0.5,
        )
    return contract


@torch.no_grad()
def sample_shortlists(scores, valid, *, count, size, hard_count, hard_pool_size):
    """Return indices/mask [B,T,K] and hard membership, using only mean scores.

    Partition each query's allowed pool into its highest-scoring H documents and
    the remainder. Randomly permute each stratum ONCE and interleave chunks at the
    requested ratio. When a stratum runs out, fill from the remaining UNUSED
    candidates. Wrap only after the ENTIRE pool is exhausted: coverage is exactly
    min(pool_size, T*K), with no within-list duplicates. Later lists can have a
    different hard fraction; coverage takes priority over a fixed hard quota.
    The first list and RNG consumption do not depend on T.

    size=0 retains the shortlist objective with only own candidates.
    On scarcity, transfer unavailable quota to the other stratum, then pad if
    the entire pool has fewer than K items. hard_count=0 is uniform over the
    entire allowed pool, not just the low-scoring remainder.
    """
    validate_shortlist_sampling(count, size, hard_count, hard_pool_size)
    if count == 0 or scores.ndim != 2 or valid.shape != scores.shape or valid.dtype != torch.bool:
        raise ValueError("Sampling needs count > 0 and matching [batch,pool] scores/bool mask")
    if not torch.isfinite(scores[valid]).all():
        raise ValueError("Valid shortlist mean scores must be finite")
    batch = scores.size(0)
    indices = torch.zeros((batch, count, size), dtype=torch.long, device=scores.device)
    mask = torch.zeros_like(indices, dtype=torch.bool)
    hard_mask = torch.zeros_like(mask)
    if size == 0:
        return indices, mask, hard_mask
    for b in range(batch):
        allowed = valid[b].nonzero(as_tuple=True)[0]
        if not allowed.numel():
            continue
        if hard_count:
            order = scores[b, allowed].argsort(descending=True, stable=True)
            allowed = allowed[order]
        nh = min(hard_pool_size, allowed.numel()) if hard_count else 0
        n = min(size, allowed.numel())
        take_hard = min(hard_count, nh, n)
        take_rest = min(n - take_hard, allowed.numel() - nh)
        take_hard = n - take_rest
        hard = allowed[:nh][torch.randperm(nh, device=scores.device)]
        rest = allowed[nh:][torch.randperm(allowed.numel() - nh, device=scores.device)]
        items = torch.cat((hard, rest))
        membership = torch.arange(items.numel(), device=scores.device) < nh
        if take_hard and take_rest:
            ih = torch.arange(nh, device=scores.device)
            ir = torch.arange(rest.numel(), device=scores.device)
            priority = torch.cat((ih // take_hard * n + ih % take_hard,
                                  ir // take_rest * n + take_hard + ir % take_rest))
            order = priority.argsort(stable=True)
            items, membership = items[order], membership[order]
        positions = torch.arange(count * n, device=scores.device).reshape(count, n) % items.numel()
        indices[b, :, :n] = items[positions]
        mask[b, :, :n] = True
        hard_mask[b, :, :n] = membership[positions]
    return indices, mask, hard_mask


@torch.no_grad()
def shortlist_statistics(scores, allowed, indices, mask, hard_mask):
    counts = mask.sum(-1).float()
    pool_counts = allowed.sum(-1).float()
    unique = torch.stack([indices[b][mask[b]].unique().numel() * scores.new_ones(())
                          for b in range(scores.size(0))])
    total = counts.sum(-1)
    selected_scores = scores.gather(1, indices.flatten(1)) if scores.size(1) else counts.new_zeros(
        scores.size(0), indices.size(1) * indices.size(2))
    selected_scores = selected_scores.masked_fill(~mask.flatten(1), 0)
    return {
        "shortlist/count": counts.new_tensor(indices.size(1)),
        "shortlist/pool_candidates_mean": pool_counts.mean(),
        "shortlist/pool_candidates_max": pool_counts.max(),
        "shortlist/unique_candidates_mean": unique.mean(),
        "shortlist/coverage_mean": (unique / pool_counts.clamp_min(1)).mean(),
        "shortlist/repeat_fraction": ((total - unique) / total.clamp_min(1)).mean(),
        "shortlist/hard_candidates_mean": hard_mask.sum(-1).float().mean(),
        "shortlist/selected_score_mean": selected_scores.sum() / total.sum().clamp_min(1),
        "reward_pool/cross_candidates_mean": counts.mean(),
        "reward_pool/cross_candidates_max": counts.max(),
    }
