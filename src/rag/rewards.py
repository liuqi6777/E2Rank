from __future__ import annotations

from collections.abc import Sequence

import torch


class RetrievalRewardProvider:
    """One configurable provider for answer-MRR and source-aware-MRR."""

    def __init__(self, mode: str = "source_aware_mrr", k: int = 20):
        if mode not in {"source_aware_mrr", "answer_mrr"}:
            raise ValueError(f"Unsupported retrieval reward mode: {mode}")
        if k <= 0:
            raise ValueError("retrieval reward cutoff must be positive")
        self.mode = mode
        self.k = int(k)

    def __call__(
        self,
        answer_mask: torch.Tensor,
        sources: Sequence[str],
        evidence_memberships: torch.Tensor | None = None,
        evidence_group_counts: Sequence[int] | None = None,
    ) -> torch.Tensor:
        if self.mode == "answer_mrr":
            return answer_mrr_from_mask(answer_mask, self.k)
        return source_aware_mrr(
            answer_mask,
            sources,
            evidence_memberships,
            evidence_group_counts,
            self.k,
        )


def answer_mrr_from_mask(answer_mask: torch.Tensor, k: int = 20) -> torch.Tensor:
    """MRR of the first answer-containing result for [..., retrieved]."""
    if answer_mask.dim() < 1:
        raise ValueError("answer_mask must have a retrieval dimension")
    cutoff = min(k, answer_mask.size(-1))
    mask = answer_mask[..., :cutoff].bool()
    ranks = torch.arange(1, cutoff + 1, device=mask.device, dtype=torch.float32)
    reciprocal = torch.where(mask, ranks.reciprocal(), torch.zeros_like(ranks))
    return reciprocal.max(dim=-1).values


def source_aware_mrr(
    answer_mask: torch.Tensor,
    sources: Sequence[str],
    evidence_group_ids: torch.Tensor | None,
    evidence_group_counts: Sequence[int] | None = None,
    k: int = 20,
) -> torch.Tensor:
    """NQ answer MRR; Hotpot mean reciprocal rank over evidence groups.

    ``evidence_group_ids`` may either have the same shape as ``answer_mask`` with
    one integer group id per result, or be a boolean tensor shaped
    ``[batch, ..., evidence_group, retrieved]``. The latter preserves the case in
    which one passage satisfies multiple supporting facts.
    """
    answer_rewards = answer_mrr_from_mask(answer_mask, k=k)
    if len(sources) != answer_mask.shape[0]:
        raise ValueError("sources length must match the leading batch dimension")
    if evidence_group_ids is None:
        if any(source.lower() == "hotpotqa" for source in sources):
            raise ValueError("HotpotQA source-aware MRR requires evidence_group_ids")
        return answer_rewards
    boolean_membership = evidence_group_ids.dtype == torch.bool
    if boolean_membership:
        if evidence_group_ids.shape[:-2] != answer_mask.shape[:-1] or evidence_group_ids.shape[-1] != answer_mask.shape[-1]:
            raise ValueError("boolean evidence memberships must be [batch, ..., group, retrieved]")
    elif evidence_group_ids.shape != answer_mask.shape:
        raise ValueError("integer evidence_group_ids must match answer_mask")

    rewards = answer_rewards.clone()
    cutoff = min(k, answer_mask.size(-1))
    ranks = torch.arange(1, cutoff + 1, device=answer_mask.device, dtype=torch.float32)
    for batch_index, source in enumerate(sources):
        if source.lower() != "hotpotqa":
            continue
        groups = evidence_group_ids[batch_index, ..., :cutoff]
        expected_groups = (
            int(evidence_group_counts[batch_index])
            if evidence_group_counts is not None
            else (groups.size(-2) if boolean_membership else int(torch.unique(groups[groups >= 0]).numel()))
        )
        if expected_groups <= 0:
            rewards[batch_index] = 0.0
            continue
        group_rewards = []
        for group_id in range(expected_groups):
            matches = groups[..., group_id, :] if boolean_membership else groups == group_id
            reciprocal = torch.where(matches, ranks.reciprocal(), torch.zeros_like(ranks))
            group_rewards.append(reciprocal.max(dim=-1).values)
        rewards[batch_index] = torch.stack(group_rewards).mean(dim=0)
    return rewards


def retrieval_recall(answer_mask: torch.Tensor, k: int) -> torch.Tensor:
    cutoff = min(k, answer_mask.size(-1))
    return answer_mask[..., :cutoff].bool().any(dim=-1).float()


def evidence_recall(evidence_group_ids: torch.Tensor, num_groups: torch.Tensor, k: int) -> torch.Tensor:
    """Fraction of gold evidence groups present in top-k for each query."""
    cutoff = min(k, evidence_group_ids.size(-1))
    values = []
    for row, expected in zip(evidence_group_ids[..., :cutoff], num_groups):
        found = (
            int(row.any(dim=-1).sum())
            if row.dtype == torch.bool
            else torch.unique(row[row >= 0]).numel()
        )
        values.append(float(found) / max(int(expected), 1))
    return torch.tensor(values, device=evidence_group_ids.device, dtype=torch.float32)
