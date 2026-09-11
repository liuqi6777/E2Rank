"""Shared dynamic retrieval environment for query-action policies."""

from __future__ import annotations

from typing import Any, Protocol

import torch


class SearchableFrozenIndex(Protocol):
    """Minimal index contract used by dynamic retrieval policies."""

    def search(
        self,
        query_vectors: torch.Tensor,
        k: int,
        route_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class DynamicRetrievalEnvironment:
    """Map sampled query vectors to rewards through a live corpus search.

    The environment owns retrieval mechanics only. Task-specific semantics live in
    the injected result reward provider, which receives ranked IDs and scores.
    """

    def __init__(
        self,
        index: SearchableFrozenIndex,
        result_reward_provider,
        retrieval_k: int,
    ) -> None:
        if retrieval_k <= 0:
            raise ValueError("retrieval_k must be positive")
        self.index = index
        self.result_reward_provider = result_reward_provider
        self.retrieval_k = int(retrieval_k)

    def __call__(
        self,
        actions: torch.Tensor,
        route_ids: torch.Tensor | None = None,
        **reward_inputs: Any,
    ) -> torch.Tensor:
        if route_ids is None:
            result_scores, result_ids = self.index.search(actions, self.retrieval_k)
        else:
            result_scores, result_ids = self.index.search(
                actions,
                self.retrieval_k,
                route_ids=route_ids,
            )
        rewards = self.result_reward_provider(
            result_ids=result_ids,
            result_scores=result_scores,
            index=self.index,
            route_ids=route_ids,
            **reward_inputs,
        )
        if not isinstance(rewards, torch.Tensor):
            raise TypeError("Dynamic retrieval reward provider must return a tensor")
        expected_shape = actions.shape[:-1]
        if rewards.shape != expected_shape:
            raise ValueError(
                "Dynamic retrieval rewards must match the action leading shape: "
                f"expected {tuple(expected_shape)}, got {tuple(rewards.shape)}"
            )
        rewards = rewards.detach().to(actions.device, dtype=torch.float32)
        if not torch.isfinite(rewards).all():
            raise ValueError("Dynamic retrieval rewards must be finite")
        return rewards


class KnownQrelsNDCGRewardProvider:
    """nDCG@k for live results, treating documents absent from qrels as unjudged."""

    def __init__(self, k: int) -> None:
        if k <= 0:
            raise ValueError("nDCG cutoff must be positive")
        self.k = int(k)

    def __call__(
        self,
        *,
        result_ids: torch.Tensor,
        result_scores: torch.Tensor,
        candidate_ordinals: torch.Tensor,
        relevance_labels: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        if result_ids.shape != result_scores.shape or result_ids.dim() != 3:
            raise ValueError("Retrieval IDs and scores must be [batch, group, depth]")
        if candidate_ordinals.shape != relevance_labels.shape or candidate_ordinals.dim() != 2:
            raise ValueError("Qrel ordinals and relevance labels must be [batch, candidates]")
        if candidate_ordinals.size(0) != result_ids.size(0):
            raise ValueError("Retrieval results and qrels must share their batch size")
        if candidate_mask is None:
            candidate_mask = candidate_ordinals >= 0
        elif candidate_mask.shape != candidate_ordinals.shape:
            raise ValueError("candidate_mask must match qrel ordinals")

        matches = result_ids.unsqueeze(-1) == candidate_ordinals[:, None, None, :]
        matches &= candidate_mask[:, None, None, :].bool()
        labels = relevance_labels[:, None, None, :].float()
        retrieved_labels = torch.where(matches, labels, 0.0).amax(dim=-1)

        cutoff = min(self.k, result_ids.size(-1))
        discounts = torch.arange(
            2, cutoff + 2, device=result_ids.device, dtype=torch.float32
        ).log2().reciprocal()
        dcg = (
            (2.0**retrieved_labels[..., :cutoff] - 1.0) * discounts
        ).sum(dim=-1)

        qrel_labels = relevance_labels.float().masked_fill(~candidate_mask.bool(), 0.0)
        ideal_depth = min(cutoff, qrel_labels.size(-1))
        ideal = qrel_labels.topk(ideal_depth, dim=-1).values
        idcg = (
            (2.0**ideal - 1.0) * discounts[:ideal_depth]
        ).sum(dim=-1)
        return torch.where(
            idcg[:, None] > 0,
            dcg / idcg.clamp_min(torch.finfo(torch.float32).eps)[:, None],
            torch.zeros_like(dcg),
        )
