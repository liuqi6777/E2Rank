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
