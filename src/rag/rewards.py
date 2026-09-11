from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.distributed as dist

from rag.generator import FrozenGeneratorClient
from rag.metrics import max_token_f1, passage_contains_answer


class RAGResultRewardProvider:
    """Convert single-corpus RAG search results into the configured RL reward."""

    def __init__(
        self,
        reward_type: str = "source_aware_mrr",
        retrieval_k: int = 20,
        generator: FrozenGeneratorClient | None = None,
        generator_top_k: int = 5,
    ) -> None:
        if reward_type not in {"source_aware_mrr", "answer_mrr", "answer_f1"}:
            raise ValueError(f"Unsupported reward_type={reward_type}")
        if retrieval_k <= 0:
            raise ValueError("retrieval_k must be positive")
        if generator_top_k <= 0:
            raise ValueError("generator_top_k must be positive")
        self.reward_type = reward_type
        self.generator = generator
        self.generator_top_k = int(generator_top_k)
        self.retrieval_reward = (
            RetrievalRewardProvider(reward_type, retrieval_k)
            if reward_type in {"source_aware_mrr", "answer_mrr"}
            else None
        )

    def _build_masks(
        self,
        index,
        result_ids: torch.Tensor,
        golden_answers: Sequence[Sequence[str]],
        evidence_passage_groups: Sequence[Sequence[Sequence[int]]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        contents = index.lookup_text(result_ids)
        batch, group, depth = result_ids.shape
        answer_mask = torch.zeros(
            (batch, group, depth), dtype=torch.bool, device=result_ids.device
        )
        max_evidence_groups = max(
            (len(groups) for groups in evidence_passage_groups), default=0
        )
        evidence_hits = torch.zeros(
            (batch, group, max_evidence_groups, depth),
            dtype=torch.bool,
            device=result_ids.device,
        )
        offset = 0
        for batch_index in range(batch):
            for group_index in range(group):
                row_ids = result_ids[batch_index, group_index].detach().cpu().tolist()
                row_contents = contents[offset : offset + depth]
                offset += depth
                answer_mask[batch_index, group_index] = torch.tensor(
                    [
                        passage_contains_answer(text, golden_answers[batch_index])
                        for text in row_contents
                    ],
                    device=result_ids.device,
                )
                for evidence_index, passage_group in enumerate(
                    evidence_passage_groups[batch_index]
                ):
                    evidence_hits[
                        batch_index, group_index, evidence_index
                    ] = torch.tensor(
                        [ordinal in passage_group for ordinal in row_ids],
                        device=result_ids.device,
                    )
        return answer_mask, evidence_hits

    def _generation_requests(
        self,
        index,
        query_ids: Sequence[str],
        questions: Sequence[str],
        golden_answers: Sequence[Sequence[str]],
        result_ids: torch.Tensor,
    ) -> list[dict[str, Any]]:
        batch, group, _ = result_ids.shape
        top_ids = result_ids[..., : self.generator_top_k]
        top_depth = top_ids.size(-1)
        texts = index.lookup_text(top_ids)
        requests = []
        offset = 0
        for batch_index in range(batch):
            for group_index in range(group):
                passage_ids = top_ids[batch_index, group_index].detach().cpu().tolist()
                row_texts = texts[offset : offset + top_depth]
                offset += top_depth
                requests.append(
                    {
                        "query_id": query_ids[batch_index],
                        "question": questions[batch_index],
                        "answers": list(golden_answers[batch_index]),
                        "passage_ids": passage_ids,
                        "documents": [{"contents": text} for text in row_texts],
                    }
                )
        return requests

    def _score_generation_requests(self, requests: Sequence[dict[str, Any]]) -> list[float]:
        generations = self.generator.generate_batch(
            [
                (
                    item["query_id"],
                    item["question"],
                    item["passage_ids"],
                    item["documents"],
                )
                for item in requests
            ]
        )
        return [
            max_token_f1(output, item["answers"])
            for output, item in zip(generations, requests)
        ]

    def _distributed_generate(
        self,
        index,
        query_ids: Sequence[str],
        questions: Sequence[str],
        golden_answers: Sequence[Sequence[str]],
        result_ids: torch.Tensor,
    ) -> torch.Tensor:
        distributed = dist.is_available() and dist.is_initialized()
        is_rank_zero = not distributed or dist.get_rank() == 0
        if is_rank_zero and self.generator is None:
            raise RuntimeError("answer_f1 reward requires a FrozenGeneratorClient")

        requests = self._generation_requests(
            index, query_ids, questions, golden_answers, result_ids
        )

        if distributed:
            gathered = [None] * dist.get_world_size() if is_rank_zero else None
            dist.gather_object(requests, gathered, dst=0)
            payload = [None]
            if is_rank_zero:
                flat_scores = self._score_generation_requests(
                    [item for rank_requests in gathered for item in rank_requests]
                )
                all_outputs = []
                offset = 0
                for rank_requests in gathered:
                    all_outputs.append(flat_scores[offset : offset + len(rank_requests)])
                    offset += len(rank_requests)
                payload[0] = all_outputs
            dist.broadcast_object_list(payload, src=0)
            scores = payload[0][dist.get_rank()]
        else:
            scores = self._score_generation_requests(requests)
        batch, group, _ = result_ids.shape
        return torch.tensor(
            scores, device=result_ids.device, dtype=torch.float32
        ).reshape(batch, group)

    def __call__(
        self,
        *,
        result_ids: torch.Tensor,
        result_scores: torch.Tensor,
        index,
        route_ids: torch.Tensor | None = None,
        query_ids: Sequence[str],
        sources: Sequence[str],
        questions: Sequence[str],
        golden_answers: Sequence[Sequence[str]],
        evidence_passage_groups: Sequence[Sequence[Sequence[int]]],
        **_: Any,
    ) -> torch.Tensor:
        del result_scores
        if route_ids is not None:
            raise ValueError("RAG reward currently uses one shared corpus and no route_ids")
        if self.reward_type == "answer_f1":
            return self._distributed_generate(
                index,
                query_ids,
                questions,
                golden_answers,
                result_ids,
            )
        answer_mask, evidence_memberships = self._build_masks(
            index,
            result_ids,
            golden_answers,
            evidence_passage_groups,
        )
        return self.retrieval_reward(
            answer_mask,
            sources,
            evidence_memberships,
            [len(groups) for groups in evidence_passage_groups],
        )


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
