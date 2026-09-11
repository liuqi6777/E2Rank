from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel
from transformers.file_utils import ModelOutput

from rag.generator import FrozenGeneratorClient
from rag.index import FrozenDistributedIndex
from rag.metrics import max_token_f1, passage_contains_answer
from query_policy import QueryOnlyRLWrapper
from rag.rewards import RetrievalRewardProvider
from query_only import (
    QueryOnlySupervisedModel,
    multi_positive_infonce_loss as shared_multi_positive_infonce_loss,
    teacher_ranknet_loss,
)


def multi_positive_infonce_loss(
    scores: Tensor, positive_mask: Tensor, temperature: float = 0.03
) -> Tensor:
    return shared_multi_positive_infonce_loss(scores, positive_mask, temperature)


def positive_negative_ranknet_loss(
    scores: Tensor, positive_mask: Tensor, temperature: float = 0.03
) -> Tensor:
    """RankNet over positive-negative pairs without a candidate-square tensor."""
    if scores.shape != positive_mask.shape:
        raise ValueError("scores and positive_mask must have equal shapes")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return teacher_ranknet_loss(scores, positive_mask.float(), temperature)


@dataclass
class RAGModelOutput(ModelOutput):
    loss: Optional[Tensor] = None
    reward_mean: Optional[Tensor] = None
    reward_std: Optional[Tensor] = None
    degenerate_fraction: Optional[Tensor] = None


class RAGSupervisedModel(QueryOnlySupervisedModel):
    def __init__(
        self,
        model: PreTrainedModel,
        index: FrozenDistributedIndex,
        objective: str,
        temperature: float = 0.03,
        pooling_method: str = "last",
    ):
        super().__init__(
            model=model,
            index=index,
            objective=objective,
            temperature=temperature,
            pooling_method=pooling_method,
        )
        if objective not in {"infonce", "ranknet"}:
            raise ValueError("Supervised RAG objective must be infonce or ranknet")

    def forward(
        self,
        query: dict[str, Tensor],
        candidate_passage_ids: Tensor,
        training_positive_mask: Tensor,
        **_: Any,
    ) -> RAGModelOutput:
        output = super().forward(
            query=query,
            candidate_ordinals=candidate_passage_ids,
            positive_mask=training_positive_mask,
            relevance_labels=training_positive_mask.float(),
        )
        return RAGModelOutput(loss=output.loss)


class RAGRLModel(QueryOnlyRLWrapper):
    def __init__(
        self,
        model: PreTrainedModel,
        index: FrozenDistributedIndex,
        reward_type: str = "source_aware_mrr",
        retrieval_k: int = 20,
        group_size: int = 32,
        kappa: float = 755.0,
        pooling_method: str = "last",
        generator: FrozenGeneratorClient | None = None,
        generator_top_k: int = 5,
        normalize_advantages: bool = True,
    ):
        if reward_type not in {"source_aware_mrr", "answer_mrr", "answer_f1"}:
            raise ValueError(f"Unsupported reward_type={reward_type}")
        super().__init__(
            model,
            self._reward_actions,
            group_size=group_size,
            kappa=kappa,
            pooling_method=pooling_method,
            normalize_advantages=normalize_advantages,
        )
        self.index = index
        self.reward_type = reward_type
        self.retrieval_k = retrieval_k
        self.retrieval_reward_provider = (
            RetrievalRewardProvider(reward_type, retrieval_k)
            if reward_type in {"source_aware_mrr", "answer_mrr"}
            else None
        )
        self.generator = generator
        self.generator_top_k = generator_top_k

    def _build_masks(
        self,
        result_ids: Tensor,
        golden_answers: Sequence[Sequence[str]],
        evidence_passage_groups: Sequence[Sequence[Sequence[int]]],
    ) -> tuple[Tensor, Tensor]:
        flat_ids = result_ids.detach().cpu().reshape(-1).tolist()
        contents = self.index.lookup_text(flat_ids)
        batch, group, depth = result_ids.shape
        answer_mask = torch.zeros((batch, group, depth), dtype=torch.bool, device=result_ids.device)
        max_evidence_groups = max((len(groups) for groups in evidence_passage_groups), default=0)
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
                    [passage_contains_answer(text, golden_answers[batch_index]) for text in row_contents],
                    device=result_ids.device,
                )
                for evidence_index, passage_group in enumerate(evidence_passage_groups[batch_index]):
                    evidence_hits[batch_index, group_index, evidence_index] = torch.tensor(
                        [ordinal in passage_group for ordinal in row_ids],
                        device=result_ids.device,
                    )
        return answer_mask, evidence_hits

    def _distributed_generate(
        self,
        query_ids: Sequence[str],
        questions: Sequence[str],
        golden_answers: Sequence[Sequence[str]],
        result_ids: Tensor,
    ) -> Tensor:
        is_rank_zero = not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0
        if is_rank_zero and self.generator is None:
            raise RuntimeError("answer_f1 reward requires a FrozenGeneratorClient")
        local_requests = []
        batch, group, _ = result_ids.shape
        for batch_index in range(batch):
            for group_index in range(group):
                ids = result_ids[batch_index, group_index, : self.generator_top_k].detach().cpu().tolist()
                local_requests.append({
                    "query_id": query_ids[batch_index],
                    "question": questions[batch_index],
                    "answers": list(golden_answers[batch_index]),
                    "passage_ids": ids,
                    "documents": [{"contents": text} for text in self.index.lookup_text(ids)],
                })

        if dist.is_available() and dist.is_initialized():
            gathered = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
            dist.gather_object(local_requests, gathered, dst=0)
            payload = [None]
            if dist.get_rank() == 0:
                flat_requests = [item for rank_requests in gathered for item in rank_requests]
                generations = self.generator.generate_batch([
                    (item["query_id"], item["question"], item["passage_ids"], item["documents"])
                    for item in flat_requests
                ])
                flat_scores = [
                    max_token_f1(output, item["answers"])
                    for output, item in zip(generations, flat_requests)
                ]
                all_outputs = []
                offset = 0
                for rank_requests in gathered:
                    all_outputs.append(flat_scores[offset : offset + len(rank_requests)])
                    offset += len(rank_requests)
                payload[0] = all_outputs
            dist.broadcast_object_list(payload, src=0)
            scores = payload[0][dist.get_rank()]
        else:
            generations = self.generator.generate_batch([
                (item["query_id"], item["question"], item["passage_ids"], item["documents"])
                for item in local_requests
            ])
            scores = [
                max_token_f1(output, item["answers"])
                for output, item in zip(generations, local_requests)
            ]
        return torch.tensor(scores, device=result_ids.device, dtype=torch.float32).reshape(batch, group)

    def _reward_actions(
        self,
        actions: Tensor,
        query_ids: Sequence[str],
        sources: Sequence[str],
        questions: Sequence[str],
        golden_answers: Sequence[Sequence[str]],
        evidence_passage_groups: Sequence[Sequence[Sequence[int]]],
        **_: Any,
    ) -> Tensor:
        _, result_ids = self.index.search(actions, self.retrieval_k)
        if self.reward_type == "answer_f1":
            rewards = self._distributed_generate(
                query_ids, questions, golden_answers, result_ids
            )
        else:
            answer_mask, evidence_ids = self._build_masks(
                result_ids, golden_answers, evidence_passage_groups
            )
            rewards = self.retrieval_reward_provider(
                answer_mask,
                sources,
                evidence_ids,
                [len(groups) for groups in evidence_passage_groups],
            )
        return rewards

    def forward(self, query: dict[str, Tensor], **reward_inputs: Any) -> RAGModelOutput:
        output = self.policy_step(query, **reward_inputs)
        return RAGModelOutput(
            loss=output.loss,
            reward_mean=output.rewards.mean().detach(),
            reward_std=output.rewards.std(unbiased=False).detach(),
            degenerate_fraction=output.degenerate_fraction.detach(),
        )
