from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from torch import Tensor
from transformers import PreTrainedModel
from transformers.file_utils import ModelOutput

from fixed_corpus.environment import DynamicRetrievalEnvironment
from fixed_corpus.models import (
    QueryOnlySupervisedModel,
    multi_positive_infonce_loss as shared_multi_positive_infonce_loss,
    teacher_ranknet_loss,
)
from fixed_corpus.policy import QueryOnlyRLWrapper
from rag.generator import FrozenGeneratorClient
from rag.index import FrozenDistributedIndex
from rag.rewards import RAGResultRewardProvider


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
        result_reward_provider = RAGResultRewardProvider(
            reward_type=reward_type,
            retrieval_k=retrieval_k,
            generator=generator,
            generator_top_k=generator_top_k,
        )
        retrieval_environment = DynamicRetrievalEnvironment(
            index=index,
            result_reward_provider=result_reward_provider,
            retrieval_k=retrieval_k,
        )
        super().__init__(
            model,
            retrieval_environment,
            group_size=group_size,
            kappa=kappa,
            pooling_method=pooling_method,
            normalize_advantages=normalize_advantages,
        )
    def forward(self, query: dict[str, Tensor], **reward_inputs: Any) -> RAGModelOutput:
        output = self.policy_step(query, **reward_inputs)
        return RAGModelOutput(
            loss=output.loss,
            reward_mean=output.rewards.mean().detach(),
            reward_std=output.rewards.std(unbiased=False).detach(),
            degenerate_fraction=output.degenerate_fraction.detach(),
        )
