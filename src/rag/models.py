from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import Tensor
from transformers import PreTrainedModel
from transformers.file_utils import ModelOutput

from fixed_corpus.environment import DynamicRetrievalEnvironment
from fixed_corpus.models import (
    QueryOnlySupervisedModel,
    lambda_loss,
    multi_positive_infonce_loss as shared_multi_positive_infonce_loss,
    teacher_ranknet_loss,
)
from fixed_corpus.policy import QueryOnlyRLWrapper
from rag.generator import FrozenGeneratorClient
from rag.index import FrozenDistributedIndex
from rag.negatives import build_cross_query_pool
from rag.relevance import RELEVANCE_SCHEMES, RelevanceView, build_relevance
from rag.rewards import RAGResultRewardProvider


def multi_positive_infonce_loss(
    scores: Tensor,
    positive_mask: Tensor,
    temperature: float = 0.03,
    candidate_mask: Tensor | None = None,
) -> Tensor:
    return shared_multi_positive_infonce_loss(scores, positive_mask, temperature, candidate_mask)


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
    exploration_metrics: Optional[dict[str, Tensor]] = None


class RAGSupervisedModel(QueryOnlySupervisedModel):
    """Query-only contrastive training against a frozen corpus index.

    Beyond the base class this adds the two levers RAG was missing: a
    configurable relevance scheme (see ``rag.relevance``) and a cross-query
    negative pool that can span data-parallel ranks.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        index: FrozenDistributedIndex,
        objective: str,
        temperature: float = 0.03,
        pooling_method: str = "last",
        relevance_scheme: str = "binary",
        ndcg_k: int = 10,
        lambdaloss_sigma: float = 1.0,
        use_in_batch_candidates: bool = False,
        in_batch_include_negatives: bool = False,
        cross_device_negatives: bool = False,
        in_batch_pool_size: int = 15,
        anchor_coef: float = 0.0,
    ):
        super().__init__(
            model=model,
            index=index,
            objective=objective,
            temperature=temperature,
            pooling_method=pooling_method,
            ndcg_k=ndcg_k,
            lambdaloss_sigma=lambdaloss_sigma,
        )
        if objective not in {"infonce", "ranknet", "lambdaloss"}:
            raise ValueError("Supervised RAG objective must be infonce, ranknet or lambdaloss")
        if relevance_scheme not in RELEVANCE_SCHEMES:
            raise ValueError(f"Unsupported relevance scheme {relevance_scheme!r}")
        self.relevance_scheme = relevance_scheme
        self.use_in_batch_candidates = use_in_batch_candidates
        self.in_batch_include_negatives = in_batch_include_negatives
        self.cross_device_negatives = cross_device_negatives
        self.in_batch_pool_size = int(in_batch_pool_size)
        self.anchor_coef = float(anchor_coef)

    def _append_cross_query_negatives(
        self,
        queries: Tensor,
        scored,
        view,
        candidate_passage_ids: Tensor,
        judged_mask: Tensor,
    ):
        pool_ordinals, pool_mask = build_cross_query_pool(
            candidate_ordinals=candidate_passage_ids,
            candidate_mask=scored.candidate_mask,
            judged_mask=judged_mask,
            pool_size=self.in_batch_pool_size,
            include_negatives=self.in_batch_include_negatives,
            cross_device=self.cross_device_negatives,
        )
        batch = queries.size(0)
        width = pool_ordinals.numel()
        pooled = self.scorer(
            queries,
            pool_ordinals.unsqueeze(0).expand(batch, width),
            pool_mask,
        )
        zeros_bool = torch.zeros((batch, width), device=queries.device, dtype=torch.bool)
        scores = torch.cat((scored.scores, pooled.scores), dim=-1)
        return scores, RelevanceView(
            labels=torch.cat((view.labels, view.labels.new_zeros((batch, width))), dim=-1),
            positive_mask=torch.cat((view.positive_mask, zeros_bool), dim=-1),
            denominator_mask=torch.cat((view.denominator_mask, pool_mask), dim=-1),
            scoring_mask=torch.cat((view.scoring_mask, pool_mask), dim=-1),
        )

    def forward(
        self,
        query: dict[str, Tensor],
        candidate_passage_ids: Tensor,
        training_positive_mask: Tensor,
        answer_positive_mask: Tensor | None = None,
        evidence_positive_mask: Tensor | None = None,
        anchor_embeddings: Tensor | None = None,
        **_: Any,
    ) -> RAGModelOutput:
        if answer_positive_mask is None:
            answer_positive_mask = training_positive_mask
        queries = self.encode_query(query)
        valid = candidate_passage_ids >= 0
        scored = self.scorer(queries, candidate_passage_ids, valid)
        view = build_relevance(
            scheme=self.relevance_scheme,
            training_positive_mask=training_positive_mask,
            answer_positive_mask=answer_positive_mask,
            evidence_positive_mask=evidence_positive_mask,
            candidate_mask=scored.candidate_mask,
        )
        scores = scored.scores
        if self.use_in_batch_candidates:
            # Anything this query judges positive or answer-bearing must not come
            # back as somebody else's negative.
            judged = view.positive_mask | answer_positive_mask.bool()
            scores, view = self._append_cross_query_negatives(
                queries, scored, view, candidate_passage_ids, judged
            )

        if self.objective == "infonce":
            losses = multi_positive_infonce_loss(
                scores, view.positive_mask, self.temperature, view.denominator_mask
            )
        elif self.objective == "ranknet":
            losses = teacher_ranknet_loss(
                scores, view.labels, self.temperature, view.scoring_mask
            )
        else:
            losses = lambda_loss(
                scores, view.labels, self.ndcg_k, self.lambdaloss_sigma, view.scoring_mask
            )
        loss = losses.mean()
        if anchor_embeddings is not None and self.anchor_coef > 0:
            loss = loss + self.anchor_coef * anchor_penalty(queries, anchor_embeddings)
        return RAGModelOutput(loss=loss)


def anchor_penalty(means: Tensor, anchors: Tensor) -> Tensor:
    """``mean(1 - cos(e_theta(q), e_E0(q)))``: the drift round 3 measured."""
    if anchors.shape != means.shape:
        raise ValueError(
            f"anchor embeddings must match query embeddings: {tuple(anchors.shape)} vs "
            f"{tuple(means.shape)}"
        )
    similarity = torch.nn.functional.cosine_similarity(
        means.float(), anchors.float().to(means.device), dim=-1
    )
    return (1.0 - similarity).mean()


class RAGRLModel(QueryOnlyRLWrapper):
    def __init__(
        self,
        model: PreTrainedModel,
        index: FrozenDistributedIndex,
        reward_type: str = "mrr",
        retrieval_k: int = 10,
        group_size: int = 32,
        kappa: float = 755.0,
        pooling_method: str = "last",
        generator: FrozenGeneratorClient | None = None,
        generator_top_k: int = 10,
        normalize_advantages: bool | None = None,
        advantage_baseline: str = "leave_one_out",
        advantage_norm: str = "none",
        advantage_baseline_momentum: float = 0.99,
        target_alignment: float | None = None,
        final_alignment: float | None = None,
        exploration_schedule: str = "fixed",
        relevance_scheme: str = "binary",
        anchor_coef: float = 0.0,
    ):
        result_reward_provider = RAGResultRewardProvider(
            reward_type=reward_type,
            retrieval_k=retrieval_k,
            generator=generator,
            generator_top_k=generator_top_k,
            relevance_scheme=relevance_scheme,
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
            advantage_baseline=advantage_baseline, advantage_norm=advantage_norm,
            advantage_baseline_momentum=advantage_baseline_momentum,
            target_alignment=target_alignment, final_alignment=final_alignment,
            exploration_schedule=exploration_schedule,
        )
        self.anchor_coef = float(anchor_coef)

    def policy_step(self, query: dict[str, Tensor], **reward_inputs: Any):
        # The anchor rides in with the collator's batch and must not reach the
        # reward provider, which validates its keyword contract strictly.
        anchors = reward_inputs.pop("anchor_embeddings", None)
        if anchors is None or self.anchor_coef <= 0:
            return super().policy_step(query, **reward_inputs)
        means = self.encode_query(query)
        output = self.policy_step_from_embeddings(means, **reward_inputs)
        output.loss = output.loss + self.anchor_coef * anchor_penalty(means, anchors)
        return output

    def forward(self, query: dict[str, Tensor], **reward_inputs: Any) -> RAGModelOutput:
        output = self.policy_step(query, **reward_inputs)
        return RAGModelOutput(
            loss=output.loss,
            reward_mean=output.rewards.mean().detach(),
            reward_std=output.rewards.std(unbiased=False).detach(),
            degenerate_fraction=output.degenerate_fraction.detach(),
            exploration_metrics=self.policy.exploration_metrics(),
        )
