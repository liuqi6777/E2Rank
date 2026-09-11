"""Shared query-only models over immutable document embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.file_utils import ModelOutput

from embedding_protocol import pool_embeddings
from frozen_corpus import FrozenCorpusIndex


@dataclass
class CandidateScoreOutput:
    scores: Tensor
    candidate_mask: Tensor


class FrozenCandidateScorer(nn.Module):
    """Cosine scorer backed by ordinal lookup from a read-only fp16 corpus."""

    def __init__(self, index: FrozenCorpusIndex):
        super().__init__()
        self.index = index

    def forward(
        self,
        query_embeddings: Tensor,
        candidate_ordinals: Tensor,
        candidate_mask: Tensor | None = None,
    ) -> CandidateScoreOutput:
        if candidate_ordinals.dim() != 2:
            raise ValueError("candidate_ordinals must have shape [batch, candidates]")
        if query_embeddings.dim() != 2 or query_embeddings.size(0) != candidate_ordinals.size(0):
            raise ValueError("query embeddings and candidate ordinals must share their batch size")
        if candidate_mask is None:
            candidate_mask = candidate_ordinals >= 0
        if candidate_mask.shape != candidate_ordinals.shape or candidate_mask.dtype != torch.bool:
            raise ValueError("candidate_mask must be boolean and match candidate_ordinals")
        if not candidate_mask.any(dim=-1).all():
            raise ValueError("Every query must retain at least one candidate")
        safe_ordinals = candidate_ordinals.masked_fill(~candidate_mask, 0)
        documents = self.index.lookup_embeddings(safe_ordinals).detach()
        documents = F.normalize(documents.float(), dim=-1)
        queries = F.normalize(query_embeddings.float(), dim=-1)
        scores = torch.einsum("bd,bkd->bk", queries, documents)
        return CandidateScoreOutput(
            scores=scores.masked_fill(~candidate_mask.to(scores.device), float("-inf")),
            candidate_mask=candidate_mask.to(scores.device),
        )

    def append_in_batch(
        self,
        query_embeddings: Tensor,
        base: CandidateScoreOutput,
        batch_candidate_ordinals: Tensor,
        cross_candidate_mask: Tensor,
    ) -> CandidateScoreOutput:
        """Append a batch-wide candidate grid with caller-defined duplicate filtering."""
        if batch_candidate_ordinals.dim() != 2:
            raise ValueError("batch_candidate_ordinals must be [batch, candidates]")
        batch, width = batch_candidate_ordinals.shape
        if query_embeddings.size(0) != batch:
            raise ValueError("In-batch candidate and query batch sizes differ")
        if cross_candidate_mask.shape != (batch, batch, width):
            raise ValueError("cross_candidate_mask must be [batch, batch, candidates]")
        flat_ordinals = batch_candidate_ordinals.reshape(1, -1).expand(batch, -1)
        flat_mask = cross_candidate_mask.reshape(batch, -1).bool()
        safe = flat_ordinals.masked_fill(~flat_mask, 0)
        documents = self.index.lookup_embeddings(safe).detach()
        scores = torch.einsum(
            "bd,bkd->bk",
            F.normalize(query_embeddings.float(), dim=-1),
            F.normalize(documents.float(), dim=-1),
        ).masked_fill(~flat_mask.to(query_embeddings.device), float("-inf"))
        return CandidateScoreOutput(
            scores=torch.cat((base.scores, scores), dim=-1),
            candidate_mask=torch.cat((base.candidate_mask, flat_mask.to(base.scores.device)), dim=-1),
        )


def multi_positive_infonce_loss(
    scores: Tensor,
    positive_mask: Tensor,
    temperature: float = 0.03,
    candidate_mask: Tensor | None = None,
) -> Tensor:
    if scores.shape != positive_mask.shape or temperature <= 0:
        raise ValueError("InfoNCE inputs must have equal shapes and positive temperature")
    mask = torch.ones_like(positive_mask, dtype=torch.bool) if candidate_mask is None else candidate_mask.bool()
    positives = positive_mask.bool() & mask
    if not positives.any(dim=-1).all():
        raise ValueError("Every query must have at least one unmasked positive")
    scaled = scores.float().masked_fill(~mask, float("-inf")) / float(temperature)
    return torch.logsumexp(scaled, dim=-1) - torch.logsumexp(
        scaled.masked_fill(~positives, float("-inf")), dim=-1
    )


def teacher_ranknet_loss(
    scores: Tensor,
    rank_labels: Tensor,
    temperature: float = 0.03,
    candidate_mask: Tensor | None = None,
) -> Tensor:
    if scores.shape != rank_labels.shape or temperature <= 0:
        raise ValueError("RankNet inputs must have equal shapes and positive temperature")
    mask = torch.ones_like(rank_labels, dtype=torch.bool) if candidate_mask is None else candidate_mask.bool()
    safe_scores = scores.float().masked_fill(~mask, 0)
    differences = safe_scores.unsqueeze(2) - safe_scores.unsqueeze(1)
    ordered = rank_labels.unsqueeze(2) > rank_labels.unsqueeze(1)
    ordered &= mask.unsqueeze(2) & mask.unsqueeze(1)
    losses = F.softplus(-differences / float(temperature)) * ordered
    counts = ordered.sum((1, 2))
    return torch.where(counts > 0, losses.sum((1, 2)) / counts.clamp_min(1), losses.sum((1, 2)))


def lambda_loss(
    scores: Tensor,
    relevance_labels: Tensor,
    k: int = 10,
    sigma: float = 1.0,
    candidate_mask: Tensor | None = None,
) -> Tensor:
    if scores.shape != relevance_labels.shape or k <= 0 or sigma <= 0:
        raise ValueError("LambdaLoss inputs/configuration are invalid")
    mask = torch.ones_like(relevance_labels, dtype=torch.bool) if candidate_mask is None else candidate_mask.bool()
    labels = relevance_labels.float().masked_fill(~mask, 0)
    ranking = scores.masked_fill(~mask, float("-inf")).argsort(
        dim=-1, descending=True, stable=True
    )
    positions = torch.empty_like(ranking)
    positions.scatter_(1, ranking, torch.arange(scores.size(1), device=scores.device).expand_as(ranking))
    cutoff = min(k, scores.size(1))
    discounts = scores.new_zeros(scores.size(1), dtype=torch.float32)
    discounts[:cutoff] = 1.0 / torch.log2(torch.arange(2, cutoff + 2, device=scores.device).float())
    item_discounts = discounts[positions].masked_fill(~mask, 0)
    gains = 2.0**labels - 1.0
    ideal = labels.masked_fill(~mask, float("-inf")).topk(cutoff, dim=-1).values
    ideal = ideal.masked_fill(~torch.isfinite(ideal), 0)
    idcg = ((2.0**ideal - 1.0) * discounts[:cutoff]).sum(-1)
    preferred = labels.unsqueeze(2) > labels.unsqueeze(1)
    preferred &= mask.unsqueeze(2) & mask.unsqueeze(1)
    weights = (
        (gains.unsqueeze(2) - gains.unsqueeze(1)).abs()
        * (item_discounts.unsqueeze(2) - item_discounts.unsqueeze(1)).abs()
    )
    weights = torch.where(
        idcg[:, None, None] > 0,
        weights / idcg.clamp_min(torch.finfo(torch.float32).eps)[:, None, None],
        torch.zeros_like(weights),
    ) * preferred
    safe_scores = scores.float().masked_fill(~mask, 0)
    pair_losses = F.softplus(-float(sigma) * (safe_scores.unsqueeze(2) - safe_scores.unsqueeze(1)))
    totals = weights.sum((1, 2))
    return torch.where(totals > 0, (pair_losses * weights).sum((1, 2)) / totals.clamp_min(1e-12), totals)


@dataclass
class QueryOnlyModelOutput(ModelOutput):
    loss: Tensor | None = None


class QueryEncoderMixin:
    model: nn.Module
    pooling_method: str

    def encode_query(self, inputs: dict[str, Tensor]) -> Tensor:
        return pool_embeddings(
            self.model(**inputs).last_hidden_state,
            inputs["attention_mask"],
            pooling_method=self.pooling_method,
            normalize=True,
        )

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.model.gradient_checkpointing_enable(*args, **kwargs)

    def enable_input_require_grads(self):
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()


class QueryOnlySupervisedModel(QueryEncoderMixin, nn.Module):
    """Shared single/multi-positive InfoNCE, teacher RankNet and LambdaLoss wrapper."""

    def __init__(
        self,
        model: nn.Module,
        index: FrozenCorpusIndex,
        objective: str,
        temperature: float = 0.03,
        pooling_method: str = "last",
        ndcg_k: int = 10,
        lambdaloss_sigma: float = 1.0,
        use_in_batch_candidates: bool = False,
    ):
        super().__init__()
        if objective not in {"infonce", "ranknet", "lambdaloss"}:
            raise ValueError(f"Unsupported query-only supervised objective: {objective}")
        self.model = model
        self.config = model.config
        self.index = index
        self.scorer = FrozenCandidateScorer(index)
        self.objective = objective
        self.temperature = temperature
        self.pooling_method = pooling_method
        self.ndcg_k = ndcg_k
        self.lambdaloss_sigma = lambdaloss_sigma
        self.use_in_batch_candidates = use_in_batch_candidates

    def forward(
        self,
        query: dict[str, Tensor],
        candidate_ordinals: Tensor | None = None,
        candidate_mask: Tensor | None = None,
        relevance_labels: Tensor | None = None,
        rank_labels: Tensor | None = None,
        positive_mask: Tensor | None = None,
        in_batch_positive_mask: Tensor | None = None,
        candidate_passage_ids: Tensor | None = None,
        training_positive_mask: Tensor | None = None,
        **_: Any,
    ) -> QueryOnlyModelOutput:
        # RAG's old batch field names are accepted at this adapter boundary.
        ordinals = candidate_ordinals if candidate_ordinals is not None else candidate_passage_ids
        if ordinals is None:
            raise ValueError("candidate_ordinals are required")
        mask = ordinals >= 0 if candidate_mask is None else candidate_mask.bool()
        queries = self.encode_query(query)
        scored = self.scorer(queries, ordinals, mask)
        if positive_mask is None:
            positive_mask = training_positive_mask
        if positive_mask is None and relevance_labels is not None:
            # G1 collation places its once-selected positive first; graded teacher
            # labels remain separate supervision for LambdaLoss.
            positive_mask = torch.zeros_like(relevance_labels, dtype=torch.bool)
            positive_mask[:, 0] = True
        if self.objective == "infonce" and positive_mask is None:
            raise ValueError("InfoNCE requires positive labels")

        if self.use_in_batch_candidates:
            batch = ordinals.size(0)
            cross = in_batch_positive_mask
            if cross is None:
                cross = ~torch.eye(batch, device=ordinals.device, dtype=torch.bool)
            cross = cross.reshape(batch, batch, 1)
            scored = self.scorer.append_in_batch(queries, scored, ordinals[:, :1], cross)
            zeros = relevance_labels.new_zeros((batch, batch)) if relevance_labels is not None else None
            if relevance_labels is not None:
                relevance_labels = torch.cat((relevance_labels, zeros), dim=-1)
            if rank_labels is not None:
                rank_labels = torch.cat((rank_labels, rank_labels.new_zeros((batch, batch))), dim=-1)
            positive_mask = torch.cat(
                (positive_mask.bool(), torch.zeros((batch, batch), device=ordinals.device, dtype=torch.bool)),
                dim=-1,
            )

        if self.objective == "infonce":
            losses = multi_positive_infonce_loss(scored.scores, positive_mask, self.temperature, scored.candidate_mask)
        elif self.objective == "ranknet":
            labels = rank_labels if rank_labels is not None else relevance_labels
            if labels is None:
                raise ValueError("RankNet requires rank_labels or relevance_labels")
            losses = teacher_ranknet_loss(scored.scores, labels, self.temperature, scored.candidate_mask)
        else:
            if relevance_labels is None:
                raise ValueError("LambdaLoss requires relevance_labels")
            losses = lambda_loss(
                scored.scores,
                relevance_labels,
                self.ndcg_k,
                self.lambdaloss_sigma,
                scored.candidate_mask,
            )
        return QueryOnlyModelOutput(loss=losses.mean())


class FixedCorpusGRPOModel(QueryEncoderMixin, nn.Module):
    """G1 reward semantics with query actions and immutable document candidates."""

    def __init__(self, model: nn.Module, index: FrozenCorpusIndex, rl_args, pooling_method: str = "last"):
        super().__init__()
        from grpo import GRPO

        if tuple(rl_args.action_components) != (("query",),):
            raise ValueError("Frozen-document GRPO requires action_components=[[query]]")
        self.model = model
        self.config = model.config
        self.pooling_method = pooling_method
        self.scorer = FrozenCandidateScorer(index)
        self.grpo = GRPO(
            action_components=(("query",),),
            group_size=rl_args.group_size,
            sigma=rl_args.sigma,
            kappa=rl_args.kappa,
            sigma_learnable=rl_args.sigma_learnable,
            sigma_min=rl_args.sigma_min,
            sigma_max=rl_args.sigma_max,
            reward_type=rl_args.reward_type,
            reward_terms=rl_args.reward_terms,
            reward_combine=rl_args.reward_combine,
            reward_ndcg_k=rl_args.reward_ndcg_k,
            reward_rbo_p=rl_args.reward_rbo_p,
            ndcg_in_batch_include_negatives=rl_args.ndcg_in_batch_include_negatives,
            contrastive_use_in_batch_negatives=rl_args.contrastive_use_in_batch_negatives,
            contrastive_temperature=rl_args.contrastive_temperature,
            advantage_norm=rl_args.advantage_norm,
            sampling_law=rl_args.sampling_law,
            rollout=rl_args.rollout,
            frozen_doc_rescale=rl_args.frozen_doc_rescale,
            advantage_baseline=rl_args.advantage_baseline,
            advantage_baseline_momentum=rl_args.advantage_baseline_momentum,
            in_batch_use_sampled_documents=False,
            kl_coef=rl_args.kl_coef,
        )

    def forward(
        self,
        query: dict[str, Tensor],
        candidate_ordinals: Tensor,
        relevance_labels: Tensor,
        rank_labels: Tensor | None = None,
        candidate_mask: Tensor | None = None,
        in_batch_positive_mask: Tensor | None = None,
        in_batch_candidate_mask: Tensor | None = None,
        **_: Any,
    ):
        from grpo import GRPOModelOutput

        mask = candidate_ordinals >= 0 if candidate_mask is None else candidate_mask.bool()
        policy_queries = self.encode_query(query)
        rollout_queries = policy_queries.detach()
        safe = candidate_ordinals.masked_fill(~mask, 0)
        documents = self.scorer.index.lookup_embeddings(safe).detach().float()
        documents = F.normalize(documents, dim=-1).masked_fill(~mask.unsqueeze(-1), 0)
        reference_queries = None
        if self.grpo.kl_coef > 0:
            if not hasattr(self.model, "disable_adapter"):
                raise RuntimeError("kl_coef > 0 requires a PEFT model exposing disable_adapter()")
            with torch.no_grad(), self.model.disable_adapter():
                reference_queries = self.encode_query(query)
        loss, reward_stats, advantage_stats, sigma, kl = self.grpo(
            rollout_query_embeddings=rollout_queries,
            rollout_positive_document_embeddings=documents[:, :1],
            rollout_negative_document_embeddings=documents[:, 1:],
            relevance_labels=relevance_labels,
            rank_labels=rank_labels,
            candidate_mask=mask,
            in_batch_positive_mask=in_batch_positive_mask,
            in_batch_candidate_mask=in_batch_candidate_mask,
            policy_query_embeddings=policy_queries,
            reference_query_embeddings=reference_queries,
        )
        term_metrics = {key: value for key, value in reward_stats.items() if "/" in key}
        aggregate = {key: value for key, value in reward_stats.items() if "/" not in key}
        return GRPOModelOutput(
            loss=loss,
            reward=reward_stats["reward_mean"],
            **aggregate,
            **advantage_stats,
            sigma=sigma,
            kl=kl,
            reward_terms=term_metrics or None,
        )
