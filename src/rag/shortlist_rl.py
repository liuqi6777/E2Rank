"""Static-shortlist RL for the RAG query encoder.

The shipped RAG RL path samples a query action and runs a live ANN search over
the 21M-passage corpus for every rollout: at batch 128 and group 32 that is
4,096 searches per optimizer step. Two things follow, and both are bad.

*Cost.* The search dominates the step.

*Estimator.* The reward then depends on every passage that could cross the
top-k boundary -- in principle the whole corpus -- so the conditional-projection
(CP) estimator is unavailable: projecting onto the returned top-k would delete
directions the reward genuinely responds to, making CP biased rather than merely
lower-variance (paper/METHOD_REDESIGN.md:133, gated at src/config.py:122). G1
measured this configuration gap directly: dynamic query-only retrieval 17.52 vs
static CP 20.72 BRIGHT nDCG@10.

This module ranks a fixed slate instead. Each query gets a small slate drawn
from its own mined depth-1000 candidates, plus K negatives borrowed from other
queries in the batch. The reward is a deterministic function of that enumerable
set's scores, which is exactly the condition CP needs. No ANN search runs.

Slate width is the whole point and is not a free parameter. CP's variance
reduction is proportional to ``rank(span)/D``; with D=1024 a depth-1000 slate
saturates the span, the projector becomes the identity, and CP degenerates into
the score-function estimator it was meant to improve on. G1's working recipe
spans roughly 36 columns. ``projection/query_span_rank_mean`` is logged so this
is observable rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.file_utils import ModelOutput

from conditional_projection import project_span
from embedding_protocol import pool_embeddings
from grpo import sample_vmf
from policy_math import ExplorationSchedule, group_advantages, mean_alignment
from score_precision import fp32_scores
from rag.negatives import build_cross_query_pool
from rag.relevance import build_relevance


GRADIENT_ESTIMATORS = ("score_function", "conditional_projection")


@dataclass
class ShortlistRLOutput(ModelOutput):
    """Field names match RAGModelOutput so RAGTrainer logs them unchanged."""

    loss: Tensor | None = None
    reward_mean: Tensor | None = None
    reward_std: Tensor | None = None
    degenerate_fraction: Tensor | None = None
    span_rank: Tensor | None = None
    exploration_metrics: dict[str, Tensor] | None = None


def build_slate(
    candidate_ordinals: Tensor,
    labels: Tensor,
    scoring_mask: Tensor,
    slate_size: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compact the depth-1000 candidate list into a narrow, gold-bearing slate.

    Judged candidates come first so the slate always has non-zero ideal DCG --
    a query whose qrel positive sits at depth 800 would otherwise contribute a
    constant-zero reward and no gradient. The remainder is filled from the head
    of the frozen retrieval order, which is where the hard negatives live.

    Returns ``(ordinals, labels, mask)``, each ``[batch, slate_size]``.
    """
    batch, depth = candidate_ordinals.shape
    width = min(slate_size, depth)
    valid = scoring_mask.bool()
    judged = valid & (labels > 0)
    # Rank key: judged candidates (2) before valid unjudged (1) before padding
    # (0); ties broken by the frozen retrieval order via a stable sort.
    priority = judged.int() * 2 + (valid & ~judged).int()
    order = priority.argsort(dim=-1, descending=True, stable=True)[:, :width]
    return (
        candidate_ordinals.gather(1, order),
        labels.gather(1, order),
        valid.gather(1, order),
    )


def graded_ndcg(
    scores: Tensor,
    labels: Tensor,
    mask: Tensor,
    k: int,
) -> Tensor:
    """nDCG@k with exponential gains over a fixed slate.

    ``scores`` is ``[batch, group, width]``; ``labels``/``mask`` are
    ``[batch, width]``. Borrowed negatives enter as gain-0 columns, so they can
    only push gold down -- they never raise the ideal ceiling.
    """
    width = scores.size(-1)
    cutoff = min(k, width)
    ranked = scores.masked_fill(~mask[:, None, :], float("-inf"))
    top = ranked.topk(cutoff, dim=-1).indices
    gains = (2.0 ** labels.float() - 1.0).masked_fill(~mask, 0.0)
    discounts = torch.arange(
        2, cutoff + 2, device=scores.device, dtype=torch.float32
    ).log2().reciprocal()
    dcg = (gains[:, None, :].expand_as(ranked).gather(-1, top) * discounts).sum(-1)
    ideal = gains.topk(cutoff, dim=-1).values
    idcg = (ideal * discounts).sum(-1)
    return torch.where(
        idcg[:, None] > 0, dcg / idcg.clamp_min(torch.finfo(torch.float32).eps)[:, None],
        torch.zeros_like(dcg),
    )


def score_function_loss(
    means: Tensor, actions: Tensor, advantages: Tensor, kappa: float
) -> Tensor:
    """REINFORCE surrogate: the estimator the shipped RAG RL path uses."""
    log_prob = kappa * (actions.detach().float() * means.unsqueeze(1)).sum(-1)
    return -(advantages.detach() * log_prob).mean()


def conditional_projection_loss(
    means: Tensor,
    actions: Tensor,
    advantages: Tensor,
    kappa: float,
    columns: Tensor,
) -> tuple[Tensor, Tensor]:
    """Query-only conditional projection.

    ``src.conditional_projection.conditional_projection_loss`` is the joint
    query+document form and hard-requires two sampled document draws, so it
    cannot be called here. This is its query-only specialization: identical
    sampling, rewards and leave-one-out advantages, with the detached
    coefficient of the live mean replaced by its projection onto the span of
    everything the reward actually saw.

    Because the score-function surrogate is linear in the actions, the whole
    group collapses into one vector before projecting, which avoids forming a
    projector per rollout.

    ``columns`` is ``[batch, span, dim]``: the rollout mean followed by every
    frozen document vector that entered the reward. Omitting any of them drops a
    reward-visible direction and biases the estimator.
    """
    group = actions.size(1)
    with torch.no_grad():
        weighted = torch.einsum("bi,bid->bd", advantages.detach().float(), actions.detach().float())
        projected, rank = project_span(weighted, columns.detach().float().transpose(-2, -1))
    # Only this contraction is differentiable; means carries the policy gradient.
    loss = -(kappa / group) * (means.float() * projected).sum(-1).mean()
    return loss, rank


class RAGShortlistRLModel(nn.Module):
    """Query-only policy over a fixed slate plus cross-query shortlist negatives."""

    def __init__(
        self,
        model: nn.Module,
        index: Any,
        *,
        relevance_scheme: str = "graded",
        reward_k: int = 10,
        slate_size: int = 20,
        shortlist_size: int = 15,
        group_size: int = 32,
        kappa: float = 755.0,
        pooling_method: str = "last",
        gradient_estimator: str = "conditional_projection",
        advantage_baseline: str = "leave_one_out",
        advantage_norm: str = "none",
        target_alignment: float | None = None,
        final_alignment: float | None = None,
        exploration_schedule: str = "fixed",
        cross_device_negatives: bool = True,
        anchor_coef: float = 0.0,
    ):
        super().__init__()
        if gradient_estimator not in GRADIENT_ESTIMATORS:
            raise ValueError(f"Unsupported gradient_estimator={gradient_estimator!r}")
        if group_size <= 1:
            raise ValueError("group_size must exceed one")
        if slate_size <= 1 or shortlist_size < 0:
            raise ValueError("slate_size must exceed one and shortlist_size be non-negative")
        if gradient_estimator == "conditional_projection" and advantage_baseline != "leave_one_out":
            # CP's unbiasedness is derived for the leave-one-out baseline.
            raise ValueError("conditional_projection requires the leave_one_out baseline")
        if gradient_estimator == "conditional_projection" and advantage_norm != "none":
            raise ValueError("conditional_projection requires advantage_norm=none")
        self.model = model
        self.config = model.config
        self.index = index
        self.relevance_scheme = relevance_scheme
        self.reward_k = int(reward_k)
        self.slate_size = int(slate_size)
        self.shortlist_size = int(shortlist_size)
        self.group_size = int(group_size)
        self.kappa = float(kappa)
        self.pooling_method = pooling_method
        self.gradient_estimator = gradient_estimator
        self.advantage_baseline = advantage_baseline
        self.advantage_norm = advantage_norm
        self.cross_device_negatives = cross_device_negatives
        self.anchor_coef = float(anchor_coef)
        self.exploration = ExplorationSchedule(
            kappa, target_alignment, final_alignment, exploration_schedule
        )
        self.dimension = None

    @property
    def grpo(self):
        """RAGTrainer steps ``.grpo.exploration`` once per optimizer step."""
        return self

    def encode_query(self, inputs: dict[str, Tensor]) -> Tensor:
        return pool_embeddings(
            self.model(**inputs).last_hidden_state,
            inputs["attention_mask"],
            pooling_method=self.pooling_method,
            normalize=True,
        )

    def _lookup(self, ordinals: Tensor, mask: Tensor) -> Tensor:
        safe = ordinals.masked_fill(~mask, 0)
        documents = self.index.lookup_embeddings(safe).detach()
        documents = F.normalize(documents.float(), dim=-1)
        return documents.masked_fill(~mask.unsqueeze(-1), 0.0)

    def exploration_metrics(self) -> dict[str, Tensor]:
        device = next(self.model.parameters()).device
        return {
            "exploration/kappa": torch.tensor(self.kappa, device=device),
            "exploration/mean_alignment": torch.tensor(
                mean_alignment(self.dimension, self.kappa) if self.dimension else 0.0,
                device=device,
            ),
        }

    def forward(
        self,
        query: dict[str, Tensor],
        candidate_passage_ids: Tensor,
        training_positive_mask: Tensor,
        answer_positive_mask: Tensor | None = None,
        evidence_positive_mask: Tensor | None = None,
        anchor_embeddings: Tensor | None = None,
        **_: Any,
    ) -> ShortlistRLOutput:
        if answer_positive_mask is None:
            answer_positive_mask = training_positive_mask
        valid = candidate_passage_ids >= 0
        view = build_relevance(
            scheme=self.relevance_scheme,
            training_positive_mask=training_positive_mask,
            answer_positive_mask=answer_positive_mask,
            evidence_positive_mask=evidence_positive_mask,
            candidate_mask=valid,
        )
        slate_ids, slate_labels, slate_mask = build_slate(
            candidate_passage_ids, view.labels, view.scoring_mask, self.slate_size
        )

        means = self.encode_query(query)
        self.dimension = means.size(-1)
        self.kappa = self.exploration.resolve(self.dimension)
        actions = sample_vmf(
            F.normalize(means.float(), dim=-1).detach(), self.kappa, self.group_size
        ).to(means.dtype)

        slate_docs = self._lookup(slate_ids, slate_mask)
        columns = [F.normalize(means.float(), dim=-1).detach().unsqueeze(1), slate_docs]
        scores = [torch.einsum("bid,bmd->bim", actions.float(), slate_docs)]
        labels = [slate_labels]
        masks = [slate_mask]

        if self.shortlist_size:
            # Negatives borrowed from other queries, with this query's own judged
            # passages filtered out so the pool cannot reintroduce false negatives.
            judged = view.positive_mask | answer_positive_mask.bool()
            pool_ids, pool_mask = build_cross_query_pool(
                candidate_ordinals=candidate_passage_ids,
                candidate_mask=valid,
                judged_mask=judged,
                pool_size=self.shortlist_size,
                include_negatives=True,
                cross_device=self.cross_device_negatives,
            )
            batch = means.size(0)
            short_ids, short_mask = self._sample_shortlist(means, pool_ids, pool_mask)
            short_docs = self._lookup(short_ids, short_mask)
            columns.append(short_docs)
            scores.append(torch.einsum("bid,bkd->bik", actions.float(), short_docs))
            labels.append(slate_labels.new_zeros((batch, short_ids.size(1))))
            masks.append(short_mask)

        with fp32_scores(means.device):
            rewards = graded_ndcg(
                torch.cat(scores, dim=-1),
                torch.cat(labels, dim=-1),
                torch.cat(masks, dim=-1),
                self.reward_k,
            )
        advantages, degenerate = group_advantages(
            rewards, self.advantage_baseline, self.advantage_norm
        )

        span_rank = torch.zeros((), device=means.device)
        if self.gradient_estimator == "conditional_projection":
            loss, rank = conditional_projection_loss(
                means, actions, advantages, self.kappa, torch.cat(columns, dim=1)
            )
            span_rank = rank.float().mean()
        else:
            loss = score_function_loss(means, actions, advantages, self.kappa)
        if anchor_embeddings is not None and self.anchor_coef > 0:
            similarity = torch.nn.functional.cosine_similarity(
                means.float(), anchor_embeddings.float().to(means.device), dim=-1
            )
            loss = loss + self.anchor_coef * (1.0 - similarity).mean()
        return ShortlistRLOutput(
            loss=loss,
            reward_mean=rewards.mean().detach(),
            reward_std=rewards.std(unbiased=False).detach(),
            degenerate_fraction=degenerate.float().mean(),
            span_rank=span_rank,
            exploration_metrics=self.exploration_metrics(),
        )

    def _sample_shortlist(
        self, means: Tensor, pool_ids: Tensor, pool_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Pick the K highest-scoring allowed pool entries per query.

        Selection uses the rollout *mean*, never a sampled action: a shortlist
        that moved with the draw would change what the reward conditions on and
        break the projection argument.
        """
        batch = means.size(0)
        width = min(self.shortlist_size, int(pool_mask.sum(dim=-1).max().item() or 1))
        # The pool is shared across the batch, so look it up once as [pool, dim]
        # rather than once per row.
        pool_docs = self.index.lookup_embeddings(pool_ids.clamp_min(0).unsqueeze(0)).detach()
        pool_docs = F.normalize(pool_docs.float(), dim=-1).squeeze(0)
        with torch.no_grad():
            scores = (
                F.normalize(means.float(), dim=-1).detach() @ pool_docs.t()
            ).masked_fill(~pool_mask, float("-inf"))
            chosen = scores.topk(width, dim=-1).indices
        ids = pool_ids.unsqueeze(0).expand(batch, pool_ids.numel()).gather(1, chosen)
        return ids, pool_mask.gather(1, chosen)

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.model.gradient_checkpointing_enable(*args, **kwargs)

    def enable_input_require_grads(self):
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
