"""Relevance schemes that reconcile RAG training labels with the RAG metric.

The mined candidate manifest carries three independent judgments per passage:

``training_positive_mask``
    External binary qrels (DPR human positives, mapped HotpotQA supporting
    facts). Mean 1.01 passages per query; 36% of queries have none inside the
    frozen top-10.
``evidence_group_ids``
    Multi-hop supporting-fact membership. A passage belongs to at least one
    evidence group when this list is non-empty.
``answer_positive_mask``
    The passage text contains a golden answer. Mean 56.8 passages per query,
    2.03 of them inside the frozen top-10.

``src/eval_rag.py`` scores ``answer_recall_at_k`` and ``answer_mrr_at_k``
straight off the third judgment, so supervising on the first alone makes the
top-10 answer-bearing passages hard negatives of the very metric being
reported. The schemes here expose that choice instead of hard-coding it.

Each scheme returns three tensors:

``labels``
    Graded gains for nDCG-style objectives, consumed as ``2**label - 1``.
``positive_mask``
    The InfoNCE numerator.
``denominator_mask``
    Candidates the contrastive denominator may treat as negatives. Passages
    dropped here are neither positive nor negative -- they are simply not
    scored, which is how ``answer_masked`` retires the false negatives.
``scoring_mask``
    Candidates a listwise objective ranks. Wider than ``denominator_mask``:
    under ``graded`` an answer-bearing passage is a poor InfoNCE negative but a
    perfectly good gain-1 document for nDCG, so it appears here and not there.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


RELEVANCE_SCHEMES = ("binary", "answer_masked", "graded")

#: Graded gains. The ordering matters more than the absolute values because
#: nDCG consumes them exponentially: a qrel positive is worth 7 gain, an
#: evidence passage 3, an answer-bearing passage 1.
QREL_GRADE = 3.0
EVIDENCE_GRADE = 2.0
ANSWER_GRADE = 1.0


@dataclass
class RelevanceView:
    labels: Tensor
    positive_mask: Tensor
    denominator_mask: Tensor
    scoring_mask: Tensor


def build_relevance(
    *,
    scheme: str,
    training_positive_mask: Tensor,
    answer_positive_mask: Tensor,
    evidence_positive_mask: Tensor | None = None,
    candidate_mask: Tensor | None = None,
) -> RelevanceView:
    """Resolve per-candidate judgments into one scheme's supervision tensors."""
    if scheme not in RELEVANCE_SCHEMES:
        raise ValueError(f"Unsupported relevance scheme {scheme!r}; expected {RELEVANCE_SCHEMES}")
    if training_positive_mask.shape != answer_positive_mask.shape:
        raise ValueError("training and answer masks must have equal shapes")
    if training_positive_mask.dim() != 2:
        raise ValueError("relevance masks must be [batch, candidates]")

    qrel = training_positive_mask.bool()
    answer = answer_positive_mask.bool()
    if evidence_positive_mask is None:
        evidence = torch.zeros_like(qrel)
    else:
        if evidence_positive_mask.shape != qrel.shape:
            raise ValueError("evidence mask must match the training mask shape")
        evidence = evidence_positive_mask.bool()
    valid = torch.ones_like(qrel) if candidate_mask is None else candidate_mask.bool()

    qrel = qrel & valid
    answer = answer & valid
    evidence = evidence & valid

    if scheme == "binary":
        return RelevanceView(
            labels=qrel.float(),
            positive_mask=qrel,
            denominator_mask=valid,
            scoring_mask=valid,
        )

    if scheme == "answer_masked":
        # Answer-bearing passages that the qrels do not confirm are unjudged,
        # not negative: drop them from the denominator rather than demoting them.
        unjudged = valid & ~(answer & ~qrel)
        return RelevanceView(
            labels=qrel.float(),
            positive_mask=qrel,
            denominator_mask=unjudged,
            scoring_mask=unjudged,
        )

    # graded
    labels = torch.zeros_like(qrel, dtype=torch.float32)
    labels = torch.where(answer, torch.full_like(labels, ANSWER_GRADE), labels)
    labels = torch.where(evidence, torch.full_like(labels, EVIDENCE_GRADE), labels)
    labels = torch.where(qrel, torch.full_like(labels, QREL_GRADE), labels)
    labels = labels.masked_fill(~valid, 0.0)
    # Evidence passages join the numerator; answer-only passages stay unjudged
    # for the contrastive denominator but keep their graded gain for nDCG.
    positives = qrel | evidence
    return RelevanceView(
        labels=labels,
        positive_mask=positives,
        denominator_mask=valid & ~(answer & ~positives),
        scoring_mask=valid,
    )


def ensure_positive_coverage(view: RelevanceView) -> Tensor:
    """Rows with no positive cannot form an InfoNCE numerator; report them.

    Returns the boolean per-row keep mask so callers can drop, rather than
    crash on, the 36% of queries whose qrels miss the frozen candidate list.
    """
    return (view.positive_mask & view.denominator_mask).any(dim=-1)
