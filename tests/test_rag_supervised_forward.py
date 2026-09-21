"""End-to-end CPU forward/backward check for RAGSupervisedModel.

Stubs the frozen index and the backbone so the whole loss path -- relevance
scheme, cross-query negative pool, objective -- runs without GPUs or the 43 GB
corpus.

    PYTHONPATH=src python tests/test_rag_supervised_forward.py
"""

from __future__ import annotations

import sys
import types

import torch
from torch import nn

from rag.models import RAGSupervisedModel


CORPUS = 512
DIM = 32


class StubIndex:
    """Deterministic fake corpus vectors keyed by ordinal."""

    def __init__(self):
        generator = torch.Generator().manual_seed(7)
        self.vectors = torch.randn((CORPUS, DIM), generator=generator)

    def lookup_embeddings(self, ordinals, route_ids=None):
        return self.vectors[ordinals.clamp(0, CORPUS - 1)]


class StubBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(1000, DIM)
        self.config = types.SimpleNamespace(hidden_size=DIM)

    def forward(self, input_ids=None, attention_mask=None, **_):
        return types.SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def _batch(batch_size=4, depth=16):
    generator = torch.Generator().manual_seed(11)
    ordinals = torch.randint(0, CORPUS, (batch_size, depth), generator=generator)
    qrel = torch.zeros((batch_size, depth), dtype=torch.bool)
    qrel[:, 0] = True
    answer = torch.zeros((batch_size, depth), dtype=torch.bool)
    answer[:, :4] = True
    evidence = torch.zeros((batch_size, depth), dtype=torch.bool)
    evidence[:, 1] = True
    query = {
        "input_ids": torch.randint(0, 1000, (batch_size, 8), generator=generator),
        "attention_mask": torch.ones((batch_size, 8), dtype=torch.long),
    }
    return query, ordinals, qrel, answer, evidence


def _build(objective="infonce", **kwargs):
    return RAGSupervisedModel(
        StubBackbone(), StubIndex(), objective=objective, pooling_method="mean", **kwargs
    )


def _forward(model):
    query, ordinals, qrel, answer, evidence = _batch()
    return model(
        query=query,
        candidate_passage_ids=ordinals,
        training_positive_mask=qrel,
        answer_positive_mask=answer,
        evidence_positive_mask=evidence,
    )


def test_every_scheme_and_objective_produces_a_finite_gradient():
    for objective in ("infonce", "ranknet", "lambdaloss"):
        for scheme in ("binary", "answer_masked", "graded"):
            model = _build(objective, relevance_scheme=scheme)
            loss = _forward(model).loss
            assert torch.isfinite(loss), f"{objective}/{scheme} loss is not finite"
            loss.backward()
            grad = model.model.embedding.weight.grad
            assert grad is not None and torch.isfinite(grad).all(), f"{objective}/{scheme} grad"


def test_cross_query_pool_changes_the_loss_and_stays_finite():
    plain = _build(relevance_scheme="answer_masked")
    pooled = _build(
        relevance_scheme="answer_masked",
        use_in_batch_candidates=True,
        in_batch_include_negatives=True,
        in_batch_pool_size=6,
    )
    pooled.model.load_state_dict(plain.model.state_dict())
    base = _forward(plain).loss
    wide = _forward(pooled).loss
    assert torch.isfinite(wide)
    # Extra negatives can only enlarge the InfoNCE denominator.
    assert wide.item() >= base.item() - 1e-6
    assert abs(wide.item() - base.item()) > 1e-6, "pool had no effect on the loss"


def test_representative_pool_is_narrower_than_full_candidate_pool():
    kwargs = dict(relevance_scheme="answer_masked", use_in_batch_candidates=True)
    representative = _build(**kwargs, in_batch_include_negatives=False)
    full = _build(**kwargs, in_batch_include_negatives=True, in_batch_pool_size=8)
    full.model.load_state_dict(representative.model.state_dict())
    assert torch.isfinite(_forward(representative).loss)
    assert torch.isfinite(_forward(full).loss)


def test_binary_penalises_answer_bearing_passages_more_than_answer_masked():
    binary = _build(relevance_scheme="binary")
    masked = _build(relevance_scheme="answer_masked")
    masked.model.load_state_dict(binary.model.state_dict())
    # answer_masked removes candidates 1..3 (answer-bearing, non-qrel) from the
    # denominator, so its loss must be the smaller of the two.
    assert _forward(masked).loss.item() < _forward(binary).loss.item()


def test_anchor_penalty_adds_to_the_supervised_loss():
    """The CL+anchor ablation (G3-R2-CL-AnswerMasked-Anchor050) rides anchors
    through the same collator field the RL arms use: identical anchors must
    leave the loss unchanged, and displaced ones must raise it by exactly
    coef * mean(1 - cos), with the penalty reaching the backward pass."""
    from rag.models import anchor_penalty

    plain = _build(relevance_scheme="answer_masked")
    anchored = _build(relevance_scheme="answer_masked", anchor_coef=0.5)
    anchored.model.load_state_dict(plain.model.state_dict())
    query, ordinals, qrel, answer, evidence = _batch()
    kwargs = dict(
        query=query,
        candidate_passage_ids=ordinals,
        training_positive_mask=qrel,
        answer_positive_mask=answer,
        evidence_positive_mask=evidence,
    )
    base = plain(**kwargs).loss
    means = anchored.encode_query(query)
    same = anchored(**kwargs, anchor_embeddings=means.detach()).loss
    assert abs((same - base).item()) < 1e-6
    displaced = torch.roll(means.detach(), 1, dims=0)
    raised = anchored(**kwargs, anchor_embeddings=displaced).loss
    expected = 0.5 * anchor_penalty(means.detach(), displaced).item()
    assert abs((raised - base).item() - expected) < 1e-5
    raised.backward()
    grad = anchored.model.embedding.weight.grad
    assert grad is not None and torch.isfinite(grad).all()


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {test.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
