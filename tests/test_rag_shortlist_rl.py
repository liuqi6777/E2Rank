"""CPU checks for the RAG static-shortlist RL path.

    PYTHONPATH=src python tests/test_rag_shortlist_rl.py
"""

from __future__ import annotations

import sys
import types

import torch
from torch import nn

from rag.shortlist_rl import (
    RAGShortlistRLModel,
    build_slate,
    conditional_projection_loss,
    graded_ndcg,
    score_function_loss,
)


CORPUS = 512
DIM = 16


class StubIndex:
    def __init__(self):
        generator = torch.Generator().manual_seed(3)
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


def test_build_slate_puts_judged_candidates_first():
    ordinals = torch.tensor([[10, 11, 12, 13, 14, 15]])
    labels = torch.tensor([[0.0, 0.0, 0.0, 3.0, 0.0, 1.0]])
    mask = torch.ones_like(labels, dtype=torch.bool)
    ids, slate_labels, slate_mask = build_slate(ordinals, labels, mask, 4)
    assert ids.shape == (1, 4)
    # The judged candidates (13 -> 3.0, 15 -> 1.0) must survive truncation even
    # though they sit at depth 3 and 5 of the frozen order.
    assert set(ids[0, :2].tolist()) == {13, 15}
    assert slate_labels[0, :2].tolist() == [3.0, 1.0]
    # Remaining slots come from the head of the retrieval order.
    assert ids[0, 2:].tolist() == [10, 11]
    assert slate_mask.all()


def test_build_slate_drops_padding():
    ordinals = torch.tensor([[10, 11, -1, -1]])
    labels = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([[True, True, False, False]])
    ids, _, slate_mask = build_slate(ordinals, labels, mask, 4)
    assert slate_mask[0].tolist() == [True, True, False, False]
    assert ids[0, :2].tolist() == [10, 11]


def test_graded_ndcg_is_bounded_and_rewards_correct_ordering():
    labels = torch.tensor([[3.0, 1.0, 0.0, 0.0]])
    mask = torch.ones_like(labels, dtype=torch.bool)
    perfect = torch.tensor([[[4.0, 3.0, 2.0, 1.0]]])
    worst = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    best_score = graded_ndcg(perfect, labels, mask, 4)
    worst_score = graded_ndcg(worst, labels, mask, 4)
    assert abs(best_score.item() - 1.0) < 1e-6
    assert 0.0 <= worst_score.item() < best_score.item()


def test_graded_ndcg_is_zero_when_no_candidate_is_relevant():
    labels = torch.zeros((1, 4))
    mask = torch.ones_like(labels, dtype=torch.bool)
    scores = torch.tensor([[[4.0, 3.0, 2.0, 1.0]]])
    assert graded_ndcg(scores, labels, mask, 4).item() == 0.0


def test_borrowed_negatives_can_only_lower_the_reward():
    labels = torch.tensor([[3.0, 0.0]])
    mask = torch.ones_like(labels, dtype=torch.bool)
    scores = torch.tensor([[[5.0, 1.0]]])
    alone = graded_ndcg(scores, labels, mask, 10)
    # A borrowed negative that outranks the gold document.
    wide_labels = torch.tensor([[3.0, 0.0, 0.0]])
    wide_mask = torch.ones_like(wide_labels, dtype=torch.bool)
    wide_scores = torch.tensor([[[5.0, 1.0, 9.0]]])
    wider = graded_ndcg(wide_scores, wide_labels, wide_mask, 10)
    assert wider.item() < alone.item()


def test_conditional_projection_matches_score_function_on_a_full_span():
    """CP is only a variance reduction: with a full-rank span it *is* SF.

    This is the load-bearing correctness check. It also demonstrates the failure
    mode the slate width guards against -- a span that covers the whole
    embedding space buys nothing.
    """
    torch.manual_seed(0)
    batch, group, dim = 2, 6, 5
    means = torch.randn((batch, dim), requires_grad=True)
    means_copy = means.detach().clone().requires_grad_(True)
    actions = torch.nn.functional.normalize(torch.randn((batch, group, dim)), dim=-1)
    advantages = torch.randn((batch, group))
    kappa = 3.0
    # dim independent columns per row => projector is the identity.
    columns = torch.eye(dim).unsqueeze(0).expand(batch, dim, dim).contiguous()

    cp_loss, rank = conditional_projection_loss(means, actions, advantages, kappa, columns)
    sf_loss = score_function_loss(means_copy, actions, advantages, kappa)
    assert rank.float().mean().item() == dim
    assert torch.allclose(cp_loss, sf_loss, atol=1e-5), f"{cp_loss} vs {sf_loss}"
    cp_loss.backward()
    sf_loss.backward()
    assert torch.allclose(means.grad, means_copy.grad, atol=1e-5)


def test_conditional_projection_reduces_the_span_when_columns_are_narrow():
    torch.manual_seed(1)
    batch, group, dim = 2, 6, 8
    means = torch.randn((batch, dim), requires_grad=True)
    actions = torch.nn.functional.normalize(torch.randn((batch, group, dim)), dim=-1)
    advantages = torch.randn((batch, group))
    columns = torch.randn((batch, 3, dim))
    loss, rank = conditional_projection_loss(means, actions, advantages, 3.0, columns)
    assert rank.float().mean().item() == 3, "narrow span should not reach full rank"
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(means.grad).all()


def _model(**kwargs):
    defaults = dict(
        relevance_scheme="graded",
        reward_k=5,
        slate_size=8,
        shortlist_size=4,
        group_size=4,
        kappa=50.0,
        pooling_method="mean",
        cross_device_negatives=False,
    )
    defaults.update(kwargs)
    return RAGShortlistRLModel(StubBackbone(), StubIndex(), **defaults)


def _batch(batch_size=3, depth=24):
    generator = torch.Generator().manual_seed(5)
    ordinals = torch.randint(0, CORPUS, (batch_size, depth), generator=generator)
    qrel = torch.zeros((batch_size, depth), dtype=torch.bool)
    qrel[:, 2] = True
    answer = torch.zeros((batch_size, depth), dtype=torch.bool)
    answer[:, :5] = True
    evidence = torch.zeros((batch_size, depth), dtype=torch.bool)
    evidence[:, 3] = True
    return dict(
        query={
            "input_ids": torch.randint(0, 1000, (batch_size, 6), generator=generator),
            "attention_mask": torch.ones((batch_size, 6), dtype=torch.long),
        },
        candidate_passage_ids=ordinals,
        training_positive_mask=qrel,
        answer_positive_mask=answer,
        evidence_positive_mask=evidence,
    )


def test_both_estimators_run_end_to_end():
    for estimator in ("score_function", "conditional_projection"):
        model = _model(gradient_estimator=estimator)
        output = model(**_batch())
        assert torch.isfinite(output.loss), estimator
        # nDCG is bounded, so the logged mean must land in [0, 1].
        assert 0.0 <= output.reward_mean.item() <= 1.0, estimator
        assert torch.isfinite(output.reward_std), estimator
        assert "exploration/kappa" in output.exploration_metrics, estimator
        output.loss.backward()
        grad = model.model.embedding.weight.grad
        assert grad is not None and torch.isfinite(grad).all(), estimator


def test_conditional_projection_rejects_incompatible_baselines():
    for kwargs in ({"advantage_baseline": "group"}, {"advantage_norm": "per_component"}):
        try:
            _model(gradient_estimator="conditional_projection", **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"should reject {kwargs}")


def test_span_rank_is_reported_and_below_the_embedding_dimension():
    model = _model(gradient_estimator="conditional_projection", slate_size=4, shortlist_size=2)
    output = model(**_batch())
    # 1 mean + 4 slate + 2 shortlist = 7 columns in DIM=16.
    assert 0 < output.span_rank.item() <= 7
    assert output.span_rank.item() < DIM


def test_shortlist_can_be_disabled():
    model = _model(shortlist_size=0)
    output = model(**_batch())
    assert torch.isfinite(output.loss)


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
