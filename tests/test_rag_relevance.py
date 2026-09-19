"""CPU checks for RAG relevance schemes and the cross-query negative pool.

Run directly (this repo has no pytest installed):
    PYTHONPATH=src python tests/test_rag_relevance.py
"""

from __future__ import annotations

import sys

import torch

from rag.negatives import build_cross_query_pool
from rag.relevance import build_relevance


def _masks():
    #                     c0     c1     c2     c3     c4
    qrel = torch.tensor([[False, True, False, False, False]])
    answer = torch.tensor([[True, True, False, True, False]])
    evidence = torch.tensor([[False, False, True, False, False]])
    return qrel, answer, evidence


def test_binary_keeps_historical_behaviour():
    qrel, answer, evidence = _masks()
    view = build_relevance(
        scheme="binary",
        training_positive_mask=qrel,
        answer_positive_mask=answer,
        evidence_positive_mask=evidence,
    )
    assert view.positive_mask.tolist() == qrel.tolist()
    # Every candidate stays a negative -- this is the behaviour that demoted
    # answer-bearing passages and put G3 below its own initialization.
    assert view.denominator_mask.all()
    assert view.labels.tolist() == [[0.0, 1.0, 0.0, 0.0, 0.0]]


def test_answer_masked_retires_false_negatives():
    qrel, answer, evidence = _masks()
    view = build_relevance(
        scheme="answer_masked",
        training_positive_mask=qrel,
        answer_positive_mask=answer,
        evidence_positive_mask=evidence,
    )
    assert view.positive_mask.tolist() == qrel.tolist()
    # c0 and c3 contain the answer but are not qrel positives -> unjudged.
    # c1 is a qrel positive and stays. c2 and c4 stay as true negatives.
    assert view.denominator_mask.tolist() == [[False, True, True, False, True]]


def test_graded_orders_qrel_above_evidence_above_answer():
    qrel, answer, evidence = _masks()
    view = build_relevance(
        scheme="graded",
        training_positive_mask=qrel,
        answer_positive_mask=answer,
        evidence_positive_mask=evidence,
    )
    assert view.labels.tolist() == [[1.0, 3.0, 2.0, 1.0, 0.0]]
    # Evidence joins the numerator alongside the qrel positive.
    assert view.positive_mask.tolist() == [[False, True, True, False, False]]
    # Answer-only passages are unjudged contrastively ...
    assert view.denominator_mask.tolist() == [[False, True, True, False, True]]
    # ... but keep their gain-1 for the listwise / nDCG view.
    assert view.scoring_mask.all()


def test_candidate_mask_suppresses_padding():
    qrel, answer, evidence = _masks()
    valid = torch.tensor([[True, True, True, False, False]])
    view = build_relevance(
        scheme="graded",
        training_positive_mask=qrel,
        answer_positive_mask=answer,
        evidence_positive_mask=evidence,
        candidate_mask=valid,
    )
    assert view.labels.tolist() == [[1.0, 3.0, 2.0, 0.0, 0.0]]
    assert not view.denominator_mask[0, 3]
    assert not view.scoring_mask[0, 4]


def test_pool_excludes_own_row_and_own_judged_passages():
    # Query 0's candidates are ordinals 10..13; query 1's are 20..23, except
    # ordinal 12 which query 1 also happens to retrieve and query 0 judges.
    ordinals = torch.tensor([[10, 11, 12, 13], [20, 21, 12, 23]])
    valid = torch.ones_like(ordinals, dtype=torch.bool)
    judged = torch.tensor(
        [[False, False, True, False], [True, False, False, False]]
    )
    pool, mask = build_cross_query_pool(
        candidate_ordinals=ordinals,
        candidate_mask=valid,
        judged_mask=judged,
        pool_size=4,
        include_negatives=True,
        cross_device=False,
        generator=torch.Generator().manual_seed(0),
    )
    assert pool.numel() == 8
    index = {int(o): i for i, o in enumerate(pool.tolist())}
    # No query borrows from its own row.
    for own in (10, 11, 12, 13):
        assert not mask[0, index[own]], f"query 0 must not use its own candidate {own}"
    for own in (20, 21, 23):
        assert not mask[1, index[own]]
    # Query 0 judges ordinal 12, so query 1's copy of 12 is also suppressed --
    # this is the false-negative guard.
    assert not mask[0, index[12]]
    # Query 1 does get query 0's unjudged candidates as negatives.
    assert mask[1, index[10]] and mask[1, index[11]] and mask[1, index[13]]
    # Query 0 judges 12; query 1 judges 20, which query 0 may still use.
    assert mask[0, index[20]]


def test_representative_pool_contributes_one_per_query():
    ordinals = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]])
    valid = torch.ones_like(ordinals, dtype=torch.bool)
    judged = torch.zeros_like(ordinals, dtype=torch.bool)
    pool, mask = build_cross_query_pool(
        candidate_ordinals=ordinals,
        candidate_mask=valid,
        judged_mask=judged,
        pool_size=4,
        include_negatives=False,
        cross_device=False,
    )
    assert pool.tolist() == [10, 20]
    assert mask.tolist() == [[False, True], [True, False]]


def test_infonce_ignores_masked_candidates():
    from fixed_corpus.models import multi_positive_infonce_loss

    scores = torch.tensor([[0.1, 0.9, 0.2, 5.0]])
    positives = torch.tensor([[False, True, False, False]])
    wide = multi_positive_infonce_loss(scores, positives, 0.03, torch.ones_like(positives))
    # Dropping the dominant distractor at index 3 must reduce the loss, which is
    # exactly what answer_masked does to answer-bearing false negatives.
    narrow = multi_positive_infonce_loss(
        scores, positives, 0.03, torch.tensor([[True, True, True, False]])
    )
    assert narrow.item() < wide.item()


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001 - surface every failure
            failures += 1
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {test.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
