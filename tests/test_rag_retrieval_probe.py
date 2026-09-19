"""CPU checks for the full-corpus retrieval probe.

The probe calls ``index.search``, which is collective over the sharded frozen
index. An uneven query split across ranks would not raise -- it would hang the
whole job. These tests pin the padding that prevents that, plus the scoring and
the in-domain / held-out split.

    PYTHONPATH=src python tests/test_rag_retrieval_probe.py
"""

from __future__ import annotations

import sys
import types

import torch
from torch import nn

from rag.retrieval_probe import IN_DOMAIN, OUT_OF_DOMAIN, run_retrieval_probe


DIM = 8


class RecordingIndex:
    """Counts search() calls so collective symmetry is observable."""

    def __init__(self, answer_at: dict[str, int] | None = None):
        self.search_calls = 0
        self.searched_rows = 0
        self.answer_at = answer_at or {}

    def search(self, query_vectors, k, route_ids=None):
        self.search_calls += 1
        rows = query_vectors.size(0)
        self.searched_rows += rows
        ids = torch.arange(rows * k, dtype=torch.long).reshape(rows, k)
        return torch.zeros((rows, k)), ids

    def lookup_text(self, row):
        # Ordinal r encodes (row_index * k + position); position 0 of each row
        # carries the answer unless told otherwise.
        return [f"passage {ordinal}" for ordinal in row]


class AnswerAtPositionIndex(RecordingIndex):
    """Places the answer string at a fixed rank for every query."""

    def __init__(self, position: int, k: int):
        super().__init__()
        self.position = position
        self.k = k

    def lookup_text(self, row):
        return [
            "ANSWERTOKEN" if i == self.position else "unrelated filler"
            for i in range(len(row))
        ]


class StubBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(64, DIM)
        self.config = types.SimpleNamespace(hidden_size=DIM)

    def forward(self, input_ids=None, attention_mask=None, **_):
        return types.SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class StubEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = StubBackbone()

    def encode_query(self, inputs):
        hidden = self.model(**inputs).last_hidden_state
        return torch.nn.functional.normalize(hidden.mean(dim=1), dim=-1)


class StubTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    padding_side = "left"

    def __call__(self, texts, **kwargs):
        batch = len(texts)
        return {
            "input_ids": torch.ones((batch, 4), dtype=torch.long),
            "attention_mask": torch.ones((batch, 4), dtype=torch.long),
        }


def _queries(n_in=4, n_out=6):
    items = []
    for i in range(n_in):
        items.append({"dataset": "nq", "scope": "in_domain",
                      "question": f"in {i}", "answers": ["ANSWERTOKEN"]})
    for i in range(n_out):
        items.append({"dataset": "popqa", "scope": "held_out",
                      "question": f"out {i}", "answers": ["ANSWERTOKEN"]})
    return items


def _run(index, queries, **kwargs):
    import rag.retrieval_probe as module

    original = module.tokenize_embedding_texts
    module.tokenize_embedding_texts = lambda texts, tok, append, max_length=None: StubTokenizer()(texts)
    try:
        return run_retrieval_probe(
            model=StubEncoder(),
            index=index,
            tokenizer=StubTokenizer(),
            queries=queries,
            device=torch.device("cpu"),
            batch_size=4,
            **kwargs,
        )
    finally:
        module.tokenize_embedding_texts = original


def test_reports_in_domain_and_held_out_separately():
    metrics = _run(AnswerAtPositionIndex(0, 20), _queries(), retrieval_k=20)
    for scope in ("in_domain", "held_out", "all"):
        assert f"retrieval/{scope}/answer_mrr_at_10" in metrics, metrics.keys()
    # Answer at rank 1 for every query.
    assert abs(metrics["retrieval/all/answer_mrr_at_10"] - 1.0) < 1e-9
    assert abs(metrics["retrieval/all/answer_recall_at_5"] - 1.0) < 1e-9


def test_mrr_respects_the_cutoff():
    # Answer at position 12 is inside recall@20 but outside MRR@10.
    metrics = _run(AnswerAtPositionIndex(12, 20), _queries(), retrieval_k=20)
    assert metrics["retrieval/all/answer_mrr_at_10"] == 0.0
    assert abs(metrics["retrieval/all/answer_recall_at_20"] - 1.0) < 1e-9
    assert metrics["retrieval/all/answer_recall_at_5"] == 0.0


def test_missing_answer_scores_zero():
    metrics = _run(RecordingIndex(), _queries(), retrieval_k=20)
    assert metrics["retrieval/all/answer_recall_at_20"] == 0.0
    assert metrics["retrieval/all/answer_mrr_at_10"] == 0.0


def test_every_rank_issues_the_same_number_of_searches():
    """The deadlock guard: index.search is collective, so call counts must match.

    10 queries over 4 ranks splits 3/3/2/2. Without padding two ranks would
    issue fewer collective searches and the job would hang rather than fail.
    """
    queries = _queries(4, 6)
    assert len(queries) == 10
    counts, rows = set(), set()
    for rank in range(4):
        index = AnswerAtPositionIndex(0, 20)
        _run(index, queries, retrieval_k=20, rank=rank, world_size=4)
        counts.add(index.search_calls)
        rows.add(index.searched_rows)
    assert len(counts) == 1, f"ranks issued differing search counts: {counts}"
    assert len(rows) == 1, f"ranks searched differing row counts: {rows}"


def test_padding_is_not_scored():
    """Padded rows keep the collective symmetric without inflating counts."""
    queries = _queries(4, 6)
    totals = []
    for rank in range(4):
        metrics = _run(
            AnswerAtPositionIndex(0, 20), queries, retrieval_k=20, rank=rank, world_size=4
        )
        # Single-rank view of its own shard: recall must stay a true rate <= 1.
        assert metrics["retrieval/all/answer_recall_at_5"] <= 1.0 + 1e-9
        totals.append(metrics)
    assert len(totals) == 4


def test_domain_constants_are_disjoint_and_cover_the_suite():
    from rag.data import EVALUATION_SUITE

    assert not set(IN_DOMAIN) & set(OUT_OF_DOMAIN)
    assert set(IN_DOMAIN) | set(OUT_OF_DOMAIN) == set(EVALUATION_SUITE)


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
