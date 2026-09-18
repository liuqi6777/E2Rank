"""The parallel evidence-title scan against the serial one it stands in for."""

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from rag.candidates import build_title_catalog, build_title_catalog_parallel
from rag.metrics import normalize_answer


TITLES = ["Evan Morris", "Horatio Hale", "USS Buffalo (1893)", "Tom Vilsack", "A The Article"]
ROWS = 5000


def write_corpus(path, rows=ROWS, skip=None):
    """A corpus in the frozen layout: contiguous ids, title as a quoted first line."""
    with open(path, "w", encoding="utf-8") as handle:
        for ordinal in range(rows):
            if ordinal == skip:
                continue
            title = TITLES[ordinal % len(TITLES)]
            contents = f'"{title}"\npassage {ordinal} about {title.lower()} and other matters'
            handle.write(json.dumps({"id": str(ordinal), "contents": contents}) + "\n")
    return str(path)


def targets():
    # Two titles that exist, one that does not, and one normalizing to nothing.
    return {normalize_answer(TITLES[0]), normalize_answer(TITLES[2]), "no such title", ""}


def test_parallel_scan_matches_the_serial_scan(tmp_path):
    corpus = write_corpus(tmp_path / "corpus.jsonl")
    wanted = targets()
    serial = build_title_catalog(corpus, wanted)

    # A worker count above and below the number of byte ranges, and the inline path.
    for workers in (1, 3, 8):
        assert build_title_catalog_parallel(corpus, wanted, workers=workers) == serial, workers


def test_every_matching_passage_is_found_in_corpus_order(tmp_path):
    corpus = write_corpus(tmp_path / "corpus.jsonl")
    catalog = build_title_catalog_parallel(corpus, targets(), workers=4)

    assert [ordinal for ordinal, _ in catalog[normalize_answer(TITLES[0])]] == list(
        range(0, ROWS, len(TITLES))
    )
    assert "no such title" not in catalog


def test_a_corpus_whose_ids_skip_a_row_is_rejected(tmp_path):
    """Ordinals come from each row's id, so the ranges have to be proven contiguous."""
    corpus = write_corpus(tmp_path / "corpus.jsonl", skip=2500)

    with pytest.raises(ValueError, match="ordinal"):
        build_title_catalog_parallel(corpus, targets(), workers=4)


def test_progress_is_reported_for_every_row(tmp_path):
    corpus = write_corpus(tmp_path / "corpus.jsonl")
    seen = []

    build_title_catalog_parallel(corpus, targets(), workers=4, progress=seen.append)

    assert sum(seen) == ROWS
