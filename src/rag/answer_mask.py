"""Answer-mask computation for candidate mining, usable inside worker processes.

Building the mask for one query means fetching its candidate passages from the
frozen corpus (latency-bound) and normalizing each passage (CPU-bound). Both are
done here so a worker only receives corpus ordinals and returns a boolean mask,
which keeps what crosses the process boundary tiny compared with the passage text.
"""

from __future__ import annotations

import os
from typing import Sequence

from fixed_corpus.corpus_reader import CorpusTextReader
from rag.metrics import normalize_answers, passage_contains_normalized_answer


_READER: CorpusTextReader | None = None


def initialize_worker(corpus_path: str, offsets_path: str, read_workers: int) -> None:
    global _READER
    _READER = CorpusTextReader(corpus_path, offsets_path, read_workers=read_workers)


def answer_mask_for_ordinals(
    reader: CorpusTextReader, ordinals: Sequence[int], answers: Sequence[str]
) -> list[bool]:
    needles = normalize_answers(answers)
    return [
        passage_contains_normalized_answer(contents, needles)
        for contents in reader.read_contents(ordinals)
    ]


def answer_mask_task(task: tuple[Sequence[int], Sequence[str]]) -> list[bool]:
    """Pool entry point: (candidate ordinals, golden answers) -> answer mask."""
    if _READER is None:
        raise RuntimeError("answer_mask worker was not initialized")
    ordinals, answers = task
    return answer_mask_for_ordinals(_READER, ordinals, answers)


def default_worker_count() -> int:
    # Measured on the wiki18 mining loop: throughput peaks near 32 processes
    # and falls off beyond it as corpus-read concurrency saturates the filesystem.
    return max(1, min(32, len(os.sched_getaffinity(0))))
