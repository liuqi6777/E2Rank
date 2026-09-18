from __future__ import annotations

import json
import multiprocessing
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from rag.data import iter_jsonl
from rag.metrics import (
    best_window_token_f1,
    extract_evidence_groups,
    normalize_answer,
    normalize_answers,
    passage_contains_normalized_answer,
)


def corpus_title(record: dict[str, Any]) -> str:
    if isinstance(record.get("title"), str):
        return record["title"]
    first_line = str(record.get("contents", "")).splitlines()[0] if record.get("contents") else ""
    return first_line.strip().strip('"')


def collect_hotpot_title_targets(records: Iterable[dict[str, Any]]) -> set[str]:
    return {
        normalize_answer(group["title"])
        for record in records
        for group in extract_evidence_groups(record)
        if group.get("title")
    }


def build_title_catalog(
    corpus_path: str,
    target_titles: set[str],
    progress: Callable[[int], None] | None = None,
    progress_interval: int = 100_000,
) -> dict[str, list[tuple[int, str]]]:
    """Index the corpus passages whose title is one a caller asks about.

    The scan parses every corpus row, so a caller that wants to show it moving
    passes ``progress`` and is told how many rows were read since the last call.
    Reporting in blocks keeps a bar from being refreshed twenty-one million times.
    """
    catalog: dict[str, list[tuple[int, str]]] = defaultdict(list)
    since_report = 0
    for ordinal, record in enumerate(iter_jsonl(corpus_path)):
        title = normalize_answer(corpus_title(record))
        if title in target_titles:
            catalog[title].append((ordinal, record["contents"]))
        if progress is not None:
            since_report += 1
            if since_report == progress_interval:
                progress(since_report)
                since_report = 0
    if progress is not None and since_report:
        progress(since_report)
    return dict(catalog)


def default_catalog_workers() -> int:
    # The scan is parse-bound rather than read-bound, so this tracks cores, with
    # the same ceiling candidate mining uses against the shared filesystem.
    return max(1, min(32, len(os.sched_getaffinity(0))))


_CATALOG_TARGETS: set[str] | None = None


def _initialize_catalog_worker(target_titles: set[str]) -> None:
    global _CATALOG_TARGETS
    _CATALOG_TARGETS = target_titles


def scan_title_catalog_chunk(
    chunk: tuple[str, int, int]
) -> tuple[int, int, list[tuple[str, int, str]]]:
    """Index one line-aligned byte range, returning its matches and its ordinal span.

    Ordinals come from each row's own ``id`` rather than from counting lines, so a
    worker does not need to know how many rows precede its range. The span it
    reports back lets the caller prove the ranges tile the corpus without a gap,
    which is what makes those ids equal the positions the serial scan would count.
    """
    if _CATALOG_TARGETS is None:
        raise RuntimeError("title-catalog worker was not initialized")
    corpus_path, start, stop = chunk
    matches: list[tuple[str, int, str]] = []
    first_ordinal = expected = None
    # Binary mode with the offset tracked by hand, as the qrels scanner does: a
    # text handle's tell() rebuilds an opaque decoder cookie, and asking for it
    # once per line costs more than parsing the line.
    with open(corpus_path, "rb") as handle:
        handle.seek(start)
        position = start
        while position < stop:
            line = handle.readline()
            if not line:
                break
            position += len(line)
            if not line.strip():
                continue
            record = json.loads(line)
            ordinal = int(record["id"])
            if expected is None:
                first_ordinal = expected = ordinal
            elif ordinal != expected:
                raise ValueError(
                    f"Corpus ids must equal ordinals: id={ordinal}, expected {expected}"
                )
            expected = ordinal + 1
            title = normalize_answer(corpus_title(record))
            if title in _CATALOG_TARGETS:
                matches.append((title, ordinal, record["contents"]))
    if first_ordinal is None:
        return 0, 0, matches
    return first_ordinal, expected, matches


def build_title_catalog_parallel(
    corpus_path: str,
    target_titles: set[str],
    workers: int,
    progress: Callable[[int], None] | None = None,
) -> dict[str, list[tuple[int, str]]]:
    """``build_title_catalog`` spread over processes, producing the same catalog.

    Scanning twenty-one million rows to find a few tens of thousands of titles is
    the longest single step of an evaluation and it holds no GPU, so it is split
    into byte ranges the same way candidate mining splits the corpus. Workers
    return only their matches, and the ranges are merged in file order, which is
    the order the serial scan appends in.
    """
    if workers <= 1:
        return build_title_catalog(corpus_path, target_titles, progress=progress)

    from rag.build_qrels import _chunk_byte_ranges

    chunks = _chunk_byte_ranges(Path(corpus_path), workers)
    # Not fork: an evaluation has already built its index by this point and so holds
    # a CUDA context, which is the same reason candidate mining spawns. Forkserver
    # over spawn because a worker that starts by importing this module pulls in
    # torch behind it, and paying that once in the server rather than in every
    # worker is the difference between 126s of startup and 6s -- against a scan
    # that only takes 5s once it is running. The server itself is exec'd fresh, so
    # no CUDA context reaches the workers it forks.
    if "forkserver" in multiprocessing.get_all_start_methods():
        context = multiprocessing.get_context("forkserver")
        context.set_forkserver_preload([__name__])
    else:
        context = multiprocessing.get_context("spawn")
    catalog: dict[str, list[tuple[int, str]]] = defaultdict(list)
    previous_stop = 0
    with context.Pool(
        processes=min(workers, len(chunks)),
        initializer=_initialize_catalog_worker,
        initargs=(target_titles,),
    ) as pool:
        for first_ordinal, stop_ordinal, matches in pool.imap(scan_title_catalog_chunk, chunks):
            if matches or stop_ordinal:
                if first_ordinal != previous_stop:
                    raise ValueError(
                        f"Corpus chunks are not contiguous: expected ordinal "
                        f"{previous_stop}, found {first_ordinal}"
                    )
                previous_stop = stop_ordinal
            for title, ordinal, contents in matches:
                catalog[title].append((ordinal, contents))
            if progress is not None:
                progress(stop_ordinal - first_ordinal)
    return dict(catalog)


def map_hotpot_evidence(
    record: dict[str, Any],
    title_catalog: dict[str, list[tuple[int, str]]],
    minimum_window_f1: float = 0.8,
) -> list[list[int]] | None:
    mapped: list[list[int]] = []
    for group in extract_evidence_groups(record):
        candidates = title_catalog.get(normalize_answer(group["title"]), [])
        sentence = group.get("sentence")
        if not candidates or not sentence:
            return None
        scored = [(best_window_token_f1(sentence, contents), ordinal) for ordinal, contents in candidates]
        best_score = max((score for score, _ in scored), default=0.0)
        if best_score < minimum_window_f1:
            return None
        # The protocol selects one deterministic passage for each supporting fact:
        # highest score first, then the smallest corpus ordinal.
        mapped.append([min(ordinal for score, ordinal in scored if score == best_score)])
    return mapped or None


def force_evidence_into_candidates(candidates: list[int], evidence_groups: list[list[int]]) -> list[int]:
    required = [group[0] for group in evidence_groups if group]
    return force_passages_into_candidates(candidates, required)


def force_passages_into_candidates(
    candidates: list[int], required_passage_ids: Iterable[int]
) -> list[int]:
    """Keep every known positive in a fixed-depth candidate list."""
    depth = len(candidates)
    result = list(dict.fromkeys(candidates))
    required = list(dict.fromkeys(int(ordinal) for ordinal in required_passage_ids))
    if len(required) > depth:
        raise ValueError("More required passages than candidate depth")
    for ordinal in required:
        if ordinal not in result:
            result.append(ordinal)
    required_set = set(required)
    while len(result) > depth:
        for index in range(len(result) - 1, -1, -1):
            if result[index] not in required_set:
                result.pop(index)
                break
        else:
            raise ValueError("More required evidence passages than candidate depth")
    if len(result) != depth:
        raise ValueError("Could not preserve candidate depth while inserting evidence")
    return result


def answer_mask_for_contents(contents: Iterable[str], answers: Sequence[str]) -> list[bool]:
    """Mark which candidate passages contain a golden answer."""
    needles = normalize_answers(answers)
    return [passage_contains_normalized_answer(text, needles) for text in contents]


def build_qrel_candidate_record(
    record: dict[str, Any],
    candidate_ids: list[int],
    answer_mask: list[bool],
) -> dict[str, Any]:
    """Attach fixed binary qrels to an immutable retrieval candidate pool.

    ``answer_mask`` comes from the caller because deriving it needs the candidate
    text, which mining fetches in parallel workers.
    """
    if len(answer_mask) != len(candidate_ids):
        raise ValueError("answer_mask must have one entry per candidate passage")
    qrel_ids = record.get("qrel_passage_ids")
    relevance = record.get("qrel_relevance")
    if (
        not isinstance(qrel_ids, list)
        or not qrel_ids
        or not all(isinstance(ordinal, int) and ordinal >= 0 for ordinal in qrel_ids)
    ):
        raise ValueError("qrel_passage_ids must contain non-negative corpus ordinals")
    if relevance != [1] * len(qrel_ids):
        raise ValueError("RAG candidate mining requires binary qrel_relevance")
    qrel_set = set(qrel_ids)
    training_mask = [ordinal in qrel_set for ordinal in candidate_ids]
    if sum(training_mask) != len(qrel_set):
        raise ValueError("Candidate pool does not contain every fixed qrel passage")
    answers = record["golden_answers"]
    evidence_groups = record.get("evidence_passage_groups") or []
    return {
        "query_id": record["query_id"],
        "source": record["source"],
        "question": record["question"],
        "golden_answers": answers,
        "candidate_passage_ids": candidate_ids,
        "answer_positive_mask": answer_mask,
        "training_positive_mask": training_mask,
        "evidence_group_ids": evidence_group_memberships_for_results(
            candidate_ids, evidence_groups
        ),
        "evidence_passage_groups": evidence_groups,
        "label_origin": record.get("label_origin"),
    }


def evidence_group_ids_for_results(result_ids: list[int], evidence_groups: list[list[int]]) -> list[int]:
    ordinal_to_group = {
        ordinal: group_index
        for group_index, group in enumerate(evidence_groups)
        for ordinal in group
    }
    return [ordinal_to_group.get(ordinal, -1) for ordinal in result_ids]


def evidence_group_memberships_for_results(
    result_ids: list[int], evidence_groups: list[list[int]]
) -> list[list[int]]:
    """All evidence groups matched by each result (a passage may satisfy several)."""
    return [
        [group_index for group_index, group in enumerate(evidence_groups) if ordinal in group]
        for ordinal in result_ids
    ]


def build_candidate_record(
    source: str,
    record: dict[str, Any],
    candidate_ids: list[int],
    candidate_contents: list[str],
    evidence_groups: list[list[int]] | None = None,
) -> dict[str, Any] | None:
    answer_mask = answer_mask_for_contents(candidate_contents, record["golden_answers"])
    if source == "nq":
        if not any(answer_mask):
            return None
        training_mask = answer_mask
        evidence_groups = []
    else:
        if not evidence_groups:
            return None
        evidence_ordinals = {ordinal for group in evidence_groups for ordinal in group}
        training_mask = [ordinal in evidence_ordinals for ordinal in candidate_ids]
        if not any(training_mask):
            return None
    return {
        "query_id": record["id"],
        "source": source,
        "question": record["question"],
        "golden_answers": record["golden_answers"],
        "candidate_passage_ids": candidate_ids,
        "answer_positive_mask": answer_mask,
        "training_positive_mask": training_mask,
        "evidence_group_ids": evidence_group_memberships_for_results(
            candidate_ids, evidence_groups
        ),
        "evidence_passage_groups": evidence_groups,
    }
