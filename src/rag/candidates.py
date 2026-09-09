from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from rag.data import iter_jsonl
from rag.metrics import best_window_token_f1, extract_evidence_groups, normalize_answer, passage_contains_answer


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


def build_title_catalog(corpus_path: str, target_titles: set[str]) -> dict[str, list[tuple[int, str]]]:
    catalog: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for ordinal, record in enumerate(iter_jsonl(corpus_path)):
        title = normalize_answer(corpus_title(record))
        if title in target_titles:
            catalog[title].append((ordinal, record["contents"]))
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
    depth = len(candidates)
    result = list(dict.fromkeys(candidates))
    required = [group[0] for group in evidence_groups if group]
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
    answer_mask = [passage_contains_answer(contents, record["golden_answers"]) for contents in candidate_contents]
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
