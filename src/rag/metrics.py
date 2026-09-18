from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any


_ARTICLES = re.compile(r"\b(a|an|the)\b")
_WHITESPACE = re.compile(r"\s+")
_PUNCTUATION = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """SQuAD/FlashRAG-style normalization used by answer metrics."""
    # str.translate does the punctuation strip in C; the equivalent generator
    # comprehension dominated the cost of scanning hundreds of millions of
    # retrieved passages. The article pattern needs no IGNORECASE because it only
    # ever sees the lowercased text.
    text = str(text).lower().translate(_PUNCTUATION)
    text = _ARTICLES.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def token_f1(prediction: str, reference: str) -> float:
    normalized_prediction = normalize_answer(prediction)
    normalized_reference = normalize_answer(reference)
    if normalized_prediction in {"yes", "no", "noanswer"} and normalized_prediction != normalized_reference:
        return 0.0
    if normalized_reference in {"yes", "no", "noanswer"} and normalized_prediction != normalized_reference:
        return 0.0
    prediction_tokens = normalized_prediction.split()
    reference_tokens = normalized_reference.split()
    if not prediction_tokens or not reference_tokens:
        return 0.0
    common = Counter(prediction_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2.0 * precision * recall / (precision + recall)


def max_token_f1(prediction: str, references: Sequence[str]) -> float:
    return max((token_f1(prediction, reference) for reference in references), default=0.0)


def exact_match(prediction: str, references: Sequence[str]) -> float:
    normalized_prediction = normalize_answer(prediction)
    return float(any(normalized_prediction == normalize_answer(reference) for reference in references))


def normalize_answers(answers: Sequence[str]) -> list[str]:
    """Normalize an answer set once so it can be reused across many passages."""
    return [needle for needle in (normalize_answer(answer) for answer in answers) if needle]


def passage_contains_normalized_answer(contents: str, normalized_answers: Sequence[str]) -> bool:
    """``passage_contains_answer`` with the answer set already normalized."""
    if not normalized_answers:
        return False
    haystack = normalize_answer(contents)
    if not haystack:
        return False
    return any(needle in haystack for needle in normalized_answers)


def passage_contains_answer(contents: str, answers: Sequence[str]) -> bool:
    """FlashRAG Retrieval_Recall: normalized answer substring in normalized passage."""
    return passage_contains_normalized_answer(contents, normalize_answers(answers))


def extract_evidence_groups(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Return title/sentence evidence groups from FlashRAG multi-hop metadata."""
    metadata = record.get("metadata") or {}
    # MuSiQue stores one supporting paragraph per decomposition step instead of
    # Hotpot/2Wiki's title + sentence-id arrays.
    decomposition = metadata.get("question_decomposition") or []
    musique_groups = []
    for index, step in enumerate(decomposition):
        paragraph = step.get("support_paragraph") or {}
        if paragraph.get("is_supporting", True) and paragraph.get("title"):
            musique_groups.append({
                "group": index,
                "title": paragraph["title"],
                "sentence": paragraph.get("paragraph_text"),
            })
    if musique_groups:
        return musique_groups

    supporting = metadata.get("supporting_facts") or {}
    titles = supporting.get("title") or []
    sentence_ids = supporting.get("sent_id") or []
    contexts = metadata.get("context") or {}
    context_titles = contexts.get("title") or []
    context_sentences = contexts.get("sentences") or contexts.get("content") or []
    by_title = {
        normalize_answer(title): sentences
        for title, sentences in zip(context_titles, context_sentences)
    }

    groups: list[dict[str, Any]] = []
    for index, title in enumerate(titles):
        sentence_id = sentence_ids[index] if index < len(sentence_ids) else None
        sentences = by_title.get(normalize_answer(title), [])
        sentence = None
        if isinstance(sentence_id, int) and 0 <= sentence_id < len(sentences):
            sentence = sentences[sentence_id]
        groups.append({"group": index, "title": title, "sentence": sentence})
    return groups


def reciprocal_rank(relevant: Iterable[bool], k: int | None = None) -> float:
    for rank, is_relevant in enumerate(relevant, start=1):
        if k is not None and rank > k:
            break
        if is_relevant:
            return 1.0 / rank
    return 0.0


def best_window_token_f1(reference: str, text: str) -> float:
    """Best token F1 against a same-width window in a longer passage.

    Every window holds exactly as many tokens as the reference, so precision and
    recall are both overlap/width and the score rises with the multiset overlap.
    A sliding counter tracks that overlap in constant time per offset, where the
    previous shape rebuilt two Counters and re-ran normalization for every one of
    them, which dominated multi-hop evidence mapping.

    The winning window is then scored by ``token_f1`` itself rather than from the
    overlap directly. Both routes agree to within rounding, but only this one is
    bit-identical, and callers compare the result against a threshold and against
    each other: ``map_hotpot_evidence`` drops a query whose best score falls below
    ``minimum_window_f1`` and breaks ties by exact equality, so a last-place-digit
    difference is enough to move which passages an evaluation reports.
    """
    reference_tokens = normalize_answer(reference).split()
    text_tokens = normalize_answer(text).split()
    if not reference_tokens or not text_tokens:
        return float(reference_tokens == text_tokens)
    width = len(reference_tokens)
    if len(text_tokens) <= width:
        return token_f1(" ".join(text_tokens), " ".join(reference_tokens))
    needed = Counter(reference_tokens)
    window: Counter = Counter()
    overlap = 0
    for token in text_tokens[:width]:
        if window[token] < needed[token]:
            overlap += 1
        window[token] += 1
    best = overlap
    best_offset = 0
    for offset in range(1, len(text_tokens) - width + 1):
        if best == width:
            break
        leaving = text_tokens[offset - 1]
        window[leaving] -= 1
        if window[leaving] < needed[leaving]:
            overlap -= 1
        entering = text_tokens[offset + width - 1]
        if window[entering] < needed[entering]:
            overlap += 1
        window[entering] += 1
        if overlap > best:
            best, best_offset = overlap, offset
    return token_f1(
        " ".join(text_tokens[best_offset : best_offset + width]),
        " ".join(reference_tokens),
    )
