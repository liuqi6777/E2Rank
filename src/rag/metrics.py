from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any


_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")


def normalize_answer(text: str) -> str:
    """SQuAD/FlashRAG-style normalization used by answer metrics."""
    text = str(text).lower()
    text = "".join(character for character in text if character not in string.punctuation)
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


def passage_contains_answer(contents: str, answers: Sequence[str]) -> bool:
    """FlashRAG Retrieval_Recall: normalized answer substring in normalized passage."""
    haystack = normalize_answer(contents)
    if not haystack:
        return False
    for answer in answers:
        needle = normalize_answer(answer)
        if not needle:
            continue
        if needle in haystack:
            return True
    return False


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
    """Best token F1 against a same-width window in a longer passage."""
    reference_tokens = normalize_answer(reference).split()
    text_tokens = normalize_answer(text).split()
    if not reference_tokens or not text_tokens:
        return float(reference_tokens == text_tokens)
    width = len(reference_tokens)
    if len(text_tokens) <= width:
        return token_f1(" ".join(text_tokens), " ".join(reference_tokens))
    return max(
        token_f1(" ".join(text_tokens[offset : offset + width]), " ".join(reference_tokens))
        for offset in range(len(text_tokens) - width + 1)
    )
