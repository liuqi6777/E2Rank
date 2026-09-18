"""The sliding-window evidence score against the shape it replaced.

``map_hotpot_evidence`` drops a query whose best window falls below a threshold
and breaks ties by exact float equality, so this compares representations rather
than values: two scores that are equal as reals but differ in the last place are
a regression here.
"""

from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from rag.metrics import best_window_token_f1, normalize_answer, token_f1


def legacy_best_window_token_f1(reference: str, text: str) -> float:
    """Score every offset with ``token_f1`` and keep the best, as this once did."""
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


def assert_identical(reference: str, text: str) -> None:
    expected = legacy_best_window_token_f1(reference, text)
    actual = best_window_token_f1(reference, text)
    assert repr(actual) == repr(expected), (reference, text, repr(expected), repr(actual))


EDGE_REFERENCES = [
    "", "  ", "-", "a", "the", "a an the",       # normalizes to nothing, or to one token
    "yes", "no", "noanswer", "Yes!", "NoAnswer",  # token_f1's short-circuit words
    "cat", "cat cat", "Hello, world",
]
EDGE_TEXTS = [
    "", "yes", "no", "noanswer", "a an the", "the cat", "cat cat cat",
    "yes no yes", "noanswer noanswer", "cat dog cat dog cat",
    "Hello, the cat -- sat on the mat",
]


def test_edge_cases_match_the_per_offset_policy():
    for reference in EDGE_REFERENCES:
        for text in EDGE_TEXTS:
            assert_identical(reference, text)


def test_random_cases_match_the_per_offset_policy():
    random.seed(7)
    vocabulary = [
        "yes", "no", "noanswer", "the", "a", "an",
        "cat", "dog", "Paris", "1977", "x", "y", "Hello,", "(b)", "--",
    ]
    for _ in range(3000):
        reference = " ".join(random.choice(vocabulary) for _ in range(random.randint(0, 6)))
        text = " ".join(random.choice(vocabulary) for _ in range(random.randint(0, 40)))
        assert_identical(reference, text)


def test_repeated_tokens_match_the_per_offset_policy():
    """A three-word vocabulary, so windows tie constantly and overlap is a multiset."""
    random.seed(11)
    for _ in range(3000):
        reference = " ".join(random.choice("abc") for _ in range(random.randint(1, 5)))
        text = " ".join(random.choice("abc") for _ in range(random.randint(1, 30)))
        assert_identical(reference, text)


def test_score_is_the_overlap_fraction():
    # Two of the reference's three tokens appear in the best window, and a third
    # copy of a token the reference holds once cannot count twice.
    assert best_window_token_f1("cat dog bird", "zzz cat dog cat zzz") == token_f1(
        "cat dog cat", "cat dog bird"
    )
    assert best_window_token_f1("cat dog", "cat dog") == 1.0
    assert best_window_token_f1("cat dog", "zzz qqq www") == 0.0
