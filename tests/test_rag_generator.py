"""Prompt rendering and cache behaviour of the frozen generator client."""

from pathlib import Path
import sys

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import rag.generator as generator_module
from rag.generator import FrozenGeneratorClient


CHAT_TEMPLATE = (
    "{% for message in messages %}<|{{ message['role'] }}|> {{ message['content'] }} "
    "{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


def client(tmp_path, max_input_length, endpoint="http://127.0.0.1:1", cache_path=None, **kwargs):
    """A client over a whitespace tokenizer, so a token is a word and budgets are exact."""
    backend = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1}, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[UNK]"
    )
    tokenizer.chat_template = CHAT_TEMPLATE
    model_dir = tmp_path / "tokenizer"
    tokenizer.save_pretrained(model_dir)
    return FrozenGeneratorClient(
        endpoint,
        str(model_dir),
        str(tmp_path / "cache.sqlite3") if cache_path is None else cache_path,
        max_input_length=max_input_length,
        **kwargs,
    )


def documents(count, words):
    return [
        {"contents": f"Title{index}\n" + " ".join(f"w{index}-{word}" for word in range(words))}
        for index in range(count)
    ]


def reference_render(generator, question, docs):
    """The one-prompt-at-a-time policy this module used to apply, kept verbatim."""
    prompt = generator._format(question, docs)
    tokens = generator.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if len(tokens) <= generator.max_input_length:
        return prompt
    if len(docs) > 1:
        return reference_render(generator, question, docs[:-1])
    encoded_question = generator.tokenizer(
        f"\nQuestion:{question}", add_special_tokens=False
    )["input_ids"]
    kept = tokens[: max(generator.max_input_length - len(encoded_question), 1)] + encoded_question
    return generator.tokenizer.decode(kept[: generator.max_input_length], skip_special_tokens=False)


def inputs_covering_every_branch():
    """Prompts that fit, that need one drop, that need several, and that must truncate."""
    return [
        ("who fits", documents(2, 3)),
        ("who needs one drop", documents(5, 6)),
        ("who needs several drops", documents(10, 8)),
        ("who must truncate", documents(1, 400)),
        ("who has no context", []),
    ]


def test_batched_rendering_matches_the_per_prompt_policy(tmp_path):
    generator = client(tmp_path, max_input_length=60)
    cases = inputs_covering_every_branch()

    rendered = generator._render_batch(cases)

    assert rendered == [reference_render(generator, question, docs) for question, docs in cases]
    # The reference is only worth comparing against if the cases exercise the
    # branches: something has to be dropped, and something has to be truncated.
    lengths = generator._token_lengths(rendered)
    assert max(lengths) <= generator.max_input_length
    assert any(rendered[index] != generator._format(*cases[index]) for index in range(len(cases)))
    generator.close()


def test_rendering_measures_once_when_every_prompt_fits(tmp_path):
    generator = client(tmp_path, max_input_length=4096)
    cases = inputs_covering_every_branch()
    rounds = []
    measure = generator._token_lengths
    generator._token_lengths = lambda prompts: rounds.append(len(prompts)) or measure(prompts)

    rendered = generator._render_batch(cases)

    assert rendered == [generator._format(question, docs) for question, docs in cases]
    assert rounds == [len(cases)]
    generator.close()


def test_rendering_remeasures_only_the_prompts_still_over_budget(tmp_path):
    generator = client(tmp_path, max_input_length=60)
    # One prompt fits immediately; the other has to shed documents for several rounds.
    cases = [("fits", documents(1, 3)), ("shrinks", documents(6, 9))]
    rounds = []
    measure = generator._token_lengths
    generator._token_lengths = lambda prompts: rounds.append(len(prompts)) or measure(prompts)

    generator._render_batch(cases)

    assert rounds[0] == 2 and len(rounds) > 1
    assert all(count == 1 for count in rounds[1:])
    generator.close()


def test_generate_batch_renders_only_cache_misses(tmp_path):
    generator = client(tmp_path, max_input_length=4096)
    docs = documents(2, 3)
    requests = [
        ("q1", "first question", [0, 1], docs),
        ("q2", "second question", [2, 3], docs),
        ("q1", "first question", [0, 1], docs),
    ]
    sent = []

    def complete(prompts):
        sent.append(list(prompts))
        return [f"answer {len(sent)}.{index}" for index in range(len(prompts))]

    generator._complete = complete
    first = generator.generate_batch(requests)

    # The repeated query is one prompt on the wire and the same answer in both slots.
    assert len(sent) == 1 and len(sent[0]) == 2
    assert first[0] == first[2] and first[0] != first[1]

    second = generator.generate_batch(requests)
    assert second == first
    assert len(sent) == 1
    # Only the second pass counts as hits: a repeat inside one batch is folded into
    # its first occurrence before the cache is consulted, so it is never looked up.
    assert generator.statistics()["generation_cache_hits"] == 3
    generator.close()


def test_oversized_batches_are_split_to_the_server_limit(tmp_path):
    generator = client(tmp_path, max_input_length=4096)
    generator.max_prompts_per_request = 2
    docs = documents(1, 2)
    requests = [(f"q{index}", f"question {index}", [index], docs) for index in range(5)]
    sizes = []

    generator._complete = lambda prompts: sizes.append(len(prompts)) or [
        f"a{index}" for index in range(len(prompts))
    ]
    outputs = generator.generate_batch(requests)

    assert sizes == [2, 2, 1]
    assert len(outputs) == 5
    generator.close()


def test_rows_written_after_startup_are_still_found(tmp_path):
    """Training shares one cache file across ranks, so the snapshot cannot be final.

    The in-memory copy taken at startup exists to keep lookups off a network
    filesystem, not to pin the cache: a key it does not hold has to be checked
    against the file before the endpoint is asked for it.
    """
    writer = client(tmp_path, max_input_length=4096)
    docs = documents(2, 3)
    requests = [("q1", "first question", [0, 1], docs)]
    writer._complete = lambda prompts: ["written by another rank"]
    writer.generate_batch(requests)
    writer.close()

    reader = client(tmp_path, max_input_length=4096)
    # A second client over the same file, started before the row it needs existed.
    reader._cache.clear()
    reader._complete = lambda prompts: pytest.fail("the endpoint was asked for a cached answer")

    assert reader.generate_batch(requests) == ["written by another rank"]
    assert reader.statistics()["generation_cache_hits"] == 1
    reader.close()


def test_a_cache_too_large_to_hold_still_answers_from_the_file(tmp_path):
    """Training appends a row per rollout, so the table can outgrow memory."""
    writer = client(tmp_path, max_input_length=4096)
    docs = documents(2, 3)
    requests = [(f"q{index}", f"question {index}", [index], docs) for index in range(4)]
    writer._complete = lambda prompts: [f"answer {index}" for index in range(len(prompts))]
    expected = writer.generate_batch(requests)
    writer.close()

    limit = generator_module.MAX_PRELOADED_ROWS
    generator_module.MAX_PRELOADED_ROWS = 1
    try:
        reader = client(tmp_path, max_input_length=4096)
    finally:
        generator_module.MAX_PRELOADED_ROWS = limit

    assert reader._cache == {}, "a table over the limit must not be held in memory"
    reader._complete = lambda prompts: pytest.fail("the endpoint was asked for a cached answer")
    assert reader.generate_batch(requests) == expected
    assert reader.statistics()["generation_cache_hits"] == 4
    reader.close()


def test_cache_lookups_do_not_run_one_query_per_request(tmp_path):
    """These caches live on network storage, where a round trip costs milliseconds."""
    generator = client(tmp_path, max_input_length=4096)
    docs = documents(1, 2)
    requests = [(f"q{index}", f"question {index}", [index], docs) for index in range(64)]
    generator._complete = lambda prompts: [f"a{index}" for index in range(len(prompts))]
    generator.generate_batch(requests)

    statements = []

    class RecordingConnection:
        def __init__(self, connection):
            self._connection = connection

        def execute(self, sql, *rest):
            statements.append(sql)
            return self._connection.execute(sql, *rest)

        def __getattr__(self, name):
            return getattr(self._connection, name)

    generator.connection = RecordingConnection(generator.connection)
    generator.generate_batch(requests)

    # Every answer is already in memory, so the file is not consulted at all.
    assert statements == []
    assert generator.statistics()["generation_cache_hits"] == 64
    generator.close()


def test_empty_endpoint_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="generator endpoint"):
        FrozenGeneratorClient("", "ignored", str(tmp_path / "cache.sqlite3"))


def test_multiple_endpoints_split_batches_and_keep_order(tmp_path):
    """The 8-GPU layout runs two vLLM instances; batches must be dealt to both
    concurrently and the answers reassembled in prompt order."""
    generator = client(
        tmp_path,
        max_input_length=4096,
        endpoint="http://10.0.0.1:8000,http://10.0.0.2:8001",
        max_prompts_per_request=2,
    )
    docs = documents(1, 3)
    requests = [
        (f"q{index}", f"question {index}", [index], docs) for index in range(6)
    ]
    seen = []

    def complete(prompts, endpoint=None):
        seen.append((endpoint, len(prompts)))
        return [f"{endpoint}#{position}" for position in range(len(prompts))]

    generator._complete = complete
    outputs = generator.generate_batch(requests)

    # Three server-sized batches, round-robin over the two endpoints.
    assert seen == [
        ("http://10.0.0.1:8000", 2),
        ("http://10.0.0.2:8001", 2),
        ("http://10.0.0.1:8000", 2),
    ]
    assert outputs == [
        "http://10.0.0.1:8000#0", "http://10.0.0.1:8000#1",
        "http://10.0.0.2:8001#0", "http://10.0.0.2:8001#1",
        "http://10.0.0.1:8000#0", "http://10.0.0.1:8000#1",
    ]
    generator.close()


def test_single_endpoint_keeps_the_sequential_path(tmp_path):
    """One endpoint must not change behaviour for the existing callers."""
    generator = client(tmp_path, max_input_length=4096, max_prompts_per_request=2)
    docs = documents(1, 3)
    requests = [(f"q{index}", f"question {index}", [index], docs) for index in range(4)]
    seen = []

    def complete(prompts, endpoint=None):
        seen.append(endpoint)
        return [f"a{index}" for index in range(len(prompts))]

    generator._complete = complete
    outputs = generator.generate_batch(requests)
    assert seen == [None, None]
    # The stub answers every batch with the same two labels; what matters is
    # that both batches went through the one-endpoint sequential path.
    assert outputs == ["a0", "a1", "a0", "a1"]
    generator.close()


def test_cache_free_mode_sends_every_request_and_writes_nothing(tmp_path):
    generator = client(tmp_path, max_input_length=4096, cache_path="")
    docs = documents(1, 3)
    requests = [("q1", "question", [0], docs), ("q1", "question", [0], docs)]

    def complete(prompts, endpoint=None):
        return [f"a{index}" for index in range(len(prompts))]

    generator._complete = complete
    assert generator.generate_batch(requests) == ["a0", "a1"]
    stats = generator.statistics()
    assert stats["generation_requests"] == 2
    assert stats["generation_cache_hits"] == 0
    assert not (tmp_path / "cache.sqlite3").exists()
    generator.close()
