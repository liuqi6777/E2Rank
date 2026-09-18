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

from rag.generator import FrozenGeneratorClient


CHAT_TEMPLATE = (
    "{% for message in messages %}<|{{ message['role'] }}|> {{ message['content'] }} "
    "{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


def client(tmp_path, max_input_length):
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
        "http://127.0.0.1:1",
        str(model_dir),
        str(tmp_path / "cache.sqlite3"),
        max_input_length=max_input_length,
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


def test_empty_endpoint_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="generator endpoint"):
        FrozenGeneratorClient("", "ignored", str(tmp_path / "cache.sqlite3"))
