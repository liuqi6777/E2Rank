"""CPU checks for the E0 anchor machinery.

    PYTHONPATH=src python tests/test_rag_anchor.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn

from rag.anchors import (
    anchor_cache_path,
    encode_anchor_rows,
    validate_anchor_cache,
    write_anchor_meta,
)
from rag.models import RAGRLModel, anchor_penalty


DIM = 16


def test_anchor_penalty_is_zero_for_identical_and_one_for_orthogonal():
    means = torch.nn.functional.normalize(torch.randn((4, DIM)), dim=-1)
    assert anchor_penalty(means, means.clone()).item() < 1e-7
    orthogonal = torch.zeros_like(means)
    orthogonal[:, 0] = 1.0
    means2 = torch.zeros_like(means)
    means2[:, 1] = 1.0
    assert abs(anchor_penalty(means2, orthogonal).item() - 1.0) < 1e-6


def test_anchor_penalty_rejects_shape_mismatch():
    means = torch.randn((4, DIM))
    try:
        anchor_penalty(means, torch.randn((3, DIM)))
    except ValueError:
        return
    raise AssertionError("shape mismatch must raise")


def test_anchor_meta_roundtrip_and_validation():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = root / "manifest.jsonl"
        manifest.write_text("{}\n")
        path = anchor_cache_path(manifest, root)
        # The vectors file must exist too: validation requires both sides.
        np.memmap(path, dtype=np.float16, mode="w+", shape=(10, DIM))
        write_anchor_meta(
            path,
            manifest_path=manifest,
            model_name_or_path="Qwen/Qwen3-Embedding-0.6B",
            model_revision="rev1",
            row_count=10,
            dimension=DIM,
        )
        assert validate_anchor_cache(
            path,
            manifest_path=manifest,
            model_name_or_path="Qwen/Qwen3-Embedding-0.6B",
            model_revision="rev1",
            row_count=10,
            dimension=DIM,
        )
        # Any drift in the dataset or the encoder invalidates the cache.
        assert not validate_anchor_cache(
            path,
            manifest_path=manifest,
            model_name_or_path="Qwen/Qwen3-Embedding-0.6B",
            model_revision="rev1",
            row_count=11,
            dimension=DIM,
        )
        assert not validate_anchor_cache(
            path,
            manifest_path=manifest,
            model_name_or_path="Qwen/Qwen3-Embedding-0.6B",
            model_revision="rev2",
            row_count=10,
            dimension=DIM,
        )
        # A changed manifest (different content hash) invalidates too.
        manifest.write_text("{}\n{}\n")
        assert not validate_anchor_cache(
            path,
            manifest_path=manifest,
            model_name_or_path="Qwen/Qwen3-Embedding-0.6B",
            model_revision="rev1",
            row_count=10,
            dimension=DIM,
        )


def test_cache_path_varies_with_the_variant():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        a = anchor_cache_path("m.jsonl", root, "train:0.05:1:None")
        b = anchor_cache_path("m.jsonl", root, "full:0.05:1:None")
        c = anchor_cache_path("m.jsonl", root)
        assert a != b and a != c and b != c


class _StubEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(64, DIM)
        self.config = types.SimpleNamespace(hidden_size=DIM)

    def forward(self, input_ids=None, attention_mask=None, **_):
        return types.SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class _StubTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, texts, **kwargs):
        return {
            "input_ids": torch.ones((len(texts), 4), dtype=torch.long),
            "attention_mask": torch.ones((len(texts), 4), dtype=torch.long),
        }


class _StubDataset:
    def __len__(self):
        return 7

    def __getitem__(self, index):
        return {"question": f"q{index}", "query_id": f"id{index}", "source": "nq"}


def test_encode_anchor_rows_writes_disjoint_rows_and_pooling_is_last_token():
    import embedding_protocol as protocol_module

    original = protocol_module.tokenize_embedding_texts
    protocol_module.tokenize_embedding_texts = (
        lambda texts, tok, append, max_length=None: _StubTokenizer()(texts)
    )
    torch.manual_seed(0)
    model = _StubEncoder()
    try:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "anchors.npy"
            # Rank 0 writes rows 0,1,2; rank 1 writes rows 3,4 (disjoint sets
            # from one shared memmap, as the distributed precompute does).
            encode_anchor_rows(
                model=model, tokenizer=_StubTokenizer(), dataset=_StubDataset(),
                row_indices=[0, 1, 2], vector_path=path, dimension=DIM,
                batch_size=2, device=torch.device("cpu"),
                query_prompt_template="q", append_token="pad",
            )
            encode_anchor_rows(
                model=model, tokenizer=_StubTokenizer(), dataset=_StubDataset(),
                row_indices=[3, 4], vector_path=path, dimension=DIM,
                batch_size=2, device=torch.device("cpu"),
                query_prompt_template="q", append_token="pad",
            )
            anchors = np.memmap(path, dtype=np.float16, mode="r", shape=(7, DIM))
            # Rows 5,6 were never written and must be exactly zero.
            assert float(np.abs(np.asarray(anchors[5:])).max()) == 0.0
            written = torch.from_numpy(np.array(anchors[:3])).float()
            norms = written.norm(dim=-1)
            # last-token pooling of an all-ones mask over unit embedding rows
            # yields unit vectors; fp16 rounding keeps them near 1.
            assert torch.allclose(norms, torch.ones(3), atol=1e-2)
    finally:
        protocol_module.tokenize_embedding_texts = original


def _rl_model(**kwargs):
    index = types.SimpleNamespace()
    return RAGRLModel(
        _StubEncoder(), index, reward_type="mrr", retrieval_k=10, group_size=4,
        kappa=50.0, pooling_method="mean", **kwargs
    )


def test_rl_anchor_penalty_scales_the_loss():
    """A zero anchor term must not change the loss; a real one must add to it."""
    torch.manual_seed(1)

    class _Env:
        def __call__(self, actions, **reward_inputs):
            return torch.rand(actions.shape[:-1])

    torch.manual_seed(3)
    plain = _rl_model(anchor_coef=0.0)
    plain.reward_provider = _Env()
    anchored = _rl_model(anchor_coef=0.5)
    anchored.load_state_dict(plain.state_dict())
    anchored.reward_provider = _Env()

    anchors = torch.nn.functional.normalize(torch.randn((2, DIM)), dim=-1)
    query = {
        "input_ids": torch.randint(0, 64, (2, 5)),
        "attention_mask": torch.ones((2, 5), dtype=torch.long),
    }
    # Same seed for the vMF draw so both runs sample identical actions.
    torch.manual_seed(11)
    base = plain(query=query).loss.item()
    torch.manual_seed(11)
    with_anchor = anchored(query=query, anchor_embeddings=anchors).loss.item()
    assert with_anchor > base, f"anchor penalty did not increase loss: {with_anchor} vs {base}"


def test_rl_identical_anchors_leave_the_loss_unchanged():
    """If the encoder has not drifted, the penalty must be exactly zero."""

    class _Env:
        def __call__(self, actions, **reward_inputs):
            return torch.rand(actions.shape[:-1])

    torch.manual_seed(5)
    model = _rl_model(anchor_coef=0.5)
    model.reward_provider = _Env()

    query = {
        "input_ids": torch.randint(0, 64, (2, 5)),
        "attention_mask": torch.ones((2, 5), dtype=torch.long),
    }
    torch.manual_seed(13)
    without = model(query=query).loss.item()
    torch.manual_seed(13)
    with_same = model(query=query).loss.item()  # no anchors passed at all
    assert abs(without - with_same) < 1e-6

    # Anchors equal to the encoder's own output: penalty zero by construction.
    torch.manual_seed(13)
    with_equal = model(
        query=query,
        anchor_embeddings=model.encode_query(
            {k: v.clone() for k, v in query.items()}
        ).detach(),
    ).loss.item()
    assert abs(without - with_equal) < 1e-4


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            import traceback

            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        else:
            print(f"ok   {test.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
