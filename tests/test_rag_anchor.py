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
    RAGAnchorPrecomputeCallback,
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
    def __init__(self):
        self.anchor_memmap = None

    def __len__(self):
        return 7

    def __getitem__(self, index):
        record = {"question": f"q{index}", "query_id": f"id{index}", "source": "nq"}
        if self.anchor_memmap is not None:
            record["anchor_embedding"] = torch.from_numpy(
                np.array(self.anchor_memmap[index])
            )
        return record

    def attach_anchors(self, vector_path, dimension):
        self.anchor_memmap = np.memmap(
            vector_path, dtype=np.float16, mode="r", shape=(len(self), dimension)
        )


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


class _FakeArgs:
    device = torch.device("cpu")


def test_precompute_callback_full_flow(tmp_path=None):
    """Exercise the exact on_train_begin path that failed on GPU.

    Two production failures shape this test:
    - 2081553/2081554: encoding before the Trainer exists finds ZeRO-3
      partitions ('weight' must be 2-D), so the callback must encode through
      the model it is handed at on_train_begin.
    - 2088952/2088953: that model is the engine wrapping RAGRLModel, whose
      forward takes (query, reward_inputs) -- a raw token batch has to go
      through encode_query instead.

    The stub backbone stores its parameter 1-D (as ZeRO-3 leaves it), and the
    stub wrapper's forward rejects token batches, so either wrong path raises.
    """
    import embedding_protocol as protocol_module
    from embedding_protocol import pool_embeddings

    class PartitionedBackbone(nn.Module):
        """Parameters stored 1-D, as ZeRO-3 leaves them before the Trainer."""

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(64 * DIM))
            self.config = types.SimpleNamespace(hidden_size=DIM)

        def forward(self, input_ids=None, attention_mask=None, **_):
            return types.SimpleNamespace(
                last_hidden_state=self.weight.view(64, DIM)[input_ids]
            )

    class RLWrapper(nn.Module):
        """The Trainer's model for an RL run, minus the engine shell."""

        def __init__(self):
            super().__init__()
            self.model = PartitionedBackbone()

        def forward(self, query, **reward_inputs):
            raise TypeError("forward() missing 1 required positional argument: 'query'")

        def encode_query(self, inputs):
            return pool_embeddings(
                self.model(**inputs).last_hidden_state,
                inputs["attention_mask"],
                pooling_method="last",
                normalize=True,
            )

    original = protocol_module.tokenize_embedding_texts
    protocol_module.tokenize_embedding_texts = (
        lambda texts, tok, append, max_length=None: _StubTokenizer()(texts)
    )
    try:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "m.jsonl"
            manifest.write_text("{}\n")
            dataset = _StubDataset()
            callback = __import__("rag.anchors", fromlist=["RAGAnchorPrecomputeCallback"]).RAGAnchorPrecomputeCallback(
                dataset=dataset,
                tokenizer=_StubTokenizer(),
                training_args=types.SimpleNamespace(per_device_train_batch_size=2),
                data_args=types.SimpleNamespace(
                    rag_candidate_manifest=str(manifest),
                    rag_query_max_length=128,
                ),
                model_args=types.SimpleNamespace(
                    model_name_or_path="stub", model_revision="rev1",
                    query_prompt_template="q", embedding_max_length=512,
                    append_token="pad",
                ),
                dimension=DIM,
                vector_path=root / "anchors.npy",
            )
            engine = RLWrapper()  # stands in for the DS engine wrapping RAGRLModel
            callback.on_train_begin(_FakeArgs(), state=None, control=None, model=engine)

            anchors = np.memmap(callback.vector_path, dtype=np.float16, mode="r", shape=(7, DIM))
            written = torch.from_numpy(np.array(anchors)).float()
            # Rows 0-6 were all encoded (single rank, world 1): none left zero.
            norms = written.norm(dim=-1)
            assert (norms > 0.9).all() and (norms < 1.1).all(), norms
            # The dataset is attached and serves anchors per row.
            record = dataset[3]
            assert record["anchor_embedding"].shape == (DIM,)
            assert torch.allclose(
                record["anchor_embedding"].float(), written[3], atol=1e-3
            )
            # The meta pins the manifest so a changed manifest rebuilds.
            meta = json.loads((root / "anchors.npy.meta.json").read_text())
            assert meta["row_count"] == 7 and meta["dimension"] == DIM
    finally:
        protocol_module.tokenize_embedding_texts = original


def test_callback_survives_trainer_init_events():
    """Trials 2082629/2082630: Trainer.__init__ fires on_init_end on every
    registered callback before on_train_begin ever runs, so the callback must
    carry the whole TrainerCallback event interface, not just on_train_begin."""
    from transformers import TrainerControl, TrainerState, TrainingArguments
    from transformers.trainer_callback import CallbackHandler

    with tempfile.TemporaryDirectory() as directory:
        callback = RAGAnchorPrecomputeCallback(
            dataset=_StubDataset(),
            tokenizer=_StubTokenizer(),
            training_args=types.SimpleNamespace(per_device_train_batch_size=2),
            data_args=types.SimpleNamespace(
                rag_candidate_manifest=str(Path(directory) / "m.jsonl"),
                rag_query_max_length=128,
            ),
            model_args=types.SimpleNamespace(
                model_name_or_path="stub",
                model_revision=None,
                query_prompt_template="q",
                embedding_max_length=512,
                append_token="pad",
            ),
            dimension=DIM,
            vector_path=Path(directory) / "anchors.npy",
        )
        handler = CallbackHandler([callback], None, None, None, None)
        handler.on_init_end(
            TrainingArguments(output_dir=directory), TrainerState(), TrainerControl()
        )


def test_real_dataset_and_collator_serve_anchors():
    """Trial 2089107 died at step 0: the real dataset's __getitem__ referenced
    numpy through an import that only existed inside attach_anchors. The stub
    dataset used by the other tests imports numpy itself, so only the real
    class can catch this. Drives dataset -> collator, the two pieces the
    precompute hands its anchors to."""
    import rag.data as data_module
    from rag.data import CandidateManifestDataset, RAGQueryCollator

    original = data_module.tokenize_embedding_texts
    data_module.tokenize_embedding_texts = (
        lambda texts, tok, append, max_length=None: _StubTokenizer()(texts)
    )
    try:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            records = [
                {
                    "query_id": f"q{i}",
                    "source": "nq",
                    "question": f"question {i}",
                    "golden_answers": [f"answer {i}"],
                    "candidate_passage_ids": [10 * i, 10 * i + 1],
                    "answer_positive_mask": [True, False],
                }
                for i in range(5)
            ]
            manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
            dataset = CandidateManifestDataset(str(manifest), split="full")
            vectors = np.memmap(root / "a.npy", dtype=np.float16, mode="w+", shape=(5, DIM))
            vectors[:] = np.arange(5 * DIM, dtype=np.float16).reshape(5, DIM)
            vectors.flush()
            dataset.attach_anchors(str(root / "a.npy"), DIM)

            row = dataset[3]
            assert row["anchor_embedding"].shape == (DIM,)
            batch = RAGQueryCollator(_StubTokenizer())([dataset[i] for i in (0, 3)])
            assert batch["anchor_embeddings"].shape == (2, DIM)
            assert torch.allclose(
                batch["anchor_embeddings"][1].float(),
                torch.arange(5 * DIM, dtype=torch.float32).reshape(5, DIM)[3],
            )
    finally:
        data_module.tokenize_embedding_texts = original


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
