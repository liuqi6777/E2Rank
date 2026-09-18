"""Handing back the search storage while corpus text stays readable."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from embedding_protocol import POOLING_COMPUTE_DTYPE, TOKENIZATION_VERSION
from fixed_corpus.index import FrozenCorpusIndex, sha256_file


def build_index(directory: Path, rows: int = 24, dimension: int = 8, shards: int = 3) -> Path:
    """Write the smallest artifact set FrozenCorpusIndex will accept."""
    directory.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(0)
    vectors = generator.standard_normal((rows, dimension)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)

    offsets, keys = [], {}
    with (directory / "corpus.jsonl").open("wb") as corpus:
        for ordinal in range(rows):
            offsets.append(corpus.tell())
            keys[f"key-{ordinal}"] = ordinal
            corpus.write(json.dumps(
                {"document_key": f"key-{ordinal}", "contents": f"passage {ordinal}"},
                ensure_ascii=False, separators=(",", ":"),
            ).encode("utf-8") + b"\n")
    np.save(directory / "corpus_offsets.npy", np.asarray(offsets, dtype=np.int64))
    (directory / "document_key_to_ordinal.json").write_text(json.dumps(keys) + "\n")

    per_shard = rows // shards
    shard_entries = []
    for shard_id in range(shards):
        start = shard_id * per_shard
        path = directory / f"vectors-{shard_id:05d}-of-{shards:05d}.npy"
        np.save(path, vectors[start : start + per_shard].astype(np.float16))
        shard_entries.append({
            "path": path.name, "start": start, "count": per_shard, "sha256": sha256_file(path),
        })

    manifest = {
        "format_version": 1,
        "artifact_type": "frozen_document_index",
        "corpus_path": "corpus.jsonl",
        "corpus_offsets_path": "corpus_offsets.npy",
        "document_key_to_ordinal_path": "document_key_to_ordinal.json",
        "corpus_sha256": sha256_file(directory / "corpus.jsonl"),
        "corpus_offsets_sha256": sha256_file(directory / "corpus_offsets.npy"),
        "document_key_to_ordinal_sha256": sha256_file(directory / "document_key_to_ordinal.json"),
        "tokenization_version": TOKENIZATION_VERSION,
        "pooling_compute_dtype": POOLING_COMPUTE_DTYPE,
        "dimension": dimension,
        "count": rows,
        "shards": shard_entries,
        "normalized": True,
        "dtype": "float16",
    }
    manifest_path = directory / "index_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def test_release_keeps_corpus_lookups_and_refuses_further_search(tmp_path):
    manifest_path = build_index(tmp_path / "index")
    index = FrozenCorpusIndex(str(manifest_path), backend="torch", device="cpu")
    queries = torch.nn.functional.normalize(torch.randn(4, index.dimension), dim=-1)

    scores, ordinals = index.search(queries, 5)
    before = index.lookup_records(ordinals[0])
    assert index.search_device_count == 0  # cpu holds nothing to give back
    assert scores.shape == (4, 5)

    index.release_search_backend()

    assert index.backend == "lookup"
    assert index._torch_vectors is None and index._faiss_indexes == []
    # The passages behind the ordinals found before the release still read back.
    assert index.lookup_records(ordinals[0]) == before
    assert index.lookup_text([0, 1]) == ["passage 0", "passage 1"]
    assert index.lookup_ordinals(["key-3"]) == [3]
    with pytest.raises(RuntimeError, match="lookup backend does not support corpus search"):
        index.search(queries, 5)
    index.close()


def test_release_is_idempotent(tmp_path):
    manifest_path = build_index(tmp_path / "index")
    index = FrozenCorpusIndex(str(manifest_path), backend="torch", device="cpu")
    index.release_search_backend()
    index.release_search_backend()
    assert index.lookup_text([0]) == ["passage 0"]
    index.close()


def test_artifact_hashes_survive_a_release(tmp_path):
    """A released index still proves which frozen artifacts it was reading."""
    manifest_path = build_index(tmp_path / "index")
    index = FrozenCorpusIndex(str(manifest_path), backend="torch", device="cpu")
    before = index.artifact_hashes()
    index.release_search_backend()
    assert index.artifact_hashes() == before
    assert before["corpus"] == hashlib.sha256(
        (tmp_path / "index" / "corpus.jsonl").read_bytes()
    ).hexdigest()
    index.close()
