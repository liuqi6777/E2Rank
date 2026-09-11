"""Query/checkpoint + immutable per-subset corpus embeddings for MTEB retrieval."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from mteb.encoder_interface import PromptType


INDEX_FORMAT_VERSION = 1
_SAFE_SUBSET = re.compile(r"^[A-Za-z0-9_.-]+$")


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _texts_sha256(texts: Sequence[str]) -> str:
    """Hash a text sequence without separator ambiguity."""
    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def corpus_encoder_identity(model_name_or_path: str, model: Any) -> dict[str, object]:
    """Extract the document-side identity that makes an index reproducible."""
    embedder = getattr(model, "model", None)
    backbone = getattr(embedder, "base_model", None)
    config = getattr(backbone, "config", None)
    resolved_revision = getattr(config, "_commit_hash", None)
    configured_revision = getattr(
        getattr(model, "mteb_model_meta", None), "revision", None
    )
    revision = resolved_revision or configured_revision
    if not revision:
        raise ValueError(
            "Fixed-corpus evaluation requires a resolved E0 model revision. "
            "Pass --fixed_corpus_model_revision with an immutable commit SHA."
        )
    try:
        parameter_dtype = str(next(embedder.parameters()).dtype)
    except (AttributeError, StopIteration):
        parameter_dtype = None
    return {
        "model_name_or_path": model_name_or_path,
        "resolved_revision": revision,
        "pooling_method": getattr(embedder, "pooler_type", None),
        "truncate_dim": getattr(embedder, "truncate_dim", None),
        "normalize": getattr(embedder, "do_norm", None),
        "parameter_dtype": parameter_dtype,
        "autocast_dtype": str(getattr(model, "amp_dtype", None)),
        "padding_side": getattr(
            getattr(embedder, "tokenizer", None), "padding_side", None
        ),
        "append_token": getattr(embedder, "append_token", None),
        "document_prompt_template": getattr(model, "document_prompt_template", None),
        "document_max_length": getattr(model, "max_doc_length", None),
    }


class PerSubsetCorpusIndex:
    """Content-addressed immutable embedding shards for exactly one active subset.

    MTEB sorts each subset's corpus and submits it in chunks. Each chunk becomes one
    shard. A completed index records the exact shard sequence, so a changed corpus,
    changed order, or changed chunking cannot silently reuse or extend an old index.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        task_name: str,
        encoder_identity: dict[str, object],
    ) -> None:
        self.root = Path(root).resolve()
        self.task_name = task_name
        self.encoder_identity = copy.deepcopy(encoder_identity)
        self.subset: str | None = None
        self.subset_dir: Path | None = None
        self.manifest_path: Path | None = None
        self.manifest: dict[str, Any] | None = None
        self._observed_shards: list[str] = []

    def begin(self, subset: str) -> None:
        if self.subset is not None:
            raise RuntimeError(f"Corpus subset {self.subset!r} is still active")
        if not _SAFE_SUBSET.fullmatch(subset):
            raise ValueError(f"Unsafe corpus subset name: {subset!r}")

        subset_dir = self.root / subset
        manifest_path = subset_dir / "index_manifest.json"
        if manifest_path.is_file():
            with manifest_path.open(encoding="utf-8") as handle:
                manifest = json.load(handle)
            expected = {
                "format_version": INDEX_FORMAT_VERSION,
                "task_name": self.task_name,
                "subset": subset,
                "encoder": self.encoder_identity,
            }
            actual = {key: manifest.get(key) for key in expected}
            if actual != expected:
                raise ValueError(
                    f"Frozen corpus index protocol mismatch: {manifest_path}. "
                    "Use a new --fixed_corpus_index_dir for the changed E0 protocol."
                )
            if not isinstance(manifest.get("shards"), dict):
                raise ValueError(
                    f"Invalid frozen corpus shard manifest: {manifest_path}"
                )
        else:
            subset_dir.mkdir(parents=True, exist_ok=True)
            (subset_dir / "shards").mkdir(exist_ok=True)
            manifest = {
                "format_version": INDEX_FORMAT_VERSION,
                "task_name": self.task_name,
                "subset": subset,
                "encoder": self.encoder_identity,
                "complete": False,
                "count": 0,
                "dimension": None,
                "shard_sequence": [],
                "shards": {},
            }
            self._write_manifest(manifest_path, manifest)

        self.subset = subset
        self.subset_dir = subset_dir
        self.manifest_path = manifest_path
        self.manifest = manifest
        self._observed_shards = []

    def encode(
        self,
        texts: Sequence[str],
        encode_fn: Callable[[], np.ndarray | torch.Tensor],
    ) -> np.ndarray:
        if (
            self.manifest is None
            or self.subset_dir is None
            or self.manifest_path is None
        ):
            raise RuntimeError("Select a BRIGHT subset before encoding its corpus")
        if not texts:
            raise ValueError("A frozen corpus shard cannot be empty")

        text_hash = _texts_sha256(texts)
        position = len(self._observed_shards)
        expected_sequence = self.manifest.get("shard_sequence", [])
        if self.manifest.get("complete"):
            if (
                position >= len(expected_sequence)
                or expected_sequence[position] != text_hash
            ):
                raise ValueError(
                    f"BRIGHT subset {self.subset!r} no longer matches its completed frozen index; "
                    "use a new --fixed_corpus_index_dir"
                )

        metadata = self.manifest["shards"].get(text_hash)
        if metadata is None:
            if self.manifest.get("complete"):
                raise ValueError(
                    f"Missing shard {text_hash} in completed index {self.manifest_path}"
                )
            values = self._as_numpy(encode_fn())
            if values.shape[0] != len(texts):
                raise ValueError(
                    f"Corpus encoder returned {values.shape[0]} rows for {len(texts)} documents"
                )
            relative_path = Path("shards") / f"{text_hash}.npy"
            shard_path = self.subset_dir / relative_path
            temporary = shard_path.with_suffix(".npy.partial")
            with temporary.open("wb") as handle:
                np.save(handle, values, allow_pickle=False)
            os.replace(temporary, shard_path)
            metadata = {
                "path": str(relative_path),
                "text_sha256": text_hash,
                "embeddings_sha256": _sha256_file(shard_path),
                "count": int(values.shape[0]),
                "dimension": int(values.shape[1]),
                "dtype": str(values.dtype),
            }
            self.manifest["shards"][text_hash] = metadata
            self._write_manifest(self.manifest_path, self.manifest)
        else:
            shard_path = self.subset_dir / metadata["path"]
            if not shard_path.is_file():
                raise ValueError(f"Frozen corpus shard is missing: {shard_path}")
            if _sha256_file(shard_path) != metadata.get("embeddings_sha256"):
                raise ValueError(f"Frozen corpus shard hash mismatch: {shard_path}")
            values = np.load(shard_path, allow_pickle=False)
            if list(values.shape) != [metadata.get("count"), metadata.get("dimension")]:
                raise ValueError(f"Frozen corpus shard shape mismatch: {shard_path}")
            if str(values.dtype) != metadata.get("dtype"):
                raise ValueError(f"Frozen corpus shard dtype mismatch: {shard_path}")

        self._observed_shards.append(text_hash)
        return values

    def finish(self) -> None:
        if self.manifest is None or self.manifest_path is None or self.subset is None:
            raise RuntimeError("No BRIGHT corpus subset is active")
        if not self._observed_shards:
            # MTEB may reuse an existing result without invoking the retriever.
            if self.manifest.get("complete"):
                self.cancel()
                return
            raise RuntimeError(f"BRIGHT subset {self.subset!r} did not encode a corpus")

        expected_sequence = self.manifest.get("shard_sequence", [])
        if self.manifest.get("complete") and self._observed_shards != expected_sequence:
            raise ValueError(
                f"BRIGHT subset {self.subset!r} corpus shard sequence changed; "
                "use a new --fixed_corpus_index_dir"
            )

        shards = [self.manifest["shards"][key] for key in self._observed_shards]
        dimensions = {int(shard["dimension"]) for shard in shards}
        if len(dimensions) != 1:
            raise ValueError(
                f"Inconsistent embedding dimensions in BRIGHT subset {self.subset!r}"
            )
        dtypes = {shard["dtype"] for shard in shards}
        if len(dtypes) != 1:
            raise ValueError(
                f"Inconsistent embedding dtypes in BRIGHT subset {self.subset!r}"
            )
        self.manifest.update(
            complete=True,
            count=sum(int(shard["count"]) for shard in shards),
            dimension=dimensions.pop(),
            dtype=dtypes.pop(),
            shard_sequence=list(self._observed_shards),
        )
        self._write_manifest(self.manifest_path, self.manifest)
        self.cancel()

    def cancel(self) -> None:
        """Release active-subset state without marking an incomplete build complete."""
        self.subset = None
        self.subset_dir = None
        self.manifest_path = None
        self.manifest = None
        self._observed_shards = []

    @staticmethod
    def _as_numpy(values: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(values, torch.Tensor):
            values = values.detach().float().cpu().numpy()
        values = np.asarray(values)
        if values.ndim != 2 or not np.issubdtype(values.dtype, np.floating):
            raise ValueError(
                f"Corpus embeddings must be a floating matrix, got shape={values.shape} "
                f"dtype={values.dtype}"
            )
        return values

    @staticmethod
    def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
        temporary = path.with_suffix(".json.partial")
        temporary.write_bytes(_json_bytes(manifest) + b"\n")
        os.replace(temporary, path)


class FixedCorpusMTEBModel:
    """Route queries to a trained checkpoint and passages to per-subset E0 indexes."""

    def __init__(
        self,
        query_model: Any,
        corpus_model: Any,
        *,
        index_dir: str | os.PathLike[str],
        corpus_model_name_or_path: str,
        task_name: str,
    ) -> None:
        self.query_model = query_model
        self.corpus_model = corpus_model
        query_config = getattr(
            getattr(getattr(query_model, "model", None), "base_model", None),
            "config",
            None,
        )
        corpus_config = getattr(
            getattr(getattr(corpus_model, "model", None), "base_model", None),
            "config",
            None,
        )
        query_dimension = getattr(query_config, "hidden_size", None)
        corpus_dimension = getattr(corpus_config, "hidden_size", None)
        if (
            query_dimension is not None
            and corpus_dimension is not None
            and query_dimension != corpus_dimension
        ):
            raise ValueError(
                "Query checkpoint and fixed E0 corpus encoder dimensions differ: "
                f"{query_dimension} != {corpus_dimension}"
            )
        self.index = PerSubsetCorpusIndex(
            index_dir,
            task_name=task_name,
            encoder_identity=corpus_encoder_identity(
                corpus_model_name_or_path, corpus_model
            ),
        )
        self.mteb_model_meta = copy.copy(query_model.mteb_model_meta)
        identity_hash = hashlib.sha256(
            _json_bytes(self.index.encoder_identity)
        ).hexdigest()[:12]
        self.mteb_model_meta.name = (
            f"{self.mteb_model_meta.name}__fixed-corpus-{identity_hash}"
        )
        self.world_size = getattr(query_model, "world_size", 1)

    def begin_corpus_subset(self, subset: str) -> None:
        self.index.begin(subset)

    def finish_corpus_subset(self) -> None:
        self.index.finish()

    def cancel_corpus_subset(self) -> None:
        self.index.cancel()

    def encode(self, sentences, *, prompt_type=None, **kwargs):
        if prompt_type == PromptType.passage:
            return self.index.encode(
                sentences,
                lambda: self.corpus_model.encode(
                    sentences,
                    prompt_type=prompt_type,
                    **kwargs,
                ),
            )
        return self.query_model.encode(sentences, prompt_type=prompt_type, **kwargs)

    def start(self) -> None:
        if hasattr(self.query_model, "start"):
            self.query_model.start()
        if hasattr(self.corpus_model, "start"):
            self.corpus_model.start()

    def stop(self) -> None:
        errors = []
        for model in (self.query_model, self.corpus_model):
            if hasattr(model, "stop"):
                try:
                    model.stop()
                except Exception as exc:  # pragma: no cover - cleanup best effort
                    errors.append(exc)
        if errors:
            raise errors[0]
