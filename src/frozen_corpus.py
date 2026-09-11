from __future__ import annotations

import hashlib
import json
import os
from bisect import bisect_right
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_index_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    required = {"format_version", "dimension", "count", "shards", "corpus_path"}
    missing = sorted(required - manifest.keys())
    if missing:
        raise ValueError(f"Index manifest is missing fields: {', '.join(missing)}")
    if manifest["format_version"] != 1:
        raise ValueError(f"Unsupported index manifest version: {manifest['format_version']}")
    return manifest


def validate_frozen_protocol(
    manifest: dict[str, Any],
    *,
    model_name_or_path: str,
    resolved_model_revision: str | None,
    pooling_method: str,
    padding_side: str,
    append_token: str,
    document_prompt_template: str | None = None,
    document_max_length: int | None = None,
    query_prompt_template: str | None = None,
    embedding_max_length: int | None = None,
) -> None:
    expected_model = manifest.get("model_name_or_path")
    if expected_model and expected_model != model_name_or_path:
        raise ValueError(
            f"Model {model_name_or_path!r} differs from frozen document model {expected_model!r}"
        )
    expected_revision = manifest.get("resolved_model_revision")
    if expected_revision and resolved_model_revision and expected_revision != resolved_model_revision:
        raise ValueError("Model revision differs from the frozen document encoder revision")
    for key, actual in (
        ("pooling_method", pooling_method),
        ("padding_side", padding_side),
        ("append_token", append_token),
        ("document_prompt_template", document_prompt_template),
        ("document_max_length", document_max_length),
        ("query_prompt_template", query_prompt_template),
        ("embedding_max_length", embedding_max_length),
    ):
        expected = manifest.get(key)
        if actual is not None and expected is not None and expected != actual:
            raise ValueError(
                f"Frozen document protocol mismatch for {key}: {actual!r} != {expected!r}"
            )


class FrozenCorpusIndex:
    """Read-only exact inner-product index over immutable normalized shards.

    Storage shards are assigned round-robin to distributed ranks. Each rank searches
    every gathered query against its local shards, then the local top-k lists are
    gathered and merged. In a non-distributed process all shards are searched locally.
    """

    def __init__(
        self,
        manifest_path: str,
        backend: str = "faiss",
        device: str | torch.device = "cuda",
        verify_hashes: bool = True,
        search_batch_size: int = 1024,
    ):
        self.manifest_path = str(Path(manifest_path).resolve())
        self.manifest = load_index_manifest(self.manifest_path)
        self.root = Path(self.manifest_path).parent
        self.backend = backend.lower()
        if self.backend not in {"faiss", "torch", "lookup"}:
            raise ValueError("backend must be faiss, torch, or lookup")
        self.device = torch.device(device)
        self.search_batch_size = int(search_batch_size)
        if self.search_batch_size <= 0:
            raise ValueError("search_batch_size must be positive")
        self.dimension = int(self.manifest["dimension"])
        self.count = int(self.manifest["count"])
        self.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        self.shards = [self._resolve_shard(shard) for shard in self.manifest["shards"]]
        self.local_shards = self.shards[self.rank :: self.world_size]
        if not self.local_shards and self.backend != "lookup":
            raise ValueError(
                f"Distributed rank {self.rank} received no index shard; "
                f"world_size={self.world_size}, shards={len(self.shards)}"
            )
        if verify_hashes:
            self.verify()
        self._all_memmaps: list[np.ndarray] | None = None
        self._corpus_offsets: np.ndarray | None = None
        self._corpus_handle = None
        self._build_local_index()

    def _resolve_shard(self, shard: dict[str, Any]) -> dict[str, Any]:
        resolved = dict(shard)
        path = Path(resolved["path"])
        resolved["path"] = str(path if path.is_absolute() else self.root / path)
        return resolved

    def verify(self) -> None:
        total = 0
        expected_start = 0
        for shard in sorted(self.shards, key=lambda value: int(value["start"])):
            path = shard["path"]
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
            start = int(shard["start"])
            count = int(shard["count"])
            if start != expected_start:
                raise ValueError(f"Non-contiguous shard at {path}: start={start}, expected={expected_start}")
            if shard.get("sha256") and sha256_file(path) != shard["sha256"]:
                raise ValueError(f"Index shard hash mismatch: {path}")
            vectors = np.load(path, mmap_mode="r")
            if vectors.shape != (count, self.dimension):
                raise ValueError(
                    f"Shard shape mismatch for {path}: {vectors.shape}, "
                    f"expected {(count, self.dimension)}"
                )
            if self.manifest.get("dtype") == "float16" and vectors.dtype != np.float16:
                raise ValueError(f"Shard dtype mismatch for {path}: {vectors.dtype} != float16")
            if not np.isfinite(vectors).all():
                raise ValueError(f"Index shard contains non-finite vectors: {path}")
            if self.manifest.get("normalized"):
                norms = np.linalg.norm(np.asarray(vectors, dtype=np.float32), axis=-1)
                if not np.allclose(norms, 1.0, rtol=2e-3, atol=2e-3):
                    raise ValueError(f"Index shard contains non-normalized vectors: {path}")
            expected_start += count
            total += count
        if total != self.count:
            raise ValueError(f"Shard count {total} does not match manifest count {self.count}")
        for path_key, hash_key in (
            ("corpus_path", "corpus_sha256"),
            ("corpus_offsets_path", "corpus_offsets_sha256"),
            ("document_key_to_ordinal_path", "document_key_to_ordinal_sha256"),
        ):
            expected_hash = self.manifest.get(hash_key)
            if expected_hash:
                target = Path(self.manifest[path_key])
                if not target.is_absolute():
                    target = self.root / target
                if sha256_file(target) != expected_hash:
                    raise ValueError(f"Frozen artifact hash mismatch: {target}")
        offsets_path = self.manifest.get("corpus_offsets_path")
        if offsets_path:
            target = Path(offsets_path)
            if not target.is_absolute():
                target = self.root / target
            offsets = np.load(target, mmap_mode="r")
            if offsets.shape != (self.count,) or not np.issubdtype(offsets.dtype, np.integer):
                raise ValueError("Corpus offsets must contain one integer offset per corpus row")
        if self.manifest.get("document_key_to_ordinal_path"):
            self._ensure_key_mapping()

    def validate_protocol(
        self,
        *,
        model_name_or_path: str,
        resolved_model_revision: str | None,
        pooling_method: str,
        padding_side: str,
        append_token: str,
        document_prompt_template: str | None = None,
        document_max_length: int | None = None,
        query_prompt_template: str | None = None,
        embedding_max_length: int | None = None,
    ) -> None:
        """Reject embedding-space drift between the index and its consuming run."""
        validate_frozen_protocol(
            self.manifest,
            model_name_or_path=model_name_or_path,
            resolved_model_revision=resolved_model_revision,
            pooling_method=pooling_method,
            padding_side=padding_side,
            append_token=append_token,
            document_prompt_template=document_prompt_template,
            document_max_length=document_max_length,
            query_prompt_template=query_prompt_template,
            embedding_max_length=embedding_max_length,
        )

    def _local_vectors_and_ordinals(self) -> tuple[np.ndarray, np.ndarray]:
        arrays = [np.asarray(np.load(shard["path"], mmap_mode="r"), dtype=np.float32) for shard in self.local_shards]
        vectors = np.concatenate(arrays, axis=0)
        ordinals = np.concatenate(
            [np.arange(int(shard["start"]), int(shard["start"]) + int(shard["count"]), dtype=np.int64)
             for shard in self.local_shards]
        )
        return vectors, ordinals

    def _build_local_index(self) -> None:
        if self.backend == "lookup":
            self._local_ordinals = torch.empty(0, device=self.device, dtype=torch.long)
            self._torch_vectors = None
            self._faiss_index = None
            return
        ordinals = np.concatenate(
            [
                np.arange(
                    int(shard["start"]),
                    int(shard["start"]) + int(shard["count"]),
                    dtype=np.int64,
                )
                for shard in self.local_shards
            ]
        )
        self._local_ordinals = torch.from_numpy(ordinals).to(self.device)
        if self.backend == "torch":
            vectors = np.concatenate(
                [np.load(shard["path"], mmap_mode="r") for shard in self.local_shards], axis=0
            )
            self._torch_vectors = torch.from_numpy(vectors).to(self.device, dtype=torch.float16)
            self._faiss_index = None
            return

        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "FAISS backend requested but faiss is unavailable. Install a CUDA-compatible "
                "faiss-gpu build on the training host, or use rag_index_backend=torch for tests."
            ) from exc
        cpu_index = faiss.IndexFlatIP(self.dimension)
        for shard in self.local_shards:
            # Add one storage shard at a time. This avoids a second full float32
            # concatenation while keeping an exact FlatIP index.
            cpu_index.add(np.asarray(np.load(shard["path"], mmap_mode="r"), dtype=np.float32))
        if self.device.type == "cuda":
            resources = faiss.StandardGpuResources()
            options = faiss.GpuClonerOptions()
            options.useFloat16 = True
            self._faiss_resources = resources
            self._faiss_index = faiss.index_cpu_to_gpu(resources, self.device.index or 0, cpu_index, options)
        else:
            self._faiss_index = cpu_index
        self._torch_vectors = None

    def _search_local(self, queries: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.backend == "lookup":
            raise RuntimeError("The lookup backend does not support corpus search")
        local_k = min(k, self._local_ordinals.numel())
        score_batches = []
        position_batches = []
        for start in range(0, queries.size(0), self.search_batch_size):
            query_batch = queries[start : start + self.search_batch_size]
            if self.backend == "torch":
                batch_scores = torch.matmul(query_batch.float(), self._torch_vectors.float().T)
                batch_scores, batch_positions = batch_scores.topk(local_k, dim=-1)
            else:
                query_array = query_batch.detach().float().cpu().numpy()
                score_array, position_array = self._faiss_index.search(query_array, local_k)
                batch_scores = torch.from_numpy(score_array).to(self.device)
                batch_positions = torch.from_numpy(position_array).to(self.device)
            score_batches.append(batch_scores)
            position_batches.append(batch_positions)
        scores = torch.cat(score_batches, dim=0)
        positions = torch.cat(position_batches, dim=0)
        ordinals = self._local_ordinals[positions.long()]
        if local_k < k:
            padding = k - local_k
            scores = F.pad(scores, (0, padding), value=float("-inf"))
            ordinals = F.pad(ordinals, (0, padding), value=-1)
        return scores, ordinals

    def _gather_queries(self, queries: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
        if self.world_size == 1:
            return queries, [queries.size(0)]
        count = torch.tensor([queries.size(0)], device=self.device, dtype=torch.long)
        gathered_counts = [torch.zeros_like(count) for _ in range(self.world_size)]
        dist.all_gather(gathered_counts, count)
        counts = [int(value.item()) for value in gathered_counts]
        max_count = max(counts)
        padded = F.pad(queries, (0, 0, 0, max_count - queries.size(0)))
        gathered = [torch.empty_like(padded) for _ in range(self.world_size)]
        dist.all_gather(gathered, padded)
        return torch.cat([value[:count] for value, count in zip(gathered, counts)], dim=0), counts

    def search(self, query_vectors: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return exact cosine/IP scores and global corpus ordinals."""
        if k <= 0:
            raise ValueError("k must be positive")
        original_shape = query_vectors.shape[:-1]
        if query_vectors.size(-1) != self.dimension:
            raise ValueError(
                f"Query dim {query_vectors.size(-1)} does not match index dim {self.dimension}"
            )
        flat_queries = F.normalize(query_vectors.detach().to(self.device).float(), dim=-1).reshape(-1, self.dimension)
        all_queries, counts = self._gather_queries(flat_queries)
        local_scores, local_ids = self._search_local(all_queries, k)

        if self.world_size > 1:
            score_parts = [torch.empty_like(local_scores) for _ in range(self.world_size)]
            id_parts = [torch.empty_like(local_ids) for _ in range(self.world_size)]
            dist.all_gather(score_parts, local_scores)
            dist.all_gather(id_parts, local_ids)
            combined_scores = torch.cat(score_parts, dim=-1)
            combined_ids = torch.cat(id_parts, dim=-1)
        else:
            combined_scores, combined_ids = local_scores, local_ids

        # Stable secondary ordering by ordinal, followed by primary score ordering.
        id_order = torch.argsort(combined_ids, dim=-1, stable=True)
        combined_scores = combined_scores.gather(-1, id_order)
        combined_ids = combined_ids.gather(-1, id_order)
        score_order = torch.argsort(combined_scores, dim=-1, descending=True, stable=True)[..., :k]
        merged_scores = combined_scores.gather(-1, score_order)
        merged_ids = combined_ids.gather(-1, score_order)

        offset = sum(counts[: self.rank])
        own_count = counts[self.rank]
        merged_scores = merged_scores[offset : offset + own_count]
        merged_ids = merged_ids[offset : offset + own_count]
        return (
            merged_scores.reshape(*original_shape, k),
            merged_ids.reshape(*original_shape, k),
        )

    def _ensure_memmaps(self) -> list[np.ndarray]:
        if self._all_memmaps is None:
            self._all_memmaps = [np.load(shard["path"], mmap_mode="r") for shard in self.shards]
            self._shard_starts = [int(shard["start"]) for shard in self.shards]
        return self._all_memmaps

    def lookup_embeddings(self, ordinals: torch.Tensor) -> torch.Tensor:
        """Fetch immutable vectors by corpus ordinal from CPU shards."""
        memmaps = self._ensure_memmaps()
        flat = ordinals.detach().cpu().reshape(-1).tolist()
        rows = []
        for ordinal in flat:
            if ordinal < 0 or ordinal >= self.count:
                raise IndexError(f"Corpus ordinal out of range: {ordinal}")
            shard_index = bisect_right(self._shard_starts, ordinal) - 1
            local_index = ordinal - self._shard_starts[shard_index]
            rows.append(np.asarray(memmaps[shard_index][local_index], dtype=np.float32).copy())
        result = torch.from_numpy(np.stack(rows)).reshape(*ordinals.shape, self.dimension)
        return result.to(self.device)

    def _ensure_key_mapping(self) -> dict[str, int]:
        if not hasattr(self, "_key_to_ordinal"):
            mapping_path = self.manifest.get("document_key_to_ordinal_path")
            if not mapping_path:
                raise ValueError("Index manifest does not contain document_key_to_ordinal_path")
            mapping_path = Path(mapping_path)
            if not mapping_path.is_absolute():
                mapping_path = self.root / mapping_path
            with mapping_path.open(encoding="utf-8") as handle:
                mapping = json.load(handle)
            if not isinstance(mapping, dict) or len(mapping) != self.count:
                raise ValueError("Document-key mapping must contain exactly one entry per corpus row")
            values = list(mapping.values())
            if sorted(values) != list(range(self.count)):
                raise ValueError("Document-key mapping ordinals must be a complete 0-based permutation")
            self._key_to_ordinal = {str(key): int(value) for key, value in mapping.items()}
        return self._key_to_ordinal

    def lookup_ordinals(self, document_keys: list[str]) -> list[int]:
        mapping = self._ensure_key_mapping()
        missing = [key for key in document_keys if key not in mapping]
        if missing:
            preview = ", ".join(missing[:3])
            raise KeyError(f"Unknown document_key(s): {preview}")
        return [mapping[key] for key in document_keys]

    @property
    def document_key_to_ordinal(self) -> dict[str, int]:
        return dict(self._ensure_key_mapping())

    def artifact_hashes(self) -> dict[str, str]:
        """Return actual hashes for every immutable file referenced by the index."""
        artifacts = {"index_manifest": sha256_file(self.manifest_path)}
        for key in ("corpus_path", "corpus_offsets_path", "document_key_to_ordinal_path"):
            value = self.manifest.get(key)
            if value:
                path = Path(value)
                if not path.is_absolute():
                    path = self.root / path
                artifacts[key.removesuffix("_path")] = sha256_file(path)
        for shard in self.shards:
            artifacts[f"vector_shard:{Path(shard['path']).name}"] = sha256_file(shard["path"])
        return artifacts

    def _ensure_corpus_offsets(self) -> np.ndarray:
        if self._corpus_offsets is None:
            offsets_path = self.manifest.get("corpus_offsets_path")
            if not offsets_path:
                raise ValueError("Index manifest does not contain corpus_offsets_path")
            offsets_path = Path(offsets_path)
            if not offsets_path.is_absolute():
                offsets_path = self.root / offsets_path
            self._corpus_offsets = np.load(offsets_path, mmap_mode="r")
            corpus_path = Path(self.manifest["corpus_path"])
            if not corpus_path.is_absolute():
                corpus_path = self.root / corpus_path
            self._corpus_handle = open(corpus_path, "rb")
        return self._corpus_offsets

    def lookup_records(self, ordinals: torch.Tensor | list[int]) -> list[dict[str, Any]]:
        offsets = self._ensure_corpus_offsets()
        values = ordinals.detach().cpu().reshape(-1).tolist() if isinstance(ordinals, torch.Tensor) else ordinals
        records = []
        for ordinal in values:
            self._corpus_handle.seek(int(offsets[ordinal]))
            records.append(json.loads(self._corpus_handle.readline()))
        return records

    def lookup_text(self, ordinals: torch.Tensor | list[int]) -> list[str]:
        """Fetch immutable corpus contents by ordinal."""
        return [record["contents"] for record in self.lookup_records(ordinals)]

    def close(self) -> None:
        if self._corpus_handle is not None:
            self._corpus_handle.close()
            self._corpus_handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# Historical name retained for callers that used it outside ``rag``.
FrozenDistributedIndex = FrozenCorpusIndex


def write_frozen_training_audit(
    output_dir: str | os.PathLike[str],
    index: FrozenCorpusIndex,
    hashes_before: dict[str, str],
) -> Path:
    """Verify immutability and persist the query-only run's artifact evidence."""
    index.verify()
    hashes_after = index.artifact_hashes()
    if hashes_after != hashes_before:
        changed = sorted(set(hashes_before) | set(hashes_after))
        changed = [name for name in changed if hashes_before.get(name) != hashes_after.get(name)]
        raise RuntimeError("Frozen document artifacts changed during training: " + ", ".join(changed))
    payload = {
        "format_version": 1,
        "document_encoder_mode": "frozen_index",
        "index_manifest": index.manifest_path,
        "artifact_hashes_before": hashes_before,
        "artifact_hashes_after": hashes_after,
        "document_reencoding_count": 0,
        "index_rebuild_count": 0,
    }
    output = Path(output_dir) / "frozen_document_run_manifest.json"
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return output
