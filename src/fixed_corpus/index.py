"""Immutable corpus storage, lookup, search, verification and run auditing."""

from __future__ import annotations

import hashlib
import json
import os
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from embedding_protocol import TOKENIZATION_VERSION, POOLING_COMPUTE_DTYPE
from fixed_corpus.corpus_reader import CorpusTextReader, DEFAULT_READ_WORKERS


# FAISS reserves a per-device scratch pool on top of the index storage; the default
# is min(1.5 GiB, 18% of the device). Budgeted so the capacity pre-flight below
# fails with an explanation instead of a raw cudaMalloc error.
FAISS_TEMP_MEMORY_BYTES = 1536 << 20

# The torch backend materialises a float32 view of the corpus to score against.
# Doing that for the whole index needs twice the index size, so it runs in blocks
# of roughly this many rows (~2 GiB of float32 at dimension 1024).
TORCH_CORPUS_BLOCK_ROWS = 1 << 19

# Validating float16 vectors needs a float32 view of them. Only manifests without
# a per-shard hash reach that path, but block it so a shard of any size fits.
VECTOR_VALIDATION_BLOCK_ROWS = 1 << 17


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
    if manifest.get("tokenization_version") != TOKENIZATION_VERSION:
        raise ValueError(
            f"Frozen document tokenization_version must be {TOKENIZATION_VERSION}; "
            "rebuild the index with the current embedding protocol."
        )
    if manifest.get("pooling_compute_dtype") != POOLING_COMPUTE_DTYPE:
        raise ValueError("Frozen document pooling_compute_dtype must be float32; rebuild the index")
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
        faiss_gpu_devices: Sequence[int] | None = None,
        corpus_read_workers: int = DEFAULT_READ_WORKERS,
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
        self.faiss_gpu_devices = None if faiss_gpu_devices is None else [int(d) for d in faiss_gpu_devices]
        self.corpus_read_workers = int(corpus_read_workers)
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
        self._corpus_reader: CorpusTextReader | None = None
        self._search_executor: ThreadPoolExecutor | None = None
        self._build_local_index()

    def _resolve_shard(self, shard: dict[str, Any]) -> dict[str, Any]:
        resolved = dict(shard)
        path = Path(resolved["path"])
        resolved["path"] = str(path if path.is_absolute() else self.root / path)
        return resolved

    def resolved_manifest_path(self, key: str) -> Path:
        """Absolute path for a manifest entry that may be stored relative to the root."""
        value = self.manifest.get(key)
        if not value:
            raise ValueError(f"Index manifest does not contain {key}")
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def _verify_shard_contents(self, shard: dict[str, Any]) -> None:
        """Confirm one shard still holds the bytes the manifest was written for.

        A recorded sha256 already pins every byte, so matching it makes a second
        pass over the values redundant: finiteness and unit norm are established
        once, when the shard is written. Only a manifest without a shard hash has
        to pay for the full scan here.
        """
        path = shard["path"]
        expected_hash = shard.get("sha256")
        if expected_hash:
            if sha256_file(path) != expected_hash:
                raise ValueError(f"Index shard hash mismatch: {path}")
            return
        vectors = np.load(path, mmap_mode="r")
        for start in range(0, len(vectors), VECTOR_VALIDATION_BLOCK_ROWS):
            block = np.asarray(
                vectors[start : start + VECTOR_VALIDATION_BLOCK_ROWS], dtype=np.float32
            )
            if not np.isfinite(block).all():
                raise ValueError(f"Index shard contains non-finite vectors: {path}")
            if self.manifest.get("normalized"):
                norms = np.linalg.norm(block, axis=-1)
                if not np.allclose(norms, 1.0, rtol=2e-3, atol=2e-3):
                    raise ValueError(f"Index shard contains non-normalized vectors: {path}")

    def verify(self) -> None:
        """Validate the frozen artifacts, spreading the file reads across ranks.

        Layout checks stay global because they only read .npy headers. Hashing is
        split so the group covers every shard exactly once instead of each rank
        re-reading all of them; each rank draws the same slice it will load, so a
        corrupt shard is reported by the rank that would have used it. A raising
        rank exits non-zero and torchrun tears down the remaining workers.
        """
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
            vectors = np.load(path, mmap_mode="r")
            if vectors.shape != (count, self.dimension):
                raise ValueError(
                    f"Shard shape mismatch for {path}: {vectors.shape}, "
                    f"expected {(count, self.dimension)}"
                )
            if self.manifest.get("dtype") == "float16" and vectors.dtype != np.float16:
                raise ValueError(f"Shard dtype mismatch for {path}: {vectors.dtype} != float16")
            expected_start += count
            total += count
        if total != self.count:
            raise ValueError(f"Shard count {total} does not match manifest count {self.count}")
        for shard in self.shards[self.rank :: self.world_size]:
            self._verify_shard_contents(shard)
        hashed_artifacts = [
            (path_key, hash_key)
            for path_key, hash_key in (
                ("corpus_path", "corpus_sha256"),
                ("corpus_offsets_path", "corpus_offsets_sha256"),
                ("document_key_to_ordinal_path", "document_key_to_ordinal_sha256"),
            )
            if self.manifest.get(hash_key)
        ]
        for position, (path_key, hash_key) in enumerate(hashed_artifacts):
            # The corpus alone is larger than a shard, so deal these out too.
            if position % self.world_size != self.rank:
                continue
            target = Path(self.manifest[path_key])
            if not target.is_absolute():
                target = self.root / target
            if sha256_file(target) != self.manifest[hash_key]:
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

    def _shard_ordinals(self, shards: list[dict[str, Any]]) -> torch.Tensor:
        return torch.from_numpy(
            np.concatenate(
                [
                    np.arange(
                        int(shard["start"]),
                        int(shard["start"]) + int(shard["count"]),
                        dtype=np.int64,
                    )
                    for shard in shards
                ]
            )
        ).to(self.device)

    def _resolve_faiss_devices(self) -> list[int]:
        """CUDA devices this process may spread the flat index over.

        A corpus large enough to need several devices only arises in single-process
        jobs asking for a generic ``cuda`` placement, so that is the one case that
        fans out. An explicit ``cuda:N`` and every distributed rank keep one device:
        peers there are owned by other ranks or other indexes.
        """
        if self.device.type != "cuda":
            return []
        if self.faiss_gpu_devices is not None:
            return self.faiss_gpu_devices
        if self.device.index is not None or self.world_size > 1:
            return [self.device.index or 0]
        import faiss

        return list(range(faiss.get_num_gpus())) or [0]

    def _check_faiss_gpu_capacity(
        self, devices: list[int], groups: list[list[dict[str, Any]]]
    ) -> None:
        for device_id, shard_group in zip(devices, groups):
            rows = sum(int(shard["count"]) for shard in shard_group)
            storage = rows * self.dimension * 2
            # GpuIndexFlat grows its storage geometrically, so a resize transiently
            # holds the old and the new buffer, and every add() stages its input on
            # the device in float32 before converting it to the float16 storage.
            staging = max(int(shard["count"]) for shard in shard_group) * self.dimension * 4
            required = 2 * storage + staging + FAISS_TEMP_MEMORY_BYTES
            free = torch.cuda.mem_get_info(device_id)[0]
            if required <= free:
                continue
            raise RuntimeError(
                f"Frozen index does not fit on cuda:{device_id}: {rows} rows need about "
                f"{required / 1e9:.0f} GB (float16 storage {storage / 1e9:.0f} GB plus resize "
                f"and float32 staging headroom) but only {free / 1e9:.0f} GB is free. "
                f"Spread the index over more devices (currently {len(devices)}) via "
                "faiss_gpu_devices/CUDA_VISIBLE_DEVICES, or use backend='torch'."
            )

    def _build_local_index(self) -> None:
        self._faiss_indexes: list[Any] = []
        self._faiss_ordinals: list[torch.Tensor] = []
        self._faiss_resources: list[Any] = []
        if self.backend == "lookup":
            self._local_ordinals = torch.empty(0, device=self.device, dtype=torch.long)
            self._torch_vectors = None
            return
        self._local_ordinals = self._shard_ordinals(self.local_shards)
        if self.backend == "torch":
            vectors = np.concatenate(
                [np.load(shard["path"], mmap_mode="r") for shard in self.local_shards], axis=0
            )
            self._torch_vectors = torch.from_numpy(vectors).to(self.device, dtype=torch.float16)
            return
        self._torch_vectors = None

        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "FAISS backend requested but faiss is unavailable. Install a CUDA-compatible "
                "faiss-gpu build on the training host, or use rag_index_backend=torch for tests."
            ) from exc

        devices = self._resolve_faiss_devices()
        if not devices:
            cpu_index = faiss.IndexFlatIP(self.dimension)
            for shard in self.local_shards:
                # One storage shard per add keeps the float32 conversion proportional
                # to a shard rather than to the whole corpus.
                cpu_index.add(np.asarray(np.load(shard["path"], mmap_mode="r"), dtype=np.float32))
            self._faiss_indexes = [cpu_index]
            self._faiss_ordinals = [self._local_ordinals]
            return

        # Spread the shards over the devices instead of cloning one whole-corpus index
        # onto a single device: index_cpu_to_gpu stages the entire corpus in float32 on
        # the target device before converting it, which needs 2x the index size on one
        # card and cannot hold a 21M x 1024 corpus.
        groups = [group for group in (self.local_shards[i :: len(devices)] for i in range(len(devices))) if group]
        devices = devices[: len(groups)]
        self._check_faiss_gpu_capacity(devices, groups)
        for device_id, shard_group in zip(devices, groups):
            resources = faiss.StandardGpuResources()
            config = faiss.GpuIndexFlatConfig()
            config.device = device_id
            config.useFloat16 = True
            sub_index = faiss.GpuIndexFlatIP(resources, self.dimension, config)
            for shard in shard_group:
                sub_index.add(np.asarray(np.load(shard["path"], mmap_mode="r"), dtype=np.float32))
            # StandardGpuResources must outlive the index that borrows it.
            self._faiss_resources.append(resources)
            self._faiss_indexes.append(sub_index)
            self._faiss_ordinals.append(self._shard_ordinals(shard_group))

    @staticmethod
    def _merge_topk(
        scores: torch.Tensor, ordinals: torch.Tensor, k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reduce concatenated candidate lists to exactly k, breaking ties by ordinal."""
        width = scores.size(-1)
        if width < k:
            scores = F.pad(scores, (0, k - width), value=float("-inf"))
            ordinals = F.pad(ordinals, (0, k - width), value=-1)
        # Stable secondary ordering by ordinal, followed by primary score ordering.
        # Applied unconditionally so ties resolve by ordinal no matter how many
        # sub-indexes, corpus blocks or ranks contributed the candidates.
        ordinal_order = torch.argsort(ordinals, dim=-1, stable=True)
        scores = scores.gather(-1, ordinal_order)
        ordinals = ordinals.gather(-1, ordinal_order)
        score_order = torch.argsort(scores, dim=-1, descending=True, stable=True)[..., :k]
        return scores.gather(-1, score_order), ordinals.gather(-1, score_order)

    def _search_pool(self) -> ThreadPoolExecutor:
        # FAISS releases the GIL inside search(), so fanning the per-device searches
        # out over threads keeps every GPU busy at once. Running them in a plain loop
        # instead drives the devices one at a time and costs most of the speedup.
        if self._search_executor is None:
            self._search_executor = ThreadPoolExecutor(
                max_workers=len(self._faiss_indexes), thread_name_prefix="faiss-search"
            )
        return self._search_executor

    def _search_faiss(self, queries: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        score_batches = []
        ordinal_batches = []
        jobs = [
            (sub_index, min(k, sub_ordinals.numel()))
            for sub_index, sub_ordinals in zip(self._faiss_indexes, self._faiss_ordinals)
        ]
        for start in range(0, queries.size(0), self.search_batch_size):
            # Convert the query block once and reuse it for every device's sub-index.
            query_array = queries[start : start + self.search_batch_size].detach().float().cpu().numpy()
            if len(jobs) > 1:
                raw = list(self._search_pool().map(lambda job: job[0].search(query_array, job[1]), jobs))
            else:
                raw = [jobs[0][0].search(query_array, jobs[0][1])]
            part_scores = []
            part_ordinals = []
            for (score_array, position_array), sub_ordinals in zip(raw, self._faiss_ordinals):
                scores = torch.from_numpy(score_array).to(self.device)
                positions = torch.from_numpy(position_array).to(self.device).long()
                # FAISS pads with position -1 when a sub-index holds fewer than k rows.
                found = positions >= 0
                part_scores.append(torch.where(found, scores, torch.full_like(scores, float("-inf"))))
                part_ordinals.append(
                    torch.where(found, sub_ordinals[positions.clamp(min=0)], torch.full_like(positions, -1))
                )
            batch_scores, batch_ordinals = self._merge_topk(
                torch.cat(part_scores, dim=-1), torch.cat(part_ordinals, dim=-1), k
            )
            score_batches.append(batch_scores)
            ordinal_batches.append(batch_ordinals)
        return torch.cat(score_batches, dim=0), torch.cat(ordinal_batches, dim=0)

    def _search_torch(self, queries: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        score_batches = []
        ordinal_batches = []
        total_rows = self._torch_vectors.size(0)
        for start in range(0, queries.size(0), self.search_batch_size):
            query_batch = queries[start : start + self.search_batch_size].float()
            batch_scores: torch.Tensor | None = None
            batch_ordinals: torch.Tensor | None = None
            for block_start in range(0, total_rows, TORCH_CORPUS_BLOCK_ROWS):
                block = self._torch_vectors[block_start : block_start + TORCH_CORPUS_BLOCK_ROWS]
                # float() one block at a time: a float32 view of the whole corpus is
                # twice the index size and does not fit beside it on the device.
                block_scores = torch.matmul(query_batch, block.float().T)
                block_scores, block_positions = block_scores.topk(
                    min(k, block_scores.size(-1)), dim=-1
                )
                block_ordinals = self._local_ordinals[block_positions + block_start]
                if batch_scores is None:
                    batch_scores, batch_ordinals = block_scores, block_ordinals
                else:
                    batch_scores, batch_ordinals = self._merge_topk(
                        torch.cat([batch_scores, block_scores], dim=-1),
                        torch.cat([batch_ordinals, block_ordinals], dim=-1),
                        min(k, total_rows),
                    )
            score_batches.append(batch_scores)
            ordinal_batches.append(batch_ordinals)
        scores = torch.cat(score_batches, dim=0)
        ordinals = torch.cat(ordinal_batches, dim=0)
        return self._merge_topk(scores, ordinals, k)

    def _search_local(self, queries: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.backend == "lookup":
            raise RuntimeError("The lookup backend does not support corpus search")
        if self.backend == "torch":
            return self._search_torch(queries, k)
        return self._search_faiss(queries, k)

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

    def search(
        self,
        query_vectors: torch.Tensor,
        k: int,
        route_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return exact cosine/IP scores and global corpus ordinals."""
        if k <= 0:
            raise ValueError("k must be positive")
        if route_ids is not None:
            raise ValueError("A single frozen index does not accept route_ids")
        original_shape = query_vectors.shape[:-1]
        if query_vectors.size(-1) != self.dimension:
            raise ValueError(
                f"Query dim {query_vectors.size(-1)} does not match index dim {self.dimension}"
            )
        flat_queries = F.normalize(query_vectors.detach().to(self.device).float(), dim=-1).reshape(-1, self.dimension)
        all_queries, counts = self._gather_queries(flat_queries)
        if all_queries.size(0) == 0:
            return (
                torch.empty((*original_shape, k), device=self.device, dtype=torch.float32),
                torch.empty((*original_shape, k), device=self.device, dtype=torch.long),
            )
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

        merged_scores, merged_ids = self._merge_topk(combined_scores, combined_ids, k)

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

    def lookup_embeddings(
        self,
        ordinals: torch.Tensor,
        route_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fetch immutable vectors by corpus ordinal from CPU shards."""
        if route_ids is not None:
            raise ValueError("A single frozen index does not accept route_ids")
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

    def corpus_reader(self) -> CorpusTextReader:
        """Shared reader over the frozen corpus text backing this index."""
        if self._corpus_reader is None:
            self._corpus_reader = CorpusTextReader(
                self.resolved_manifest_path("corpus_path"),
                self.resolved_manifest_path("corpus_offsets_path"),
                read_workers=self.corpus_read_workers,
            )
            if self._corpus_reader.count != self.count:
                raise ValueError(
                    f"Corpus offsets describe {self._corpus_reader.count} rows but the index "
                    f"holds {self.count}"
                )
        return self._corpus_reader

    def lookup_records(self, ordinals: torch.Tensor | list[int]) -> list[dict[str, Any]]:
        values = ordinals.detach().cpu().reshape(-1).tolist() if isinstance(ordinals, torch.Tensor) else ordinals
        return self.corpus_reader().read_records(values)

    def lookup_text(self, ordinals: torch.Tensor | list[int]) -> list[str]:
        """Fetch immutable corpus contents by ordinal."""
        return [record["contents"] for record in self.lookup_records(ordinals)]

    def close(self) -> None:
        if self._corpus_reader is not None:
            self._corpus_reader.close()
            self._corpus_reader = None
        if self._search_executor is not None:
            self._search_executor.shutdown(wait=False)
            self._search_executor = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class FrozenCorpusIndexRouter:
    """Route each training row to one immutable per-domain corpus index."""

    def __init__(
        self,
        manifest_path: str,
        backend: str = "lookup",
        device: str | torch.device = "cuda",
        verify_hashes: bool = True,
        search_batch_size: int = 1024,
        faiss_gpu_devices: Sequence[int] | None = None,
    ) -> None:
        self.manifest_path = str(Path(manifest_path).resolve())
        self.root = Path(self.manifest_path).parent
        with open(self.manifest_path, encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        if self.manifest.get("format_version") != 1:
            raise ValueError("Unsupported routed frozen-index manifest version")
        if self.manifest.get("artifact_type") != "frozen_document_index_router":
            raise ValueError("Manifest is not a frozen document index router")
        routes = self.manifest.get("routes")
        if not isinstance(routes, dict) or not routes:
            raise ValueError("Routed frozen-index manifest requires a non-empty routes mapping")
        aliases = self.manifest.get("aliases", {})
        if not isinstance(aliases, dict):
            raise ValueError("Routed frozen-index aliases must be a mapping")

        self.route_names = sorted(routes)
        self.route_name_to_id = {name: index for index, name in enumerate(self.route_names)}
        self.source_to_route_name = {name: name for name in self.route_names}
        for source, route in aliases.items():
            if route not in routes:
                raise ValueError(f"Alias {source!r} targets unknown route {route!r}")
            self.source_to_route_name[str(source)] = str(route)
        self.source_to_route_id = {
            source: self.route_name_to_id[route]
            for source, route in self.source_to_route_name.items()
        }

        self.indices: dict[str, FrozenCorpusIndex] = {}
        for route in self.route_names:
            item = routes[route]
            if not isinstance(item, dict) or not item.get("manifest"):
                raise ValueError(f"Route {route!r} requires a manifest path")
            child_path = Path(item["manifest"])
            if not child_path.is_absolute():
                child_path = self.root / child_path
            expected_hash = item.get("sha256")
            if verify_hashes and expected_hash and sha256_file(child_path) != expected_hash:
                raise ValueError(f"Frozen route manifest hash mismatch: {child_path}")
            child = FrozenCorpusIndex(
                str(child_path),
                backend=backend,
                device=device,
                verify_hashes=verify_hashes,
                search_batch_size=search_batch_size,
                # Routed corpora are per-domain and individually small. Pin each to
                # one device so a dozen routes do not each fan out over every GPU.
                faiss_gpu_devices=(
                    faiss_gpu_devices
                    if faiss_gpu_devices is not None
                    else [torch.device(device).index or 0]
                ),
            )
            if child.manifest.get("source_name") not in {None, route}:
                raise ValueError(
                    f"Route {route!r} points to source {child.manifest.get('source_name')!r}"
                )
            self.indices[route] = child
        dimensions = {index.dimension for index in self.indices.values()}
        if len(dimensions) != 1:
            raise ValueError(f"Routed frozen indexes have inconsistent dimensions: {dimensions}")
        self.dimension = dimensions.pop()
        self.device = torch.device(device)
        self.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

    def _expand_route_ids(
        self,
        route_ids: torch.Tensor | None,
        target_shape: torch.Size | tuple[int, ...],
    ) -> torch.Tensor:
        if route_ids is None:
            raise ValueError("Routed frozen-index operation requires route_ids")
        target_shape = tuple(target_shape)
        route_ids = route_ids.detach().to(self.device)
        if route_ids.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise TypeError("route_ids must use an integer dtype")
        if tuple(route_ids.shape) == target_shape:
            return route_ids
        route_shape = tuple(route_ids.shape)
        if route_shape and route_shape == target_shape[: len(route_shape)]:
            view_shape = route_shape + (1,) * (len(target_shape) - len(route_shape))
            return route_ids.reshape(view_shape).expand(target_shape)
        raise ValueError(
            f"route_ids shape {route_shape} must be a leading prefix of target shape {target_shape}"
        )

    def _validate_route_ids(
        self,
        expanded_routes: torch.Tensor,
        *,
        distributed: bool = False,
    ) -> None:
        unknown = (expanded_routes < 0) | (expanded_routes >= len(self.route_names))
        has_unknown = torch.tensor(
            int(unknown.any()), device=self.device, dtype=torch.int32
        )
        if distributed and self.world_size > 1:
            dist.all_reduce(has_unknown, op=dist.ReduceOp.MAX)
        if has_unknown.item():
            local_values = sorted(set(expanded_routes[unknown].detach().cpu().tolist()))
            detail = f": {local_values}" if local_values else " on another distributed rank"
            raise ValueError(f"Unknown frozen index route id(s){detail}")

    def route_name(self, source: str) -> str:
        try:
            return self.source_to_route_name[source]
        except KeyError as exc:
            raise KeyError(
                f"No frozen index route for source {source!r}; "
                f"known sources: {sorted(self.source_to_route_name)}"
            ) from exc

    @property
    def document_key_to_ordinal_by_source(self) -> dict[str, dict[str, int]]:
        mappings: dict[str, dict[str, int]] = {}
        for source, route in self.source_to_route_name.items():
            # Aliases intentionally share one mapping instead of copying million-row
            # BRIGHT ID dictionaries into every DataLoader worker.
            mappings[source] = self.indices[route]._ensure_key_mapping()
        return mappings

    def lookup_embeddings(
        self,
        ordinals: torch.Tensor,
        route_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        expanded_routes = self._expand_route_ids(route_ids, ordinals.shape)
        self._validate_route_ids(expanded_routes)
        result = torch.empty(
            (*ordinals.shape, self.dimension),
            device=self.device,
            dtype=torch.float32,
        )
        for route_id, route in enumerate(self.route_names):
            selected = expanded_routes == route_id
            if selected.any():
                result[selected] = self.indices[route].lookup_embeddings(ordinals[selected])
        return result

    def search(
        self,
        query_vectors: torch.Tensor,
        k: int,
        route_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Search each query only in its routed corpus and restore input order."""
        if k <= 0:
            raise ValueError("k must be positive")
        if query_vectors.size(-1) != self.dimension:
            raise ValueError(
                f"Query dim {query_vectors.size(-1)} does not match routed index dim {self.dimension}"
            )
        original_shape = query_vectors.shape[:-1]
        expanded_routes = self._expand_route_ids(route_ids, original_shape)
        self._validate_route_ids(expanded_routes, distributed=True)
        flat_queries = query_vectors.detach().to(self.device).reshape(-1, self.dimension)
        flat_routes = expanded_routes.reshape(-1)
        scores = torch.empty((flat_queries.size(0), k), device=self.device, dtype=torch.float32)
        ordinals = torch.empty((flat_queries.size(0), k), device=self.device, dtype=torch.long)
        # Every distributed rank enters every child search in the same route order.
        # A child search gathers variable local query counts and safely accepts zero.
        for route_id, route in enumerate(self.route_names):
            selected = flat_routes == route_id
            global_count = torch.tensor(
                int(selected.sum()), device=self.device, dtype=torch.long
            )
            if self.world_size > 1:
                dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
            if global_count.item() == 0:
                continue
            route_scores, route_ordinals = self.indices[route].search(
                flat_queries[selected], k
            )
            scores[selected] = route_scores
            ordinals[selected] = route_ordinals
        return scores.reshape(*original_shape, k), ordinals.reshape(*original_shape, k)

    def validate_protocol(self, **kwargs) -> None:
        for route, index in self.indices.items():
            try:
                index.validate_protocol(**kwargs)
            except ValueError as exc:
                raise ValueError(f"Frozen index route {route!r}: {exc}") from exc

    def verify(self) -> None:
        for index in self.indices.values():
            index.verify()

    def artifact_hashes(self) -> dict[str, str]:
        hashes = {"index_router_manifest": sha256_file(self.manifest_path)}
        for route, index in self.indices.items():
            for name, value in index.artifact_hashes().items():
                hashes[f"{route}:{name}"] = value
        return hashes

    def close(self) -> None:
        for index in self.indices.values():
            index.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def load_frozen_index(
    manifest_path: str,
    **kwargs,
) -> FrozenCorpusIndex | FrozenCorpusIndexRouter:
    """Load either a legacy single index or a routed index manifest."""
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("artifact_type") == "frozen_document_index_router":
        return FrozenCorpusIndexRouter(manifest_path, **kwargs)
    return FrozenCorpusIndex(manifest_path, **kwargs)


# Historical name retained for callers that used it outside ``rag``.
FrozenDistributedIndex = FrozenCorpusIndex


def write_frozen_training_audit(
    output_dir: str | os.PathLike[str],
    index: FrozenCorpusIndex | FrozenCorpusIndexRouter,
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
        "index_routes": getattr(index, "source_to_route_name", None),
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
