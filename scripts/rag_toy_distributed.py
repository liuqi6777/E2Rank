from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from rag.index import FrozenDistributedIndex, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--backend", choices=("faiss", "torch"), default="faiss")
    args = parser.parse_args()
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the 2-GPU acceptance test")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    root = Path(args.work_dir).resolve()
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        generator = np.random.default_rng(42)
        vectors = generator.normal(size=(32, 8)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors.astype(np.float16)
        shards = []
        for shard_id in range(4):
            start, end = shard_id * 8, (shard_id + 1) * 8
            path = root / f"vectors-{shard_id}.npy"
            np.save(path, vectors[start:end])
            shards.append({
                "path": path.name, "start": start, "count": end - start,
                "sha256": sha256_file(path),
            })
        corpus = root / "corpus.jsonl"
        offsets = []
        with open(corpus, "wb") as handle:
            for ordinal in range(len(vectors)):
                offsets.append(handle.tell())
                handle.write((json.dumps({"id": str(ordinal), "contents": f"doc {ordinal}"}) + "\n").encode())
        np.save(root / "offsets.npy", np.asarray(offsets, dtype=np.int64))
        manifest = {
            "format_version": 1, "dimension": 8, "count": 32, "shards": shards,
            "corpus_path": corpus.name, "corpus_offsets_path": "offsets.npy",
            "corpus_sha256": sha256_file(corpus),
            "corpus_offsets_sha256": sha256_file(root / "offsets.npy"),
        }
        with open(root / "index_manifest.json", "w", encoding="utf-8") as handle:
            json.dump(manifest, handle)
    dist.barrier()

    index = FrozenDistributedIndex(
        str(root / "index_manifest.json"), backend=args.backend, device=device
    )
    query = torch.zeros((1, 8), device=device)
    query[0, rank % 8] = 1
    scores, ids = index.search(query, 7)
    all_vectors = torch.from_numpy(np.concatenate([
        np.load(root / f"vectors-{shard_id}.npy") for shard_id in range(4)
    ]).astype(np.float32)).to(device)
    expected_scores, expected_ids = (query @ all_vectors.T).topk(7, dim=-1)
    torch.testing.assert_close(scores, expected_scores)
    torch.testing.assert_close(ids, expected_ids)
    index.verify()
    dist.barrier()
    if rank == 0:
        print("distributed FlatIP toy acceptance passed")
    index.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
