from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoModel, AutoTokenizer

from embedding_protocol import append_configured_token, pool_embeddings
from rag.index import sha256_file


def _distributed_context() -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return rank, world_size, device


def _load_source_manifest(path: Path) -> tuple[dict, Path]:
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return manifest, path.parent


def _read_batch(corpus_handle, offsets: np.ndarray, start: int, end: int) -> list[str]:
    records = []
    for ordinal in range(start, end):
        corpus_handle.seek(int(offsets[ordinal]))
        records.append(json.loads(corpus_handle.readline())["contents"])
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flashrag-manifest", default="data/rag/flashrag_manifest.json")
    parser.add_argument("--output-dir", default="data/rag/qwen3_e0_index")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--num-shards", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--pooling-method", default="last")
    parser.add_argument("--padding-side", default="left")
    parser.add_argument("--append-token", default="pad")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rank, world_size, device = _distributed_context()
    source_manifest, source_root = _load_source_manifest(Path(args.flashrag_manifest).resolve())
    corpus_path = source_root / source_manifest["corpus"]["path"]
    offsets_path = source_root / source_manifest["corpus"]["offsets_path"]
    offsets = np.load(offsets_path, mmap_mode="r")
    count = len(offsets)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "index_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Frozen index already exists: {manifest_path}. Use --overwrite to rebuild explicitly."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        padding_side=args.padding_side,
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device).eval()
    dimension = int(model.config.hidden_size)
    per_shard = math.ceil(count / args.num_shards)

    with open(corpus_path, "rb") as corpus_handle:
        for shard_id in range(rank, args.num_shards, world_size):
            start = shard_id * per_shard
            end = min(start + per_shard, count)
            if start >= end:
                continue
            output_path = output_dir / f"vectors-{shard_id:05d}-of-{args.num_shards:05d}.npy"
            if output_path.exists() and not args.overwrite:
                raise FileExistsError(f"Index shard already exists: {output_path}")
            partial_path = output_path.with_suffix(output_path.suffix + ".partial")
            vectors = np.lib.format.open_memmap(
                partial_path, mode="w+", dtype=np.float16, shape=(end - start, dimension)
            )
            for batch_start in range(start, end, args.batch_size):
                batch_end = min(batch_start + args.batch_size, end)
                texts = _read_batch(corpus_handle, offsets, batch_start, batch_end)
                texts = append_configured_token(texts, tokenizer, args.append_token)
                inputs = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                    return_tensors="pt",
                ).to(device)
                with torch.inference_mode():
                    embeddings = pool_embeddings(
                        model(**inputs).last_hidden_state,
                        inputs["attention_mask"],
                        pooling_method=args.pooling_method,
                        normalize=True,
                    )
                vectors[batch_start - start : batch_end - start] = embeddings.float().cpu().numpy()
            vectors.flush()
            del vectors
            os.replace(partial_path, output_path)

    if world_size > 1:
        dist.barrier()
    if rank == 0:
        shards = []
        for shard_id in range(args.num_shards):
            start = shard_id * per_shard
            end = min(start + per_shard, count)
            if start >= end:
                continue
            path = output_dir / f"vectors-{shard_id:05d}-of-{args.num_shards:05d}.npy"
            shards.append({
                "path": path.name,
                "start": start,
                "count": end - start,
                "sha256": sha256_file(path),
            })
        index_manifest = {
            "format_version": 1,
            "source_manifest": os.path.relpath(Path(args.flashrag_manifest).resolve(), output_dir),
            "source_manifest_sha256": sha256_file(Path(args.flashrag_manifest).resolve()),
            "corpus_path": os.path.relpath(corpus_path, output_dir),
            "corpus_offsets_path": os.path.relpath(offsets_path, output_dir),
            "corpus_sha256": source_manifest["corpus"]["sha256"],
            "corpus_offsets_sha256": sha256_file(offsets_path),
            "model_name_or_path": args.model,
            "model_revision": args.revision,
            "resolved_model_revision": getattr(model.config, "_commit_hash", None),
            "pooling_method": args.pooling_method,
            "padding_side": args.padding_side,
            "append_token": args.append_token,
            "document_max_length": args.max_length,
            "normalized": True,
            "dtype": "float16",
            "dimension": dimension,
            "count": count,
            "shards": shards,
            "distributed_shard_assignment": "round_robin",
        }
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(index_manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
