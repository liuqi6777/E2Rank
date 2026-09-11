"""RAG-specific manifest wrapper around the shared corpus shard encoder."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch.distributed as dist

from fixed_corpus.encode import distributed_context, encode_corpus_shards
from fixed_corpus.index import sha256_file


def _load_source_manifest(path: Path) -> tuple[dict, Path]:
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return manifest, path.parent


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

    rank, world_size, device = distributed_context()
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

    dimension, resolved_revision, shards = encode_corpus_shards(
        corpus_path=corpus_path,
        offsets_path=offsets_path,
        output_dir=output_dir,
        model_name_or_path=args.model,
        revision=args.revision,
        num_shards=args.num_shards,
        batch_size=args.batch_size,
        max_length=args.max_length,
        pooling_method=args.pooling_method,
        padding_side=args.padding_side,
        append_token=args.append_token,
        document_prompt_template="{document}",
        rank=rank,
        world_size=world_size,
        device=device,
        overwrite=args.overwrite,
    )
    if rank == 0:
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
            "resolved_model_revision": resolved_revision,
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
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
