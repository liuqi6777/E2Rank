"""Build the immutable G1 document corpus and fp16 embedding shards."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import unicodedata

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModel, AutoTokenizer

from embedding_protocol import append_configured_token, format_embedding_text, pool_embeddings
from frozen_corpus import FrozenCorpusIndex, sha256_file, validate_frozen_protocol


def normalize_document(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _distributed_context() -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return rank, world_size, device


def build_corpus(source_path: Path, output_dir: Path) -> int:
    """Deduplicate in first-seen order and reject key/text inconsistencies."""
    corpus_path = output_dir / "corpus.jsonl"
    offsets: list[int] = []
    key_to_ordinal: dict[str, int] = {}
    normalized_by_key: dict[str, str] = {}
    with source_path.open(encoding="utf-8") as source, corpus_path.open("wb") as corpus:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {source_path}:{line_number}") from exc
            documents = record.get("document")
            keys = record.get("document_keys")
            if not isinstance(documents, list) or not isinstance(keys, list) or len(documents) != len(keys):
                raise ValueError(f"Candidate text/key mismatch at {source_path}:{line_number}")
            for text, key in zip(documents, keys):
                if not isinstance(text, str) or not isinstance(key, str):
                    raise ValueError(f"Non-string document/key at {source_path}:{line_number}")
                normalized = normalize_document(text)
                if key in normalized_by_key:
                    if normalized_by_key[key] != normalized:
                        raise ValueError(f"Conflicting normalized text for document_key={key}")
                    continue
                ordinal = len(key_to_ordinal)
                key_to_ordinal[key] = ordinal
                normalized_by_key[key] = normalized
                offsets.append(corpus.tell())
                payload = json.dumps(
                    {"document_key": key, "contents": text},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8") + b"\n"
                corpus.write(payload)
        corpus.flush()
        os.fsync(corpus.fileno())
    if not key_to_ordinal:
        raise ValueError(f"No documents found in {source_path}")
    np.save(output_dir / "corpus_offsets.npy", np.asarray(offsets, dtype=np.int64))
    with (output_dir / "document_key_to_ordinal.json").open("w", encoding="utf-8") as handle:
        json.dump(key_to_ordinal, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
    return len(key_to_ordinal)


def _read_texts(handle, offsets: np.ndarray, start: int, end: int) -> list[str]:
    values = []
    for ordinal in range(start, end):
        handle.seek(int(offsets[ordinal]))
        values.append(json.loads(handle.readline())["contents"])
    return values


def _validate_existing(args, manifest_path: Path) -> None:
    index = FrozenCorpusIndex(str(manifest_path), backend="lookup", device="cpu", verify_hashes=True)
    if index.manifest.get("model_revision") != args.revision:
        raise ValueError("Existing frozen index uses a different configured model revision")
    model_config = AutoConfig.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True
    )
    validate_frozen_protocol(
        index.manifest,
        model_name_or_path=args.model,
        resolved_model_revision=getattr(model_config, "_commit_hash", None),
        pooling_method=args.pooling_method,
        padding_side=args.padding_side,
        append_token=args.append_token,
        document_prompt_template=args.document_prompt_template,
        document_max_length=args.max_length,
        query_prompt_template=args.query_prompt_template,
        embedding_max_length=args.embedding_max_length,
    )
    if index.manifest.get("source_data_sha256") != sha256_file(args.input):
        raise ValueError("Existing frozen index was built from different training data")
    index.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--num-shards", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--pooling-method", default="last")
    parser.add_argument("--padding-side", default="left")
    parser.add_argument("--append-token", default="pad")
    parser.add_argument("--document-prompt-template", default="{document}")
    parser.add_argument("--query-prompt-template", default="Instruct: {task_description}\nQuery:{query}")
    parser.add_argument("--embedding-max-length", type=int, required=True)
    args = parser.parse_args()
    if any(value <= 0 for value in (
        args.num_shards, args.batch_size, args.max_length, args.embedding_max_length
    )):
        raise ValueError("shard, batch, and embedding length settings must be positive")

    rank, world_size, device = _distributed_context()
    source_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    manifest_path = output_dir / "index_manifest.json"
    if manifest_path.is_file():
        _validate_existing(args, manifest_path)
        if rank == 0:
            print(f"Frozen document index already valid: {manifest_path}")
        if dist.is_initialized():
            dist.destroy_process_group()
        return
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite incomplete frozen index directory: {output_dir}")

    building = output_dir.with_name(output_dir.name + ".building")
    if rank == 0:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        if building.exists():
            raise FileExistsError(f"Remove or inspect stale build directory: {building}")
        building.mkdir()
        count = build_corpus(source_path, building)
        (building / "build_state.json").write_text(json.dumps({"count": count}) + "\n")
    if dist.is_initialized():
        dist.barrier()
    state = json.loads((building / "build_state.json").read_text())
    count = int(state["count"])

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        padding_side=args.padding_side,
        trust_remote_code=True,
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = (
            getattr(tokenizer, "eot_token", None) or tokenizer.eos_token or tokenizer.bos_token
        )
    model = AutoModel.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device).eval()
    dimension = int(model.config.hidden_size)
    offsets = np.load(building / "corpus_offsets.npy", mmap_mode="r")
    per_shard = math.ceil(count / args.num_shards)
    with (building / "corpus.jsonl").open("rb") as corpus:
        for shard_id in range(rank, args.num_shards, world_size):
            start = shard_id * per_shard
            end = min(start + per_shard, count)
            if start >= end:
                continue
            output_path = building / f"vectors-{shard_id:05d}-of-{args.num_shards:05d}.npy"
            partial = output_path.with_suffix(".npy.partial")
            vectors = np.lib.format.open_memmap(
                partial, mode="w+", dtype=np.float16, shape=(end - start, dimension)
            )
            for batch_start in range(start, end, args.batch_size):
                batch_end = min(batch_start + args.batch_size, end)
                texts = _read_texts(corpus, offsets, batch_start, batch_end)
                texts = [format_embedding_text(args.document_prompt_template, text) for text in texts]
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
            os.replace(partial, output_path)
    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        shards = []
        for shard_id in range(args.num_shards):
            start = shard_id * per_shard
            end = min(start + per_shard, count)
            if start >= end:
                continue
            path = building / f"vectors-{shard_id:05d}-of-{args.num_shards:05d}.npy"
            shards.append({
                "path": path.name,
                "start": start,
                "count": end - start,
                "sha256": sha256_file(path),
            })
        corpus_path = building / "corpus.jsonl"
        offsets_path = building / "corpus_offsets.npy"
        mapping_path = building / "document_key_to_ordinal.json"
        protocol = {
            "model_name_or_path": args.model,
            "model_revision": args.revision,
            "resolved_model_revision": getattr(model.config, "_commit_hash", None),
            "pooling_method": args.pooling_method,
            "padding_side": args.padding_side,
            "append_token": args.append_token,
            "document_prompt_template": args.document_prompt_template,
            "document_max_length": args.max_length,
            "query_prompt_template": args.query_prompt_template,
            "embedding_max_length": args.embedding_max_length,
            "normalized": True,
            "dtype": "float16",
        }
        manifest = {
            "format_version": 1,
            "artifact_type": "frozen_document_index",
            "source_data_path": os.path.relpath(source_path, building),
            "source_data_sha256": sha256_file(source_path),
            "corpus_path": corpus_path.name,
            "corpus_offsets_path": offsets_path.name,
            "document_key_to_ordinal_path": mapping_path.name,
            "corpus_sha256": sha256_file(corpus_path),
            "corpus_offsets_sha256": sha256_file(offsets_path),
            "document_key_to_ordinal_sha256": sha256_file(mapping_path),
            "embedding_protocol": protocol,
            **protocol,
            "dimension": dimension,
            "count": count,
            "shards": shards,
            "distributed_shard_assignment": "round_robin",
        }
        (building / "build_state.json").unlink()
        with (building / "index_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(building, output_dir)
        print(f"Built {count} frozen documents: {manifest_path}")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
