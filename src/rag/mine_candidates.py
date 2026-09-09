from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterator

from rag.candidates import (
    build_candidate_record,
    build_title_catalog,
    collect_hotpot_title_targets,
    force_evidence_into_candidates,
    map_hotpot_evidence,
)
from rag.data import TRAIN_SOURCES, iter_jsonl, validate_flashrag_record
from rag.encoder import FrozenQueryEncoder
from rag.index import FrozenDistributedIndex, sha256_file
from rag.protocol import validate_query_index_protocol


def _resolve_source_file(manifest: dict, root: Path, source: str, split: str) -> Path:
    entry = manifest["files"][f"{source}/{split}"]
    path = Path(entry["path"])
    return path if path.is_absolute() else root / path


def _batches(records: Iterator[dict], batch_size: int) -> Iterator[list[dict]]:
    batch = []
    for record in records:
        batch.append(record)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def mine(args: argparse.Namespace) -> tuple[Path, Path]:
    source_manifest_path = Path(args.flashrag_manifest).resolve()
    with open(source_manifest_path, "r", encoding="utf-8") as handle:
        source_manifest = json.load(handle)
    source_root = source_manifest_path.parent

    train_paths = {
        source: _resolve_source_file(source_manifest, source_root, source, split)
        for source, (split, _) in TRAIN_SOURCES.items()
    }
    index = FrozenDistributedIndex(
        args.index_manifest,
        backend=args.index_backend,
        device=args.device,
        verify_hashes=not args.skip_hash_verification,
    )
    indexed_corpus_path = Path(index.manifest["corpus_path"])
    if not indexed_corpus_path.is_absolute():
        indexed_corpus_path = index.root / indexed_corpus_path
    if args.corpus_path and Path(args.corpus_path).resolve() != indexed_corpus_path.resolve():
        raise ValueError("--corpus-path does not match the corpus frozen in the index manifest")
    hotpot_targets = collect_hotpot_title_targets(iter_jsonl(train_paths["hotpotqa"]))
    title_catalog = build_title_catalog(str(indexed_corpus_path), hotpot_targets)
    encoder = FrozenQueryEncoder(
        args.model,
        revision=args.model_revision,
        device=args.device,
        max_length=args.query_max_length,
        pooling_method=args.pooling_method,
        padding_side=args.padding_side,
        append_token=args.append_token,
        query_prompt_template=args.query_prompt_template,
    )
    if int(encoder.model.config.hidden_size) != index.dimension:
        raise ValueError("Query encoder dimension does not match frozen index")
    validate_query_index_protocol(
        index.manifest,
        model_name_or_path=args.model,
        resolved_model_revision=getattr(encoder.model.config, "_commit_hash", None),
        pooling_method=args.pooling_method,
        padding_side=args.padding_side,
        append_token=args.append_token,
    )

    output_path = Path(args.output).resolve()
    metadata_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Candidate manifest already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    statistics = {
        source: {"raw": 0, "kept": 0, "missing_answer": 0, "missing_evidence": 0}
        for source in TRAIN_SOURCES
    }

    with open(partial_path, "w", encoding="utf-8") as writer:
        for source in TRAIN_SOURCES:
            records = iter_jsonl(train_paths[source])
            for batch in _batches(records, args.batch_size):
                for record in batch:
                    validate_flashrag_record(record, source)
                embeddings = encoder.encode([record["question"] for record in batch])
                _, retrieved = index.search(embeddings, args.depth)
                for record, row in zip(batch, retrieved.detach().cpu().tolist()):
                    statistics[source]["raw"] += 1
                    evidence_groups = None
                    if source == "hotpotqa":
                        evidence_groups = map_hotpot_evidence(
                            record, title_catalog, args.evidence_minimum_f1
                        )
                        if evidence_groups is None:
                            statistics[source]["missing_evidence"] += 1
                            continue
                        row = force_evidence_into_candidates(row, evidence_groups)
                    contents = [item["contents"] for item in index.lookup_records(row)]
                    candidate = build_candidate_record(
                        source, record, row, contents, evidence_groups
                    )
                    if candidate is None:
                        statistics[source]["missing_answer" if source == "nq" else "missing_evidence"] += 1
                        continue
                    writer.write(json.dumps(candidate, ensure_ascii=False) + "\n")
                    statistics[source]["kept"] += 1
        writer.flush()
        os.fsync(writer.fileno())
    os.replace(partial_path, output_path)

    for values in statistics.values():
        values["coverage"] = values["kept"] / max(values["raw"], 1)
    metadata = {
        "format_version": 1,
        "artifact_origin": "E2Rank-RL",
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "index_manifest": str(Path(args.index_manifest).resolve()),
        "index_manifest_sha256": sha256_file(args.index_manifest),
        "corpus_sha256": index.manifest.get("corpus_sha256"),
        "model": args.model,
        "model_revision": args.model_revision,
        "resolved_model_revision": getattr(encoder.model.config, "_commit_hash", None),
        "depth": args.depth,
        "query_max_length": args.query_max_length,
        "pooling_method": args.pooling_method,
        "padding_side": args.padding_side,
        "append_token": args.append_token,
        "query_prompt_template": args.query_prompt_template,
        "statistics": statistics,
        "candidate_manifest": output_path.name,
        "candidate_manifest_sha256": sha256_file(output_path),
    }
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
    index.close()
    return output_path, metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine immutable E0 top-1000 RAG candidates")
    parser.add_argument("--flashrag-manifest", default="data/rag/flashrag_manifest.json")
    parser.add_argument("--corpus-path", default=None, help="Optional consistency check; normally derived from the index")
    parser.add_argument("--index-manifest", default="data/rag/qwen3_e0_index/index_manifest.json")
    parser.add_argument("--output", default="data/rag/candidates/nq_hotpotqa_train.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--index-backend", choices=("faiss", "torch"), default="faiss")
    parser.add_argument("--depth", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--query-max-length", type=int, default=128)
    parser.add_argument("--pooling-method", default="last")
    parser.add_argument("--padding-side", default="left")
    parser.add_argument("--append-token", default="pad")
    parser.add_argument(
        "--query-prompt-template",
        default="Instruct: {task_description}\nQuery:{query}",
    )
    parser.add_argument("--evidence-minimum-f1", type=float, default=0.8)
    parser.add_argument("--skip-hash-verification", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output_path, metadata_path = mine(args)
    print(output_path)
    print(metadata_path)


if __name__ == "__main__":
    main()
