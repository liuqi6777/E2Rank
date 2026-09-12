"""Evaluate a project-produced query encoder on the fixed seven-dataset suite."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterator

import torch

from rag.candidates import (
    build_title_catalog,
    collect_hotpot_title_targets,
    map_hotpot_evidence,
)
from rag.data import EVALUATION_SUITE, iter_jsonl, validate_flashrag_record
from rag.encoder import FrozenQueryEncoder
from rag.generator import FrozenGeneratorClient
from rag.index import FrozenDistributedIndex, sha256_file
from rag.metrics import exact_match, max_token_f1, passage_contains_answer
from rag.protocol import validate_query_index_protocol


MULTIHOP_SOURCES = {"hotpotqa", "2wikimultihopqa", "musique"}


def _source_path(manifest: dict, root: Path, source: str, split: str) -> Path:
    path = Path(manifest["files"][f"{source}/{split}"]["path"])
    return path if path.is_absolute() else root / path


def _batches(records: Iterator[dict], size: int) -> Iterator[list[dict]]:
    batch = []
    for record in records:
        batch.append(record)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _first_relevant_rank(mask: list[bool], cutoff: int) -> float:
    for rank, relevant in enumerate(mask[:cutoff], start=1):
        if relevant:
            return 1.0 / rank
    return 0.0


def _mean(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / max(len(rows), 1)


def _validate_project_checkpoint(checkpoint: str | None, index_manifest_path: str) -> None:
    if checkpoint is None:
        return
    checkpoint_path = Path(checkpoint).resolve()
    run_manifest_path = checkpoint_path / "rag_run_manifest.json"
    if not run_manifest_path.is_file():
        raise ValueError(
            f"Final evaluation only accepts project-produced RAG checkpoints; missing {run_manifest_path}"
        )
    with open(run_manifest_path, "r", encoding="utf-8") as handle:
        run_manifest = json.load(handle)
    if run_manifest.get("artifact_origin") != "E2Rank-RL":
        raise ValueError("Checkpoint manifest has an unsupported artifact origin")
    if run_manifest.get("index_manifest_sha256_after") != sha256_file(index_manifest_path):
        raise ValueError("Checkpoint was trained against a different frozen index")


def evaluate(args: argparse.Namespace) -> Path:
    if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        raise RuntimeError("eval_rag.py is intentionally single-process; use one 80GB GPU")
    if args.generator_top_k > args.retrieval_k:
        raise ValueError("generator_top_k cannot exceed retrieval_k")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Evaluation output is not empty: {output_dir}")
    if output_dir.exists() and args.overwrite:
        allowed_names = {"summary.json"}
        for source, (split, _) in EVALUATION_SUITE.items():
            allowed_names.add(f"{source}_{split}.retrieval.jsonl")
            allowed_names.add(f"{source}_{split}.generation.jsonl")
        unexpected = sorted(path.name for path in output_dir.iterdir() if path.name not in allowed_names)
        if unexpected:
            raise ValueError(
                "Evaluation directory contains non-RAG artifacts: " + ", ".join(unexpected)
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    source_manifest_path = Path(args.flashrag_manifest).resolve()
    with open(source_manifest_path, "r", encoding="utf-8") as handle:
        source_manifest = json.load(handle)
    source_root = source_manifest_path.parent
    dataset_paths = {
        source: _source_path(source_manifest, source_root, source, split)
        for source, (split, _) in EVALUATION_SUITE.items()
    }
    _validate_project_checkpoint(args.checkpoint, args.index_manifest)

    index = FrozenDistributedIndex(
        args.index_manifest,
        backend=args.index_backend,
        device=args.device,
        verify_hashes=not args.skip_hash_verification,
    )
    if index.manifest.get("source_manifest_sha256") != sha256_file(source_manifest_path):
        raise ValueError("Evaluation data and frozen index use different FlashRAG manifests")
    corpus_path = Path(index.manifest["corpus_path"])
    if not corpus_path.is_absolute():
        corpus_path = index.root / corpus_path
    evidence_titles = set()
    for source in MULTIHOP_SOURCES:
        evidence_titles.update(collect_hotpot_title_targets(iter_jsonl(dataset_paths[source])))
    title_catalog = build_title_catalog(str(corpus_path), evidence_titles)

    encoder = FrozenQueryEncoder(
        args.model,
        adapter_path=args.checkpoint,
        revision=args.model_revision,
        device=args.device,
        max_length=args.query_max_length,
        pooling_method=args.pooling_method,
        padding_side=args.padding_side,
        append_token=args.append_token,
        query_prompt_template=args.query_prompt_template,
    )
    validate_query_index_protocol(
        index.manifest,
        model_name_or_path=args.model,
        resolved_model_revision=getattr(encoder.model.config, "_commit_hash", None),
        pooling_method=args.pooling_method,
        padding_side=args.padding_side,
        append_token=args.append_token,
    )
    generator = None
    if not args.retrieval_only:
        generator = FrozenGeneratorClient(
            args.generator_endpoint,
            args.generator_model,
            args.generator_cache,
            revision=args.generator_revision,
            max_input_length=args.generator_max_input_length,
            max_new_tokens=args.generator_max_new_tokens,
            timeout_seconds=args.generator_timeout_seconds,
            system_prompt=args.generator_system_prompt,
        )

    started = time.perf_counter()
    retrieval_seconds = 0.0
    total_queries = 0
    summaries = {}
    for source, (split, expected_count) in EVALUATION_SUITE.items():
        rows = []
        retrieval_path = output_dir / f"{source}_{split}.retrieval.jsonl"
        generation_path = output_dir / f"{source}_{split}.generation.jsonl"
        generation_writer = open(generation_path, "w", encoding="utf-8") if generator else None
        with open(retrieval_path, "w", encoding="utf-8") as retrieval_writer:
            for batch in _batches(iter_jsonl(dataset_paths[source]), args.batch_size):
                for record in batch:
                    validate_flashrag_record(record, source)
                retrieval_started = time.perf_counter()
                embeddings = encoder.encode([record["question"] for record in batch])
                scores, result_ids = index.search(embeddings, args.retrieval_k)
                retrieval_seconds += time.perf_counter() - retrieval_started
                ids_rows = result_ids.detach().cpu().tolist()
                score_rows = scores.detach().float().cpu().tolist()
                generation_requests = []
                pending = []
                for record, ids, values in zip(batch, ids_rows, score_rows):
                    documents = index.lookup_records(ids)
                    answer_mask = [
                        passage_contains_answer(document["contents"], record["golden_answers"])
                        for document in documents
                    ]
                    row = {
                        "query_id": record["id"],
                        "source": source,
                        "question": record["question"],
                        "golden_answers": record["golden_answers"],
                        "passage_ids": ids,
                        "scores": values,
                        "answer_recall_at_5": float(any(answer_mask[:5])),
                        "answer_recall_at_10": float(any(answer_mask[:10])),
                        "answer_recall_at_20": float(any(answer_mask[:20])),
                        "answer_mrr_at_10": _first_relevant_rank(answer_mask, 10),
                        "answer_mrr_at_20": _first_relevant_rank(answer_mask, 20),
                    }
                    if source in MULTIHOP_SOURCES:
                        groups = map_hotpot_evidence(record, title_catalog, args.evidence_minimum_f1)
                        found_counts = []
                        for cutoff in (5, 10, 20):
                            found = 0 if groups is None else sum(
                                any(ordinal in set(group) for ordinal in ids[:cutoff])
                                for group in groups
                            )
                            count = len(groups) if groups else 0
                            row[f"evidence_group_recall_at_{cutoff}"] = found / max(count, 1)
                            found_counts.append((found, count))
                        for cutoff, (found, count) in zip((5, 10, 20), found_counts):
                            row[f"all_evidence_success_at_{cutoff}"] = float(
                                groups is not None and found == count
                            )
                        row["evidence_mapping_success"] = float(groups is not None)
                    retrieval_writer.write(json.dumps(row, ensure_ascii=False) + "\n")
                    if generator:
                        top_ids = ids[: args.generator_top_k]
                        generation_requests.append(
                            (record["id"], record["question"], top_ids, documents[: args.generator_top_k])
                        )
                        pending.append((record, row, top_ids))
                    rows.append(row)
                if generator:
                    generations = generator.generate_batch(generation_requests)
                    for generation, (record, row, top_ids) in zip(generations, pending):
                        row["generator_em"] = exact_match(generation, record["golden_answers"])
                        row["generator_token_f1"] = max_token_f1(generation, record["golden_answers"])
                        generation_writer.write(json.dumps({
                            "query_id": record["id"],
                            "source": source,
                            "passage_ids": top_ids,
                            "prediction": generation,
                            "golden_answers": record["golden_answers"],
                            "em": row["generator_em"],
                            "token_f1": row["generator_token_f1"],
                        }, ensure_ascii=False) + "\n")
        if generation_writer:
            generation_writer.close()
        if len(rows) != expected_count:
            raise ValueError(f"{source}/{split} has {len(rows)} rows, expected {expected_count}")
        total_queries += len(rows)
        metric_keys = [
            "answer_recall_at_5", "answer_recall_at_10", "answer_recall_at_20",
            "answer_mrr_at_10", "answer_mrr_at_20",
            *( ["generator_em", "generator_token_f1"] if generator else [] ),
            *( ["evidence_group_recall_at_5", "evidence_group_recall_at_10", "evidence_group_recall_at_20", "all_evidence_success_at_5", "all_evidence_success_at_10", "all_evidence_success_at_20", "evidence_mapping_success"] if source in MULTIHOP_SOURCES else [] ),
        ]
        summaries[source] = {key: _mean(rows, key) for key in metric_keys}
        summaries[source]["count"] = len(rows)

    elapsed = time.perf_counter() - started
    if total_queries != sum(value[1] for value in EVALUATION_SUITE.values()):
        raise RuntimeError(f"Evaluation suite count mismatch: {total_queries}")
    common_keys = [
        "answer_recall_at_5", "answer_recall_at_10", "answer_recall_at_20",
        "answer_mrr_at_10", "answer_mrr_at_20",
    ]
    if generator:
        common_keys += ["generator_em", "generator_token_f1"]
    def aggregate(sources: list[str]) -> dict[str, float]:
        return {key: sum(summaries[source][key] for source in sources) / len(sources) for key in common_keys}
    all_sources = list(EVALUATION_SUITE)
    summary = {
        "format_version": 1,
        "artifact_origin": "E2Rank-RL",
        "datasets": summaries,
        "macro_average": aggregate(all_sources),
        "training_domain_average": aggregate(["nq", "hotpotqa"]),
        "held_out_average": aggregate([source for source in all_sources if source not in {"nq", "hotpotqa"}]),
        "telemetry": {
            "wall_hours": elapsed / 3600,
            "query_index_gpu_hours": elapsed / 3600 if torch.cuda.is_available() else 0.0,
            "generator_gpu_hours": elapsed * args.generator_gpu_count / 3600 if generator else 0.0,
            "gpu_hours": (
                (elapsed / 3600 if torch.cuda.is_available() else 0.0)
                + (elapsed * args.generator_gpu_count / 3600 if generator else 0.0)
            ),
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0,
            "retrieval_seconds": retrieval_seconds,
            "retrieval_queries_per_second": total_queries / max(retrieval_seconds, 1e-9),
            **(generator.statistics() if generator else {"generation_requests": 0}),
            "corpus_reencoding_count": 0,
            "index_rebuild_count": 0,
        },
        "manifests": {
            "flashrag": str(source_manifest_path),
            "flashrag_sha256": sha256_file(source_manifest_path),
            "index": index.manifest_path,
            "index_sha256": sha256_file(index.manifest_path),
            "corpus_sha256": index.manifest.get("corpus_sha256"),
            "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
            "generator_manifest_hash": generator.manifest_hash if generator else None,
        },
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
    if generator:
        generator.close()
    index.close()
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flashrag-manifest", default="data/rag/flashrag_manifest.json")
    parser.add_argument("--index-manifest", default="data/rag/qwen3_e0_index/index_manifest.json")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Project-produced full-model checkpoint or legacy LoRA adapter; omit for E0",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--index-backend", choices=("faiss", "torch"), default="faiss")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--retrieval-k", type=int, default=20)
    parser.add_argument("--query-max-length", type=int, default=128)
    parser.add_argument("--pooling-method", default="last")
    parser.add_argument("--padding-side", default="left")
    parser.add_argument("--append-token", default="pad")
    parser.add_argument("--query-prompt-template", default="Instruct: {task_description}\nQuery:{query}")
    parser.add_argument("--evidence-minimum-f1", type=float, default=0.8)
    parser.add_argument("--generator-endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--generator-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--generator-system-prompt", default=(
        "Answer the question based on the given documents. "
        "Only give me the answer and do not output any other words."
    ))
    parser.add_argument("--generator-revision", default=None)
    parser.add_argument("--generator-cache", default="data/rag/generator_cache.sqlite3")
    parser.add_argument("--generator-top-k", type=int, default=10)
    parser.add_argument("--generator-max-input-length", type=int, default=2048)
    parser.add_argument("--generator-max-new-tokens", type=int, default=32)
    parser.add_argument("--generator-timeout-seconds", type=int, default=600)
    parser.add_argument("--generator-gpu-count", type=int, default=1)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--skip-hash-verification", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(evaluate(args))


if __name__ == "__main__":
    main()
