"""Evaluate a project-produced query encoder on the fixed seven-dataset suite."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
from tqdm import tqdm

from rag.candidates import (
    build_title_catalog,
    collect_hotpot_title_targets,
    map_hotpot_evidence,
)
from rag.data import EVALUATION_SUITE, iter_jsonl, validate_flashrag_record
from rag.encoder import FrozenQueryEncoder
from rag.generator import FrozenGeneratorClient
from rag.index import FrozenDistributedIndex, sha256_file
from rag.metrics import (
    exact_match,
    max_token_f1,
    normalize_answers,
    passage_contains_normalized_answer,
)
from rag.protocol import validate_query_index_protocol


MULTIHOP_SOURCES = {"hotpotqa", "2wikimultihopqa", "musique"}

RETRIEVAL_METRIC_KEYS = (
    "answer_recall_at_5", "answer_recall_at_10", "answer_recall_at_20",
    "answer_mrr_at_10", "answer_mrr_at_20",
)
MULTIHOP_METRIC_KEYS = (
    "evidence_group_recall_at_5", "evidence_group_recall_at_10", "evidence_group_recall_at_20",
    "all_evidence_success_at_5", "all_evidence_success_at_10", "all_evidence_success_at_20",
    "evidence_mapping_success",
)
GENERATION_METRIC_KEYS = ("generator_em", "generator_token_f1")


def _progress(total: int, description: str, unit: str = "query") -> tqdm:
    """A bar that stays readable when stderr is a log file rather than a terminal.

    A full evaluation is normally launched detached with its output redirected, and
    there tqdm's tenth-of-a-second refresh writes tens of thousands of updates that
    bury everything else in the file. Off a terminal it is slowed to once a minute,
    which is still often enough to see which phase is moving and how fast.
    """
    return tqdm(
        total=total,
        desc=description,
        unit=unit,
        dynamic_ncols=True,
        mininterval=0.5 if sys.stderr.isatty() else 60.0,
        file=sys.stderr,
    )


@contextmanager
def _stage(description: str) -> Iterator[None]:
    """Bracket a startup step that reads tens of gigabytes without reporting any of it.

    Hash verification and the faiss build each walk the whole index, and the encoder
    load can pull weights over the network; none of them can be made into a bar from
    here. Announcing them at least says which one the run is sitting in.
    """
    print(f"[eval_rag] {description}", file=sys.stderr, flush=True)
    started = time.perf_counter()
    yield
    print(
        f"[eval_rag] {description}: done in {time.perf_counter() - started:.1f}s",
        file=sys.stderr,
        flush=True,
    )


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


def _retrieve(
    args: argparse.Namespace,
    index: FrozenDistributedIndex,
    encoder: FrozenQueryEncoder,
    dataset_paths: dict[str, Path],
    title_catalog,
    output_dir: Path,
) -> tuple[dict, float]:
    """Write one retrieval record per query and return the per-source metrics.

    This is the only phase that needs the query encoder and the vector index, and
    on the seven-dataset suite it accounts for a couple of percent of a full
    evaluation. Running it to completion first is what lets the caller hand those
    devices back before the generator work begins.
    """
    summaries: dict[str, dict] = {}
    retrieval_seconds = 0.0
    for source, (split, expected_count) in EVALUATION_SUITE.items():
        rows = []
        retrieval_path = output_dir / f"{source}_{split}.retrieval.jsonl"
        with _progress(expected_count, f"retrieve {source}") as bar, open(
            retrieval_path, "w", encoding="utf-8"
        ) as retrieval_writer:
            for batch in _batches(iter_jsonl(dataset_paths[source]), args.batch_size):
                for record in batch:
                    validate_flashrag_record(record, source)
                retrieval_started = time.perf_counter()
                embeddings = encoder.encode([record["question"] for record in batch])
                scores, result_ids = index.search(embeddings, args.retrieval_k)
                retrieval_seconds += time.perf_counter() - retrieval_started
                ids_rows = result_ids.detach().cpu().tolist()
                score_rows = scores.detach().float().cpu().tolist()
                for record, ids, values in zip(batch, ids_rows, score_rows):
                    documents = index.lookup_records(ids)
                    # One answer set covers every passage of its query, so normalize
                    # it once here rather than once per retrieved passage.
                    needles = normalize_answers(record["golden_answers"])
                    answer_mask = [
                        passage_contains_normalized_answer(document["contents"], needles)
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
                    rows.append(row)
                bar.update(len(batch))
        if len(rows) != expected_count:
            raise ValueError(f"{source}/{split} has {len(rows)} rows, expected {expected_count}")
        metric_keys = list(RETRIEVAL_METRIC_KEYS)
        if source in MULTIHOP_SOURCES:
            metric_keys += list(MULTIHOP_METRIC_KEYS)
        summaries[source] = {key: _mean(rows, key) for key in metric_keys}
        summaries[source]["count"] = len(rows)
    return summaries, retrieval_seconds


def _generate(
    args: argparse.Namespace,
    generator: FrozenGeneratorClient,
    index: FrozenDistributedIndex,
    output_dir: Path,
    summaries: dict,
) -> None:
    """Answer every retrieved query and fold the answer metrics into ``summaries``.

    Retrieval has already finished, so the endpoint is the only thing left doing
    real work and the pipeline exists to keep it fed: requests stay outstanding
    across source boundaries, where the previous shape of this loop drained itself
    empty seven times. Batches are collected in submission order, which is what
    keeps each generation file in the order of its retrieval file.
    """
    pool = ThreadPoolExecutor(
        max_workers=args.generation_workers, thread_name_prefix="rag-generate"
    )
    in_flight: deque = deque()
    writers: dict[str, Any] = {}
    totals: dict[str, dict[str, float]] = {}
    # One bar for the whole phase rather than one per source: requests stay in
    # flight across source boundaries, so per-source bars would each sit finished
    # for a while before the answers behind them arrive.
    bar = _progress(sum(summary["count"] for summary in summaries.values()), "generate")

    def collect(future, source: str, pending: list[tuple[str, list, list[int]]]) -> None:
        writer = writers[source]
        running = totals[source]
        for generation, (query_id, golden_answers, top_ids) in zip(future.result(), pending):
            em = exact_match(generation, golden_answers)
            token_f1 = max_token_f1(generation, golden_answers)
            running["generator_em"] += em
            running["generator_token_f1"] += token_f1
            running["count"] += 1
            writer.write(json.dumps({
                "query_id": query_id,
                "source": source,
                "passage_ids": top_ids,
                "prediction": generation,
                "golden_answers": golden_answers,
                "em": em,
                "token_f1": token_f1,
            }, ensure_ascii=False) + "\n")
        bar.set_postfix_str(source, refresh=False)
        bar.update(len(pending))

    try:
        for source, (split, _) in EVALUATION_SUITE.items():
            writers[source] = open(
                output_dir / f"{source}_{split}.generation.jsonl", "w", encoding="utf-8"
            )
            totals[source] = {"generator_em": 0.0, "generator_token_f1": 0.0, "count": 0}
            retrieval_rows = iter_jsonl(output_dir / f"{source}_{split}.retrieval.jsonl")
            for batch in _batches(retrieval_rows, args.batch_size):
                requests = []
                pending = []
                for row in batch:
                    top_ids = row["passage_ids"][: args.generator_top_k]
                    documents = index.lookup_records(top_ids)
                    requests.append((row["query_id"], row["question"], top_ids, documents))
                    pending.append((row["query_id"], row["golden_answers"], top_ids))
                in_flight.append((pool.submit(generator.generate_batch, requests), source, pending))
                while len(in_flight) > args.generation_workers:
                    collect(*in_flight.popleft())
        while in_flight:
            collect(*in_flight.popleft())
    finally:
        pool.shutdown()
        bar.close()
        for writer in writers.values():
            writer.close()

    for source, running in totals.items():
        if running["count"] != summaries[source]["count"]:
            raise ValueError(
                f"{source} produced {running['count']} answers for "
                f"{summaries[source]['count']} retrieved queries"
            )
        for key in GENERATION_METRIC_KEYS:
            summaries[source][key] = running[key] / running["count"]


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

    with _stage(
        "loading frozen index"
        + ("" if args.skip_hash_verification else " (hashing every shard first)")
    ):
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
    # The multi-hop evidence metrics need the corpus rows behind the supporting
    # titles, and finding them parses all twenty-one million of them.
    with _progress(index.count, "scan corpus for evidence titles", unit="doc") as bar:
        title_catalog = build_title_catalog(
            str(corpus_path), evidence_titles, progress=bar.update
        )

    with _stage("loading query encoder"):
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
    summaries, retrieval_seconds = _retrieve(
        args, index, encoder, dataset_paths, title_catalog, output_dir
    )
    retrieval_wall_seconds = time.perf_counter() - started
    index_device_count = index.search_device_count

    # Everything past this point reads corpus text and talks to an HTTP endpoint,
    # so the vectors and the query encoder are handed back now. They occupy one
    # device each, or the whole visible set when a corpus is spread over several,
    # and on this suite they are busy for a few percent of the run.
    del encoder
    index.release_search_backend()
    # Both releases above drop their last reference, which is already enough for the
    # plain checkpoint path. The collection is here for the wrapped ones, where an
    # adapter or a SWIG proxy can leave the weights reachable through a cycle until
    # something walks it, and it has to happen before the cache is emptied below.
    gc.collect()
    peak_gpu_memory_bytes = 0
    if torch.cuda.is_available():
        # A high-water mark, so emptying the cache below does not move it.
        peak_gpu_memory_bytes = torch.cuda.max_memory_allocated()
        torch.cuda.empty_cache()

    if generator:
        _generate(args, generator, index, output_dir, summaries)
    elapsed = time.perf_counter() - started
    generation_wall_seconds = elapsed - retrieval_wall_seconds

    total_queries = sum(summary["count"] for summary in summaries.values())
    if total_queries != sum(value[1] for value in EVALUATION_SUITE.values()):
        raise RuntimeError(f"Evaluation suite count mismatch: {total_queries}")
    common_keys = list(RETRIEVAL_METRIC_KEYS)
    if generator:
        common_keys += list(GENERATION_METRIC_KEYS)
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
            # Charged per phase: the index devices are released once retrieval ends,
            # so billing them for the generation hours would overstate the cost of a
            # run by more than an order of magnitude. generator_gpu_count has to be
            # set to the endpoint's tensor-parallel size; nothing here can read it.
            "query_index_gpu_hours": retrieval_wall_seconds * index_device_count / 3600,
            "generator_gpu_hours": (
                generation_wall_seconds * args.generator_gpu_count / 3600 if generator else 0.0
            ),
            "gpu_hours": (
                retrieval_wall_seconds * index_device_count / 3600
                + (generation_wall_seconds * args.generator_gpu_count / 3600 if generator else 0.0)
            ),
            "peak_gpu_memory_gib": peak_gpu_memory_bytes / 1024**3,
            "retrieval_wall_seconds": retrieval_wall_seconds,
            "generation_wall_seconds": generation_wall_seconds if generator else 0.0,
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
    parser.add_argument(
        "--generator-gpu-count",
        type=int,
        default=1,
        help="Devices behind the endpoint, for cost accounting; match its tensor-parallel size",
    )
    parser.add_argument(
        "--generation-workers",
        type=int,
        default=2,
        help=(
            "Generation requests kept in flight. A second one covers the gap while a "
            "batch is being rendered and scored; more only widens the batch the server "
            "already merges, and overlapping requests let it mix prompts by arrival order"
        ),
    )
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--skip-hash-verification", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(evaluate(args))


if __name__ == "__main__":
    main()
