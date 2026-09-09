"""Evaluate checkpoints only on the fixed 5% candidate-manifest tuning split."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from rag.data import CandidateManifestDataset
from rag.encoder import FrozenQueryEncoder
from rag.index import FrozenDistributedIndex, sha256_file
from rag.metrics import passage_contains_answer
from rag.rewards import RetrievalRewardProvider
from rag.protocol import validate_query_index_protocol


def _batches(dataset, size):
    for start in range(0, len(dataset), size):
        yield [dataset[index] for index in range(start, min(start + size, len(dataset)))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", default="data/rag/candidates/nq_hotpotqa_train.jsonl")
    parser.add_argument("--index-manifest", default="data/rag/qwen3_e0_index/index_manifest.json")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--index-backend", choices=("faiss", "torch"), default="faiss")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--retrieval-k", type=int, default=20)
    parser.add_argument("--tuning-fraction", type=float, default=0.05)
    parser.add_argument("--tuning-seed", type=int, default=20260909)
    parser.add_argument("--query-max-length", type=int, default=128)
    args = parser.parse_args()

    dataset = CandidateManifestDataset(
        args.candidate_manifest,
        split="tuning",
        tuning_fraction=args.tuning_fraction,
        tuning_seed=args.tuning_seed,
    )
    index = FrozenDistributedIndex(args.index_manifest, backend=args.index_backend, device=args.device)
    candidate_metadata_path = Path(args.candidate_manifest).resolve().with_suffix(
        Path(args.candidate_manifest).suffix + ".manifest.json"
    )
    with open(candidate_metadata_path, "r", encoding="utf-8") as handle:
        candidate_metadata = json.load(handle)
    if candidate_metadata.get("index_manifest_sha256") != sha256_file(index.manifest_path):
        raise ValueError("Tuning candidates were mined against a different frozen index")
    encoder = FrozenQueryEncoder(
        args.model,
        adapter_path=args.checkpoint,
        revision=args.model_revision,
        device=args.device,
        max_length=args.query_max_length,
    )
    validate_query_index_protocol(
        index.manifest,
        model_name_or_path=args.model,
        resolved_model_revision=getattr(encoder.model.config, "_commit_hash", None),
        pooling_method="last",
        padding_side="left",
        append_token="pad",
    )
    providers = {
        mode: RetrievalRewardProvider(mode, args.retrieval_k)
        for mode in ("source_aware_mrr", "answer_mrr")
    }
    totals = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)
    for batch in _batches(dataset, args.batch_size):
        embeddings = encoder.encode([record["question"] for record in batch])
        _, ids = index.search(embeddings, args.retrieval_k)
        answer_mask = torch.zeros(ids.shape, dtype=torch.bool, device=ids.device)
        max_groups = max((len(record["evidence_passage_groups"]) for record in batch), default=0)
        evidence = torch.zeros(
            (len(batch), max_groups, args.retrieval_k), dtype=torch.bool, device=ids.device
        )
        for row_index, (record, row_ids) in enumerate(zip(batch, ids.detach().cpu().tolist())):
            texts = index.lookup_text(row_ids)
            answer_mask[row_index] = torch.tensor([
                passage_contains_answer(text, record["golden_answers"]) for text in texts
            ], device=ids.device)
            for group_index, passage_group in enumerate(record["evidence_passage_groups"]):
                evidence[row_index, group_index] = torch.tensor(
                    [ordinal in passage_group for ordinal in row_ids], device=ids.device
                )
        sources = [record["source"] for record in batch]
        group_counts = [len(record["evidence_passage_groups"]) for record in batch]
        rewards = {
            mode: provider(answer_mask, sources, evidence, group_counts)
            for mode, provider in providers.items()
        }
        for row_index, source in enumerate(sources):
            counts[source] += 1
            totals[source]["answer_recall_at_5"] += float(answer_mask[row_index, :5].any())
            totals[source]["answer_recall_at_20"] += float(answer_mask[row_index, :20].any())
            for mode, values in rewards.items():
                totals[source][mode] += float(values[row_index])

    per_source = {
        source: {
            "count": count,
            **{key: value / count for key, value in totals[source].items()},
        }
        for source, count in counts.items()
    }
    summary = {
        "format_version": 1,
        "artifact_origin": "E2Rank-RL",
        "selection_split": "source-wise deterministic hash 5%",
        "tuning_fraction": args.tuning_fraction,
        "tuning_seed": args.tuning_seed,
        "per_source": per_source,
        "macro_source_aware_mrr": sum(
            metrics["source_aware_mrr"] for metrics in per_source.values()
        ) / max(len(per_source), 1),
        "candidate_manifest_sha256": sha256_file(args.candidate_manifest),
        "index_manifest_sha256": sha256_file(args.index_manifest),
        "checkpoint": str(Path(args.checkpoint).resolve()),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
    index.close()
    print(output)


if __name__ == "__main__":
    main()
