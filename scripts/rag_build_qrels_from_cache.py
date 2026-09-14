"""Rebuild HotpotQA/NQ binary qrels for any F1 threshold from the alignment cache.

Consumes the cache produced by ``scripts/rag_export_alignment_cache.py`` and reuses
``rag.build_qrels`` label loading + ``materialize_qrels`` so the output format, NQ
handling, manifest, and unresolved list are IDENTICAL to a direct ``build_qrels`` run —
only the (expensive) corpus scan is replaced by a cache lookup, so a new threshold costs
seconds instead of a full 14 GB pass.

Verified property: with ``--hotpot-minimum-f1 0.8`` this reproduces the strict qrels
byte-for-byte (same qrels_sha256) as long as the cache was built from the same corpus.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "src")

from rag.build_qrels import (  # noqa: E402
    EXPECTED_DPR_NQ_TRAIN,
    EXPECTED_HOTPOTQA_TRAIN,
    _atomic_jsonl,
    load_dpr_nq,
    load_hotpotqa,
    materialize_qrels,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="data/rag/qrels_alignment_cache")
    parser.add_argument("--flashrag-manifest", default="data/rag/flashrag_manifest.json")
    parser.add_argument("--dpr-nq-train", default="data/rag/raw/biencoder-nq-train.json.gz")
    parser.add_argument("--hotpot-minimum-f1", type=float, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    with open(args.flashrag_manifest, encoding="utf-8") as handle:
        source_manifest = json.load(handle)
    hotpot_entry = source_manifest["files"]["hotpotqa/train"]
    hotpot_path = Path(args.flashrag_manifest).parent / hotpot_entry["path"]
    cache_manifest = json.loads((cache_dir / "cache_manifest.json").read_text())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "nq_hotpotqa_train.jsonl"
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    unresolved_path = output_dir / "nq_hotpotqa_train.unresolved.jsonl"
    for path in (output_path, manifest_path, unresolved_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists: {path}; pass --overwrite")

    nq_rows, nq_input_stats = load_dpr_nq(Path(args.dpr_nq_train), EXPECTED_DPR_NQ_TRAIN)
    hotpot_rows, hotpot_input_stats = load_hotpotqa(hotpot_path, EXPECTED_HOTPOTQA_TRAIN)

    nq_matches = {k: list(v) for k, v in json.loads((cache_dir / "nq_matches.json").read_text()).items()}

    thr = args.hotpot_minimum_f1
    hotpot_matches: dict[tuple[str, str], tuple[float, int]] = {}
    with (cache_dir / "hotpot_best.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["best_f1"] >= thr:
                hotpot_matches[(rec["title"], rec["sentence"])] = (rec["best_f1"], rec["best_ordinal"])

    rows, output_stats, unresolved = materialize_qrels(nq_rows, hotpot_rows, nq_matches, hotpot_matches)
    if not rows:
        raise RuntimeError("No qrel-aligned records produced")

    _atomic_jsonl(output_path, rows)
    _atomic_jsonl(unresolved_path, unresolved)
    manifest = {
        "format_version": 1,
        "artifact_origin": "E2Rank-RL",
        "artifact_type": "rag_binary_passage_qrels",
        "built_from_cache": True,
        "hotpot_minimum_f1": thr,
        "label_policy": {
            "relevance": "binary",
            "nq": "DPR positive_ctxs with score == 1000 only",
            "hotpotqa": f"all supporting facts, title-exact + window-token-F1 >= {thr}",
            "retrieval_time_relabeling": False,
        },
        "inputs": {
            "dpr_nq_train": {"path": str(Path(args.dpr_nq_train)), "sha256": sha256_file(Path(args.dpr_nq_train))},
            "hotpotqa_train": {"path": str(hotpot_path), "sha256": hotpot_entry["sha256"]},
            "corpus": {"path": cache_manifest["corpus_path"], "sha256": cache_manifest["corpus_sha256"]},
        },
        "statistics": {
            "input": {"nq": nq_input_stats, "hotpotqa": hotpot_input_stats},
            "output": output_stats,
            "total_kept": len(rows),
            "total_unresolved": len(unresolved),
        },
        "qrels_path": output_path.name,
        "qrels_sha256": sha256_file(output_path),
        "unresolved_path": unresolved_path.name,
        "unresolved_sha256": sha256_file(unresolved_path),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"kept={len(rows)} unresolved={len(unresolved)} -> {output_path}")
    print(f"qrels_sha256={manifest['qrels_sha256']}")


if __name__ == "__main__":
    main()
