"""One-shot corpus scan that caches alignment intermediates for HotpotQA/NQ qrels.

Rationale
---------
`rag.build_qrels` rescans the 14 GB frozen corpus for every F1 threshold. This helper
scans ONCE and writes a compact cache from which qrels for *any* HotpotQA F1 threshold
can be rebuilt in seconds (see ``scripts/rag_build_qrels_from_cache.py``), without ever
touching the corpus again.

Cache contents (under ``--cache-dir``):
  - ``nq_matches.json``     : {dpr_strict_key: [corpus_ordinal, ...]}   (threshold-independent)
  - ``hotpot_best.jsonl``   : one row per (title, sentence) fact whose title matched a
                              corpus passage: {title, sentence, best_f1, best_ordinal}
  - ``cache_manifest.json`` : input hashes + corpus sha256 so downstream can verify
                              the cache matches the frozen corpus/inputs.

This does the exact same matching as ``build_qrels.align_to_corpus`` but records the
best window-token-F1 per fact instead of applying a fixed cutoff, so it is a superset
sufficient to reproduce the strict (0.8) qrels and any looser variant.

Run it ONLY when the GPU corpus encode is idle: a full-width parallel scan competes with
the encoder's CPU-side tokenization. Use ``E2RANK_QRELS_WORKERS`` to cap parallelism.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "src")

from rag.build_qrels import (  # noqa: E402
    EXPECTED_DPR_NQ_TRAIN,
    EXPECTED_HOTPOTQA_TRAIN,
    _chunk_byte_ranges,
    _corpus_sha256,
    load_dpr_nq,
    load_hotpotqa,
    passage_fingerprint,
    sha256_file,
    split_corpus_contents,
)
from rag.metrics import best_window_token_f1, normalize_answer  # noqa: E402

_STATE: dict = {}


def _init(nq_strict, nq_relaxed, facts_by_title):
    _STATE["nq_strict"] = nq_strict
    _STATE["nq_relaxed"] = nq_relaxed
    _STATE["facts_by_title"] = facts_by_title


def _scan(args):
    corpus_path, start, end = args
    nq_strict = _STATE["nq_strict"]
    nq_relaxed = _STATE["nq_relaxed"]
    facts_by_title = _STATE["facts_by_title"]
    nq_matches: dict[str, list[int]] = defaultdict(list)
    hotpot_best: dict[tuple[str, str], tuple[float, int]] = {}
    with open(corpus_path, "rb") as handle:
        handle.seek(start)
        pos = start
        while pos < end:
            raw = handle.readline()
            if not raw:
                break
            pos += len(raw)
            rec = json.loads(raw)
            ordinal = int(rec["id"])
            title, text = split_corpus_contents(rec["contents"])
            strict_key = passage_fingerprint(title, text)
            if strict_key in nq_strict:
                for tk in nq_strict[strict_key]:
                    nq_matches[tk].append(ordinal)
            else:
                relaxed_key = passage_fingerprint(title, text, relaxed=True)
                if relaxed_key in nq_relaxed:
                    for tk in nq_relaxed[relaxed_key]:
                        nq_matches[tk].append(ordinal)
            ntitle = normalize_answer(title)
            sentences = facts_by_title.get(ntitle)
            if not sentences:
                continue
            ncontents = normalize_answer(rec["contents"])
            for sentence in sentences:
                score = 1.0 if sentence in ncontents else best_window_token_f1(sentence, rec["contents"])
                key = (ntitle, sentence)
                cur = hotpot_best.get(key)
                if cur is None or score > cur[0] or (score == cur[0] and ordinal < cur[1]):
                    hotpot_best[key] = (score, ordinal)
    return dict(nq_matches), hotpot_best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-path", default="data/rag/corpus/wiki18_100w.jsonl")
    parser.add_argument("--flashrag-manifest", default="data/rag/flashrag_manifest.json")
    parser.add_argument("--dpr-nq-train", default="data/rag/raw/biencoder-nq-train.json.gz")
    parser.add_argument("--cache-dir", default="data/rag/qrels_alignment_cache")
    args = parser.parse_args()

    corpus_path = Path(args.corpus_path)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    with open(args.flashrag_manifest, encoding="utf-8") as handle:
        source_manifest = json.load(handle)
    hotpot_entry = source_manifest["files"]["hotpotqa/train"]
    hotpot_path = Path(args.flashrag_manifest).parent / hotpot_entry["path"]

    print("Loading DPR NQ + HotpotQA labels...", flush=True)
    nq_rows, _ = load_dpr_nq(Path(args.dpr_nq_train), EXPECTED_DPR_NQ_TRAIN)
    hotpot_rows, _ = load_hotpotqa(hotpot_path, EXPECTED_HOTPOTQA_TRAIN)

    nq_strict: dict[str, set[str]] = defaultdict(set)
    nq_relaxed: dict[str, set[str]] = defaultdict(set)
    for row in nq_rows:
        for t in row["positive_targets"]:
            nq_strict[t["strict_key"]].add(t["strict_key"])
            nq_relaxed[t["relaxed_key"]].add(t["strict_key"])
    facts_by_title: dict[str, set[str]] = defaultdict(set)
    for row in hotpot_rows:
        for title, sentence in row["fact_keys"]:
            facts_by_title[title].add(sentence)

    workers = int(os.environ.get("E2RANK_QRELS_WORKERS", os.cpu_count() or 1))
    chunks = _chunk_byte_ranges(corpus_path, workers)
    print(f"Scanning corpus once with {workers} workers over {len(chunks)} chunks...", flush=True)
    ctx = mp.get_context("fork")
    with ctx.Pool(
        min(workers, len(chunks)),
        initializer=_init,
        initargs=(dict(nq_strict), dict(nq_relaxed), dict(facts_by_title)),
    ) as pool:
        results = pool.map(_scan, chunks)

    nq_matches: dict[str, list[int]] = defaultdict(list)
    hotpot_best: dict[tuple[str, str], tuple[float, int]] = {}
    for chunk_nq, chunk_hp in results:
        for tk, ords in chunk_nq.items():
            nq_matches[tk].extend(ords)
        for key, val in chunk_hp.items():
            cur = hotpot_best.get(key)
            score, ordinal = val
            if cur is None or score > cur[0] or (score == cur[0] and ordinal < cur[1]):
                hotpot_best[key] = val
    for tk in nq_matches:
        nq_matches[tk] = sorted(set(nq_matches[tk]))

    print(f"scan done: nq_targets_hit={len(nq_matches)} hotpot_facts_hit={len(hotpot_best)}", flush=True)

    (cache_dir / "nq_matches.json").write_text(
        json.dumps(nq_matches, separators=(",", ":")), encoding="utf-8"
    )
    with (cache_dir / "hotpot_best.jsonl").open("w", encoding="utf-8") as handle:
        for (title, sentence), (f1, ordinal) in hotpot_best.items():
            handle.write(json.dumps(
                {"title": title, "sentence": sentence, "best_f1": f1, "best_ordinal": ordinal},
                ensure_ascii=False, separators=(",", ":"),
            ) + "\n")

    manifest = {
        "artifact_type": "rag_qrels_alignment_cache",
        "corpus_path": str(corpus_path),
        "corpus_sha256": _corpus_sha256(corpus_path),
        "dpr_nq_train_sha256": sha256_file(Path(args.dpr_nq_train)),
        "hotpotqa_train_sha256": hotpot_entry["sha256"],
        "flashrag_resolved_revision": source_manifest.get("resolved_revision"),
        "nq_targets_hit": len(nq_matches),
        "hotpot_facts_hit": len(hotpot_best),
    }
    (cache_dir / "cache_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"cache written to {cache_dir}", flush=True)


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
