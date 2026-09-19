#!/usr/bin/env python3
"""E0 reference line for the in-training tuning probe.

``rag/tuning_eval.py`` re-ranks each held-out query's frozen depth-1000
candidate list and reports answer recall/MRR off the stored masks. Those numbers
are only interpretable against two reference lines:

* **E0** -- the frozen retrieval order, i.e. what the probe reports at step 0
  before training changes anything. A run below this is actively regressing,
  which is exactly what every round-1 G3 arm did on the real evaluation.
* **Oracle** -- perfect re-ranking of the same depth-1000 pool. The gap between
  E0 and oracle is the headroom a query-only re-ranker can possibly capture;
  nothing beyond it is reachable without changing the frozen index.

CPU only, no GPU and no model: it reads the manifest sequentially and scores the
stored ordering directly.

    PYTHONPATH=src python scripts/rag_tuning_reference.py
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag.data import belongs_to_tuning_split  # noqa: E402


CUTOFFS = (5, 10, 20)


def _reciprocal_rank(flags, cutoff):
    for position, flag in enumerate(flags[:cutoff], start=1):
        if flag:
            return 1.0 / position
    return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/rag/candidates/nq_hotpotqa_train.jsonl")
    parser.add_argument("--split", default="tuning", choices=("tuning", "train", "full"))
    parser.add_argument("--tuning-fraction", type=float, default=0.05)
    parser.add_argument("--tuning-seed", type=int, default=20260909)
    parser.add_argument("--limit", type=int, default=None, help="Stop after N matching queries")
    args = parser.parse_args()

    totals = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)
    with open(args.manifest, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            in_tuning = belongs_to_tuning_split(
                record["source"], record["query_id"], args.tuning_fraction, args.tuning_seed
            )
            if args.split == "tuning" and not in_tuning:
                continue
            if args.split == "train" and in_tuning:
                continue
            source = record["source"]
            answer = record["answer_positive_mask"]
            qrel = record["training_positive_mask"]
            counts[source] += 1
            bucket = totals[source]
            for cutoff in CUTOFFS:
                bucket[f"answer_recall_at_{cutoff}"] += float(any(answer[:cutoff]))
            bucket["answer_mrr_at_10"] += _reciprocal_rank(answer, 10)
            bucket["qrel_mrr_at_10"] += _reciprocal_rank(qrel, 10)
            # Oracle: perfect re-ranking of the same depth-1000 pool.
            bucket["oracle_answer_recall_at_5"] += float(any(answer))
            bucket["oracle_answer_mrr_at_10"] += float(any(answer))
            bucket["oracle_qrel_mrr_at_10"] += float(any(qrel))
            bucket["answer_positives"] += float(sum(answer))
            bucket["qrel_positives"] += float(sum(qrel))
            if args.limit and sum(counts.values()) >= args.limit:
                break

    if not counts:
        print("No matching queries", file=sys.stderr)
        return 1

    keys = [
        *(f"answer_recall_at_{c}" for c in CUTOFFS),
        "answer_mrr_at_10",
        "qrel_mrr_at_10",
        "oracle_answer_recall_at_5",
        "oracle_answer_mrr_at_10",
        "oracle_qrel_mrr_at_10",
        "answer_positives",
        "qrel_positives",
    ]
    print(f"\nE0 frozen-order reference, split={args.split} "
          f"(what rag/tuning_eval.py reports before training)\n")
    width = max(len(key) for key in keys) + 2
    header = "metric".ljust(width) + "".join(f"{s:>12}" for s in sorted(counts)) + f"{'macro':>12}"
    print(header)
    print("-" * len(header))
    for key in keys:
        cells = ""
        values = []
        for source in sorted(counts):
            value = totals[source][key] / counts[source]
            values.append(value)
            cells += f"{value:>12.4f}"
        cells += f"{sum(values) / len(values):>12.4f}"
        print(key.ljust(width) + cells)
    print("\n" + "queries".ljust(width) + "".join(f"{counts[s]:>12d}" for s in sorted(counts))
          + f"{sum(counts.values()):>12d}")
    print(
        "\nHeadroom is oracle_* minus the matching E0 row: that is everything a "
        "query-only\nre-ranker can win on this pool. A probe below the E0 row is regressing."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
