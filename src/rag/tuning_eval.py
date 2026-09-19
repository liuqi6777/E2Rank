"""In-training monitoring for RAG query-encoder runs.

``src/eval_rag_tuning.py`` is the offline probe: it runs a real ANN search and
then reads every retrieved passage back out of the corpus to test it for the
answer string. On this cluster the corpus lives on JuiceFS, where a random row
costs roughly 38 ms, so 5.4k queries x 10 passages is over half an hour --
unusable between training steps.

This callback measures the same quantities without touching corpus text. The
candidate manifest already stores, per query, the frozen top-1000 passage
ordinals and their ``answer_positive_mask``. Re-ranking that fixed list with the
live query encoder and reading the stored mask yields answer recall and MRR
directly, using only the vector lookups the training step already performs.

The numbers are therefore re-ranking metrics over a depth-1000 pool, not
full-corpus retrieval: they are bounded by the frozen index's recall@1000 and
will read a little higher than ``eval_rag.py``. They move with the real metric,
which is what a live regression signal needs -- and regression is the live risk,
since every previous G3 run finished below its own initialization.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import TrainerCallback

from fixed_corpus.models import FrozenCandidateScorer


RECALL_CUTOFFS = (5, 10, 20)
MRR_CUTOFF = 10


def _reciprocal_rank(hits: torch.Tensor, cutoff: int) -> torch.Tensor:
    """First-relevant reciprocal rank over a [batch, depth] boolean hit matrix."""
    depth = min(cutoff, hits.size(-1))
    ranks = torch.arange(1, depth + 1, device=hits.device, dtype=torch.float32)
    scored = torch.where(hits[:, :depth], ranks.reciprocal(), torch.zeros_like(ranks))
    return scored.max(dim=-1).values


@torch.no_grad()
def score_tuning_split(
    *,
    model: Any,
    index: Any,
    dataset: Any,
    collator: Any,
    batch_size: int,
    device: torch.device,
    rank: int = 0,
    world_size: int = 1,
) -> dict[str, float]:
    """Re-rank each tuning query's frozen candidates and summarise the ordering."""
    scorer = FrozenCandidateScorer(index)
    encoder = model.module if hasattr(model, "module") else model
    # Both the supervised and RL wrappers expose encode_query over the backbone.
    encode = encoder.encode_query

    shard = list(range(rank, len(dataset), world_size))
    loader = DataLoader(
        torch.utils.data.Subset(dataset, shard),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    totals: dict[str, torch.Tensor] = defaultdict(
        lambda: torch.zeros((), device=device, dtype=torch.float64)
    )
    per_source: dict[str, dict[str, torch.Tensor]] = {}
    was_training = encoder.training
    encoder.eval()
    try:
        for batch in loader:
            query = {key: value.to(device) for key, value in batch["query"].items()}
            ordinals = batch["candidate_passage_ids"].to(device)
            answer = batch["answer_positive_mask"].to(device)
            qrel = batch["training_positive_mask"].to(device)
            valid = ordinals >= 0
            scores = scorer(encode(query), ordinals, valid).scores
            order = scores.argsort(dim=-1, descending=True)
            answer_hits = answer.gather(1, order)
            qrel_hits = qrel.gather(1, order)

            rows = {}
            for cutoff in RECALL_CUTOFFS:
                rows[f"answer_recall_at_{cutoff}"] = answer_hits[:, :cutoff].any(dim=-1).float()
            rows[f"answer_mrr_at_{MRR_CUTOFF}"] = _reciprocal_rank(answer_hits, MRR_CUTOFF)
            rows[f"qrel_mrr_at_{MRR_CUTOFF}"] = _reciprocal_rank(qrel_hits, MRR_CUTOFF)

            totals["count"] += float(ordinals.size(0))
            for key, value in rows.items():
                totals[key] += value.sum().double()
            for position, source in enumerate(batch["sources"]):
                bucket = per_source.setdefault(
                    source,
                    defaultdict(lambda: torch.zeros((), device=device, dtype=torch.float64)),
                )
                bucket["count"] += 1.0
                for key, value in rows.items():
                    bucket[key] += value[position].double()
    finally:
        encoder.train(was_training)

    # Sources are a fixed, tiny vocabulary; sorting keeps the reduction order
    # identical on every rank.
    sources = sorted(per_source)
    if world_size > 1:
        names = [None] * world_size
        dist.all_gather_object(names, sources)
        sources = sorted({name for group in names for name in group})

    keys = ["count"] + [f"answer_recall_at_{c}" for c in RECALL_CUTOFFS]
    keys += [f"answer_mrr_at_{MRR_CUTOFF}", f"qrel_mrr_at_{MRR_CUTOFF}"]
    flat = [totals[key] for key in keys]
    for source in sources:
        bucket = per_source.get(source)
        flat += [
            bucket[key] if bucket is not None else torch.zeros((), device=device, dtype=torch.float64)
            for key in keys
        ]
    packed = torch.stack(flat)
    if world_size > 1:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)

    values = packed.tolist()
    metrics: dict[str, float] = {}

    def _emit(prefix: str, offset: int) -> None:
        count = values[offset]
        if count <= 0:
            return
        metrics[f"{prefix}count"] = count
        for position, key in enumerate(keys[1:], start=1):
            metrics[f"{prefix}{key}"] = values[offset + position] / count

    _emit("tuning/", 0)
    for position, source in enumerate(sources):
        _emit(f"tuning/{source}/", (position + 1) * len(keys))
    return metrics


class RAGTuningEvalCallback(TrainerCallback):
    """Log re-ranking metrics on the held-out tuning split at each save point."""

    def __init__(self, *, model, index, dataset, collator, batch_size: int = 64):
        self.model = model
        self.index = index
        self.dataset = dataset
        self.collator = collator
        self.batch_size = batch_size
        #: Set by the caller once the Trainer exists, so metrics reach W&B
        #: through the same path as the training scalars.
        self.trainer = None

    def _run(self, args, state, control, **kwargs) -> None:
        distributed = dist.is_available() and dist.is_initialized()
        metrics = score_tuning_split(
            model=kwargs.get("model", self.model),
            index=self.index,
            dataset=self.dataset,
            collator=self.collator,
            batch_size=self.batch_size,
            device=torch.device(args.device),
            rank=dist.get_rank() if distributed else 0,
            world_size=dist.get_world_size() if distributed else 1,
        )
        metrics.pop("tuning/count", None)
        if self.trainer is not None:
            # Every rank must call log(): Trainer.log is collective under
            # DeepSpeed and only rank zero forwards to the reporters.
            self.trainer.log({key: round(value, 6) for key, value in metrics.items()})
        if state.is_world_process_zero:
            print(f"[rag-tuning-eval] step={state.global_step} {metrics}", flush=True)

    def on_save(self, args, state, control, **kwargs):
        self._run(args, state, control, **kwargs)

    def on_train_end(self, args, state, control, **kwargs):
        self._run(args, state, control, **kwargs)
