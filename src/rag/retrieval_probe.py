"""Full-corpus retrieval probe over in-domain and out-of-domain QA slices.

``rag/tuning_eval.py`` re-ranks each query's stored depth-1000 candidate list.
Round 2 showed that measures the wrong thing: across four CL arms it rose from
0.5448 to 0.5718 while the real QA metric fell 0.024 below the untrained
baseline. Two defects compound there.

*Wrong operation.* Re-ranking a frozen pool is not retrieval. A query encoder
can reorder E0's top-1000 better and still pull a worse top-20 out of the 21M
corpus, because nothing in that measurement -- or in the training loss -- sees
the other ~21M passages.

*Wrong distribution.* The tuning split is held-out in queries but drawn from
nq/hotpotqa, the two sources being trained on. The round-2 collapse was
concentrated in the five datasets never trained on (-0.033 vs -0.001).

This probe fixes both: it runs a real ANN search and reports in-domain and
out-of-domain slices separately, so the held-out regression is visible while a
run is still training rather than only in the final evaluation.

It costs a corpus text read per retrieved passage -- roughly 38 ms each on
JuiceFS -- so it is deliberately small and sampled deterministically. At the
default 96 queries per dataset and k=20 it is a few thousand lookups per probe.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.distributed as dist
from transformers import TrainerCallback

from embedding_protocol import format_embedding_text, tokenize_embedding_texts
from rag.data import EVALUATION_SUITE, RAG_TASK_DESCRIPTION
from rag.metrics import passage_contains_answer


#: nq and hotpotqa are the training sources; the rest are never trained on and
#: are where round 2's regression concentrated.
IN_DOMAIN = ("nq", "hotpotqa")
OUT_OF_DOMAIN = ("triviaqa", "popqa", "2wikimultihopqa", "musique", "bamboogle")


def _stable_sample(records: Sequence[dict], limit: int, salt: str) -> list[dict]:
    """Deterministic subsample, stable across runs and checkpoints."""
    if len(records) <= limit:
        return list(records)
    keyed = sorted(
        records,
        key=lambda record: hashlib.sha256(
            f"{salt}:{record.get('id')}".encode("utf-8")
        ).digest(),
    )
    return keyed[:limit]


def load_probe_queries(
    manifest_path: str | Path,
    *,
    datasets: Sequence[str] | None = None,
    per_dataset: int = 96,
) -> list[dict[str, Any]]:
    """Read a fixed, reproducible slice of the evaluation suite."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    wanted = tuple(datasets) if datasets else IN_DOMAIN + OUT_OF_DOMAIN
    selected: list[dict[str, Any]] = []
    for key, entry in sorted(manifest.get("files", {}).items()):
        dataset = entry.get("dataset")
        if dataset not in wanted:
            continue
        # Only the split the evaluation suite uses. The manifest also carries
        # nq/train and hotpotqa/train, which are the training data itself --
        # including them would make the in-domain slice a training-set probe.
        expected = EVALUATION_SUITE.get(dataset)
        if expected is None or entry.get("split") != expected[0]:
            continue
        path = root / entry["path"]
        if not path.is_file():
            continue
        records = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
        for record in _stable_sample(records, per_dataset, key):
            answers = record.get("golden_answers") or []
            if not answers:
                continue
            selected.append(
                {
                    "dataset": dataset,
                    "scope": "in_domain" if dataset in IN_DOMAIN else "held_out",
                    "question": record["question"],
                    "answers": answers,
                }
            )
    return selected


@torch.no_grad()
def run_retrieval_probe(
    *,
    model: Any,
    index: Any,
    tokenizer: Any,
    queries: Sequence[dict[str, Any]],
    device: torch.device,
    retrieval_k: int = 20,
    batch_size: int = 32,
    query_max_length: int = 128,
    append_token: str = "pad",
    query_prompt_template: str = "Instruct: {task_description}\nQuery:{query}",
    rank: int = 0,
    world_size: int = 1,
) -> dict[str, float]:
    """Search the live index and score answer containment on the results."""
    encoder = model.module if hasattr(model, "module") else model
    shard = [queries[i] for i in range(rank, len(queries), world_size)]
    # index.search is collective over the sharded index, so every rank must call
    # it the same number of times. An uneven split would not raise -- it would
    # hang the whole job. Pad short ranks and drop the padding from the totals.
    per_rank = -(-len(queries) // world_size) if world_size > 1 else len(shard)
    real = len(shard)
    if world_size > 1 and real < per_rank:
        shard = shard + [shard[0] if shard else queries[0]] * (per_rank - real)

    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    was_training = encoder.training
    encoder.eval()
    try:
        for start in range(0, len(shard), batch_size):
            batch = shard[start : start + batch_size]
            texts = [
                format_embedding_text(
                    query_prompt_template, item["question"], task_description=RAG_TASK_DESCRIPTION
                )
                for item in batch
            ]
            tokenized = tokenize_embedding_texts(
                texts, tokenizer, append_token, max_length=query_max_length
            )
            inputs = {key: value.to(device) for key, value in dict(tokenized).items()}
            embeddings = encoder.encode_query(inputs)
            _, ids = index.search(embeddings.float(), retrieval_k)
            for offset, (row, item) in enumerate(zip(ids.detach().cpu().tolist(), batch)):
                if start + offset >= real:
                    continue  # padding: searched for collective symmetry, not scored
                passages = index.lookup_text(row)
                hits = [passage_contains_answer(text, item["answers"]) for text in passages]
                rank_of_first = next((i for i, hit in enumerate(hits) if hit), None)
                for scope in (item["scope"], "all"):
                    bucket = totals[scope]
                    bucket["count"] += 1.0
                    bucket["answer_recall_at_5"] += float(any(hits[:5]))
                    bucket["answer_recall_at_20"] += float(any(hits[:20]))
                    bucket["answer_mrr_at_10"] += (
                        0.0 if rank_of_first is None or rank_of_first >= 10
                        else 1.0 / (rank_of_first + 1)
                    )
    finally:
        encoder.train(was_training)

    scopes = ("in_domain", "held_out", "all")
    keys = ("count", "answer_recall_at_5", "answer_recall_at_20", "answer_mrr_at_10")
    packed = torch.tensor(
        [[totals[scope][key] for key in keys] for scope in scopes],
        device=device,
        dtype=torch.float64,
    )
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)

    metrics: dict[str, float] = {}
    for position, scope in enumerate(scopes):
        count = packed[position, 0].item()
        if count <= 0:
            continue
        for offset, key in enumerate(keys[1:], start=1):
            metrics[f"retrieval/{scope}/{key}"] = packed[position, offset].item() / count
    return metrics


class RAGRetrievalProbeCallback(TrainerCallback):
    """Log real-retrieval metrics, split in-domain vs held-out, at each save."""

    def __init__(
        self,
        *,
        model,
        index,
        tokenizer,
        queries,
        retrieval_k: int = 20,
        batch_size: int = 32,
        query_max_length: int = 128,
        append_token: str = "pad",
        query_prompt_template: str = "Instruct: {task_description}\nQuery:{query}",
    ):
        self.model = model
        self.index = index
        self.tokenizer = tokenizer
        self.queries = queries
        self.retrieval_k = retrieval_k
        self.batch_size = batch_size
        self.query_max_length = query_max_length
        self.append_token = append_token
        self.query_prompt_template = query_prompt_template
        self.trainer = None

    def _run(self, args, state, control, **kwargs) -> None:
        distributed = dist.is_available() and dist.is_initialized()
        metrics = run_retrieval_probe(
            model=kwargs.get("model", self.model),
            index=self.index,
            tokenizer=self.tokenizer,
            queries=self.queries,
            device=torch.device(args.device),
            retrieval_k=self.retrieval_k,
            batch_size=self.batch_size,
            query_max_length=self.query_max_length,
            append_token=self.append_token,
            query_prompt_template=self.query_prompt_template,
            rank=dist.get_rank() if distributed else 0,
            world_size=dist.get_world_size() if distributed else 1,
        )
        if self.trainer is not None:
            self.trainer.log({key: round(value, 6) for key, value in metrics.items()})
        if state.is_world_process_zero:
            print(f"[rag-retrieval-probe] step={state.global_step} {metrics}", flush=True)

    def on_train_begin(self, args, state, control, **kwargs):
        # At step 0 the encoder is still the initialization, so this records the
        # run's own untrained reference line. Every later probe is then readable
        # without importing a number from another run's evaluation.
        self._run(args, state, control, **kwargs)

    def on_save(self, args, state, control, **kwargs):
        self._run(args, state, control, **kwargs)

    def on_train_end(self, args, state, control, **kwargs):
        self._run(args, state, control, **kwargs)
