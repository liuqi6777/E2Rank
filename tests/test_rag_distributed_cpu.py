"""Multi-process CPU checks for the RAG collective paths.

The cross-device negative pool and the tuning probe both call collectives
inside forward. A shape mismatch there does not raise -- it deadlocks, and a
deadlock costs a whole 8-GPU queue slot before anyone notices. These run the
real code over gloo on CPU so the failure shows up here instead.

    PYTHONPATH=src python tests/test_rag_distributed_cpu.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import types

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn


CORPUS = 256
DIM = 16
WORLD = 3


class StubIndex:
    def __init__(self):
        generator = torch.Generator().manual_seed(7)
        self.vectors = torch.randn((CORPUS, DIM), generator=generator)

    def lookup_embeddings(self, ordinals, route_ids=None):
        return self.vectors[ordinals.clamp(0, CORPUS - 1)]


class StubBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(512, DIM)
        self.config = types.SimpleNamespace(hidden_size=DIM)

    def forward(self, input_ids=None, attention_mask=None, **_):
        return types.SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def _rank_batch(rank, batch_size=4, depth=12):
    generator = torch.Generator().manual_seed(100 + rank)
    # Disjoint ordinal ranges per rank so pooled candidates are identifiable.
    base = rank * 50
    ordinals = base + torch.randint(0, 40, (batch_size, depth), generator=generator)
    qrel = torch.zeros((batch_size, depth), dtype=torch.bool)
    qrel[:, 1] = True
    answer = torch.zeros((batch_size, depth), dtype=torch.bool)
    answer[:, :3] = True
    evidence = torch.zeros((batch_size, depth), dtype=torch.bool)
    evidence[:, 2] = True
    return dict(
        query={
            "input_ids": torch.randint(0, 512, (batch_size, 5), generator=generator),
            "attention_mask": torch.ones((batch_size, 5), dtype=torch.long),
        },
        candidate_passage_ids=ordinals,
        training_positive_mask=qrel,
        answer_positive_mask=answer,
        evidence_positive_mask=evidence,
    )


def _worker(rank, world_size, rendezvous, result_queue):
    try:
        dist.init_process_group(
            backend="gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world_size
        )
        from rag.models import RAGSupervisedModel
        from rag.negatives import build_cross_query_pool
        from rag.shortlist_rl import RAGShortlistRLModel

        batch = _rank_batch(rank)
        ordinals = batch["candidate_passage_ids"]
        valid = torch.ones_like(ordinals, dtype=torch.bool)
        judged = batch["training_positive_mask"] | batch["answer_positive_mask"]

        # 1. The pool must be identical on every rank and must span all ranks.
        pool, mask = build_cross_query_pool(
            candidate_ordinals=ordinals,
            candidate_mask=valid,
            judged_mask=judged,
            pool_size=5,
            include_negatives=True,
            cross_device=True,
        )
        expected_width = world_size * ordinals.size(0) * 5
        assert pool.numel() == expected_width, f"pool width {pool.numel()} != {expected_width}"
        assert mask.shape == (ordinals.size(0), expected_width)
        # Candidates from other ranks must be usable as negatives.
        foreign = (pool // 50) != rank
        assert bool((mask & foreign).any()), "rank sees no foreign negatives"
        # No query may borrow from its own row.
        own_rows = slice(rank * ordinals.size(0) * 5, (rank + 1) * ordinals.size(0) * 5)
        own_block = mask[:, own_rows].reshape(ordinals.size(0), ordinals.size(0), 5)
        assert not bool(own_block.diagonal(dim1=0, dim2=1).any()), "query borrowed from itself"

        # 2. Contrastive forward with the cross-device pool.
        torch.manual_seed(0)
        model = RAGSupervisedModel(
            StubBackbone(),
            StubIndex(),
            objective="infonce",
            pooling_method="mean",
            relevance_scheme="answer_masked",
            use_in_batch_candidates=True,
            in_batch_include_negatives=True,
            cross_device_negatives=True,
            in_batch_pool_size=5,
        )
        loss = model(**batch).loss
        assert torch.isfinite(loss), "cross-device InfoNCE loss is not finite"
        loss.backward()

        # 3. Shortlist RL with a cross-device pool, both estimators.
        for estimator in ("score_function", "conditional_projection"):
            torch.manual_seed(1)
            rl = RAGShortlistRLModel(
                StubBackbone(),
                StubIndex(),
                relevance_scheme="graded",
                reward_k=5,
                slate_size=6,
                shortlist_size=5,
                group_size=4,
                kappa=40.0,
                pooling_method="mean",
                gradient_estimator=estimator,
                cross_device_negatives=True,
            )
            output = rl(**batch)
            assert torch.isfinite(output.loss), f"{estimator} loss not finite"
            output.loss.backward()

        # 4. Tuning probe reduction: every rank must agree on the result.
        from rag.tuning_eval import score_tuning_split

        class _Dataset:
            def __init__(self, size):
                self.size = size

            def __len__(self):
                return self.size

            def __getitem__(self, index):
                generator = torch.Generator().manual_seed(index)
                return {
                    "question": f"q{index}",
                    "source": "nq" if index % 2 else "hotpotqa",
                    "candidate_passage_ids": (
                        torch.randint(0, CORPUS, (8,), generator=generator).tolist()
                    ),
                    "answer_positive_mask": [index % 3 == 0] * 4 + [False] * 4,
                    "training_positive_mask": [False, True] + [False] * 6,
                    "evidence_group_ids": [[]] * 8,
                    "golden_answers": ["a"],
                    "query_id": f"id{index}",
                    "evidence_passage_groups": [],
                }

        def collate(records):
            return {
                "query": {
                    "input_ids": torch.zeros((len(records), 5), dtype=torch.long),
                    "attention_mask": torch.ones((len(records), 5), dtype=torch.long),
                },
                "candidate_passage_ids": torch.tensor(
                    [r["candidate_passage_ids"] for r in records], dtype=torch.long
                ),
                "answer_positive_mask": torch.tensor(
                    [r["answer_positive_mask"] for r in records], dtype=torch.bool
                ),
                "training_positive_mask": torch.tensor(
                    [r["training_positive_mask"] for r in records], dtype=torch.bool
                ),
                "sources": [r["source"] for r in records],
            }

        # 11 rows over 3 ranks is deliberately ragged: the reduction must still
        # agree, which is the property a padded all_gather would silently break.
        metrics = score_tuning_split(
            model=model,
            index=StubIndex(),
            dataset=_Dataset(11),
            collator=collate,
            batch_size=2,
            device=torch.device("cpu"),
            rank=rank,
            world_size=world_size,
        )
        assert "tuning/answer_mrr_at_10" in metrics, metrics.keys()
        assert "tuning/nq/answer_recall_at_5" in metrics, metrics.keys()
        gathered = [None] * world_size
        dist.all_gather_object(gathered, metrics)
        for other in gathered:
            assert other.keys() == metrics.keys(), "ranks disagree on metric keys"
            for key in metrics:
                assert abs(other[key] - metrics[key]) < 1e-9, f"ranks disagree on {key}"

        result_queue.put((rank, "ok", float(loss.item()), metrics["tuning/answer_mrr_at_10"]))
    except Exception as exc:  # noqa: BLE001
        import traceback

        result_queue.put((rank, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}", 0.0, 0.0))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(1)
    context = mp.get_context("spawn")
    queue = context.Queue()
    with tempfile.TemporaryDirectory() as directory:
        rendezvous = os.path.join(directory, "rendezvous")
        processes = [
            context.Process(target=_worker, args=(rank, WORLD, rendezvous, queue))
            for rank in range(WORLD)
        ]
        for process in processes:
            process.start()
        results = []
        for _ in range(WORLD):
            try:
                results.append(queue.get(timeout=300))
            except Exception:
                break
        for process in processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                print("FAIL: a rank deadlocked (collective shape mismatch?)")
                return 1

    failures = 0
    for rank, status, loss, mrr in sorted(results):
        if status == "ok":
            print(f"ok   rank {rank}: infonce_loss={loss:.4f} tuning_mrr={mrr:.4f}")
        else:
            failures += 1
            print(f"FAIL rank {rank}: {status}")
    if len(results) != WORLD:
        print(f"FAIL: only {len(results)}/{WORLD} ranks reported")
        return 1
    print(f"\n{WORLD - failures}/{WORLD} ranks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
