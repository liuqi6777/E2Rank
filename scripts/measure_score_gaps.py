"""Can the policy reorder the slate at all? Measure it before booking GPUs for RL.

A ranking reward only varies when a sampled perturbation flips the order of two candidates,
so whether the RL stage can work at all is a property of the checkpoint's score geometry --
measurable in minutes, rather than discovered after a training run.

For a query-side vMF policy, the score difference of two documents under one draw has

    mean  A_d(kappa) * g        std  ||delta_perp|| * sqrt(A_d(kappa)/kappa)

where g is the deterministic gap and delta_perp is the part of d_i - d_j orthogonal to the
query mean (the parallel part is not perturbed, since vMF noise is tangential). Hence

    P(flip) = Phi(-z),   z = g * sqrt(A_d * kappa) / ||delta_perp||

and the gap a pair must exceed to survive a rollout is g* = ||delta_perp|| / sqrt(A_d*kappa).

Two things follow, and only the first is obvious. (i) g* is the interpretable axis, not kappa
and not A_d: A_d(755) = 0.53 says nothing about whether a slate reorders, while g* = 0.06 says
that pairs closer than about six score-points in the second decimal are in play. (ii) The
absolute per-document perturbation saturates at 1/sqrt(d) as kappa -> 0, but z does NOT
saturate -- it scales as sqrt(A_d*kappa), monotonically. Low kappa therefore does not buy more
useful exploration past a point: everything reorders, but at random, and the reward decorrelates
from the ordering the policy mean actually induces.

Read the output as a prediction of degenerate_frac: a slate whose adjacent pairs all have
P(flip) ~ 0 returns the same reward for every rollout in the group, and contributes no
gradient no matter how G or kappa are set.

    uv run python scripts/measure_score_gaps.py \
        --model checkpoints/stage1-0.6b-s42-merged \
        --data_path /path/to/bge-m3-data --num_batches 16
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from config import DataArguments
from embedding_data import EmbeddingDataCollator, EmbeddingDataset, build_slate_inputs
from grpo import bessel_ratio, pool_last_token_embedding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="merged checkpoint or base model id")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--slate_size", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_batches", type=int, default=16)
    parser.add_argument("--q_max_len", type=int, default=512)
    parser.add_argument("--d_max_len", type=int, default=1024)
    parser.add_argument("--file_glob", default="*_len-0-500.jsonl")
    parser.add_argument("--include_sources", default=None)
    parser.add_argument(
        "--kappa",
        type=float,
        nargs="+",
        default=[200.0, 400.0, 755.0, 1500.0, 4000.0],  # the sweep in EXPERIMENT_PLAN.md
        help="exploration scales to report reorder probabilities for",
    )
    return parser.parse_args()


@torch.no_grad()
def encode(model, tokenizer, inputs, device) -> torch.Tensor:
    inputs = {key: value.to(device) for key, value in inputs.items()}
    hidden = model(**inputs).last_hidden_state
    return pool_last_token_embedding(hidden, inputs["attention_mask"], normalize=True)


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=torch.float32,  # fp32 throughout: the gaps being measured are ~1e-2
        trust_remote_code=True,
    ).to(device).eval()

    data_args = DataArguments(
        data_path=args.data_path,
        per_dataset_max_samples=args.num_batches * args.batch_size,
        q_max_len=args.q_max_len,
        d_max_len=args.d_max_len,
        relevance_scheme="binary",
        slate_size=args.slate_size,
        file_glob=args.file_glob,
        include_sources=args.include_sources,
    )
    dataset = EmbeddingDataset(data_args=data_args, batch_size=args.batch_size, split="train")
    collator = EmbeddingDataCollator(
        tokenizer=tokenizer,
        query_max_length=args.q_max_len,
        doc_max_length=args.d_max_len,
        relevance_scheme="binary",
    )

    gaps: list[torch.Tensor] = []
    norms: list[torch.Tensor] = []
    gold_ranks: list[torch.Tensor] = []

    for batch_index in range(args.num_batches):
        start = batch_index * args.batch_size
        rows = [dataset[i] for i in range(start, min(start + args.batch_size, len(dataset)))]
        if not rows:
            break
        batch = collator(rows)
        batch_size = batch["relevance_labels"].size(0)
        slate = batch["relevance_labels"].size(1)

        query = encode(model, tokenizer, batch["query"], device)
        documents = encode(
            model,
            tokenizer,
            build_slate_inputs(batch["positive_document"], batch["negative_document"], batch_size, slate),
            device,
        ).reshape(batch_size, slate, -1)

        scores = torch.einsum("bd,bnd->bn", query, documents)
        order = scores.argsort(dim=-1, descending=True)
        ordered_scores = scores.gather(1, order)
        ordered_docs = documents.gather(1, order.unsqueeze(-1).expand_as(documents))

        gaps.append((ordered_scores[:, :-1] - ordered_scores[:, 1:]).flatten().cpu())
        # Only the component orthogonal to the query mean is perturbed: vMF noise is tangential,
        # so the parallel part of the difference contributes no variance to the score gap.
        delta = ordered_docs[:, :-1] - ordered_docs[:, 1:]
        parallel = torch.einsum("bnd,bd->bn", delta, query).unsqueeze(-1) * query.unsqueeze(1)
        norms.append((delta - parallel).norm(dim=-1).flatten().cpu())
        # The gold positive is slate index 0 (the collator puts it there); 1-indexed rank.
        gold_ranks.append((order == 0).float().argmax(dim=-1).add(1).cpu())

    if not gaps:
        raise SystemExit("no batches read -- check --data_path and --file_glob")

    gap = torch.cat(gaps)
    norm = torch.cat(norms).clamp_min(1e-6)
    rank = torch.cat(gold_ranks)
    dim = model.config.hidden_size
    ceiling = 1.0 / math.sqrt(dim)

    print(f"\nmodel {args.model}   d={dim}   slate={args.slate_size}   "
          f"{rank.numel()} slates, {gap.numel()} adjacent pairs\n")
    print("adjacent-rank score gaps (scored order):")
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        print(f"  p{int(q*100):<3} {gap.quantile(q).item():.4f}")
    print(f"  mean {gap.mean().item():.4f}")
    print(f"\ngold positive's rank: mean {rank.mean().item():.2f}, "
          f"top-1 in {(rank == 1).float().mean().item():.1%} of slates")

    print(f"\nmedian ||delta_perp|| = {norm.median().item():.3f}   "
          f"(per-document perturbation saturates at 1/sqrt(d) = {ceiling:.4f}, "
          f"but the pair SNR does not)\n")
    normal = torch.distributions.Normal(0.0, 1.0)
    median_norm = norm.median().item()
    print(f"{'kappa':>8} {'A_d':>7} {'g* (1 sigma)':>13} {'mean P(flip)':>13} {'pairs P>0.05':>13}")
    for kappa in sorted(args.kappa):
        a_d = bessel_ratio(dim / 2, torch.tensor(float(kappa))).item()
        z = gap * math.sqrt(a_d * kappa) / norm
        flip = normal.cdf(-z)
        g_star = median_norm / math.sqrt(a_d * kappa)
        print(f"{kappa:>8.0f} {a_d:>7.3f} {g_star:>13.4f} "
              f"{flip.mean().item():>13.4f} {(flip > 0.05).float().mean().item():>12.1%}")

    print("\nHow to read this. g* is the gap at which a pair flips with ~16% probability, so it")
    print("is directly comparable with the quantiles above: the kappa to run is the one whose g*")
    print("sits inside the bulk of the gap distribution. 'mean P(flip)' near 0 at every kappa")
    print("means the reward is constant across a rollout group by construction and")
    print("degenerate_frac will be ~1 however G and kappa are set -- in that regime the ranking")
    print("term cannot drive the RL stage alone, and the continuous companion term in the")
    print("default mixture is what carries the gradient. 'mean P(flip)' near 0.5 is the opposite")
    print("failure: the ordering is being randomized rather than explored.")


if __name__ == "__main__":
    main()
