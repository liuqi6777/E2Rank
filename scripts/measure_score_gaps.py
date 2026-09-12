"""Inspect query-side vMF score gaps without training.

Conditional on a fixed document pair delta, the exact score variance is
A_d(kappa)/kappa * ||delta_perp||^2 + A_d'(kappa) * g^2.
The Gaussian flip estimate uses this full variance. g* is only the near-boundary
approximation ||delta_perp|| / sqrt(A_d*kappa); neither statistic proves that a
finite group will be degenerate, or predicts joint document exploration exactly.
Only valid adjacent different-grade pairs touching top-K are summarized.
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
from embedding_protocol import load_embedding_protocol, pool_embeddings
from policy_math import mean_alignment, score_gap_variance
from utils import load_raw_config_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="merged checkpoint or base model id")
    parser.add_argument("--model_config", help="model YAML carrying the embedding protocol")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--reward_k", type=int, default=10)
    parser.add_argument("--relevance_scheme", choices=["binary", "graded"], default="graded")
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
def encode(model, inputs, device, pooling_method) -> torch.Tensor:
    inputs = {key: value.to(device) for key, value in inputs.items()}
    hidden = model(**inputs).last_hidden_state
    return pool_embeddings(
        hidden,
        inputs["attention_mask"],
        pooling_method=pooling_method,
        normalize=True,
    )


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    protocol = {
        "pooling_method": "last",
        "padding_side": "left",
        "append_token": "pad",
        "query_prompt_template": "Instruct: {task_description}\nQuery:{query}",
        "document_prompt_template": "{document}",
        "max_length": 8192,
    }
    protocol.update(load_embedding_protocol(args.model))
    if args.model_config:
        raw_model_config = load_raw_config_file(args.model_config)
        protocol.update(
            {
                key: raw_model_config[key]
                for key in protocol
                if key in raw_model_config
            }
        )
        protocol["max_length"] = raw_model_config.get(
            "embedding_max_length", protocol["max_length"]
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        padding_side=protocol["padding_side"],
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=torch.float32,  # fp32 throughout: the gaps being measured are ~1e-2
        trust_remote_code=True,
    ).to(device).eval()

    data_args = DataArguments(
        data_path=args.data_path,
        per_dataset_max_samples=args.num_batches * args.batch_size,
        q_max_len=args.q_max_len,
        d_max_len=min(args.d_max_len, protocol["max_length"]),
        relevance_scheme=args.relevance_scheme,
        slate_size=args.slate_size,
        file_glob=args.file_glob,
        include_sources=args.include_sources,
    )
    dataset = EmbeddingDataset(
        data_args=data_args,
        batch_size=args.batch_size,
        split="train",
        query_prompt_template=protocol["query_prompt_template"],
    )
    collator = EmbeddingDataCollator(
        tokenizer=tokenizer,
        query_max_length=min(args.q_max_len, protocol["max_length"]),
        doc_max_length=min(args.d_max_len, protocol["max_length"]),
        relevance_scheme=args.relevance_scheme,
        document_prompt_template=protocol["document_prompt_template"],
        append_token=protocol["append_token"],
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

        query = encode(model, batch["query"], device, protocol["pooling_method"])
        documents = encode(
            model,
            build_slate_inputs(batch["positive_document"], batch["negative_document"], batch_size, slate),
            device,
            protocol["pooling_method"],
        ).reshape(batch_size, slate, -1)

        valid = batch.get("candidate_mask", torch.ones(batch_size, slate, dtype=torch.bool)).to(device)
        scores = torch.einsum("bd,bnd->bn", query, documents).masked_fill(~valid, -torch.inf)
        order = scores.argsort(dim=-1, descending=True)
        ordered_scores = scores.gather(1, order)
        ordered_docs = documents.gather(1, order.unsqueeze(-1).expand_as(documents))

        ordered_valid = valid.gather(1, order)
        labels = batch["relevance_labels"].to(device).gather(1, order)
        eligible = ordered_valid[:, :-1] & ordered_valid[:, 1:] & (labels[:, :-1] != labels[:, 1:])
        eligible[:, args.reward_k:] = False
        difference = ordered_scores[:, :-1] - ordered_scores[:, 1:]
        eligible &= difference.abs() > 1e-8  # deterministic ties have no signed flip event
        gaps.append(difference[eligible].cpu())
        delta = ordered_docs[:, :-1] - ordered_docs[:, 1:]
        parallel = torch.einsum("bnd,bd->bn", delta, query).unsqueeze(-1) * query.unsqueeze(1)
        norms.append((delta - parallel).norm(dim=-1)[eligible].cpu())
        # The gold positive is slate index 0 (the collator puts it there); 1-indexed rank.
        gold_ranks.append((order == 0).float().argmax(dim=-1).add(1).cpu())

    if not gaps or not any(x.numel() for x in gaps):
        raise SystemExit("No eligible non-tied pairs; check data, labels and cutoff")

    gap = torch.cat(gaps).double()
    norm = torch.cat(norms).double()
    rank = torch.cat(gold_ranks)
    dim = query.size(-1)
    ceiling = 1.0 / math.sqrt(dim)

    print(f"\nmodel {args.model}   d={dim}   slate={args.slate_size}   "
          f"{rank.numel()} slates, {gap.numel()} eligible adjacent pairs\n")
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
        a_d = mean_alignment(dim, float(kappa))
        variance = score_gap_variance(dim, kappa, gap, norm)
        z = a_d * gap / variance.sqrt().clamp_min(1e-15)
        flip = normal.cdf(-z)
        g_star = median_norm / math.sqrt(a_d * kappa)
        print(f"{kappa:>8.0f} {a_d:>7.3f} {g_star:>13.4f} "
              f"{flip.mean().item():>13.4f} {(flip > 0.05).float().mean().item():>12.1%}")

    print("\ng* is a near-boundary scale; P(flip) is a Gaussian approximation using full variance.")
    print("Small probabilities suggest weak local signal, not guaranteed group degeneracy.")
    print("These are own-query, different-grade top-K pairs with documents fixed; full-corpus")
    print("and joint exploration can behave differently. No configuration is selected here.")


if __name__ == "__main__":
    main()
