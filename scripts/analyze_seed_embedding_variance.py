"""Diagnose why same-recipe seed repeats diverge on downstream retrieval.

Motivation: `G1-A-MRRAlign090` seed repeats (s42 / 3407 / 2026) share every setting
except the random seed, yet their BRIGHT ndcg@10 spans 0.162-0.220. Their final
weights differ by only ~0.065% (relative Frobenius), so the gap is invisible at the
weight level. This script encodes a common set of queries/documents with each
checkpoint and quantifies the *embedding geometry* that actually drives retrieval:

  * weight_diff   : per-checkpoint relative update vs base, and pairwise seed diffs
  * consistency   : cosine of the same document embedded by two models (amplification)
  * anisotropy    : document covariance spectrum -> effective dimension (collapse)
  * separability  : pos/neg similarity, raw gap, and scale-normalized SNR + MRR

Findings (see docs/seed_variance_analysis.md): the worst seed (3407) shows the
strongest representation collapse (lowest effective dim, narrowest doc-doc cone),
which lifts *all* query-doc similarities -- negatives more than positives -- and
compresses the pos-neg gap. The geometry ranking matches the ndcg@10 ranking.

Example:
  python scripts/analyze_seed_embedding_variance.py \
    --seeds s42:checkpoints/iclr2027/G1-A-MRRAlign090-s42 \
            3407:checkpoints/iclr2027/G1-A-MRRAlign090-Seed3407-s3407 \
            2026:checkpoints/iclr2027/G1-A-MRRAlign090-Seed2026-s2026 \
    --base Qwen/Qwen3-Embedding-0.6B \
    --data data/processed/reasonrank_multi/train.ready.jsonl \
    --source biology --max-queries 200
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from embedding_protocol import (  # noqa: E402
    tokenize_embedding_texts,
    format_embedding_text,
    load_embedding_protocol,
    pool_embeddings,
)

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
DEFAULT_INSTRUCTION = (
    "Given a post from an online forum, retrieve relevant documents "
    "that help answer the post."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--seeds", nargs="+", required=True,
        help="NAME:PATH entries, e.g. s42:checkpoints/.../G1-A-MRRAlign090-s42",
    )
    p.add_argument("--base", default=None, help="Optional base model (path or HF id) for weight-diff stage")
    p.add_argument("--base-revision", default=None, help="Base model revision when --base is an HF id")
    p.add_argument("--protocol-from", default=None,
                   help="Checkpoint dir to read embedding_protocol.json from (default: first seed)")
    p.add_argument("--data", default="data/processed/reasonrank_multi/train.ready.jsonl")
    p.add_argument("--source", default="biology", help="Filter records by this `source` field")
    p.add_argument("--max-queries", type=int, default=200)
    p.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    p.add_argument("--device", default="cuda")
    p.add_argument("--precision", choices=list(DTYPES), default="fp16",
                   help="Match the eval precision (BRIGHT eval uses fp16)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--json-out", default=None, help="Optional path to dump metrics as JSON")
    return p.parse_args()


def parse_seed_specs(items: list[str]) -> dict[str, str]:
    specs: dict[str, str] = {}
    for item in items:
        if ":" not in item:
            raise ValueError(f"--seeds entry must be NAME:PATH, got {item!r}")
        name, path = item.split(":", 1)
        specs[name] = path
    return specs


def load_records(path: str, source: str, limit: int) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("source") == source:
                rows.append(d)
            if len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"No records with source={source!r} in {path}")
    return rows


def flatten_candidates(rows: list[dict]) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    queries = [r["query"] for r in rows]
    docs, owner, rel = [], [], []
    for qi, r in enumerate(rows):
        for di, doc in enumerate(r["document"]):
            docs.append(doc)
            owner.append(qi)
            rel.append(r["relevance"][di])
    return queries, docs, np.asarray(owner), np.asarray(rel)


def encode(model_path, texts, role, protocol, args):
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tok.padding_side = protocol["padding_side"]
    model = (
        AutoModel.from_pretrained(model_path, torch_dtype=DTYPES[args.precision], trust_remote_code=True)
        .to(args.device)
        .eval()
    )
    template = protocol[f"{role}_prompt_template"]
    out = []
    for s in range(0, len(texts), args.batch_size):
        batch = [
            format_embedding_text(
                template, t,
                task_description=args.instruction if role == "query" else "",
            )
            for t in texts[s : s + args.batch_size]
        ]
        inp = tokenize_embedding_texts(batch, tok, protocol["append_token"], max_length=args.max_length).to(args.device)
        with torch.inference_mode():
            hidden = model(**inp).last_hidden_state
            vecs = pool_embeddings(
                hidden, inp["attention_mask"],
                pooling_method=protocol["pooling_method"], normalize=True,
            ).float().cpu().numpy()
        out.append(vecs)
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return np.concatenate(out, 0)


def weight_diff_stage(specs, args):
    """Relative Frobenius update vs base and pairwise seed differences."""
    from safetensors import safe_open

    def resolve_weights(path):
        p = pathlib.Path(path)
        cand = p / "model.safetensors"
        if cand.is_file():
            return str(cand)
        if p.is_file():
            return str(p)
        return None  # HF id / no local safetensors: skip

    openers = {}
    for name, path in specs.items():
        w = resolve_weights(path)
        if w:
            openers[name] = safe_open(w, framework="pt")
    if len(openers) < 2:
        print("  (weight-diff skipped: need >=2 local safetensors checkpoints)")
        return {}

    keys = list(next(iter(openers.values())).keys())
    result = {"vs_base": {}, "pairwise": {}}

    fb = None
    if args.base:
        bw = resolve_weights(args.base)
        if bw:
            fb = safe_open(bw, framework="pt")
    if fb is not None:
        for name, fa in openers.items():
            num = den = 0.0
            for k in keys:
                a = fa.get_tensor(k).float(); b = fb.get_tensor(k).float()
                num += (a - b).pow(2).sum().item(); den += b.pow(2).sum().item()
            result["vs_base"][name] = (num ** 0.5) / (den ** 0.5)
        print("  relative update vs base  ||seed-base||/||base||:")
        for n, v in result["vs_base"].items():
            print(f"    {n:<10}{v*100:.4f}%")

    names = list(openers)
    print("  pairwise seed diff  ||a-b||/rms:")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            fa, fc = openers[names[i]], openers[names[j]]
            num = den = 0.0
            for k in keys:
                a = fa.get_tensor(k).float(); c = fc.get_tensor(k).float()
                num += (a - c).pow(2).sum().item()
                den += 0.5 * (a.pow(2).sum().item() + c.pow(2).sum().item())
            v = (num ** 0.5) / (den ** 0.5)
            result["pairwise"][f"{names[i]}|{names[j]}"] = v
            print(f"    {names[i]:<9} vs {names[j]:<9}: {v*100:.4f}%")
    return result


def main() -> None:
    args = parse_args()
    specs = parse_seed_specs(args.seeds)
    protocol = load_embedding_protocol(args.protocol_from or next(iter(specs.values())))

    rows = load_records(args.data, args.source, args.max_queries)
    queries, docs, owner, rel = flatten_candidates(rows)
    print(f"source={args.source}  queries={len(queries)}  docs={len(docs)}  "
          f"positives={int((rel == 1).sum())}  precision={args.precision}")

    print("\n=== Weight diff ===")
    metrics = {"weight": weight_diff_stage(specs, args)}

    emb = {}
    for name, path in specs.items():
        q = encode(path, queries, "query", protocol, args)
        d = encode(path, docs, "document", protocol, args)
        emb[name] = (q, d)
        print(f"[encoded] {name}: Q{q.shape} D{d.shape}")

    rng = np.random.default_rng(0)
    metrics["geometry"] = {}
    metrics["separability"] = {}

    print("\n=== Embedding consistency (same doc across models, mean cosine) ===")
    names = list(specs)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            cos = (emb[names[i]][1] * emb[names[j]][1]).sum(1)
            print(f"  {names[i]:<9} vs {names[j]:<9}: mean_cos={cos.mean():.4f}  min={cos.min():.4f}")

    print("\n=== Document geometry (anisotropy / effective dim) ===")
    print(f"{'model':<10}{'top1_evr':>10}{'top10_evr':>11}{'eff_dim':>9}{'doc-doc_cos':>13}")
    for name in names:
        d = emb[name][1]
        dc = d - d.mean(0)
        cov = dc.T @ dc / len(dc)
        ev = np.linalg.eigvalsh(cov)[::-1]
        ev = ev / ev.sum()
        eff = float((ev.sum() ** 2) / (ev ** 2).sum())
        idx = rng.integers(0, len(d), (8000, 2))
        dd = float((d[idx[:, 0]] * d[idx[:, 1]]).sum(1).mean())
        metrics["geometry"][name] = {
            "top1_evr": float(ev[0]), "top10_evr": float(ev[:10].sum()),
            "eff_dim": eff, "doc_doc_cos": dd,
        }
        print(f"{name:<10}{ev[0]:>10.4f}{ev[:10].sum():>11.4f}{eff:>9.1f}{dd:>13.4f}")

    print("\n=== Separability (in-candidate, per query) ===")
    print(f"{'model':<10}{'pos_sim':>9}{'neg_sim':>9}{'gap':>9}{'SNR':>8}{'MRR':>8}{'frac_neg':>10}")
    for name in names:
        q, d = emb[name]
        pos, neg, gaps, snr, rr, negm = [], [], [], [], [], []
        for qi in range(len(q)):
            mask = owner == qi
            cand = d[mask]; rl = rel[mask]
            sims = cand @ q[qi]
            if (rl == 1).sum() == 0 or (rl == 0).sum() == 0:
                continue
            p = sims[rl == 1]; n = sims[rl == 0]
            gap = p.mean() - n.mean()
            pos.append(p.mean()); neg.append(n.mean()); gaps.append(gap)
            snr.append(gap / (sims.std() + 1e-9))
            order = np.argsort(-sims)
            first = np.where(rl[order] == 1)[0][0]
            rr.append(1.0 / (first + 1))
            negm.append(1.0 if (p.max() - n.max()) < 0 else 0.0)
        metrics["separability"][name] = {
            "pos_sim": float(np.mean(pos)), "neg_sim": float(np.mean(neg)),
            "gap": float(np.mean(gaps)), "snr": float(np.mean(snr)),
            "mrr": float(np.mean(rr)), "frac_neg_margin": float(np.mean(negm)),
        }
        m = metrics["separability"][name]
        print(f"{name:<10}{m['pos_sim']:>9.4f}{m['neg_sim']:>9.4f}{m['gap']:>9.4f}"
              f"{m['snr']:>8.4f}{m['mrr']:>8.4f}{m['frac_neg_margin']:>10.3f}")

    if args.json_out:
        pathlib.Path(args.json_out).write_text(json.dumps(metrics, indent=2))
        print(f"\nWrote metrics to {args.json_out}")


if __name__ == "__main__":
    main()
