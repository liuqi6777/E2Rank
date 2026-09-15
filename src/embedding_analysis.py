"""Cached, ID-aligned retrieval and representation diagnostics.

Analysis stages only require NumPy. Encoding and vMF sampling load their heavier
dependencies lazily. All ranks use score descending, then document ID descending
(the trec_eval tie convention). nDCG uses linear relevance gains, like trec_eval.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import logging
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np


LOG = logging.getLogger(__name__)
VERSION = 1


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        if rows:
            fields = list(dict.fromkeys(key for row in rows for key in row))
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


@contextmanager
def artifact(path, identity):
    """Publish only complete artifacts; never overwrite incompatible outputs."""
    path = Path(path)
    identity = {"version": VERSION, **identity}
    signature = digest(identity)
    if path.exists():
        manifest = read_json(path / "manifest.json")
        if manifest.get("signature") != signature:
            raise ValueError(
                f"Artifact inputs changed: {path}; select a new output directory"
            )
        for name, expected in manifest["files"].items():
            if file_digest(path / name) != expected:
                raise ValueError(f"Artifact checksum mismatch: {path / name}")
        LOG.info("Reusing %s", path)
        yield None
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{path.name}-", dir=path.parent))
    try:
        yield staging
        files = {
            str(p.relative_to(staging)): file_digest(p)
            for p in sorted(staging.rglob("*"))
            if p.is_file()
        }
        write_json(
            staging / "manifest.json",
            {"signature": signature, "identity": identity, "files": files},
        )
        staging.rename(path)
        LOG.info("Saved %s", path)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def checked_manifest(path):
    manifest = read_json(Path(path) / "manifest.json")
    if manifest.get("signature") != digest(manifest["identity"]):
        raise ValueError(f"Invalid artifact manifest: {path}")
    for name, expected in manifest["files"].items():
        if file_digest(Path(path) / name) != expected:
            raise ValueError(f"Artifact checksum mismatch: {Path(path) / name}")
    return manifest


def validate_data(queries, corpus, qrels):
    for rows, key in ((queries, "query_id"), (corpus, "doc_id")):
        ids = [r[key] for r in rows]
        if not ids or any(not isinstance(i, str) or not i for i in ids):
            raise ValueError(f"Nonempty string {key}s required")
        if len(ids) != len(set(ids)) or any(
            not isinstance(r["text"], str) for r in rows
        ):
            raise ValueError(f"Duplicate {key} or invalid text")
    qids, dids = {q["query_id"] for q in queries}, {d["doc_id"] for d in corpus}
    seen, positive_queries = set(), set()
    for row in qrels:
        pair = row["query_id"], row["doc_id"]
        grade = row["relevance"]
        if pair[0] not in qids or pair[1] not in dids or pair in seen:
            raise ValueError(f"Unknown or duplicate qrel: {pair}")
        if (
            isinstance(grade, bool)
            or not isinstance(grade, (int, float))
            or not np.isfinite(grade)
            or grade < 0
        ):
            raise ValueError(f"Invalid relevance: {row}")
        seen.add(pair)
        if grade > 0:
            positive_queries.add(pair[0])
    if positive_queries != qids:
        raise ValueError(
            "Every analysis query must have at least one positive qrel in corpus"
        )


def prepare_data(config, subset):
    data = config["data"]
    if data["kind"] == "local":
        source = Path(data["subsets"][subset])
        queries = read_jsonl(source / "queries.jsonl")
        corpus = read_jsonl(source / "corpus.jsonl")
        qrels = read_jsonl(source / "qrels.jsonl")
        instruction = data.get("instructions", {}).get(subset, "")
        source_identity = {"kind": "local", "path": str(source.resolve())}
    elif data["kind"] == "bright":
        import mteb
        from types import SimpleNamespace
        from eval_mteb.run_mteb import (
            BRIGHT_DATASET_REVISION,
            BRIGHT_INSTRUCTIONS,
            BRIGHT_SPLIT,
            load_official_bright,
        )

        task = mteb.get_tasks(tasks=["BrightRetrieval"])[0]
        revision = data.get("revision", BRIGHT_DATASET_REVISION)
        query_set = data.get("query_set", "original")
        load_official_bright(
            task,
            [subset],
            SimpleNamespace(
                bright_dataset_revision=revision,
                bright_query_set=query_set,
                bright_cache_dir=data.get("cache_dir"),
            ),
        )
        queries = [
            {"query_id": qid, "text": text}
            for qid, text in task.queries[subset][BRIGHT_SPLIT].items()
        ]
        corpus = [
            {"doc_id": did, "text": (doc.get("title", "") + " " + doc["text"]).strip()}
            for did, doc in task.corpus[subset][BRIGHT_SPLIT].items()
        ]
        qrels = [
            {"query_id": qid, "doc_id": did, "relevance": grade}
            for qid, docs in task.relevant_docs[subset][BRIGHT_SPLIT].items()
            for did, grade in docs.items()
        ]
        instruction = BRIGHT_INSTRUCTIONS[subset]
        source_identity = {
            "kind": "bright",
            "revision": revision,
            "query_set": query_set,
        }
    else:
        raise ValueError("data.kind must be local or bright")
    validate_data(queries, corpus, qrels)
    queries.sort(key=lambda r: r["query_id"])
    # Index ascending is the deterministic tie priority used throughout analysis.
    corpus.sort(key=lambda r: r["doc_id"], reverse=True)
    qrels.sort(key=lambda r: (r["query_id"], r["doc_id"]))
    identity = {
        "source": source_identity,
        "subset": subset,
        "instruction": instruction,
        "content": digest([queries, corpus, qrels]),
    }
    destination = Path(config["output_dir"]) / subset / "data"
    with artifact(destination, identity) as out:
        if out is not None:
            for name, rows in (
                ("queries", queries),
                ("corpus", corpus),
                ("qrels", qrels),
            ):
                write_jsonl(out / f"{name}.jsonl", rows)
    return destination


def load_data(root):
    root = Path(root)
    manifest = checked_manifest(root / "data")
    queries = read_jsonl(root / "data/queries.jsonl")
    corpus = read_jsonl(root / "data/corpus.jsonl")
    qrels = read_jsonl(root / "data/qrels.jsonl")
    validate_data(queries, corpus, qrels)
    lookup = {d["doc_id"]: i for i, d in enumerate(corpus)}
    relevance = {q["query_id"]: {} for q in queries}
    for row in qrels:
        if row["relevance"] > 0:
            relevance[row["query_id"]][lookup[row["doc_id"]]] = float(row["relevance"])
    return queries, corpus, [relevance[q["query_id"]] for q in queries], manifest


def checkpoint_identity(model):
    path = Path(model["path"])
    if not path.is_dir():
        return {"path": model["path"], "revision": model.get("revision")}
    names = sorted(
        {
            *path.glob("*.safetensors"),
            *path.glob("pytorch_model*.bin"),
            *path.glob("*.json"),
            *path.glob("*.model"),
            *path.glob("merges.txt"),
            *path.glob("vocab.txt"),
        }
    )
    if not names:
        raise ValueError(f"No checkpoint files: {path}")
    return {
        "path": str(path.resolve()),
        "files": {p.name: file_digest(p) for p in names},
    }


def encode_stage(config, subset, names):
    import torch
    from transformers import AutoModel, AutoTokenizer
    from embedding_protocol import (
        TOKENIZATION_VERSION,
        tokenize_embedding_texts,
        format_embedding_text,
        load_embedding_protocol,
        pool_embeddings,
        validate_embedding_protocol,
    )
    from utils import load_raw_config_file

    for name in names:
        spec = config["models"][name]
        if not Path(spec["path"]).is_dir() and not re.fullmatch(
            r"[0-9a-f]{40}", spec.get("revision", "")
        ):
            raise ValueError(
                f"Model {name}: provide an existing local checkpoint or a remote model with a 40-character commit revision"
            )
    root = prepare_data(config, subset).parent
    queries, corpus, _, data_manifest = load_data(root)
    settings = config.get("encoding", {})
    device = settings.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    precision = settings.get("precision", "fp32")
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
        precision
    ]
    batch_size = positive_int(settings.get("batch_size", 16))
    for name in names:
        spec = config["models"][name]
        protocol = load_embedding_protocol(spec["path"])
        if spec.get("model_config"):
            raw = load_raw_config_file(spec["model_config"])
            protocol.update(
                {
                    key: raw[key]
                    for key in (
                        "pooling_method",
                        "padding_side",
                        "append_token",
                        "query_prompt_template",
                        "document_prompt_template",
                    )
                    if key in raw
                }
            )
            protocol["max_length"] = raw.get(
                "embedding_max_length", protocol.get("max_length", 8192)
            )
        validate_embedding_protocol(
            **{
                key: protocol[key]
                for key in (
                    "pooling_method",
                    "padding_side",
                    "append_token",
                    "query_prompt_template",
                    "document_prompt_template",
                )
            }
        )
        protocol["tokenization_version"] = TOKENIZATION_VERSION
        identity = {
            "data": data_manifest["signature"],
            "checkpoint": checkpoint_identity(spec),
            "protocol": protocol,
            "encoding": {**settings, "device": device, "precision": precision},
        }
        with artifact(root / name / "embeddings", identity) as out:
            if out is None:
                continue
            kwargs = {"trust_remote_code": True}
            if spec.get("revision"):
                kwargs["revision"] = spec["revision"]
            tokenizer = AutoTokenizer.from_pretrained(spec["path"], **kwargs)
            tokenizer.padding_side = protocol["padding_side"]
            model = (
                AutoModel.from_pretrained(spec["path"], torch_dtype=dtype, **kwargs)
                .to(device)
                .eval()
            )
            for role, rows in (("query", queries), ("document", corpus)):
                limit = positive_int(
                    settings.get(f"max_{role}_length", protocol.get("max_length", 8192))
                )
                values = None
                for start in range(0, len(rows), batch_size):
                    texts = [
                        format_embedding_text(
                            protocol[f"{role}_prompt_template"],
                            r["text"],
                            task_description=data_manifest["identity"]["instruction"]
                            if role == "query"
                            else "",
                        )
                        for r in rows[start : start + batch_size]
                    ]
                    inputs = tokenize_embedding_texts(
                        texts, tokenizer, protocol["append_token"],
                        max_length=limit,
                    ).to(device)
                    with torch.inference_mode():
                        hidden = model(**inputs).last_hidden_state
                        vectors = (
                            pool_embeddings(
                                hidden,
                                inputs["attention_mask"],
                                pooling_method=protocol["pooling_method"],
                                normalize=True,
                            )
                            .float()
                            .cpu()
                            .numpy()
                        )
                    vectors = unit(vectors)
                    if values is None:
                        values = np.lib.format.open_memmap(
                            out / f"{role}_embeddings.npy",
                            mode="w+",
                            dtype="float32",
                            shape=(len(rows), vectors.shape[1]),
                        )
                    values[start : start + len(vectors)] = vectors
                    LOG.info(
                        "%s/%s %s %d/%d",
                        subset,
                        name,
                        role,
                        start + len(vectors),
                        len(rows),
                    )
                values.flush()
                del values
            write_json(
                out / "encoder.json",
                {
                    "resolved_revision": getattr(model.config, "_commit_hash", None),
                    "protocol": protocol,
                },
            )
            del model
            if device.startswith("cuda"):
                torch.cuda.empty_cache()


def unit(values):
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norms <= 0):
        raise ValueError("Embedding contains nonfinite values or zero vectors")
    return values / norms


def load_embeddings(root, name, data_signature, nq, nd):
    folder = Path(root) / name / "embeddings"
    manifest = checked_manifest(folder)
    if manifest["identity"]["data"] != data_signature:
        raise ValueError(f"Embedding data identity mismatch: {folder}")
    q, d = (
        np.load(folder / f"{role}_embeddings.npy", mmap_mode="r", allow_pickle=False)
        for role in ("query", "document")
    )
    if (
        q.ndim != 2
        or d.ndim != 2
        or q.shape[0] != nq
        or d.shape[0] != nd
        or q.shape[1] != d.shape[1]
    ):
        raise ValueError(f"Embedding shape mismatch: {folder}")
    for values in (q, d):
        for start in range(0, len(values), 8192):
            chunk = values[start : start + 8192]
            if not np.isfinite(chunk).all() or not np.allclose(
                np.linalg.norm(chunk, axis=1), 1, atol=2e-5
            ):
                raise ValueError(f"Expected finite unit embeddings: {folder}")
    return q, d, manifest["signature"]


def ranked(scores, ids, k):
    """Bounded top-k with exact, deterministic cutoff ties."""
    if len(scores) > k:
        threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
        greater = np.flatnonzero(scores > threshold)
        equal = np.flatnonzero(scores == threshold)
        equal = equal[np.argsort(ids[equal], kind="stable")[: k - len(greater)]]
        keep = np.concatenate([greater, equal])
        scores, ids = scores[keep], ids[keep]
    order = np.lexsort((ids, -scores))[:k]
    return ids[order], scores[order]


def search(query, documents, k, block_size=8192):
    ids, scores = np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    for start in range(0, len(documents), block_size):
        block_ids = np.arange(start, min(start + block_size, len(documents)))
        block_scores = (
            np.asarray(documents[start : start + block_size], dtype=np.float32) @ query
        )
        ids, scores = ranked(
            np.concatenate([scores, block_scores]), np.concatenate([ids, block_ids]), k
        )
    return ids, scores


def metrics(order, relevance, k):
    gains = np.array([relevance.get(int(i), 0) for i in order[:k]], dtype=np.float64)
    ideal = np.array(sorted(relevance.values(), reverse=True)[:k], dtype=np.float64)
    dcg = (gains / np.log2(np.arange(len(gains)) + 2)).sum()
    idcg = (ideal / np.log2(np.arange(len(ideal)) + 2)).sum()
    hits = np.flatnonzero(gains > 0)
    return {
        "ndcg": float(dcg / idcg) if idcg else 0.0,
        "mrr": float(1 / (hits[0] + 1)) if len(hits) else 0.0,
        "hit": float(bool(len(hits))),
    }


def positive_ranks(query, documents, positives, block_size):
    """Two block passes keep positive scores identical to the ranking GEMV."""
    ids = np.array(sorted(positives), dtype=np.int64)
    scores = np.zeros(len(ids), dtype=np.float32)
    for start in range(0, len(documents), block_size):
        mask = (ids >= start) & (ids < start + block_size)
        if mask.any():
            block = (
                np.asarray(documents[start : start + block_size], dtype=np.float32)
                @ query
            )
            scores[mask] = block[ids[mask] - start]
    ranks = np.ones(len(ids), dtype=np.int64)
    for start in range(0, len(documents), block_size):
        block = (
            np.asarray(documents[start : start + block_size], dtype=np.float32) @ query
        )
        block_ids = np.arange(start, start + len(block))
        for j, (did, score) in enumerate(zip(ids, scores)):
            ranks[j] += np.count_nonzero(
                (block > score) | ((block == score) & (block_ids < did))
            )
    return ids, scores, ranks


def retrieval_stage(config, subset, names):
    root = Path(config["output_dir"]) / subset
    queries, corpus, relevance, data = load_data(root)
    settings = config.get("retrieval", {})
    k = positive_int(settings.get("k", 10))
    top_k = max(k, positive_int(settings.get("top_k", 1000)))
    block = positive_int(settings.get("block_size", 8192))
    for name in names:
        q, d, signature = load_embeddings(
            root, name, data["signature"], len(queries), len(corpus)
        )
        with artifact(
            root / name / "retrieval",
            {
                "embeddings": signature,
                "settings": settings,
                "k": k,
                "top_k": top_k,
                "ties": "doc_id_desc",
                "gain": "linear",
            },
        ) as out:
            if out is None:
                continue
            rows, rankings, positives = [], [], []
            for i, query in enumerate(queries):
                # Include enough entries to always identify the highest unlabelled candidate.
                order, scores = search(
                    q[i], d, min(len(d), max(top_k, len(relevance[i]) + 1)), block
                )
                pos_ids, pos_scores, ranks = positive_ranks(
                    q[i], d, relevance[i], block
                )
                negatives = [
                    (int(j), float(s))
                    for j, s in zip(order, scores)
                    if int(j) not in relevance[i]
                ]
                negative_score = negatives[0][1] if negatives else None
                rows.append(
                    {
                        "query_id": query["query_id"],
                        **metrics(order, relevance[i], k),
                        "best_positive_rank": int(ranks.min()),
                        "best_positive_score": float(pos_scores.max()),
                        "mean_positive_score": float(pos_scores.mean()),
                        "hardest_unlabelled_score": negative_score,
                        "margin": float(pos_scores.max() - negative_score)
                        if negative_score is not None
                        else None,
                    }
                )
                rankings.append(
                    {
                        "query_id": query["query_id"],
                        "indices": order[:top_k].tolist(),
                        "scores": scores[:top_k].tolist(),
                    }
                )
                positives.extend(
                    {
                        "query_id": query["query_id"],
                        "doc_id": corpus[j]["doc_id"],
                        "score": float(s),
                        "rank": int(r),
                    }
                    for j, s, r in zip(pos_ids, pos_scores, ranks)
                )
            write_csv(out / "queries.csv", rows)
            write_jsonl(out / "queries.jsonl", rows)
            write_jsonl(out / "rankings.jsonl", rankings)
            write_jsonl(out / "positive_ranks.jsonl", positives)


def shared_candidates(root, names, queries, relevance, candidate_k):
    candidates = [set(r) for r in relevance]
    signatures = {}
    for name in names:
        path = root / name / "retrieval"
        manifest = checked_manifest(path)
        if manifest["identity"]["top_k"] < candidate_k:
            raise ValueError("retrieval.top_k must be >= candidate_k")
        signatures[name] = manifest["signature"]
        rows = read_jsonl(path / "rankings.jsonl")
        if [r["query_id"] for r in rows] != [r["query_id"] for r in queries]:
            raise ValueError("Retrieval query order mismatch")
        for candidate, row in zip(candidates, rows):
            candidate.update(row["indices"][:candidate_k])
    return [np.array(sorted(c), dtype=np.int64) for c in candidates], signatures


def distribution_geometry(values, pair_ids, block=8192):
    """Stable pair sampling, float64 covariance, entropy effective rank."""
    n, dim = values.shape
    total, second = np.zeros(dim), np.zeros((dim, dim))
    for start in range(0, n, block):
        x = np.asarray(values[start : start + block], dtype=np.float64)
        total += x.sum(axis=0)
        second += x.T @ x
    mean = total / n
    cov = second / n - np.outer(mean, mean)
    eigenvalues = np.maximum(np.linalg.eigvalsh(cov), 0)[::-1]
    spectrum_sum = eigenvalues.sum()
    probs = (
        eigenvalues[eigenvalues > 0] / spectrum_sum
        if spectrum_sum > 0
        else np.array([])
    )
    effective_rank = (
        float(np.exp(-np.sum(probs * np.log(probs)))) if len(probs) else 0.0
    )
    kernel_sum, count = 0.0, len(pair_ids)
    for start in range(0, count, block):
        pairs = pair_ids[start : start + block]
        delta = np.asarray(values[pairs[:, 0]], dtype=np.float64) - values[pairs[:, 1]]
        kernel_sum += np.exp(-2 * np.sum(delta * delta, axis=1)).sum()
    return {
        "uniformity": float(np.log(kernel_sum / count)) if count else None,
        "mean_vector_norm": float(np.linalg.norm(mean)),
        "effective_rank": effective_rank,
        "eigenvalues": eigenvalues.tolist(),
    }


def random_pairs(n, count, rng):
    if n < 2:
        return np.empty((0, 2), dtype=np.int64)
    a = rng.integers(n, size=count)
    b = rng.integers(n - 1, size=count)
    b += b >= a
    return np.column_stack([a, b])


def neighbor_sets(values, anchors, k, block):
    result = []
    for index in anchors:
        ids, _ = search(values[index], values, min(k + 1, len(values)), block)
        result.append(set([int(i) for i in ids if i != index][:k]))
    return result


def procrustes_rotation(source, target):
    """One uncentered orthogonal transform, shared by query and document roles."""
    cross = np.asarray(source, dtype=np.float64).T @ np.asarray(
        target, dtype=np.float64
    )
    u, singular, vh = np.linalg.svd(cross, full_matrices=False)
    threshold = singular.max() * max(cross.shape) * np.finfo(np.float64).eps
    return u @ vh, int(np.count_nonzero(singular > threshold))


def geometry_stage(config, subset, names):
    root = Path(config["output_dir"]) / subset
    queries, corpus, relevance, data = load_data(root)
    settings = config.get("geometry", {})
    reference = config.get("reference", "E0")
    if reference not in names:
        raise ValueError("geometry requires the reference in --models")
    count = positive_int(settings.get("pairs", 100000))
    candidate_k = positive_int(settings.get("candidate_k", 100))
    neighbor_k = positive_int(settings.get("neighbor_k", 10))
    block = positive_int(config.get("retrieval", {}).get("block_size", 8192))
    rng = np.random.default_rng(config.get("seed", 42))
    pairs = {
        "query": random_pairs(len(queries), count, rng),
        "document": random_pairs(len(corpus), count, rng),
    }
    anchors = {
        "query": np.arange(len(queries)),
        "document": np.sort(
            rng.choice(
                len(corpus),
                min(len(corpus), positive_int(settings.get("anchors", 2000))),
                replace=False,
            )
        ),
    }
    candidates, retrieval_signatures = shared_candidates(
        root, names, queries, relevance, candidate_k
    )
    embeddings = {
        name: load_embeddings(root, name, data["signature"], len(queries), len(corpus))
        for name in names
    }
    for name in names:
        retrieval = read_json(root / name / "retrieval/manifest.json")
        if retrieval["identity"]["embeddings"] != embeddings[name][2]:
            raise ValueError("Retrieval refers to different embeddings")
    with artifact(
        root / "geometry",
        {
            "models": {n: v[2] for n, v in embeddings.items()},
            "retrieval": retrieval_signatures,
            "reference": reference,
            "settings": settings,
            "seed": config.get("seed", 42),
        },
    ) as out:
        if out is None:
            return
        baseline_neighbors = {
            role: neighbor_sets(
                embeddings[reference][j], anchors[role], neighbor_k, block
            )
            for j, role in enumerate(("query", "document"))
        }
        rows, summaries, movements, neighborhood_rows = [], [], [], []
        # E0 chooses one fixed positive / competing unlabelled document per query.
        fixed_pairs = []
        q0, d0, _ = embeddings[reference]
        for i, candidate in enumerate(candidates):
            scores = d0[candidate] @ q0[i]
            order, _ = ranked(scores, candidate, len(candidate))
            p = next(int(j) for j in order if int(j) in relevance[i])
            negative = next((int(j) for j in order if int(j) not in relevance[i]), None)
            fixed_pairs.append((p, negative))
        for name, (q, d, _) in embeddings.items():
            for role, values in (("query", q), ("document", d)):
                stats = distribution_geometry(values, pairs[role], block)
                write_json(
                    out / f"{name}_{role}_spectrum.json", stats.pop("eigenvalues")
                )
                neighbors = neighbor_sets(values, anchors[role], neighbor_k, block)
                retention = [
                    len(a & b) / len(a)
                    for a, b in zip(baseline_neighbors[role], neighbors)
                    if a
                ]
                for index, base_neighbors, new_neighbors in zip(
                    anchors[role], baseline_neighbors[role], neighbors
                ):
                    neighborhood_rows.append(
                        {
                            "model": name,
                            "role": role,
                            "sample_id": queries[index]["query_id"]
                            if role == "query"
                            else corpus[index]["doc_id"],
                            "retention": len(base_neighbors & new_neighbors)
                            / len(base_neighbors)
                            if base_neighbors
                            else None,
                        }
                    )
                summaries.append(
                    {
                        "model": name,
                        "role": role,
                        **stats,
                        "neighbor_retention": float(np.mean(retention))
                        if retention
                        else None,
                    }
                )
            if settings.get("procrustes", False):
                if d.shape[1] != d0.shape[1]:
                    raise ValueError(
                        "Orthogonal Procrustes requires equal embedding dimensions"
                    )
                # Split fixed document anchors; fitting never uses query embeddings.
                fit_ids, heldout_ids = (
                    anchors["document"][::2],
                    anchors["document"][1::2],
                )
                rotation, fit_rank = procrustes_rotation(d[fit_ids], d0[fit_ids])
                for role, values, baseline, indices in (
                    ("query", q, q0, anchors["query"]),
                    ("document", d, d0, heldout_ids),
                ):
                    for start in range(0, len(indices), block):
                        batch_ids = indices[start : start + block]
                        aligned = (
                            np.asarray(values[batch_ids], dtype=np.float64) @ rotation
                        )
                        shifts = np.linalg.norm(aligned - baseline[batch_ids], axis=1)
                        movements.extend(
                            {
                                "model": name,
                                "role": role,
                                "sample_id": queries[index]["query_id"]
                                if role == "query"
                                else corpus[index]["doc_id"],
                                "aligned_l2": float(shift),
                                "fit_rank": fit_rank,
                                "dimension": q.shape[1],
                            }
                            for index, shift in zip(batch_ids, shifts)
                        )
            for i, (p, negative) in enumerate(fixed_pairs):
                ps = float(q[i] @ d[p])
                ns = float(q[i] @ d[negative]) if negative is not None else None
                delta_norm = (
                    float(np.linalg.norm(d[p] - d[negative]))
                    if negative is not None
                    else 0
                )
                rows.append(
                    {
                        "model": name,
                        "query_id": queries[i]["query_id"],
                        "positive_id": corpus[p]["doc_id"],
                        "unlabelled_id": corpus[negative]["doc_id"]
                        if negative is not None
                        else None,
                        "positive_score": ps,
                        "unlabelled_score": ns,
                        "margin": ps - ns if ns is not None else None,
                        "boundary_distance": (ps - ns) / delta_norm
                        if delta_norm > 1e-12
                        else None,
                        "mean_positive_cosine": float(
                            (d[sorted(relevance[i])] @ q[i]).mean()
                        ),
                    }
                )
        write_csv(out / "summary.csv", summaries)
        write_jsonl(out / "summary.jsonl", summaries)
        write_csv(out / "fixed_pairs.csv", rows)
        write_jsonl(out / "fixed_pairs.jsonl", rows)
        write_jsonl(out / "neighborhoods.jsonl", neighborhood_rows)
        write_csv(out / "neighborhoods.csv", neighborhood_rows)
        write_csv(out / "procrustes.csv", movements)
        write_jsonl(
            out / "candidates.jsonl",
            [
                {"query_id": query["query_id"], "indices": c.tolist()}
                for query, c in zip(queries, candidates)
            ],
        )
        write_json(
            out / "anchors.json", {role: ids.tolist() for role, ids in anchors.items()}
        )


def perturb_stage(config, subset, names):
    import torch
    from grpo import sample_vmf
    from policy_math import kappa_for_alignment

    root = Path(config["output_dir"]) / subset
    queries, corpus, relevance, data = load_data(root)
    settings = config.get("perturb", {})
    scope = settings.get("scope", "candidates")
    if scope not in {"candidates", "corpus"}:
        raise ValueError("perturb.scope must be candidates or corpus")
    alignments = settings.get("alignments", [0.99, 0.95, 0.90, 0.80])
    if (
        not alignments
        or any(not 0 < a < 1 for a in alignments)
        or len(set(alignments)) != len(alignments)
    ):
        raise ValueError("Unique perturb alignments strictly between 0 and 1 required")
    samples = positive_int(settings.get("samples", 32))
    k = positive_int(config.get("retrieval", {}).get("k", 10))
    block = positive_int(config.get("retrieval", {}).get("block_size", 8192))
    candidate_k = positive_int(settings.get("candidate_k", 100))
    candidates, signatures = shared_candidates(
        root, names, queries, relevance, candidate_k
    )
    rng = np.random.default_rng(config.get("seed", 42))
    selected = np.sort(
        rng.choice(
            len(queries),
            min(len(queries), positive_int(settings.get("max_queries", len(queries)))),
            replace=False,
        )
    )
    embeddings = {
        name: load_embeddings(root, name, data["signature"], len(queries), len(corpus))
        for name in names
    }
    for name in names:
        retrieval_identity = read_json(root / name / "retrieval/manifest.json")[
            "identity"
        ]
        if retrieval_identity["embeddings"] != embeddings[name][2]:
            raise ValueError("Retrieval refers to different embeddings")
        if retrieval_identity["k"] != k:
            raise ValueError(
                "Perturbation cutoff must match the cached retrieval cutoff"
            )
    for name, (q, d, signature) in embeddings.items():
        with artifact(
            root / name / "perturb",
            {
                "embeddings": signature,
                "retrieval": signatures,
                "settings": settings,
                "k": k,
                "seed": config.get("seed", 42),
            },
        ) as out:
            if out is None:
                continue
            rows = []
            for i in selected:
                ids = candidates[i]
                local_docs = np.asarray(d[ids], dtype=np.float32)
                clean_order, _ = ranked(local_docs @ q[i], ids, len(ids))
                p = next(int(j) for j in clean_order if int(j) in relevance[i])
                negative = next(
                    (int(j) for j in clean_order if int(j) not in relevance[i]), None
                )
                clean_correct = (
                    negative is not None and float(q[i] @ (d[p] - d[negative])) > 0
                )
                clean_metrics = None
                for ai, alignment in enumerate([1.0, *alignments]):
                    if alignment == 1:
                        noisy = np.asarray(q[i : i + 1])
                    else:
                        # Matched seed schedule, independent of model iteration order.
                        seed = (
                            int(config.get("seed", 42)) + int(i) * 1009 + ai * 1000003
                        ) % (2**63 - 1)
                        with torch.random.fork_rng(devices=[]):
                            torch.manual_seed(seed)
                            noisy = (
                                sample_vmf(
                                    torch.tensor(np.asarray(q[i : i + 1]).copy()),
                                    kappa_for_alignment(q.shape[1], alignment),
                                    samples,
                                )[0]
                                .float()
                                .numpy()
                            )
                        noisy = unit(noisy)
                    observed, flips = [], []
                    for vector in noisy:
                        if scope == "corpus":
                            order, _ = search(vector, d, k, block)
                        else:
                            order, _ = ranked(local_docs @ vector, ids, k)
                        observed.append(metrics(order, relevance[i], k))
                        if clean_correct:
                            flips.append(float(vector @ (d[p] - d[negative]) < 0))
                    averages = {
                        key: float(np.mean([r[key] for r in observed]))
                        for key in ("ndcg", "mrr", "hit")
                    }
                    if alignment == 1:
                        clean_metrics = averages
                    rows.append(
                        {
                            "query_id": queries[i]["query_id"],
                            "scope": scope,
                            "alignment": alignment,
                            "samples": len(noisy),
                            **averages,
                            "ndcg_drop": clean_metrics["ndcg"] - averages["ndcg"],
                            "mrr_drop": clean_metrics["mrr"] - averages["mrr"],
                            "ndcg_mc_se": float(
                                np.std([r["ndcg"] for r in observed], ddof=1)
                                / np.sqrt(len(noisy))
                            )
                            if len(noisy) > 1
                            else 0.0,
                            "correct_pair_flip_rate": float(np.mean(flips))
                            if flips
                            else None,
                        }
                    )
            write_csv(out / "queries.csv", rows)
            write_jsonl(out / "queries.jsonl", rows)


def mean_present(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(values)) if values else None


def rank_bin(rank):
    return (
        "1"
        if rank == 1
        else "2-10"
        if rank <= 10
        else "11-100"
        if rank <= 100
        else "101+"
    )


def report_stage(config, subsets, names):
    reference = config.get("reference", "E0")
    if reference not in names:
        raise ValueError("report requires the reference in --models")
    root = Path(config["output_dir"])
    summaries, paired, binned, perturb, geometry, inputs = [], [], [], [], [], {}
    cutoff = None
    for subset in subsets:
        queries, _, _, data_manifest = load_data(root / subset)
        query_ids = [row["query_id"] for row in queries]
        retrieval_signatures = {
            name: checked_manifest(root / subset / name / "retrieval")["signature"]
            for name in names
        }
        per_model = {}
        for name in names:
            path = root / subset / name / "retrieval"
            manifest = checked_manifest(path)
            embedding_manifest = read_json(
                root / subset / name / "embeddings/manifest.json"
            )
            if (
                embedding_manifest["signature"]
                != digest(embedding_manifest["identity"])
                or embedding_manifest["identity"]["data"] != data_manifest["signature"]
                or manifest["identity"]["embeddings"] != embedding_manifest["signature"]
            ):
                raise ValueError("Report retrieval/data/embedding identities differ")
            inputs[f"{subset}/{name}/retrieval"] = manifest["signature"]
            current_cutoff = manifest["identity"]["k"]
            if cutoff is not None and cutoff != current_cutoff:
                raise ValueError("Cannot compare different retrieval metric cutoffs")
            cutoff = current_cutoff
            rows = read_jsonl(path / "queries.jsonl")
            if [row["query_id"] for row in rows] != query_ids:
                raise ValueError("Report retrieval query order differs from data")
            per_model[name] = {r["query_id"]: r for r in rows}
            summaries.append(
                {
                    "subset": subset,
                    "model": name,
                    "queries": len(rows),
                    **{
                        key: mean_present(rows, key)
                        for key in ("ndcg", "mrr", "hit", "margin")
                    },
                }
            )
            path = root / subset / name / "perturb"
            if path.exists():
                noisy_manifest = checked_manifest(path)
                if (
                    noisy_manifest["identity"]["embeddings"]
                    != embedding_manifest["signature"]
                    or noisy_manifest["identity"]["retrieval"] != retrieval_signatures
                    or noisy_manifest["identity"]["k"] != cutoff
                ):
                    raise ValueError(
                        "Report perturbation and retrieval identities differ"
                    )
                inputs[f"{subset}/{name}/perturb"] = noisy_manifest["signature"]
                noisy = read_jsonl(path / "queries.jsonl")
                for a in sorted({r["alignment"] for r in noisy}, reverse=True):
                    group = [r for r in noisy if r["alignment"] == a]
                    perturb.append(
                        {
                            "subset": subset,
                            "model": name,
                            "scope": group[0]["scope"],
                            "alignment": a,
                            "queries": len(group),
                            "eligible_flip_queries": sum(
                                r["correct_pair_flip_rate"] is not None for r in group
                            ),
                            **{
                                key: mean_present(group, key)
                                for key in (
                                    "ndcg",
                                    "mrr",
                                    "hit",
                                    "ndcg_drop",
                                    "correct_pair_flip_rate",
                                )
                            },
                        }
                    )
        for name, rows in per_model.items():
            if rows.keys() != per_model[reference].keys():
                raise ValueError("Cannot pair different query sets")
            for qid, row in rows.items():
                base = per_model[reference][qid]
                paired.append(
                    {
                        "subset": subset,
                        "model": name,
                        "query_id": qid,
                        "reference_rank_bin": rank_bin(base["best_positive_rank"]),
                        **{
                            f"delta_{key}": row[key] - base[key]
                            if row[key] is not None and base[key] is not None
                            else None
                            for key in (
                                "ndcg",
                                "mrr",
                                "best_positive_score",
                                "hardest_unlabelled_score",
                                "margin",
                            )
                        },
                    }
                )
        path = root / subset / "geometry"
        if path.exists():
            geo_manifest = checked_manifest(path)
            if (
                geo_manifest["identity"]["reference"] != reference
                or set(geo_manifest["identity"]["models"]) != set(names)
                or geo_manifest["identity"]["retrieval"] != retrieval_signatures
            ):
                raise ValueError("Geometry reference/model set differs from report")
            inputs[f"{subset}/geometry"] = geo_manifest["signature"]
            geometry.extend(
                {"subset": subset, **r} for r in read_jsonl(path / "summary.jsonl")
            )
            fixed = {
                (r["model"], r["query_id"]): r
                for r in read_jsonl(path / "fixed_pairs.jsonl")
            }
            neighborhoods = {
                (r["model"], r["sample_id"]): r["retention"]
                for r in read_jsonl(path / "neighborhoods.jsonl")
                if r["role"] == "query"
            }
            for row in paired:
                if row["subset"] != subset:
                    continue
                current = fixed[row["model"], row["query_id"]]
                baseline = fixed[reference, row["query_id"]]
                for key in ("margin", "boundary_distance", "mean_positive_cosine"):
                    row[f"delta_fixed_{key}"] = (
                        current[key] - baseline[key]
                        if current[key] is not None and baseline[key] is not None
                        else None
                    )
                row["query_neighbor_retention"] = neighborhoods[
                    row["model"], row["query_id"]
                ]
    # Optional geometry may be present for only some domains: keep a rectangular CSV.
    extra_keys = {key for row in paired for key in row} - set(paired[0])
    all_keys = set(paired[0]) | extra_keys
    for row in paired:
        for key in sorted(all_keys - row.keys()):
            row[key] = None
    for subset in subsets:
        for name in names:
            for bucket in ("1", "2-10", "11-100", "101+"):
                rows = [
                    r
                    for r in paired
                    if r["subset"] == subset
                    and r["model"] == name
                    and r["reference_rank_bin"] == bucket
                ]
                if rows:
                    binned.append(
                        {
                            "subset": subset,
                            "model": name,
                            "reference_rank_bin": bucket,
                            "queries": len(rows),
                            **{
                                key: mean_present(rows, key)
                                for key in rows[0]
                                if key.startswith("delta_")
                            },
                        }
                    )
    # Content-addressed report snapshots allow retrieval-only then fuller reports.
    report_id = digest(
        {"inputs": inputs, "reference": reference, "models": names, "subsets": subsets}
    )[:16]
    destination = root / "reports" / report_id
    with artifact(
        destination,
        {"inputs": inputs, "reference": reference, "models": names, "subsets": subsets},
    ) as out:
        if out is not None:
            for filename, rows in (
                ("retrieval", summaries),
                ("paired_queries", paired),
                ("rank_bins", binned),
                ("perturb", perturb),
                ("geometry", geometry),
            ):
                write_csv(out / f"{filename}.csv", rows)
            lines = [
                "# Embedding analysis",
                "",
                "Scores are in [0, 1]; nDCG uses linear relevance gains.",
                "Reference: "
                + reference
                + ". Unlabelled candidates are not verified semantic negatives.",
                "",
                f"| Model | Domains | Macro nDCG@{cutoff} | Macro MRR@{cutoff} |",
                "|---|---:|---:|---:|",
            ]
            for name in names:
                rows = [r for r in summaries if r["model"] == name]
                lines.append(
                    f"| {name} | {len(rows)} | {mean_present(rows, 'ndcg'):.6f} | {mean_present(rows, 'mrr'):.6f} |"
                )
            lines += [
                "",
                "Per-domain retrieval: [retrieval.csv](retrieval.csv).",
                "Paired changes: [paired_queries.csv](paired_queries.csv); reference-rank bins: [rank_bins.csv](rank_bins.csv).",
                "Geometry: [geometry.csv](geometry.csv); fixed document pairs are in each subset's geometry directory.",
                "Perturbations: [perturb.csv](perturb.csv). Candidate-scope results are local reranking, not full-corpus retrieval.",
                "",
                "No significance claim is made from a single training seed. Monte Carlo SE measures sampling error only.",
                "Absent optional stages produce empty CSVs. Exact ties use descending document ID.",
                "",
            ]
            (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(destination / "report.md")


def positive_int(value):
    if isinstance(value, bool) or int(value) != value or value <= 0:
        raise ValueError(f"Expected positive integer, got {value!r}")
    return int(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["encode", "retrieval", "geometry", "perturb", "report", "all"]
    )
    parser.add_argument(
        "--config",
        required=True,
        help="JSON config; paths are relative to working directory",
    )
    parser.add_argument(
        "--models", nargs="+", help="Default: all configured model aliases"
    )
    parser.add_argument("--subsets", nargs="+", help="Default: all configured subsets")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = read_json(args.config)
    names = args.models or list(config["models"])
    subsets = args.subsets or list(config["data"]["subsets"])
    if (
        len(set(names)) != len(names)
        or len(set(subsets)) != len(subsets)
        or not names
        or not subsets
    ):
        raise ValueError("Nonempty, unique models and subsets required")
    for name in [*names, *subsets]:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
            raise ValueError(f"Unsafe model/subset name: {name!r}")
    if not set(names) <= config["models"].keys() or not set(subsets) <= set(
        config["data"]["subsets"]
    ):
        raise ValueError("Unknown model or subset")
    if set(names) & {"data", "geometry", "reports"}:
        raise ValueError("Reserved model alias")
    stages = (
        ["encode", "retrieval", "geometry", "perturb", "report"]
        if args.stage == "all"
        else [args.stage]
    )
    for stage in stages:
        if stage == "report":
            report_stage(config, subsets, names)
        else:
            for subset in subsets:
                globals()[f"{stage}_stage"](config, subset, names)
