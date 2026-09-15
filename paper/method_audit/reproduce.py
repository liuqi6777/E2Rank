"""Read-only method audit. Uses cached tokenizer/model; never trains or downloads.

Run: .venv/bin/python paper/method_audit/reproduce.py [--model-probe]
The JSON contains aggregates only, not training texts. Training artifacts are untouched.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, set_seed

from embedding_data import EmbeddingDataset, SingleSourceBatchSampler, DEFAULT_TASK_PROMPTS
from embedding_protocol import pool_embeddings
from grpo import GRPO, sample_vmf
from policy_math import group_advantages, kappa_for_alignment, mean_alignment


def data_probe(rows):
    sources = Counter(r["source"] for r in rows)
    pos_count = sum(sum(r["relevance"]) for r in rows)
    candidates = sum(len(r["document"]) for r in rows)
    teacher_top1_negative = 0
    positive_zero_grade = 0
    negative_nonzero_grade = 0
    pair_inversions = pair_count = 0
    for r in rows:
        rel, grades, ranks = r["relevance"], r["graded_relevance"], r["rank_labels"]
        teacher_top1_negative += rel[max(range(len(rel)), key=ranks.__getitem__)] == 0
        positive_zero_grade += sum(p == 1 and g == 0 for p, g in zip(rel, grades))
        negative_nonzero_grade += sum(p == 0 and g > 0 for p, g in zip(rel, grades))
        for i, p in enumerate(rel):
            if p:
                for j, n in enumerate(rel):
                    if not n:
                        pair_count += 1
                        pair_inversions += ranks[j] > ranks[i]
    set_seed(42)
    args = SimpleNamespace(data_path=str(ROOT / "data/processed/reasonrank_multi/train.ready.jsonl"),
                           per_dataset_max_samples=None, file_glob="*.jsonl")
    dataset = EmbeddingDataset(args, batch_size=16)
    sampler = SingleSourceBatchSampler(dataset, batch_size=16, seed=42)
    def groups(epoch):
        sampler.set_epoch(epoch)
        order = list(sampler)
        return {frozenset(order[i:i+16]) for i in range(0, len(order), 16)}
    same_groups = groups(0) == groups(1)
    kept = Counter(dataset._read_record(i)["source"] for i in dataset.entries)
    dataset.close()
    return dict(records=len(rows), candidate_occurrences=candidates,
                positive_occurrences=pos_count,
                multi_positive_records=sum(sum(r["relevance"]) > 1 for r in rows),
                mean_candidate_count=candidates/len(rows), source_counts=dict(sources),
                source_records_kept_at_microbatch16=dict(kept),
                permanently_dropped_records=len(rows)-len(dataset),
                unchanged_in_batch_groups_across_epochs=same_groups,
                teacher_top1_binary_negative_records=teacher_top1_negative,
                binary_positive_with_zero_teacher_grade=positive_zero_grade,
                binary_negative_with_nonzero_teacher_grade=negative_nonzero_grade,
                binary_positive_negative_pairs=pair_count, teacher_inverted_binary_pairs=pair_inversions)


def tokenizer_probe(rows, tokenizer):
    unique_docs = list(dict.fromkeys(d for r in rows for d in r["document"]))
    doc_lengths = {}
    for start in range(0, len(unique_docs), 256):
        texts = unique_docs[start:start+256]
        encoded = tokenizer([x + tokenizer.pad_token for x in texts], verbose=False)
        doc_lengths.update(zip(texts, map(len, encoded["input_ids"])))
    query_texts = [f'Instruct: {DEFAULT_TASK_PROMPTS}\nQuery:{r["query"]}' for r in rows]
    query_lengths = []
    for start in range(0, len(query_texts), 256):
        encoded = tokenizer([x + tokenizer.pad_token for x in query_texts[start:start+256]], verbose=False)
        query_lengths.extend(map(len, encoded["input_ids"]))
    docs = [doc_lengths[d] for r in rows for d in r["document"]]
    simple = {name: tokenizer(text)["input_ids"] for name, text in (
        ("native", "test"), ("manual_append", "test" + tokenizer.pad_token))}
    truncated = tokenizer("test " * 100 + tokenizer.pad_token, truncation=True, max_length=16)
    return dict(simple_input_ids=simple, truncated_manual_tail_ids=truncated["input_ids"][-4:],
                tokenizer_post_processor=str(tokenizer.backend_tokenizer.post_processor),
                unique_documents=len(unique_docs),
                query_token_lengths=dict(median=statistics.median(query_lengths), maximum=max(query_lengths),
                                         above512=sum(n > 512 for n in query_lengths), above8192=sum(n > 8192 for n in query_lengths)),
                document_token_lengths=dict(median=statistics.median(docs), maximum=max(docs),
                                            above1024=sum(n > 1024 for n in docs), above8192=sum(n > 8192 for n in docs)))


def policy_probe():
    """Check actual sampling/log-prob/LOO against an analytic linear-reward gradient.

    This isolates estimator correctness. It does not test discrete-reward generalization.
    """
    torch.manual_seed(718)
    dim, group, repeats = 8, 32, 4096
    kappa = kappa_for_alignment(dim, 0.9)
    raw = torch.randn(dim, requires_grad=True)
    direction = F.normalize(raw, dim=-1)
    coefficient = F.normalize(torch.randn(dim), dim=-1)
    samples = sample_vmf(direction.detach()[None], kappa, group * repeats).reshape(repeats, group, dim)
    rewards = (samples * coefficient).sum(-1)
    advantages, _ = group_advantages(rewards, baseline="leave_one_out", normalization="none")
    log_prob = GRPO._vmf_log_prob(direction[None].expand(repeats, -1), samples, torch.tensor(kappa))
    estimate = torch.autograd.grad((advantages.detach() * log_prob).mean(), raw, retain_graph=True)[0]
    expected = torch.autograd.grad(mean_alignment(dim, kappa) * (direction * coefficient).sum(), raw)[0]
    return dict(dim=dim, group=group, repeats=repeats, kappa=kappa,
                expected_alignment=0.9, sampled_alignment=float((samples * direction.detach()).sum(-1).mean()),
                gradient_cosine=float(F.cosine_similarity(estimate, expected, dim=0)),
                gradient_relative_error=float((estimate-expected).norm()/expected.norm()),
                g1_dim=1024, g1_kappa_at_alignment090=kappa_for_alignment(1024, 0.9),
                expected_squared_distance_at_alignment090=0.2)


def baseline_precision_probe():
    """Inspect score dtypes using the operations in the two training paths."""
    q = torch.tensor([[1., 0., 0.]], dtype=torch.bfloat16)
    docs = torch.tensor([[[0.6, 0.8, 0.001], [0.6, 0.8, 0.002]]], dtype=torch.bfloat16)
    # Both operands of the real baseline matmul stay bf16 without an explicit cast.
    baseline_scores = torch.matmul(docs, q.unsqueeze(-1)).squeeze(-1)
    from grpo import _ActionComponent
    qc = _ActionComponent(role="query", rollout_embeddings=q)
    dc = _ActionComponent(role="document", rollout_embeddings=docs)
    grpo_scores = GRPO._compute_score_table(qc, dc)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        autocast_scores = GRPO._compute_score_table(qc, dc)
    return dict(baseline_dtype=str(baseline_scores.dtype), grpo_without_autocast_dtype=str(grpo_scores.dtype),
                grpo_with_cpu_autocast_dtype=str(autocast_scores.dtype),
                note="CPU dtype probe only; historical CUDA/DeepSpeed runtime autocast was not inspected.")


def conditional_projection_probe():
    """An exact two-document MRR toy at d=1024; no encoder or training.

    With h=e1 and document difference .01*e1+.5*e2, the reward observes
    only coordinates 1 and 2. Reflection symmetry makes every other coordinate
    of E[R(e) * e] exactly zero. Projecting the score's tangent vector onto e2
    therefore preserves its expectation. Compare G64 LOO estimates across repeats.
    """
    torch.manual_seed(2031)
    dim, group, repeats = 1024, 64, 256
    kappa = kappa_for_alignment(dim, 0.9)
    mean = torch.zeros(1, dim)
    mean[0, 0] = 1
    samples = sample_vmf(mean, kappa, group*repeats).reshape(repeats, group, dim)
    margin = .01*samples[..., 0] + .5*samples[..., 1]
    rewards = torch.where(margin > 0, 1., .5)
    advantages, _ = group_advantages(rewards, baseline="leave_one_out", normalization="none")
    gradients = (kappa * advantages[..., None] * samples).mean(1)
    gradients[:, 0] = 0  # Tangent score at h=e1.
    projected = torch.zeros_like(gradients)
    projected[:, 1] = gradients[:, 1]
    def stats(values):
        average = values.mean(0)
        variance = float((values-average).square().sum()/(repeats-1))
        return dict(noise_variance=variance, mean_norm=float(average.norm()),
                    mean_on_known_signal_axis=float(average[1]),
                    off_signal_mean_norm=float(average[2:].norm()))
    full, projected_stats = stats(gradients), stats(projected)
    return dict(dimension=dim, group_size=group, repeats=repeats,
                full=full, conditional_projection=projected_stats,
                variance_ratio=projected_stats["noise_variance"]/full["noise_variance"],
                positive_beats_negative_fraction=float((margin>0).float().mean()),
                note="Synthetic single-query-action embedding gradient only. Does not establish encoder-gradient variance reduction in the joint G1 product estimator.")


def result_probe():
    with (ROOT / "paper/_summary/g1_bright/run_summary.csv").open() as f:
        runs = list(csv.DictReader(f))
    scores = {r["run"]:float(r["retrieval"]) for r in runs}
    prefixes = ["G1-S-MRR32", "G1-S-MRR64", "G1-S-GradedNDCG64"]
    selected = {}
    for prefix in prefixes:
        values = [scores[f"{prefix}-Seed{s}-s{s}"] for s in (42, 3407, 2026)]
        selected[prefix] = dict(values=values, mean=statistics.mean(values), sample_sd=statistics.stdev(values))
    values = [scores[n] for n in ("G1-J-CL-s42", "G1-J-CL-Seed3407-s3407", "G1-J-CL-Seed2026-s2026")]
    selected["CL"] = dict(values=values, mean=statistics.mean(values), sample_sd=statistics.stdev(values))
    selected["LL_graded_scaled_seed42"] = scores["G1-J-LL-Scaled-s42"]
    return dict(run_count=len(runs), selected=selected)


def model_probe(tokenizer, model_id):
    from transformers import AutoModel
    model = AutoModel.from_pretrained(model_id, local_files_only=True, torch_dtype=torch.float32).eval()
    texts = [
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:What is the capital of China?",
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:Explain gravity",
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ]
    vectors = []
    for append in (False, True):
        inputs = tokenizer([t+tokenizer.pad_token if append else t for t in texts], padding=True, return_tensors="pt")
        with torch.inference_mode():
            vectors.append(pool_embeddings(model(**inputs).last_hidden_state, inputs["attention_mask"], pooling_method="last"))
    return dict(model_revision=model.config._commit_hash, dtype="float32", device="cpu",
                native_similarity=(vectors[0][:2]@vectors[0][2:].T).tolist(),
                double_terminal_similarity=(vectors[1][:2]@vectors[1][2:].T).tolist(),
                same_text_protocol_cosines=(vectors[0]*vectors[1]).sum(-1).tolist(),
                note="Four official model-card example texts; confirms sensitivity, not a BRIGHT improvement.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-probe", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    path = ROOT / "data/processed/reasonrank_multi/train.ready.jsonl"
    with path.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    model_id = "Qwen/Qwen3-Embedding-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True, padding_side="left")
    result = dict(data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  torch_version=torch.__version__, data=data_probe(rows))
    print("Data audit completed", flush=True)
    result["tokenization"] = tokenizer_probe(rows, tokenizer)
    print("Token length audit completed", flush=True)
    result["policy"] = policy_probe()
    result["conditional_projection_toy"] = conditional_projection_probe()
    result["precision"] = baseline_precision_probe()
    result["results"] = result_probe()
    if args.model_probe:
        result["model_protocol_sensitivity"] = model_probe(tokenizer, model_id)
    output = Path(__file__).with_name("evidence.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
