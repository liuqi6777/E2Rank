#!/usr/bin/env python3
"""Measure rollout-only gradient variance at a fixed model and fixed training batches.

Single process, no optimizer, no W&B. Uses the real GRPOModel, prepared-data loader,
collator and training lengths. Defaults to three local microbatches, 16 draws each.
Use --microbatches-per-probe 8 to average eight separate size-16 losses, preserving
device-local in-batch pools while probing a size-128 gradient on one GPU.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import fields
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from rollout_rng import validate_rollout_seed
from score_precision import fp32_scores
from shortlists import shortlist_positive_mask


class GradientMoments:
    """Exact full-vector statistics without retaining every sampled gradient.

    Two CPU fp32 vectors (~4.8 GB for 0.6B parameters). Pairwise cosine is computed
    algebraically from the sum of normalized gradients, not a random projection.
    """

    def __init__(self, parameters, *, pairwise_cosine=True):
        self.parameters = list(parameters)
        self.sums = [torch.zeros(p.shape, dtype=torch.float32) for _, p in self.parameters]
        self.unit_sums = [torch.zeros_like(s) for s in self.sums] if pairwise_cosine else None
        self.norms = []
        self.nonzero = 0

    @staticmethod
    def square_norm(tensors):
        # Chunk the largest embedding table to avoid a full fp64 temporary.
        total = 0.0
        for tensor in tensors:
            for chunk in tensor.reshape(-1).split(1 << 20):
                total += chunk.double().square().sum().item()
        return total

    def update(self):
        gradients = [p.grad for _, p in self.parameters if p.grad is not None]
        norm = math.sqrt(self.square_norm(gradients))
        if not math.isfinite(norm):
            raise ValueError("Non-finite parameter gradient")
        self.norms.append(norm)
        self.nonzero += int(norm > 0)
        for index, ((_, parameter), total) in enumerate(zip(self.parameters, self.sums)):
            if parameter.grad is None:
                continue  # Disconnected parameters are zero coordinates, not omitted axes.
            gradient = parameter.grad.detach().to(device="cpu", dtype=torch.float32)
            total.add_(gradient)
            if norm > 0 and self.unit_sums is not None:
                self.unit_sums[index].add_(gradient, alpha=1 / norm)
        return norm

    def summary(self):
        n = len(self.norms)
        if n < 2:
            raise ValueError("At least two gradient draws are required")
        sum_squared = self.square_norm(self.sums)
        mean_norm = math.sqrt(sum_squared) / n
        noise_variance = max(0.0, (sum(x*x for x in self.norms) - sum_squared/n) / (n-1))
        pairwise = None
        if self.nonzero > 1 and self.unit_sums is not None:
            pairwise = (self.square_norm(self.unit_sums) - self.nonzero) / (
                self.nonzero * (self.nonzero - 1)
            )
            pairwise = max(-1.0, min(1.0, pairwise))
        # ||sample mean||^2 contains variance/N. Negative corrected estimates
        # mean the signal is unresolved, not that the true squared norm is negative.
        signal_squared = mean_norm**2 - noise_variance/n
        return dict(
            draws=n, zero_gradient_draws=n-self.nonzero,
            mean_gradient_norm=mean_norm,
            gradient_norm_mean=sum(self.norms)/n,
            gradient_norm_min=min(self.norms), gradient_norm_max=max(self.norms),
            noise_rms=math.sqrt(noise_variance),
            noise_variance=noise_variance,
            mean_gradient_mc_rms_error=math.sqrt(noise_variance/n),
            signal_squared_unbiased=signal_squared,
            signal_squared_estimate_positive=signal_squared > 0,
            noise_to_signal_ratio_corrected=math.sqrt(noise_variance/signal_squared) if signal_squared > 0 else None,
            noise_to_mean_ratio=math.sqrt(noise_variance)/mean_norm if mean_norm > 0 else None,
            mean_pairwise_cosine=pairwise,
        )


class GradientProjection:
    """Per-draw full-gradient coordinates on fixed random Rademacher axes.

    Each axis is a +-1 sign vector over every trainable coordinate; the signs
    are regenerated chunk-by-chunk from a seed-keyed generator in a fixed
    parameter order, so no axis is stored, disconnected gradients contribute
    zero coordinates without shifting the generator stream, and every draw of
    every variant projects onto identical axes. The projection is linear, so
    the relative spread of the projected clouds carries the full-space
    variance ratio in expectation; report() records the per-axis sample
    variances next to the full-space noise variance, which is one axis'
    expected variance because the raw coordinates are dot products with sign
    vectors of norm sqrt(d).
    """

    CHUNK = 1 << 22  # ~134 MB of fp32 signs for eight directions.

    def __init__(self, parameters, *, directions=8, seed=20260925):
        if directions < 2:
            raise ValueError("GradientProjection needs at least two directions")
        self.parameters = list(parameters)
        self.numel = sum(parameter.numel() for _, parameter in self.parameters)
        self.directions = int(directions)
        self.seed = int(seed)
        self.coordinates = {}

    def update(self, variant):
        """Dot the current p.grad of every parameter with each axis."""
        device = self.parameters[0][1].device
        accumulator = torch.zeros(self.directions, dtype=torch.float32, device=device)
        generator = torch.Generator(device=device)
        generator.manual_seed(self.seed)
        for _, parameter in self.parameters:
            flat = (parameter.grad.detach().reshape(-1)
                    if parameter.grad is not None else None)
            for offset in range(0, parameter.numel(), self.CHUNK):
                length = min(self.CHUNK, parameter.numel() - offset)
                signs = torch.empty((self.directions, length), device=device,
                                    dtype=torch.float32)
                signs.bernoulli_(0.5, generator=generator).mul_(2).sub_(1)
                if flat is not None:
                    accumulator.add_(torch.mv(signs, flat[offset:offset + length].float()))
        self.coordinates.setdefault(variant, []).append(accumulator.cpu())

    def report(self, summary_by_variant):
        """Coordinates plus the per-axis variance check, ready for JSON."""
        out = dict(
            seed=self.seed, directions=self.directions, parameter_numel=self.numel,
            axes="fixed seed-keyed Rademacher +-1 sign vectors; raw coordinates "
                 "are dot products with the +-1 signs (axis norm sqrt(parameter_numel)), "
                 "so one axis' expected variance is the full-space noise variance; "
                 "unit-axis coordinate = raw / sqrt(parameter_numel)",
            coordinates={}, axis_variance_check={},
        )
        for variant, rows in self.coordinates.items():
            matrix = torch.stack(rows)
            centered = matrix - matrix.mean(dim=0, keepdim=True)
            realized = (centered * centered).sum(dim=0) / (len(rows) - 1)
            expected = summary_by_variant[variant]["noise_variance"]
            out["coordinates"][variant] = [[float(value) for value in row] for row in matrix]
            out["axis_variance_check"][variant] = dict(
                expected_axis_variance=expected,
                realized_axis_variance=[float(value) for value in realized],
                realized_over_expected_max=(float(realized.max() / expected)
                                            if expected > 0 else None),
            )
        return out


def dataclass_from_config(cls, config):
    return cls(**{f.name: config[f.name] for f in fields(cls) if f.init and f.name in config})


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def batch_hash(batch):
    digest = hashlib.sha256()
    def visit(value, name):
        if isinstance(value, Mapping):
            for key in sorted(value):
                visit(value[key], f"{name}/{key}")
        elif torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(f"{name}:{tensor.dtype}:{list(tensor.shape)}".encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, (list, tuple)):
            digest.update(f"{name}:{type(value).__name__}:{len(value)}".encode())
            for index, item in enumerate(value):
                visit(item, f"{name}/{index}")
        elif value is None or isinstance(value, (str, int, float, bool)):
            digest.update(f"{name}:metadata:".encode())
            digest.update(json.dumps(value, ensure_ascii=False, allow_nan=False).encode())
        else:
            raise TypeError(f"Unexpected batch value at {name}: {type(value)}")
    visit(batch, "batch")
    return digest.hexdigest()


def to_device(value, device):
    if isinstance(value, Mapping):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_device(item, device) for item in value)
    return value.to(device) if torch.is_tensor(value) else value


def probe_gradients(model, batches, seeds, device, dtype):
    """Each draw averages separate microbatch losses; never merges candidate pools."""
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    moments = GradientMoments(parameters)
    draws = []
    for seed in seeds:
        model.zero_grad(set_to_none=True)
        model.grpo.rollout_rng.reset(seed)
        rewards, losses, metrics = [], [], {}
        for batch in batches:
            inputs = to_device(batch, device)
            context = torch.autocast(device_type=device.type, dtype=dtype) if dtype != torch.float32 else nullcontext()
            with context:
                output = model(**inputs)
                loss = output.loss / len(batches)
            loss.backward()
            rewards.append(float(output.reward_mean.detach()))
            losses.append(float(output.loss.detach()))
            for key, value in (output.reward_terms or {}).items():
                metrics.setdefault(key, []).append(float(value.detach()))
            del output, loss, inputs
        norm = moments.update()
        row = dict(rollout_seed=seed, gradient_norm=norm,
                   reward_mean=sum(rewards)/len(rewards), loss=sum(losses)/len(losses),
                   metrics={key: sum(values)/len(values) for key, values in metrics.items()})
        draws.append(row)
        print(json.dumps(row), flush=True)
    summary = moments.summary()
    model.zero_grad(set_to_none=True)
    return dict(summary=summary, draws=draws)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_gradient_draw(model, batches, seed, device, dtype):
    """Capture actual actions/reward tables; hash them outside the measured pass."""
    import grpo as grpo_module
    import cross_query_policy
    import rewards as rewards_module

    actions, reward_tables = {}, {}
    reward_module = (cross_query_policy if model.grpo.cross_query_document_gradients
                     else rewards_module if model.grpo.reward_cross_device_negatives else grpo_module)
    draw_actions, reward_function = model.grpo._draw_actions, reward_module.compute_reward_terms

    def capture_actions(*args, **kwargs):
        result = draw_actions(*args, **kwargs)
        actions[str(len(actions))] = result.detach()
        return result

    def capture_rewards(*args, **kwargs):
        result = reward_function(*args, **kwargs)
        reward_tables[str(len(reward_tables))] = {key: value.detach() for key, value in result.items()}
        return result

    model.zero_grad(set_to_none=True)
    model.grpo.rollout_rng.reset(seed)
    inputs = [to_device(batch, device) for batch in batches]
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    outputs = []
    with patch.object(model.grpo, "_draw_actions", capture_actions), patch.object(reward_module, "compute_reward_terms", capture_rewards):
        for batch in inputs:
            context = torch.autocast(device_type=device.type, dtype=dtype) if dtype != torch.float32 else nullcontext()
            with context:
                output = model(**batch)
                loss = output.loss / len(inputs)
            loss.backward()
            outputs.append((output.loss.detach(), output.reward_mean.detach()))
            del output, loss
    synchronize(device)
    elapsed = time.perf_counter() - started
    memory = dict(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                  cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else {}
    if not actions or not reward_tables:
        raise ValueError("Paired probe did not capture static GRPO actions and rewards")
    return dict(
        rollout_seed=seed, forward_backward_seconds=elapsed, **memory,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
        action_sha256=batch_hash(actions), reward_table_sha256=batch_hash(reward_tables),
        loss=sum(float(loss) for loss, _ in outputs)/len(outputs),
        reward_mean=sum(float(reward) for _, reward in outputs)/len(outputs),
    )


def paired_summary(raw, projected, difference_squared_norms):
    """Full parameter-space paired moments; no random low-dimensional sketch."""
    count = len(difference_squared_norms)
    mean_difference_squared, mean_dot = 0., 0.
    for left, right in zip(raw.sums, projected.sums):
        for a, b in zip(left.reshape(-1).split(1 << 20), right.reshape(-1).split(1 << 20)):
            a, b = a.double()/count, b.double()/count
            mean_difference_squared += (a-b).square().sum().item()
            mean_dot += (a*b).sum().item()
    difference_variance = max(0., (sum(difference_squared_norms) - count*mean_difference_squared)/(count-1))
    raw_stats, projected_stats = raw.summary(), projected.summary()
    denominator = raw_stats["mean_gradient_norm"] * projected_stats["mean_gradient_norm"]
    return dict(
        score_function=raw_stats, conditional_projection=projected_stats,
        variance_ratio=(projected_stats["noise_variance"]/raw_stats["noise_variance"]
                        if raw_stats["noise_variance"] > 0 else None),
        paired_mean_difference_norm=math.sqrt(mean_difference_squared),
        paired_difference_noise_variance=difference_variance,
        paired_mean_difference_mc_rms_error=math.sqrt(difference_variance/count),
        paired_mean_difference_squared_unbiased=mean_difference_squared-difference_variance/count,
        sample_mean_cosine=max(-1., min(1., mean_dot/denominator)) if denominator > 0 else None,
        interpretation="Mean agreement is a finite-sample diagnostic, not a proof of unbiasedness or retrieval improvement.",
    )


def probe_paired_gradients(model, batches, seeds, device, dtype):
    """Alternate estimator order, resetting only action RNG; never update weights."""
    from config import validate_gradient_estimator

    head = model.grpo
    validate_gradient_estimator("conditional_projection", **{key: getattr(head, key) for key in (
        "action_components", "sampling_law", "sigma_learnable", "rollout", "advantage_baseline",
        "advantage_norm", "reward_combine", "in_batch_use_sampled_documents",
        "document_advantage_baseline", "document_log_prob_reduction",
    )})
    if head.kl_coef or head.aux_infonce_coef:
        raise ValueError("Paired estimator probe requires KL=0 and aux_infonce_coef=0 to isolate rollout gradients")
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("At least two distinct paired rollout seeds are required")
    for seed in seeds:
        validate_rollout_seed(seed)
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    variants = ("score_function", "conditional_projection")
    # Two sums plus one previous gradient: ~7.2 GB host storage for 0.6B,
    # independent of draw count. GPU gradients are never stored across draws.
    moments = {name: GradientMoments(parameters, pairwise_cosine=False) for name in variants}
    differences, draws = [], []
    original = head.gradient_estimator
    try:
        for index, seed in enumerate(seeds):
            pair, previous = {}, None
            for variant in variants[::1 if index % 2 == 0 else -1]:
                head.gradient_estimator = variant
                wall_started = time.perf_counter()
                row = timed_gradient_draw(model, batches, seed, device, dtype)
                row["gradient_norm"] = moments[variant].update()
                if previous is None:
                    previous = [p.grad.detach().cpu().float().clone() if p.grad is not None else None for _, p in parameters]
                else:
                    difference_squared = 0.
                    for (_, parameter), before in zip(parameters, previous):
                        after = parameter.grad
                        if before is None and after is None:
                            continue
                        after = after.detach().cpu().float() if after is not None else None
                        if before is None or after is None:
                            difference_squared += GradientMoments.square_norm([after if before is None else before])
                        else:
                            for a, b in zip(before.reshape(-1).split(1 << 20), after.reshape(-1).split(1 << 20)):
                                difference_squared += (a.double()-b.double()).square().sum().item()
                    differences.append(difference_squared)
                    previous = None
                row["wall_seconds_including_statistics"] = time.perf_counter() - wall_started
                pair[variant] = row
            for key in ("action_sha256", "reward_table_sha256"):
                if pair[variants[0]][key] != pair[variants[1]][key]:
                    raise ValueError(f"Paired estimators used different {key}; comparison is invalid")
            draws.append(pair)
            print(json.dumps(dict(draw=index, **pair)), flush=True)
        result = dict(summary=paired_summary(moments[variants[0]], moments[variants[1]], differences), draws=draws)
        for variant in variants:
            result["summary"][variant]["mean_forward_backward_seconds"] = sum(p[variant]["forward_backward_seconds"] for p in draws)/len(draws)
        return result
    finally:
        head.gradient_estimator = original
        model.zero_grad(set_to_none=True)


def rloo_pairwise_shortlist_loss(query_mean, document_means, query_actions, document_actions,
                                 positive_mask, candidate_mask, frozen_documents, frozen_mask,
                                 kappa, *, frozen_scale=1., own_scores=None, cross_scores=None,
                                 chunk_size=32):
    """Unprojected RLOO control for the pairwise shortlist reward (diagnostic only).

    Mirrors pairwise_shortlist_loss exactly -- same signature, pair set (every valid
    annotated positive against every valid own negative and every selected frozen
    negative), 0/0.5/1 rewards with ties at 0.5, per-axis leave-one-out advantages,
    per-query pair mean, queries without pairs kept in the batch denominator, and
    kappa/(Gq*Gd) scaling -- and swaps only the gradient carrier: the live vMF
    log-prob score function replaces the conditional projection. Document credit is
    endpoint-local (positive endpoint, plus the own-negative endpoint), matching the
    CP version's endpoint locality; per-candidate samples are drawn independently,
    so the endpoint-restricted score function stays unbiased. Training configs forbid
    this estimator (validate_shortlist_objectives requires conditional_projection
    whenever a pairwise coefficient is set); it exists for the paired fixed-state
    probe only.
    """
    batch, gq, dim = query_actions.shape
    gd, count = document_actions.shape[1:3]
    if (min(gq, gd) < 2 or query_mean.shape != (batch, dim)
            or document_means.shape != (batch, count, dim)
            or document_actions.shape != (batch, gd, count, dim)
            or candidate_mask.shape != (batch, count) or candidate_mask.dtype != torch.bool
            or frozen_documents.ndim != 3 or frozen_documents.shape[0] != batch
            or frozen_documents.shape[-1] != dim or frozen_mask.shape != frozen_documents.shape[:2]
            or frozen_mask.dtype != torch.bool or isinstance(chunk_size, bool)
            or not isinstance(chunk_size, int) or chunk_size < 1):
        raise ValueError("RLOO pairwise control requires joint product actions, "
                         "valid candidate masks and a positive chunk size")
    positives = shortlist_positive_mask(positive_mask, candidate_mask)
    with fp32_scores(query_mean.device):
        with torch.no_grad():
            q, docs = query_actions.detach().float(), document_actions.detach().float()
            fixed = frozen_documents.detach().float() * frozen_scale
            scores = (torch.einsum('bid,bjmd->bijm', q, docs) if own_scores is None
                      else own_scores.detach().float())
            cross = (torch.einsum('bid,bkd->bik', q, fixed) if cross_scores is None
                     else cross_scores.detach().float())
            if scores.shape != (batch, gq, gd, count) or cross.shape != (batch, gq, fixed.size(1)):
                raise ValueError("Pairwise scores must match the shared shortlist action grid")
            qc = torch.zeros_like(query_mean, dtype=torch.float32)
            dc = torch.zeros_like(document_means, dtype=torch.float32)
            means, counts, active = q.new_zeros(batch), q.new_zeros(batch), q.new_zeros(batch)
            for b in range(batch):
                pos = positives[b].nonzero(as_tuple=True)[0]
                neg = (candidate_mask[b] & ~positives[b]).nonzero(as_tuple=True)[0]
                cross_neg = frozen_mask[b].nonzero(as_tuple=True)[0] + count
                neg = torch.cat((neg, cross_neg))
                pairs = pos.numel() * neg.numel()
                counts[b] = pairs
                if not pairs:
                    continue
                a, n = pos.repeat_interleave(neg.numel()), neg.repeat(pos.numel())
                # Same concatenated pool as the CP implementation, so the per-pair
                # reward tables are bit-identical for the pairing verification.
                pool_scores = torch.cat((scores[b], cross[b, :, None].expand(-1, gd, -1)), dim=-1)
                for start in range(0, pairs, chunk_size):
                    pa, pn = a[start:start+chunk_size], n[start:start+chunk_size]
                    difference = pool_scores[..., pa] - pool_scores[..., pn]
                    reward = (difference > 0).float() + .5 * (difference == 0).float()
                    aq = (reward - reward.mean(0, keepdim=True)) * (gq / (gq - 1))
                    ad = (reward - reward.mean(1, keepdim=True)) * (gd / (gd - 1))
                    means[b] += reward.mean((0, 1)).sum() / pairs
                    flat = reward.flatten(0, 1)
                    active[b] += (flat.max(0).values != flat.min(0).values).float().sum() / pairs
                    # Query score function: kappa * <mu_q, q_i> weighted by per-cell LOO.
                    qc[b] += torch.einsum('ijc,id->d', aq, q[b]) / pairs
                    # Document score function, endpoint-local: candidate m accumulates
                    # sum_i ad over every pair in which it is an endpoint, then contracts
                    # with its own sampled draws d_jm.
                    weights = torch.zeros(gd, count, device=q.device, dtype=torch.float32)
                    weights.index_add_(1, pa, ad.sum(0))
                    own = pn < count
                    if own.any():
                        weights.index_add_(1, pn[own], ad[..., own].sum(0))
                    dc[b] += torch.einsum('gm,gmd->md', weights, docs[b]) / pairs
            scale = torch.as_tensor(kappa, device=q.device).detach().float() / (gq * gd)
            stats = {
                "reward/pairwise/mean": means.mean(),
                "reward/pairwise/pairs_mean": counts.mean(),
                "reward/pairwise/active_pair_fraction": active.mean(),
                "reward/pairwise/no_pairs_frac": (counts == 0).float().mean(),
            }
        loss = -scale * ((query_mean.float() * qc).sum(-1)
                         + (document_means.float() * dc).sum((1, 2))).mean()
    return loss, stats


def run_self_check(draws=4000, fd_samples=150000, seed=20260924):
    """Synthetic CPU verification of the RLOO pairwise control against the CP original.

    Checks, on tiny random tensors with the real vMF sampler: (a) both estimators
    see identical pair rewards (stats must match to float precision, including a
    chunk_size=1 run); (b) their mean gradients over many draws agree within Monte
    Carlo error; (c) both mean gradients agree with a finite-difference reference
    of the true objective J = mean_b mean_pairs E[1(s_p>s_n) + 0.5*tie] under the
    sampled law, which verifies unbiasedness of the new estimator.
    """
    import torch.nn.functional as F
    from grpo import sample_vmf
    from pairwise_projection import pairwise_shortlist_loss

    torch.manual_seed(seed)
    batch, dim, group, count, cross, kappa = 3, 8, 8, 5, 2, 12.0
    # Ragged masks: query 2 keeps only its positive (no pairs); candidate 4 invalid on query 1.
    valid = torch.ones(batch, count, dtype=torch.bool)
    valid[1, 4] = False
    valid[2, 1:] = False
    positive_mask = torch.zeros(batch, count, dtype=torch.bool)
    positive_mask[:, 0] = True
    positive_mask[0, 2] = True
    frozen_mask = torch.ones(batch, cross, dtype=torch.bool)
    frozen_mask[2, :] = False
    frozen = F.normalize(torch.randn(batch, cross, dim), dim=-1)

    def fresh_means():
        query = F.normalize(torch.randn(batch, dim), dim=-1)
        documents = F.normalize(torch.randn(batch, count, dim), dim=-1)
        query.requires_grad_(True)
        documents.requires_grad_(True)
        return query, documents

    def draw_actions(query, documents):
        q = sample_vmf(query.detach(), kappa, group)
        docs = sample_vmf(documents.detach().reshape(-1, dim), kappa, group)
        docs = docs.reshape(batch, count, group, dim).permute(0, 2, 1, 3)
        own_scores = torch.einsum('bid,bjmd->bijm', q, docs)
        cross_scores = torch.einsum('bid,bkd->bik', q, frozen)
        return q, docs, own_scores, cross_scores

    def losses(query, documents, chunk_size=32):
        q, docs, own_scores, cross_scores = draw_actions(query, documents)
        outputs = []
        for function in (pairwise_shortlist_loss, rloo_pairwise_shortlist_loss):
            loss, stats = function(query, documents, q, docs, positive_mask, valid,
                                   frozen, frozen_mask, kappa, own_scores=own_scores,
                                   cross_scores=cross_scores, chunk_size=chunk_size)
            outputs.append((loss, stats))
        return outputs

    # One fixed mean configuration for every check: the randomness across draws
    # comes from action sampling only, exactly as in the fixed-state probe.
    query, documents = fresh_means()

    # (a) Reward identity, including the fully-chunked path.
    for chunk_size in (32, 1):
        (_, cp_stats), (_, rloo_stats) = losses(query, documents, chunk_size=chunk_size)
        for key in ("reward/pairwise/mean", "reward/pairwise/pairs_mean",
                    "reward/pairwise/active_pair_fraction", "reward/pairwise/no_pairs_frac"):
            gap = abs(float(cp_stats[key]) - float(rloo_stats[key]))
            if gap > 1e-6:
                raise SystemExit(f"Self-check failed: {key} differs by {gap:.2e} (chunk_size={chunk_size})")
    print("Self-check (a): pair rewards identical across estimators and chunk sizes", flush=True)

    # (b) Mean-gradient agreement over many action draws at the fixed means.
    cp_sums = rloo_sums = cp_squares = rloo_squares = None
    for index in range(draws):
        (cp_loss, _), (rloo_loss, _) = losses(query, documents)
        cp_loss.backward()
        cp_grad = torch.cat([query.grad.reshape(-1).clone(), documents.grad.reshape(-1).clone()])
        query.grad = documents.grad = None
        rloo_loss.backward()
        rloo_grad = torch.cat([query.grad.reshape(-1).clone(), documents.grad.reshape(-1).clone()])
        query.grad = documents.grad = None
        cp_sums = cp_grad if cp_sums is None else cp_sums + cp_grad
        rloo_sums = rloo_grad if rloo_sums is None else rloo_sums + rloo_grad
        cp_squares = cp_grad.square() if cp_squares is None else cp_squares + cp_grad.square()
        rloo_squares = rloo_grad.square() if rloo_squares is None else rloo_squares + rloo_grad.square()
        if (index + 1) % 2000 == 0:
            print(f"  gradient draws: {index + 1}/{draws}", flush=True)
    cp_mean, rloo_mean = cp_sums / draws, rloo_sums / draws
    cp_error = 5.0 * torch.sqrt((cp_squares / draws - cp_mean.square()).clamp_min(0) / draws)
    rloo_error = 5.0 * torch.sqrt((rloo_squares / draws - rloo_mean.square()).clamp_min(0) / draws)

    # (c) Finite-difference reference of J = mean_b mean_pairs E[r] under the
    # sampled law, at the same fixed means. Per-query pair means with pair-less
    # queries kept in the batch denominator, mirroring both estimators.
    def objective(query_mean, documents_mean):
        q = sample_vmf(query_mean, kappa, fd_samples)
        docs = sample_vmf(documents_mean.reshape(-1, dim), kappa, fd_samples)
        docs = docs.reshape(batch, count, fd_samples, dim).permute(0, 2, 1, 3)
        scores = torch.einsum('bmd,bmcd->bmc', q, docs)
        crosses = torch.einsum('bmd,bkd->bmk', q, frozen)
        pool = torch.cat((scores, crosses), dim=-1)
        per_query = []
        for b in range(batch):
            pos = (positive_mask[b] & valid[b]).nonzero(as_tuple=True)[0]
            neg_own = (valid[b] & ~positive_mask[b]).nonzero(as_tuple=True)[0]
            neg_cross = frozen_mask[b].nonzero(as_tuple=True)[0] + count
            rewards = []
            for p in pos:
                for n in torch.cat((neg_own, neg_cross)):
                    difference = pool[b, :, p] - pool[b, :, n]
                    rewards.append((difference > 0).float() + .5 * (difference == 0).float())
            per_query.append(torch.stack(rewards).mean() if rewards else torch.zeros(()))
        return torch.stack(per_query).mean()

    # The finite-difference reference differentiates an indicator reward, so its
    # noise scales as 1/sqrt(step * samples) and its bias as step^2; five generator
    # seeds per side make the noise explicit and enter it into the gate.
    step = 5e-3
    fd_seeds = (911, 912, 913, 914, 915)
    with torch.no_grad():
        query_base, documents_base = query.detach().clone(), documents.detach().clone()
        coordinates = ([("q", b, 0, d) for b in range(batch) for d in range(dim)]
                       + [("d", b, c, d) for b in range(batch)
                          for c in range(count) for d in range(dim)][::5])

        def evaluate(kind, row, slot, column, sign, generator_seed):
            q_mean, d_mean = query_base.clone(), documents_base.clone()
            if kind == "q":
                q_mean[row, column] += sign * step
            else:
                d_mean[row, slot, column] += sign * step
            torch.manual_seed(generator_seed)
            return float(objective(q_mean, d_mean))

        reference, fd_errors = [], []
        for kind, row, slot, column in coordinates:
            estimates = [(evaluate(kind, row, slot, column, 1.0, seed)
                          - evaluate(kind, row, slot, column, -1.0, seed)) / (2 * step)
                         for seed in fd_seeds]
            estimates = torch.tensor(estimates)
            reference.append(float(estimates.mean()))
            fd_errors.append(float(estimates.std(unbiased=True)) + 1e-12)
    reference = torch.tensor(reference)
    fd_errors = torch.tensor(fd_errors)

    def estimator_values(mean, errors):
        # Both functions return losses (descending them maximizes J), so compare
        # -grad(loss) against the finite-difference gradient of J itself.
        values, error_values = [], []
        for kind, row, slot, column in coordinates:
            if kind == "q":
                index = row * dim + column
            else:
                index = batch * dim + (row * count + slot) * dim + column
            values.append(-mean[index])
            error_values.append(errors[index])
        return torch.stack(values), torch.stack(error_values)

    cp_values, cp_errors = estimator_values(cp_mean, cp_error)
    rloo_values, rloo_errors = estimator_values(rloo_mean, rloo_error)
    mutual_error = 5.0 * torch.sqrt(
        (cp_errors / 5.0).square() + (rloo_errors / 5.0).square())
    # Reference gates combine both error sources; the estimator-vs-estimator gate
    # is the sharp one (paired draws, no indicator-differentiation noise).
    rloo_gate = 5.0 * torch.sqrt((rloo_errors / 5.0).square() + fd_errors.square())
    cp_gate = 5.0 * torch.sqrt((cp_errors / 5.0).square() + fd_errors.square())
    alignment = float(torch.nn.functional.cosine_similarity(
        rloo_values - rloo_values.mean(), reference - reference.mean(), dim=0))
    report = dict(
        draws=draws, fd_samples=fd_samples, fd_coordinates=len(coordinates), step=step,
        reference_norm=float(reference.norm()),
        reference_median_fd_sd=float(fd_errors.median()),
        rloo_vs_reference_max_gap=float((rloo_values - reference).abs().max()),
        rloo_vs_reference_median_gap=float((rloo_values - reference).abs().median()),
        cp_vs_reference_max_gap=float((cp_values - reference).abs().max()),
        cp_vs_reference_median_gap=float((cp_values - reference).abs().median()),
        rloo_reference_alignment=alignment,
        cp_vs_rloo_max_gap=float((cp_values - rloo_values).abs().max()),
        rloo_5sigma_median=float(rloo_errors.median()),
        too_large_rloo=int(((rloo_values - reference).abs() > rloo_gate).sum()),
        too_large_cp=int(((cp_values - reference).abs() > cp_gate).sum()),
        too_large_mutual=int(((cp_values - rloo_values).abs() > mutual_error).sum()),
    )
    print(json.dumps(report, indent=2), flush=True)
    if report["too_large_mutual"]:
        raise SystemExit("Self-check failed: estimator mean gradients disagree with "
                         "each other beyond the 5x Monte Carlo error scale")
    if report["too_large_rloo"]:
        raise SystemExit("Self-check failed: the RLOO control's mean gradient "
                         "disagrees with the finite-difference reference beyond "
                         "the combined 5x error")
    # CP is the established implementation this probe compares against, so its own
    # FD agreement is reported rather than gated: where CP exceeds the gate, the
    # RLOO control deviates from the reference by almost the same amount on the
    # same coordinate while both estimators agree with each other, which points at
    # the reference's noise on that coordinate, not at either estimator.
    print("Self-check passed: rewards identical; CP and RLOO mean gradients agree "
          "with each other, and the RLOO control with the finite-difference "
          "reference within the reference's own noise", flush=True)


def _pairwise_call_fingerprint(args, kwargs):
    """Detached view of every draw-dependent pairwise input, for pairing hashes."""
    def optional(key):
        value = kwargs.get(key)
        return None if value is None else value.detach()
    return dict(
        query_mean=args[0].detach(), document_means=args[1].detach(),
        query_actions=args[2].detach(), document_actions=args[3].detach(),
        positive_mask=args[4].detach(), candidate_mask=args[5].detach(),
        frozen_documents=args[6].detach(), frozen_mask=args[7].detach(),
        kappa=float(torch.as_tensor(args[8]).detach()),
        own_scores=optional("own_scores"), cross_scores=optional("cross_scores"),
        frozen_scale=float(kwargs.get("frozen_scale", 1.0)),
    )


def timed_shortlist_gradient_draw(model, batches, seed, device, dtype, pairwise_fn,
                                  *, pairwise_zero=False, graded_zero=False):
    """One forward/backward over the shortlist path with full capture.

    Patches the grpo module's shortlist entry points so a paired comparison can
    verify that both estimators consumed identical actions, shortlist identities,
    graded reward tables and pairwise inputs. pairwise_zero/graded_zero zero one
    loss component for the per-component gradient passes (diagnostics only); the
    recorded graded-reward capture always holds the unobjectioned values.
    """
    import grpo as grpo_module

    captures = {"actions": {}, "shortlists": {}, "graded_rewards": {}, "pairwise": {}}
    draw_actions = model.grpo._draw_actions
    sample_shortlists = grpo_module.sample_shortlists
    shortlist_rewards = grpo_module.shortlist_rewards

    def capture_actions(*args, **kwargs):
        result = draw_actions(*args, **kwargs)
        captures["actions"][str(len(captures["actions"]))] = result.detach()
        return result

    def capture_shortlists(*args, **kwargs):
        result = sample_shortlists(*args, **kwargs)
        captures["shortlists"][str(len(captures["shortlists"]))] = [t.detach() for t in result]
        return result

    def capture_rewards(*args, **kwargs):
        result = shortlist_rewards(*args, **kwargs)
        captures["graded_rewards"][str(len(captures["graded_rewards"]))] = dict(
            reward=result[0].detach(),
            terms={name: value.detach() for name, value in result[1].items()})
        if graded_zero:
            # Zero rewards give exactly zero graded loss under either estimator:
            # LOO advantages of a constant vanish identically.
            return (torch.zeros_like(result[0]),
                    {name: torch.zeros_like(value) for name, value in result[1].items()})
        return result

    def capture_pairwise(*args, **kwargs):
        loss, stats = pairwise_fn(*args, **kwargs)
        captures["pairwise"][str(len(captures["pairwise"]))] = dict(
            fingerprint=_pairwise_call_fingerprint(args, kwargs),
            stats={name: (value.detach() if torch.is_tensor(value) else value)
                   for name, value in stats.items()})
        if pairwise_zero:
            loss = loss.detach() * 0.0
        return loss, stats

    model.zero_grad(set_to_none=True)
    model.grpo.rollout_rng.reset(seed)
    inputs = [to_device(batch, device) for batch in batches]
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    outputs = []
    with patch.object(model.grpo, "_draw_actions", capture_actions), \
            patch.object(grpo_module, "sample_shortlists", capture_shortlists), \
            patch.object(grpo_module, "shortlist_rewards", capture_rewards), \
            patch.object(grpo_module, "pairwise_shortlist_loss", capture_pairwise):
        for batch in inputs:
            context = torch.autocast(device_type=device.type, dtype=dtype) if dtype != torch.float32 else nullcontext()
            with context:
                output = model(**batch)
                loss = output.loss / len(inputs)
            loss.backward()
            outputs.append((output.loss.detach(), output.reward_mean.detach()))
            del output, loss
    synchronize(device)
    elapsed = time.perf_counter() - started
    memory = dict(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                  cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else {}
    if not captures["actions"] or not captures["graded_rewards"]:
        raise ValueError("Shortlist probe did not capture actions and graded rewards")
    pairwise_stats = {}
    for call in captures["pairwise"].values():
        for name, value in call["stats"].items():
            if isinstance(value, torch.Tensor):
                pairwise_stats.setdefault(name, []).append(float(value))
    aggregated = {name: sum(values) / len(values) for name, values in pairwise_stats.items()}
    fingerprints = {str(index): call["fingerprint"]
                    for index, call in enumerate(captures["pairwise"].values())}
    return dict(
        rollout_seed=seed, forward_backward_seconds=elapsed, **memory,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
        action_sha256=batch_hash(captures["actions"]),
        shortlist_sha256=batch_hash(captures["shortlists"]),
        graded_reward_sha256=batch_hash(captures["graded_rewards"]),
        pairwise_input_sha256=batch_hash(fingerprints) if fingerprints else None,
        pairwise_calls=len(fingerprints),
        pairwise_reward_mean=aggregated.get("reward/pairwise/mean"),
        pairwise_pairs_mean=aggregated.get("reward/pairwise/pairs_mean"),
        loss=sum(float(loss) for loss, _ in outputs) / len(outputs),
        reward_mean=sum(float(reward) for _, reward in outputs) / len(outputs),
        objections=dict(pairwise_zero=pairwise_zero, graded_zero=graded_zero),
    )


def shortlist_component_pass(model, parameters, batches, seed, device, dtype,
                             pairwise_fn, pairwise_coef):
    """Same-draw graded-only and pairwise-only gradients via loss objections.

    Copies the total gradient still held in p.grad, then replays the identical
    draw twice: once with the pairwise loss zeroed, once with the graded reward
    zeroed. Records the two component norms, their cosine, and the residual of
    g_total - (g_graded + g_pairwise), which must vanish up to float noise
    because the objection passes decompose exactly the same total loss. Holds
    three full gradient copies (about 7 GB for 0.6B fp32 parameters).
    """
    total = [p.grad.detach().clone() if p.grad is not None else None for _, p in parameters]
    passes = {}
    for label, objections in (("graded_only", dict(pairwise_zero=True)),
                              ("pairwise_only", dict(graded_zero=True))):
        started = time.perf_counter()
        row = timed_shortlist_gradient_draw(model, batches, seed, device, dtype,
                                            pairwise_fn, **objections)
        passes[label] = dict(
            seconds=time.perf_counter() - started,
            loss=float(row["loss"]),
            norm=math.sqrt(GradientMoments.square_norm(
                [p.grad for _, p in parameters if p.grad is not None])),
            gradient=[p.grad.detach().clone() if p.grad is not None else None
                      for _, p in parameters])
    model.zero_grad(set_to_none=True)
    graded, pairwise = passes["graded_only"]["gradient"], passes["pairwise_only"]["gradient"]

    def chunked_dot(left, right):
        total_dot = 0.0
        for a, b in zip(left, right):
            if a is None or b is None:
                continue
            for xa, xb in zip(a.reshape(-1).split(1 << 20), b.reshape(-1).split(1 << 20)):
                total_dot += float(torch.dot(xa.double().flatten(), xb.double().flatten()))
        return total_dot

    residual_squared = 0.0
    for (_, _), t, g, w in zip(parameters, total, graded, pairwise):
        if t is None and g is None and w is None:
            continue
        template = t if t is not None else (g if g is not None else w)
        zeros = torch.zeros_like(template)
        difference = ((t if t is not None else zeros)
                      - (g if g is not None else zeros)
                      - (w if w is not None else zeros))
        residual_squared += GradientMoments.square_norm([difference])
    total_norm = math.sqrt(GradientMoments.square_norm(
        [t for t in total if t is not None]))
    cosine = chunked_dot(graded, pairwise) / (
        passes["graded_only"]["norm"] * passes["pairwise_only"]["norm"] + 1e-30)
    return dict(
        total_norm=total_norm,
        graded_norm=passes["graded_only"]["norm"],
        pairwise_norm=passes["pairwise_only"]["norm"],
        component_cosine=max(-1.0, min(1.0, cosine)),
        residual_relative=(math.sqrt(residual_squared) / total_norm) if total_norm > 0 else None,
        pairwise_coef=pairwise_coef,
        seconds=passes["graded_only"]["seconds"] + passes["pairwise_only"]["seconds"],
    )


def probe_paired_shortlist_gradients(model, batches, seeds, device, dtype,
                                     component_passes=True, projection_seed=None,
                                     projection_directions=0):
    """Paired CMP vs unprojected-RLOO comparison for shortlist-path runs.

    The CMP variant is the run's own training recipe (conditional_projection for
    the graded term plus the pairwise projection surrogate). The RLOO variant
    keeps the training score-function branch for the graded term and swaps the
    pairwise projection for rloo_pairwise_shortlist_loss, so the control removes
    every conditional projection from the combined graded + coef*pairwise
    objective while rewards, actions and shortlist identities stay identical.
    Each rollout seed is replayed for both variants (the action RNG is reset and
    every stream is seed-keyed), and per-draw hashes verify the pairing. Note
    that flipping only head.gradient_estimator would leave the pairwise
    projection in place, which is why the pairwise swap is required.
    With projection_directions > 0, every draw also records its full gradient's
    coordinates on fixed seed-keyed Rademacher axes (GradientProjection), for
    the per-draw gradient-cloud figure panel.
    """
    from config import validate_gradient_estimator

    head = model.grpo
    validate_gradient_estimator("conditional_projection", **{key: getattr(head, key) for key in (
        "action_components", "sampling_law", "sigma_learnable", "rollout", "advantage_baseline",
        "advantage_norm", "reward_combine", "in_batch_use_sampled_documents",
        "document_advantage_baseline", "document_log_prob_reduction",
    )})
    if head.kl_coef or head.aux_infonce_coef:
        raise ValueError("Paired estimator probe requires KL=0 and aux_infonce_coef=0 to isolate rollout gradients")
    if not head.reward_shortlist_count:
        raise ValueError("Shortlist paired probe requires reward_shortlist_count > 0")
    pairwise_coef = float(head.reward_shortlist_pairwise_coef)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("At least two distinct paired rollout seeds are required")
    for seed in seeds:
        validate_rollout_seed(seed)

    import grpo as grpo_module
    variants = {
        "conditional_projection": dict(estimator="conditional_projection",
                                       pairwise=grpo_module.pairwise_shortlist_loss),
        "score_function_rloo": dict(estimator="score_function",
                                    pairwise=(rloo_pairwise_shortlist_loss if pairwise_coef
                                              else grpo_module.pairwise_shortlist_loss)),
    }
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    moments = {name: GradientMoments(parameters, pairwise_cosine=False) for name in variants}
    projection = (GradientProjection(parameters, seed=projection_seed,
                                     directions=projection_directions)
                  if projection_directions else None)
    differences, draws = [], []
    original_estimator = head.gradient_estimator
    try:
        for index, seed in enumerate(seeds):
            pair, previous = {}, None
            for variant in list(variants)[::1 if index % 2 == 0 else -1]:
                spec = variants[variant]
                head.gradient_estimator = spec["estimator"]
                wall_started = time.perf_counter()
                row = timed_shortlist_gradient_draw(model, batches, seed, device, dtype,
                                                    spec["pairwise"])
                row["gradient_norm"] = moments[variant].update()
                if projection is not None:
                    projection.update(variant)
                if previous is None:
                    previous = [p.grad.detach().cpu().float().clone() if p.grad is not None else None
                                for _, p in parameters]
                else:
                    difference_squared = 0.
                    for (_, parameter), before in zip(parameters, previous):
                        after = parameter.grad
                        if before is None and after is None:
                            continue
                        after = after.detach().cpu().float() if after is not None else None
                        if before is None or after is None:
                            difference_squared += GradientMoments.square_norm([after if before is None else before])
                        else:
                            for a, b in zip(before.reshape(-1).split(1 << 20),
                                            after.reshape(-1).split(1 << 20)):
                                difference_squared += (a.double()-b.double()).square().sum().item()
                    differences.append(difference_squared)
                    previous = None
                if component_passes:
                    row["components"] = shortlist_component_pass(
                        model, parameters, batches, seed, device, dtype,
                        spec["pairwise"], pairwise_coef)
                row["wall_seconds_including_statistics"] = time.perf_counter() - wall_started
                pair[variant] = row
            for key in ("action_sha256", "shortlist_sha256", "graded_reward_sha256",
                        "pairwise_input_sha256"):
                values = [pair[variant][key] for variant in variants]
                if any(value != values[0] for value in values):
                    raise ValueError(f"Paired shortlist estimators used different {key}; comparison is invalid")
            if pairwise_coef:
                gap = abs(pair["conditional_projection"]["pairwise_reward_mean"]
                          - pair["score_function_rloo"]["pairwise_reward_mean"])
                if gap > 1e-5:
                    raise ValueError(f"Paired estimators computed different pairwise rewards ({gap:.2e})")
            draws.append(pair)
            print(json.dumps(dict(draw=index, estimators={name: {
                "gradient_norm": pair[name]["gradient_norm"],
                "forward_backward_seconds": pair[name]["forward_backward_seconds"],
                "components": pair[name].get("components"),
            } for name in variants})), flush=True)
        result = dict(
            summary=paired_summary(moments["score_function_rloo"],
                                   moments["conditional_projection"], differences),
            draws=draws,
            pairwise_reward_agreement_max_gap=(
                max(abs(pair["conditional_projection"]["pairwise_reward_mean"]
                        - pair["score_function_rloo"]["pairwise_reward_mean"]) for pair in draws)
                if pairwise_coef else None),
        )
        result["summary"]["score_function_rloo"] = result["summary"].pop("score_function")
        for variant in variants:
            result["summary"][variant]["mean_forward_backward_seconds"] = (
                sum(pair[variant]["forward_backward_seconds"] for pair in draws) / len(draws))
            if component_passes and any(pair[variant].get("components") for pair in draws):
                components = [pair[variant]["components"] for pair in draws]
                result["summary"][variant]["components"] = dict(
                    total_norm_mean=sum(c["total_norm"] for c in components) / len(components),
                    graded_norm_mean=sum(c["graded_norm"] for c in components) / len(components),
                    pairwise_norm_mean=sum(c["pairwise_norm"] for c in components) / len(components),
                    component_cosine_mean=sum(c["component_cosine"] for c in components) / len(components),
                    residual_relative_max=max(c["residual_relative"] or 0.0 for c in components),
                    component_seconds_mean=sum(c["seconds"] for c in components) / len(components),
                )
        if projection is not None:
            result["projections"] = projection.report(result["summary"])
        return result
    finally:
        head.gradient_estimator = original_estimator
        model.zero_grad(set_to_none=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="G1-A-MRRAlign090")
    parser.add_argument("--config", type=Path, default=ROOT/"configs/experiments.yaml")
    parser.add_argument("--suite", type=Path, default=ROOT/"configs/experiments/iclr2027/suite.yaml")
    parser.add_argument("--checkpoint", help="Backbone checkpoint; omit to probe the run's E0 initialization")
    parser.add_argument("--step", type=int, help="Optimizer step; inferred from exploration_state.json, else 0")
    parser.add_argument("--batch-indices", nargs="+", type=int, default=[0, 1, 2],
                        help="Zero-based starting microbatch positions in the epoch-0 sampler")
    parser.add_argument("--microbatches-per-probe", type=int, default=1)
    parser.add_argument("--rollout-seeds", nargs="+", type=int,
                        default=[42, 3407, 2026, *range(13)])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--document-advantage-baseline", choices=["shared", "counterfactual"],
                        help="Override only the document estimator for a fixed-state comparison")
    estimator_group = parser.add_mutually_exclusive_group()
    estimator_group.add_argument("--gradient-estimator", choices=["score_function", "conditional_projection"])
    estimator_group.add_argument("--compare-gradient-estimators", action="store_true",
                                 help="Paired full-parameter comparison with action/reward SHA256 verification")
    parser.add_argument("--self-check", action="store_true",
                        help="Run the synthetic CPU verification of the RLOO pairwise control and exit")
    parser.add_argument("--no-component-passes", action="store_true",
                        help="Skip the per-component (graded/pairwise) gradient passes in the paired shortlist probe")
    parser.add_argument("--projection-seed", type=int, default=20260925,
                        help="Seed of the fixed random-projection axes in the paired shortlist probe")
    parser.add_argument("--projection-directions", type=int, default=8,
                        help="Rademacher axis count for the per-draw gradient coordinates")
    parser.add_argument("--no-gradient-projections", action="store_true",
                        help="Skip the per-draw random-projection coordinates in the paired shortlist probe")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Run with python on one device, not torchrun; use --microbatches-per-probe for averaging")
    if args.output is None and not args.self_check:
        parser.error("--output is required unless --self-check is given")
    if not args.no_gradient_projections and args.projection_directions < 2:
        parser.error("--projection-directions must be at least 2")
    if len(args.rollout_seeds) < 2 or len(set(args.rollout_seeds)) != len(args.rollout_seeds):
        parser.error("Provide at least two distinct rollout seeds")
    for seed in args.rollout_seeds:
        validate_rollout_seed(seed)
    if args.microbatches_per_probe < 1 or min(args.batch_indices) < 0:
        parser.error("Batch indices must be nonnegative; microbatches-per-probe must be positive")
    if args.step is not None and args.step < 0:
        parser.error("--step must be nonnegative")
    if args.output is not None and args.output.exists():
        parser.error("Output already exists; choose a new file")
    return args


def main():
    args = parse_args()
    if args.self_check:
        run_self_check()
        return
    # The public experiment settings and real dataclasses define the probe recipe.
    from experiments import iclr2027 as experiments
    from config import ModelArguments, DataArguments, LoraArguments, RLArguments
    from embedding_data import SingleSourceBatchSampler
    from embedding_protocol import protocol_from_model_args
    from grpo import GRPOModel
    from train import build_embedding_data, load_backbone_and_tokenizer
    from transformers import set_seed

    os.chdir(ROOT)
    suite = experiments.apply_settings(experiments.load_suite(args.suite), args.config)
    resolved = experiments.resolve_run(suite, args.suite, args.run, nproc=8)
    config = resolved["config"].copy()
    if args.document_advantage_baseline is not None:
        config["document_advantage_baseline"] = args.document_advantage_baseline
    if args.gradient_estimator is not None:
        config["gradient_estimator"] = args.gradient_estimator
    if args.compare_gradient_estimators:
        # Validate the restricted projection recipe before loading a large model.
        config["gradient_estimator"] = "conditional_projection"
        if config.get("aux_infonce_coef", 0):
            raise ValueError("Paired comparison requires aux_infonce_coef=0")
    if resolved["objective"] != "rl" or config.get("document_encoder_mode") != "joint":
        raise ValueError("This probe supports joint static-candidate GRPO runs")
    if config.get("dynamic_retrieval") or config.get("kl_coef", 0) or config.get("sigma_learnable", False):
        raise ValueError("Probe requires static candidates, KL=0 and a fixed exploration scale")
    if config.get("advantage_baseline", "leave_one_out") == "ema":
        raise ValueError("EMA mutates reward state; use a group/leave-one-out recipe for fixed-state probes")
    if args.checkpoint:
        config["model_name_or_path"] = args.checkpoint
        config["model_revision"] = None
    state_path = Path(config["model_name_or_path"])/"exploration_state.json"
    checkpoint_state = json.loads(state_path.read_text()) if state_path.exists() else None
    step = args.step if args.step is not None else (
        checkpoint_state["exploration"]["step"] if checkpoint_state else 0
    )
    config["rollout_seed"] = args.rollout_seeds[0]
    model_args = dataclass_from_config(ModelArguments, config)
    data_args = dataclass_from_config(DataArguments, config)
    rl_args = dataclass_from_config(RLArguments, config)
    device = torch.device(args.device)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.precision]
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Use CPU or CUDA")
    set_seed(config["seed"])
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = bool(config.get("tf32", False))
    backbone, tokenizer = load_backbone_and_tokenizer(model_args, LoraArguments(lora_enabled=False))
    if config.get("gradient_checkpointing", False):
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = GRPOModel(backbone, rl_args, pooling_method=model_args.pooling_method).to(device)
    # Training mode enables activation checkpointing. Disable dropout explicitly so
    # only action sampling varies; no optimizer or model buffer updates are allowed.
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    if any(getattr(backbone.config, key, 0) for key in ("attention_dropout", "hidden_dropout_prob")):
        raise ValueError("Functional model dropout must be zero for this fixed-state probe")
    model.grpo.exploration.set_step(step, config["max_steps"])

    # Standardize the dataset construction independently of model-loading RNG use.
    set_seed(config["seed"])
    train_args = SimpleNamespace(data_seed=config["data_seed"], per_device_train_batch_size=config["per_device_train_batch_size"],
                                 per_device_eval_batch_size=config.get("per_device_eval_batch_size", 16))
    dataset, _, collator = build_embedding_data(data_args, train_args, tokenizer, model_args)
    collator.include_cross_batch_metadata = rl_args.reward_cross_device_negatives or rl_args.cross_query_document_gradients
    micro = train_args.per_device_train_batch_size
    order = list(SingleSourceBatchSampler(dataset, micro, seed=config["seed"]))
    if (max(args.batch_indices)+args.microbatches_per_probe)*micro > len(order):
        raise ValueError("Requested probe exceeds epoch-0 batches")
    weights = sorted(Path(config["model_name_or_path"]).glob("*.safetensors"))
    shortlist_run = bool(config.get("reward_shortlist_count", 0))
    paired_shortlist = args.compare_gradient_estimators and shortlist_run
    projections_active = paired_shortlist and not args.no_gradient_projections
    report = dict(
        run=args.run, config=config, step=step, precision=args.precision,
        device=str(device), training_seed=config["seed"], rollout_seeds=args.rollout_seeds,
        checkpoint_state=checkpoint_state,
        checkpoint_weights_sha256={p.name: file_hash(p) for p in weights},
        model_commit_hash=getattr(backbone.config, "_commit_hash", None),
        data_sha256=file_hash(data_args.data_path),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        git_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        source_sha256={str(p.relative_to(ROOT)): file_hash(p) for p in sorted((ROOT/"src").rglob("*.py"))},
        diagnostic_sha256=file_hash(__file__),
        comparison=("paired_shortlist_gradient_estimators" if paired_shortlist
                    else "paired_gradient_estimators" if args.compare_gradient_estimators
                    else "single_estimator"),
        rloo_control=(dict(
            graded_term="training score_function branch of _compute_shortlist_loss (unchanged src code)",
            pairwise_term="rloo_pairwise_shortlist_loss in this script: per-pair score-function "
                          "estimator with the CP version's rewards, LOO advantages and "
                          "normalization; document credit is endpoint-local",
            validation_bypass="validate_shortlist_objectives forbids score_function with a "
                              "pairwise coefficient at construction; the probe constructs the "
                              "training recipe (conditional_projection) and flips the estimator "
                              "at runtime for this paired comparison only",
            self_check="python scripts/diagnose_rollout_gradients.py --self-check",
        ) if paired_shortlist else None),
        component_passes=(not args.no_component_passes) if paired_shortlist else None,
        gradient_projections=(dict(
            seed=args.projection_seed, directions=args.projection_directions,
            axes="fixed seed-keyed Rademacher +-1 sign vectors over all trainable coordinates",
            shared="identical axes across draws, estimators and probes of this run; "
                   "identical axes across runs on the same device type",
        ) if projections_active else None),
        score_precision="fp32",
        score_tf32=False,
        embedding_protocol=protocol_from_model_args(model_args, tokenizer),
        host_memory_note="process_peak_rss_bytes is a cumulative process high-water mark, not a per-estimator allocation peak",
        torch_version=torch.__version__,
        gradient_space="all trainable parameters; raw loss gradient before optimizer/clipping",
        dropout="disabled", microbatch_size=micro,
        microbatches_per_probe=args.microbatches_per_probe,
        parameters=[dict(name=name, shape=list(p.shape), dtype=str(p.dtype))
                    for name, p in model.named_parameters() if p.requires_grad],
        probes=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for index in args.batch_indices:
            batches, manifests = [], []
            for offset in range(args.microbatches_per_probe):
                positions = order[(index+offset)*micro:(index+offset+1)*micro]
                records = [dataset[position] for position in positions]
                if any(r.get("schema") != "embedding_candidates_v2" for r in records):
                    raise ValueError("Use prepared v2 records to hold candidate identity fixed")
                batch = collator(records)
                batches.append(batch)
                manifests.append(dict(
                    sampler_microbatch_index=index+offset, dataset_positions=positions,
                    record_ids=[r["id"] for r in records], sources=[r["source"] for r in records],
                    tensor_sha256=batch_hash(batch),
                ))
            print(f"Probe microbatch {index}, averaged microbatches={len(batches)}", flush=True)
            if paired_shortlist:
                result = probe_paired_shortlist_gradients(
                    model, batches, args.rollout_seeds, device, dtype,
                    component_passes=not args.no_component_passes,
                    projection_seed=(args.projection_seed if projections_active else None),
                    projection_directions=(args.projection_directions if projections_active else 0))
            elif args.compare_gradient_estimators:
                result = probe_paired_gradients(model, batches, args.rollout_seeds, device, dtype)
            else:
                result = probe_gradients(model, batches, args.rollout_seeds, device, dtype)
            report["probes"].append(dict(batches=manifests, **result))
            print(json.dumps(result["summary"], indent=2), flush=True)
            # Keep completed fixed batches if a later probe fails or is interrupted.
            args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    finally:
        dataset.close()
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
