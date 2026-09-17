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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Run with python on one device, not torchrun; use --microbatches-per-probe for averaging")
    if len(args.rollout_seeds) < 2 or len(set(args.rollout_seeds)) != len(args.rollout_seeds):
        parser.error("Provide at least two distinct rollout seeds")
    for seed in args.rollout_seeds:
        validate_rollout_seed(seed)
    if args.microbatches_per_probe < 1 or min(args.batch_indices) < 0:
        parser.error("Batch indices must be nonnegative; microbatches-per-probe must be positive")
    if args.step is not None and args.step < 0:
        parser.error("--step must be nonnegative")
    if args.output.exists():
        parser.error("Output already exists; choose a new file")
    return args


def main():
    args = parse_args()
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
        comparison="paired_gradient_estimators" if args.compare_gradient_estimators else "single_estimator",
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
            probe = probe_paired_gradients if args.compare_gradient_estimators else probe_gradients
            result = probe(model, batches, args.rollout_seeds, device, dtype)
            report["probes"].append(dict(batches=manifests, **result))
            print(json.dumps(result["summary"], indent=2), flush=True)
            # Keep completed fixed batches if a later probe fails or is interrupted.
            args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    finally:
        dataset.close()
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
