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
import subprocess
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from rollout_rng import validate_rollout_seed


class GradientMoments:
    """Exact full-vector statistics without retaining every sampled gradient.

    Two CPU fp32 vectors (~4.8 GB for 0.6B parameters). Pairwise cosine is computed
    algebraically from the sum of normalized gradients, not a random projection.
    """

    def __init__(self, parameters):
        self.parameters = list(parameters)
        self.sums = [torch.zeros(p.shape, dtype=torch.float32) for _, p in self.parameters]
        self.unit_sums = [torch.zeros_like(s) for s in self.sums]
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
        for (_, parameter), total, unit in zip(self.parameters, self.sums, self.unit_sums):
            if parameter.grad is None:
                continue  # Disconnected parameters are zero coordinates, not omitted axes.
            gradient = parameter.grad.detach().to(device="cpu", dtype=torch.float32)
            total.add_(gradient)
            if norm > 0:
                unit.add_(gradient, alpha=1 / norm)
        return norm

    def summary(self):
        n = len(self.norms)
        if n < 2:
            raise ValueError("At least two gradient draws are required")
        sum_squared = self.square_norm(self.sums)
        mean_norm = math.sqrt(sum_squared) / n
        noise_variance = max(0.0, (sum(x*x for x in self.norms) - sum_squared/n) / (n-1))
        pairwise = None
        if self.nonzero > 1:
            pairwise = (self.square_norm(self.unit_sums) - self.nonzero) / (
                self.nonzero * (self.nonzero - 1)
            )
            pairwise = max(-1.0, min(1.0, pairwise))
        return dict(
            draws=n, zero_gradient_draws=n-self.nonzero,
            mean_gradient_norm=mean_norm,
            gradient_norm_mean=sum(self.norms)/n,
            gradient_norm_min=min(self.norms), gradient_norm_max=max(self.norms),
            noise_rms=math.sqrt(noise_variance),
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
            digest.update(tensor.numpy().tobytes())
        else:
            raise TypeError(f"Unexpected batch value at {name}: {type(value)}")
    visit(batch, "batch")
    return digest.hexdigest()


def to_device(value, device):
    if isinstance(value, Mapping):
        return {key: to_device(item, device) for key, item in value.items()}
    return value.to(device)


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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="G1-A-MRRAlign090")
    parser.add_argument("--config", type=Path, default=ROOT/"configs/experiments.yaml")
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
    from grpo import GRPOModel
    from train import build_embedding_data, load_backbone_and_tokenizer
    from transformers import set_seed

    os.chdir(ROOT)
    suite = experiments.apply_settings(experiments.load_suite(), args.config)
    resolved = experiments.resolve_run(suite, experiments.DEFAULT_SUITE, args.run, nproc=8)
    config = resolved["config"].copy()
    if args.document_advantage_baseline is not None:
        config["document_advantage_baseline"] = args.document_advantage_baseline
    if resolved["objective"] != "rl" or config.get("document_encoder_mode") != "joint":
        raise ValueError("This probe supports joint static-candidate GRPO runs")
    if config.get("dynamic_retrieval") or config.get("kl_coef", 0) or config.get("sigma_learnable", False):
        raise ValueError("Probe requires static candidates, KL=0 and a fixed exploration scale")
    if config.get("advantage_baseline", "leave_one_out") == "ema":
        raise ValueError("EMA mutates reward state; use a group/leave-one-out recipe for fixed-state probes")
    if args.checkpoint:
        config["model_name_or_path"] = args.checkpoint
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
    train_args = SimpleNamespace(per_device_train_batch_size=config["per_device_train_batch_size"],
                                 per_device_eval_batch_size=config.get("per_device_eval_batch_size", 16))
    dataset, _, collator = build_embedding_data(data_args, train_args, tokenizer, model_args)
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
            result = probe_gradients(model, batches, args.rollout_seeds, device, dtype)
            report["probes"].append(dict(batches=manifests, **result))
            print(json.dumps(result["summary"], indent=2), flush=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    finally:
        dataset.close()
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
