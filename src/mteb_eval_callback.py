import json
import logging
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from transformers import TrainerCallback

from config import MTEBEvalArguments

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_mteb.run_mteb import EvalArguments, get_model, get_tasks, run_eval


logger = logging.getLogger(__name__)

DEFAULT_MTEB_MODEL_KWARGS = {
    "max_length": 8192,
    "attn_type": "causal",
    "pooler_type": "last",
    "do_norm": True,
    "use_instruction": True,
    "instruction_template": "Instruct: {}\nQuery:",
    "instruction_dict_path": "eval_mteb/scripts/task_prompts.json",
}


def _parse_json_object(value: str | None, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must be a JSON object")
    return parsed


def _parse_tasks(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [task.strip() for task in value if task.strip()]
    return [task.strip() for task in value.split(",") if task.strip()]


def _parse_optional_list(value: str | list[str] | tuple[str, ...] | None) -> list[str] | None:
    parsed = _parse_tasks(value)
    return parsed or None


def _result_task_name(result: Any) -> str:
    task_name = getattr(result, "task_name", None)
    if task_name:
        return str(task_name)
    task = getattr(result, "task", None)
    metadata = getattr(task, "metadata", None)
    name = getattr(metadata, "name", None)
    return str(name or "unknown")


@contextmanager
def _disable_deepspeed_zero3():
    """Temporarily detach the global HfDeepSpeedConfig so that nested
    ``from_pretrained`` calls do not partition new models via ``zero.Init``.

    The trainer's HfDeepSpeedConfig registers itself globally; any subsequent
    model load in the same process inherits ``zero.Init`` partitioning, which
    leaves embedding weights as 1-D rank-local shards and breaks downstream
    forward passes. We swap the weak-ref out for the duration of the load and
    restore it afterward so the trainer keeps its zero3 state.
    """
    try:
        from transformers.integrations import deepspeed as ds_mod
    except Exception:
        yield
        return
    attr = "_hf_deepspeed_config_weak_ref"
    saved = getattr(ds_mod, attr, None)
    if saved is None:
        yield
        return
    setattr(ds_mod, attr, None)
    try:
        yield
    finally:
        setattr(ds_mod, attr, saved)


class _DistEncodeWrapper:
    """Sharded encode across DDP ranks via a gloo group.

    Rank 0 calls ``encode`` (driven by mteb); ranks > 0 sit in ``serve`` waiting
    for broadcast commands. Each rank encodes ``sentences[rank::world_size]`` on
    its local Qwen3Embedding copy, then all_gather merges the embeddings.
    """

    def __init__(self, inner, group):
        self.inner = inner
        self.group = group
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        self.mteb_model_meta = inner.mteb_model_meta

    def encode(self, sentences, *, task_name, prompt_type=None, **kwargs):
        cmd = {
            "op": "encode",
            "task_name": task_name,
            "prompt_type": prompt_type,
            "kwargs": kwargs,
            "sentences": list(sentences),
        }
        self._broadcast(cmd)
        return self._encode_local(cmd, return_full=True)

    def stop(self):
        self._broadcast({"op": "stop"})

    def serve(self):
        while True:
            cmd = self._broadcast(None)
            if cmd is None or cmd.get("op") == "stop":
                return
            self._encode_local(cmd, return_full=False)

    def _broadcast(self, cmd):
        objs = [cmd] if self.rank == 0 else [None]
        dist.broadcast_object_list(objs, src=0, group=self.group)
        return objs[0]

    def _encode_local(self, cmd, return_full):
        sentences = cmd["sentences"]
        local = sentences[self.rank :: self.world_size]
        if local:
            local_emb = self.inner.encode(
                local,
                task_name=cmd["task_name"],
                prompt_type=cmd["prompt_type"],
                **cmd["kwargs"],
            )
        else:
            local_emb = np.zeros((0, 0), dtype=np.float32)

        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, local_emb, group=self.group)
        if not return_full:
            return None

        n = len(sentences)
        ref = next((g for g in gathered if g.shape[0] > 0), None)
        if ref is None:
            return np.zeros((n, 0), dtype=np.float32)
        out = np.empty((n, ref.shape[1]), dtype=ref.dtype)
        for r, emb in enumerate(gathered):
            if emb.shape[0] > 0:
                out[r :: self.world_size] = emb
        return out


def _result_main_score(result: Any) -> float | None:
    get_score = getattr(result, "get_score", None)
    if callable(get_score):
        try:
            return float(get_score())
        except Exception as exc:
            logger.warning("Could not read MTEB main score from %s: %s", _result_task_name(result), exc)
            return None
    main_score = getattr(result, "main_score", None)
    if main_score is not None:
        return float(main_score)
    return None


class MTEBEvalCallback(TrainerCallback):
    PRETRAIN_TAG = "checkpoint-0"

    def __init__(self, eval_args: MTEBEvalArguments):
        self.eval_args = eval_args
        self.tasks = _parse_tasks(eval_args.mteb_eval_tasks)
        self.trainer = None
        self._gloo_group = None
        if dist.is_available() and dist.is_initialized():
            self._gloo_group = dist.new_group(
                backend="gloo", timeout=timedelta(hours=6)
            )

    @property
    def enabled(self) -> bool:
        return bool(self.tasks or self.eval_args.mteb_eval_benchmark)

    def bind_trainer(self, trainer) -> "MTEBEvalCallback":
        self.trainer = trainer
        return self

    def _is_world_process_zero(self, args) -> bool:
        if self.trainer is not None:
            return bool(self.trainer.is_world_process_zero())
        return bool(getattr(args, "should_save", True))

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        if not metrics:
            return
        if self.trainer is not None:
            self.trainer.log(metrics)
        else:
            logger.info("MTEB eval metrics: %s", metrics)

    def _build_eval_arguments(self, checkpoint_dir: str, output_dir: str) -> EvalArguments:
        model_kwargs = {**DEFAULT_MTEB_MODEL_KWARGS}
        model_kwargs.update(_parse_json_object(self.eval_args.mteb_eval_model_kwargs, "mteb_eval_model_kwargs"))
        encode_kwargs = _parse_json_object(self.eval_args.mteb_eval_encode_kwargs, "mteb_eval_encode_kwargs")

        return EvalArguments(
            model=checkpoint_dir,
            tasks=",".join(self.tasks) if self.tasks else None,
            benchmark=self.eval_args.mteb_eval_benchmark,
            langs=self.eval_args.mteb_eval_langs,
            output_dir=output_dir,
            batch_size=self.eval_args.mteb_eval_batch_size,
            precision=self.eval_args.mteb_eval_precision,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            run_kwargs=_parse_json_object(self.eval_args.mteb_eval_run_kwargs, "mteb_eval_run_kwargs"),
        )

    def on_train_begin(self, args, state, control, **kwargs):
        """Run one MTEB eval on the initial (pre-training) weights.

        The in-training eval loads the model from disk, so we first persist the
        untouched weights to ``checkpoint-0`` (mirrors on_save; includes the LoRA
        adapter and grpo_state) and then reuse the exact same eval path.
        """
        if not self.enabled:
            return control

        checkpoint_dir = os.path.join(args.output_dir, self.PRETRAIN_TAG)
        if self.trainer is not None:
            self.trainer.save_model(checkpoint_dir)
        if self._gloo_group is not None:
            dist.barrier(group=self._gloo_group)

        if self._gloo_group is None:
            self._run_eval_single(args, state, tag=self.PRETRAIN_TAG, **kwargs)
            return control

        try:
            self._run_eval_distributed(args, state, tag=self.PRETRAIN_TAG, **kwargs)
        finally:
            dist.barrier(group=self._gloo_group)
        return control

    def on_save(self, args, state, control, **kwargs):
        if not self.enabled:
            return control

        if self._gloo_group is None:
            # Single-process training: just run on this rank.
            self._run_eval_single(args, state, **kwargs)
            return control

        try:
            self._run_eval_distributed(args, state, **kwargs)
        finally:
            dist.barrier(group=self._gloo_group)
        return control

    def _prepare_paths(self, args, state, tag: str | None = None):
        label = tag if tag is not None else f"checkpoint-{state.global_step}"
        checkpoint_dir = os.path.join(args.output_dir, label)
        output_root = self.eval_args.mteb_eval_output_dir or os.path.join(args.output_dir, "mteb_eval")
        output_dir = os.path.join(output_root, label)
        return checkpoint_dir, output_dir

    def _load_eval_model(self, eval_arguments, local_rank):
        # Pin the eval model to this rank's GPU so each DDP rank uses its own card.
        model_kwargs = dict(eval_arguments.model_kwargs)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            model_kwargs.setdefault("device", f"cuda:{local_rank}")
        with _disable_deepspeed_zero3():
            eval_model = get_model(
                eval_arguments.model,
                precision=eval_arguments.precision,
                **model_kwargs,
            )
        # Stop run_eval from spawning its own multiprocessing workers.
        eval_model.world_size = 1
        return eval_model

    def _run_eval_distributed(self, args, state, tag=None, **kwargs):
        is_rank0 = self._is_world_process_zero(args)
        local_rank = int(getattr(args, "local_rank", 0) or 0)
        checkpoint_dir, output_dir = self._prepare_paths(args, state, tag=tag)

        if not os.path.isdir(checkpoint_dir):
            if is_rank0:
                logger.warning("Skipping MTEB eval because checkpoint does not exist: %s", checkpoint_dir)
            return

        if is_rank0:
            os.makedirs(output_dir, exist_ok=True)

        eval_arguments = self._build_eval_arguments(checkpoint_dir=checkpoint_dir, output_dir=output_dir)
        eval_arguments.encode_kwargs.setdefault("batch_size", eval_arguments.batch_size)

        if is_rank0:
            logger.info(
                "Running in-training MTEB eval at step %s (world_size=%d) with args: %s",
                state.global_step,
                dist.get_world_size(group=self._gloo_group),
                asdict(self.eval_args),
            )

        # All ranks load their own eval model copy onto cuda:local_rank.
        try:
            if is_rank0:
                logger.info("MTEB eval: loading model on all ranks from %s", checkpoint_dir)
            eval_model = self._load_eval_model(eval_arguments, local_rank)
        except Exception:
            logger.exception("MTEB eval: failed to load model on rank %d", local_rank)
            return

        wrapper = _DistEncodeWrapper(eval_model, self._gloo_group)

        training_model = kwargs.get("model")
        was_training = bool(getattr(training_model, "training", False)) if training_model is not None else False
        results = None

        try:
            if is_rank0:
                logger.info("MTEB eval: resolving tasks...")
                tasks = get_tasks(
                    eval_arguments.tasks,
                    _parse_optional_list(eval_arguments.langs),
                    eval_arguments.benchmark,
                )
                logger.info("MTEB eval: resolved %d tasks; running...", len(tasks) if tasks else 0)
                try:
                    results = run_eval(wrapper, tasks, eval_arguments, **eval_arguments.run_kwargs)
                    logger.info("MTEB eval: finished, got %d results", len(results) if results else 0)
                finally:
                    # Always release the worker ranks, even if mteb threw mid-task.
                    wrapper.stop()
            else:
                wrapper.serve()
        except Exception:
            logger.exception(
                "MTEB eval failed at step %s on rank %d; continuing training.",
                state.global_step,
                local_rank,
            )
        finally:
            del eval_model, wrapper
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if was_training and training_model is not None:
                training_model.train()

        if is_rank0 and results:
            metrics = {}
            for result in results:
                score = _result_main_score(result)
                if score is not None:
                    metrics[f"eval_mteb/{_result_task_name(result)}/main_score"] = score
            self._log_metrics(metrics)

    def _run_eval_single(self, args, state, tag=None, **kwargs):
        checkpoint_dir, output_dir = self._prepare_paths(args, state, tag=tag)
        if not os.path.isdir(checkpoint_dir):
            logger.warning("Skipping MTEB eval because checkpoint does not exist: %s", checkpoint_dir)
            return
        os.makedirs(output_dir, exist_ok=True)

        eval_arguments = self._build_eval_arguments(checkpoint_dir=checkpoint_dir, output_dir=output_dir)
        eval_arguments.encode_kwargs.setdefault("batch_size", eval_arguments.batch_size)

        training_model = kwargs.get("model")
        was_training = bool(getattr(training_model, "training", False)) if training_model is not None else False
        logger.info("Running in-training MTEB eval at step %s (single-process)", state.global_step)

        try:
            tasks = get_tasks(
                eval_arguments.tasks,
                _parse_optional_list(eval_arguments.langs),
                eval_arguments.benchmark,
            )
            with _disable_deepspeed_zero3():
                eval_model = get_model(
                    eval_arguments.model,
                    precision=eval_arguments.precision,
                    **eval_arguments.model_kwargs,
                )
            eval_model.world_size = 1
            results = run_eval(eval_model, tasks, eval_arguments, **eval_arguments.run_kwargs)
        except Exception:
            logger.exception("MTEB eval failed at step %s; continuing training.", state.global_step)
            return
        finally:
            if was_training and training_model is not None:
                training_model.train()

        metrics = {}
        for result in results or []:
            score = _result_main_score(result)
            if score is not None:
                metrics[f"eval_mteb/{_result_task_name(result)}/main_score"] = score
        self._log_metrics(metrics)
