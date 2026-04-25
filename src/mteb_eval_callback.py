import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

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
    def __init__(self, eval_args: MTEBEvalArguments):
        self.eval_args = eval_args
        self.tasks = _parse_tasks(eval_args.mteb_eval_tasks)
        self.trainer = None

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

    def on_save(self, args, state, control, **kwargs):
        if not self.enabled or not self._is_world_process_zero(args):
            return control

        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(checkpoint_dir):
            logger.warning("Skipping MTEB eval because checkpoint does not exist: %s", checkpoint_dir)
            return control

        output_root = self.eval_args.mteb_eval_output_dir or os.path.join(args.output_dir, "mteb_eval")
        output_dir = os.path.join(output_root, f"checkpoint-{state.global_step}")
        os.makedirs(output_dir, exist_ok=True)

        model = kwargs.get("model")
        was_training = bool(getattr(model, "training", False)) if model is not None else False
        eval_arguments = self._build_eval_arguments(checkpoint_dir=checkpoint_dir, output_dir=output_dir)
        logger.info(
            "Running in-training MTEB eval at step %s with args: %s",
            state.global_step,
            asdict(self.eval_args),
        )

        try:
            tasks = get_tasks(
                eval_arguments.tasks,
                _parse_optional_list(eval_arguments.langs),
                eval_arguments.benchmark,
            )
            eval_model = get_model(
                eval_arguments.model,
                precision=eval_arguments.precision,
                **eval_arguments.model_kwargs,
            )
            results = run_eval(eval_model, tasks, eval_arguments, **eval_arguments.run_kwargs)
        except Exception:
            logger.exception("MTEB eval failed at step %s; continuing training.", state.global_step)
            return control
        finally:
            if was_training and model is not None:
                model.train()

        metrics = {}
        for result in results or []:
            score = _result_main_score(result)
            if score is not None:
                metrics[f"eval_mteb/{_result_task_name(result)}/main_score"] = score
        self._log_metrics(metrics)
        return control
