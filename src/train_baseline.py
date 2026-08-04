import logging
import pathlib

from transformers import HfArgumentParser, set_seed

from baselines.config import BaselineArguments
from baselines.model import BaselineModel
from baselines.trainer import BaselineTrainer
from config import DataArguments, LoraArguments, ModelArguments, TrainingArguments
from train import (
    apply_gradient_checkpointing,
    build_embedding_data,
    guard_output_dir,
    load_backbone_and_tokenizer,
    parse_arguments,
    save_run_artifacts,
    setup_logging,
    shutdown_distributed,
)
from utils import BASELINE_CONFIG_SLOTS


logger = logging.getLogger(__name__)


def main() -> None:
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments, BaselineArguments)
    )
    model_args, data_args, training_args, lora_args, baseline_args = parse_arguments(
        parser=parser,
        base_slots=BASELINE_CONFIG_SLOTS,
    )

    guard_output_dir(training_args)
    setup_logging(training_args, {"Model": model_args, "Baseline": baseline_args})

    set_seed(training_args.seed)

    backbone, tokenizer = load_backbone_and_tokenizer(model_args, lora_args)
    model = BaselineModel(
        model=backbone,
        baseline_args=baseline_args,
    )
    model.train()

    apply_gradient_checkpointing(model, training_args, lora_args)

    train_dataset, eval_dataset, data_collator = build_embedding_data(data_args, training_args, tokenizer)

    trainer = BaselineTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=None,
        data_collator=data_collator,
        metric_k=baseline_args.baseline_ndcg_k,
    )

    resume = bool(
        list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
        and not training_args.overwrite_output_dir
    )
    trainer.train(resume_from_checkpoint=True if resume else None)

    save_run_artifacts(trainer, training_args, tokenizer, model_args=model_args, baseline_args=baseline_args)


if __name__ == "__main__":
    main()
    shutdown_distributed()
