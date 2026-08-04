from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor
from transformers import HfArgumentParser, PreTrainedModel, Trainer as HFTrainer, set_seed
from transformers.file_utils import ModelOutput

from config import DataArguments, LoraArguments, ModelArguments, TrainingArguments
from embedding_data import build_slate_inputs
from grpo import pool_last_token_embedding
from grpo_trainer import EmbeddingTrainerMixin
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

INFONCE_TEMPERATURE = 0.03


@dataclass
class BaselineModelOutput(ModelOutput):
    loss: Optional[Tensor] = None


class BaselineTrainer(EmbeddingTrainerMixin, HFTrainer):
    train_metric_names = ()


def compute_infonce_loss(
    scores: Tensor,
    relevance_labels: Tensor,
    temperature: float = 0.03,
) -> Tensor:
    if temperature <= 0:
        raise ValueError(f"infonce temperature must be positive, got {temperature}")

    positive_exists = relevance_labels.max(dim=-1).values > 0
    positive_indices = relevance_labels.argmax(dim=-1, keepdim=True)
    scaled_scores = scores / float(temperature)
    positive_scores = scaled_scores.gather(dim=1, index=positive_indices).squeeze(1)
    partition = torch.logsumexp(scaled_scores, dim=-1)
    loss = partition - positive_scores
    return torch.where(positive_exists, loss, torch.zeros_like(loss))


class BaselineModel(nn.Module):
    def __init__(self, model: PreTrainedModel):
        super().__init__()
        self.model = model
        self.config = self.model.config

    def encode(self, model_inputs: Dict[str, Tensor]) -> Tensor:
        return pool_last_token_embedding(
            self.model(**model_inputs).last_hidden_state,
            model_inputs["attention_mask"],
            normalize=True,
        )

    def forward(
        self,
        query: Dict[str, Tensor] = None,
        positive_document: Dict[str, Tensor] = None,
        negative_document: Dict[str, Tensor] = None,
        relevance_labels: Tensor = None,
    ) -> BaselineModelOutput:
        batch_size, slate_length = relevance_labels.shape
        query_embeddings = self.encode(query)
        document_inputs = build_slate_inputs(
            positive_document=positive_document,
            negative_document=negative_document,
            batch_size=batch_size,
            slate_length=slate_length,
        )
        document_embeddings = self.encode(document_inputs).reshape(batch_size, slate_length, -1)
        scores = torch.matmul(document_embeddings, query_embeddings.unsqueeze(-1)).squeeze(-1)

        per_sample_loss = compute_infonce_loss(
            scores=scores,
            relevance_labels=relevance_labels,
            temperature=INFONCE_TEMPERATURE,
        )
        return BaselineModelOutput(loss=per_sample_loss.mean())

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.model.gradient_checkpointing_enable(*args, **kwargs)

    def enable_input_require_grads(self):
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()


def main() -> None:
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    model_args, data_args, training_args, lora_args = parse_arguments(
        parser=parser,
        base_slots=BASELINE_CONFIG_SLOTS,
    )

    guard_output_dir(training_args)
    setup_logging(training_args, {"Model": model_args})

    set_seed(training_args.seed)

    backbone, tokenizer = load_backbone_and_tokenizer(model_args, lora_args)
    model = BaselineModel(model=backbone)
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
    )

    resume = bool(
        list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
        and not training_args.overwrite_output_dir
    )
    trainer.train(resume_from_checkpoint=True if resume else None)

    save_run_artifacts(trainer, training_args, tokenizer, model_args=model_args)


if __name__ == "__main__":
    main()
    shutdown_distributed()