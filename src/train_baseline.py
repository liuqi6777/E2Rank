from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import HfArgumentParser, PreTrainedModel, Trainer as HFTrainer, set_seed
from transformers.file_utils import ModelOutput

from config import (
    BaselineArguments,
    DataArguments,
    LoraArguments,
    ModelArguments,
    MTEBEvalArguments,
    TrainingArguments,
)
from embedding_data import build_slate_inputs
from embedding_protocol import pool_embeddings
from grpo_trainer import EmbeddingTrainerMixin
from mteb_eval_callback import MTEBEvalCallback
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

@dataclass
class BaselineModelOutput(ModelOutput):
    loss: Optional[Tensor] = None


class BaselineTrainer(EmbeddingTrainerMixin, HFTrainer):
    train_metric_names = ()


def compute_in_batch_positive_scores(
    query_embeddings: Tensor,
    positive_embeddings: Tensor,
) -> Tensor:
    """Score every query against the other samples' detached positives.

    The diagonal is excluded because each query's own positive is already in its
    listwise slate. Detaching only the cross-query use matches RL's frozen
    in-batch candidates; the same document still receives gradients through its
    own sample's slate.
    """
    if query_embeddings.dim() != 2 or positive_embeddings.dim() != 2:
        raise ValueError(
            "query_embeddings and positive_embeddings must both be 2D, "
            f"got {tuple(query_embeddings.shape)} and {tuple(positive_embeddings.shape)}"
        )
    if query_embeddings.shape != positive_embeddings.shape:
        raise ValueError(
            "query_embeddings and positive_embeddings must have matching shapes, "
            f"got {tuple(query_embeddings.shape)} and {tuple(positive_embeddings.shape)}"
        )

    batch_size = query_embeddings.size(0)
    if batch_size <= 1:
        return query_embeddings.new_empty((batch_size, 0))

    cross_scores = torch.matmul(query_embeddings, positive_embeddings.detach().T)
    off_diagonal = ~torch.eye(batch_size, device=cross_scores.device, dtype=torch.bool)
    return cross_scores.masked_select(off_diagonal).reshape(batch_size, batch_size - 1)


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


def compute_ranknet_loss(
    scores: Tensor,
    rank_labels: Tensor,
    temperature: float = 0.03,
) -> Tensor:
    """Return the mean RankNet loss over all strictly ordered pairs per sample.

    ``rank_labels`` normally preserve the record's full teacher permutation. Ties
    are excluded as a general fallback when graded relevance labels are supplied.
    """
    if temperature <= 0:
        raise ValueError(f"ranknet temperature must be positive, got {temperature}")
    if scores.shape != rank_labels.shape:
        raise ValueError(
            "scores and rank_labels must have the same shape, "
            f"got {tuple(scores.shape)} and {tuple(rank_labels.shape)}"
        )

    scaled_scores = scores / float(temperature)
    score_differences = scaled_scores.unsqueeze(2) - scaled_scores.unsqueeze(1)
    label_differences = rank_labels.unsqueeze(2) - rank_labels.unsqueeze(1)
    ordered_pairs = label_differences > 0
    pair_losses = F.softplus(-score_differences)
    pair_counts = ordered_pairs.sum(dim=(1, 2))
    loss_sums = (pair_losses * ordered_pairs.to(pair_losses.dtype)).sum(dim=(1, 2))
    return torch.where(
        pair_counts > 0,
        loss_sums / pair_counts.clamp_min(1).to(loss_sums.dtype),
        torch.zeros_like(loss_sums),
    )


class BaselineModel(nn.Module):
    def __init__(
        self,
        model: PreTrainedModel,
        baseline_args: BaselineArguments,
        pooling_method: str = "last",
    ):
        super().__init__()
        self.model = model
        self.config = self.model.config
        self.baseline_args = baseline_args
        self.pooling_method = pooling_method

    def encode(self, model_inputs: Dict[str, Tensor]) -> Tensor:
        return pool_embeddings(
            self.model(**model_inputs).last_hidden_state,
            model_inputs["attention_mask"],
            pooling_method=self.pooling_method,
            normalize=True,
        )

    def forward(
        self,
        query: Dict[str, Tensor] = None,
        positive_document: Dict[str, Tensor] = None,
        negative_document: Dict[str, Tensor] = None,
        relevance_labels: Tensor = None,
        rank_labels: Tensor = None,
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

        if self.baseline_args.baseline_use_in_batch_negatives:
            in_batch_scores = compute_in_batch_positive_scores(
                query_embeddings=query_embeddings,
                positive_embeddings=document_embeddings[:, 0],
            )
            scores = torch.cat((scores, in_batch_scores), dim=-1)
            relevance_labels = torch.cat(
                (relevance_labels, relevance_labels.new_zeros(in_batch_scores.shape)),
                dim=-1,
            )
            if rank_labels is not None:
                # The teacher permutation orders the complete own-query slate.
                # Cross-query positives are unrelated candidates below that slate;
                # their shared zero label also excludes pairs among themselves.
                rank_labels = torch.cat(
                    (rank_labels, rank_labels.new_zeros(in_batch_scores.shape)),
                    dim=-1,
                )

        if self.baseline_args.baseline_loss == "infonce":
            per_sample_loss = compute_infonce_loss(
                scores=scores,
                relevance_labels=relevance_labels,
                temperature=self.baseline_args.baseline_temperature,
            )
        elif self.baseline_args.baseline_loss == "ranknet":
            per_sample_loss = compute_ranknet_loss(
                scores=scores,
                rank_labels=rank_labels if rank_labels is not None else relevance_labels,
                temperature=self.baseline_args.baseline_temperature,
            )
        else:  # BaselineArguments validates this; keep the model failure explicit.
            raise ValueError(
                f"Unsupported baseline loss: {self.baseline_args.baseline_loss}"
            )
        return BaselineModelOutput(loss=per_sample_loss.mean())

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.model.gradient_checkpointing_enable(*args, **kwargs)

    def enable_input_require_grads(self):
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()


def main() -> None:
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataArguments,
            TrainingArguments,
            LoraArguments,
            BaselineArguments,
            MTEBEvalArguments,
        )
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
        baseline_args,
        mteb_eval_args,
    ) = parse_arguments(
        parser=parser,
        base_slots=BASELINE_CONFIG_SLOTS,
    )

    guard_output_dir(training_args)
    setup_logging(
        training_args,
        {
            "Model": model_args,
            "Baseline": baseline_args,
            "MTEB eval": mteb_eval_args,
        },
    )

    set_seed(training_args.seed)

    backbone, tokenizer = load_backbone_and_tokenizer(model_args, lora_args)
    model = BaselineModel(
        model=backbone,
        baseline_args=baseline_args,
        pooling_method=model_args.pooling_method,
    )
    model.train()

    apply_gradient_checkpointing(model, training_args, lora_args)

    train_dataset, eval_dataset, data_collator = build_embedding_data(
        data_args,
        training_args,
        tokenizer,
        model_args,
    )

    trainer = BaselineTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=None,
        data_collator=data_collator,
    )
    mteb_callback = MTEBEvalCallback(mteb_eval_args, model_args=model_args)
    if mteb_callback.enabled:
        trainer.add_callback(mteb_callback.bind_trainer(trainer))

    resume = bool(
        list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
        and not training_args.overwrite_output_dir
    )
    trainer.train(resume_from_checkpoint=True if resume else None)

    save_run_artifacts(
        trainer,
        training_args,
        tokenizer,
        model_args=model_args,
        baseline_args=baseline_args,
    )


if __name__ == "__main__":
    main()
    shutdown_distributed()
