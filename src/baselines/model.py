from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn
from transformers import PreTrainedModel
from transformers.file_utils import ModelOutput

from baselines.losses import SUPPORTED_BASELINE_TYPES, compute_baseline_loss
from baselines.metrics import compute_mrr_at_k, compute_ndcg_at_k
from baselines.config import BaselineArguments
from grpo import pool_last_token_embedding


@dataclass
class BaselineModelOutput(ModelOutput):
    loss: Optional[Tensor] = None
    ndcg: Optional[Tensor] = None
    mrr: Optional[Tensor] = None


class BaselineModel(nn.Module):
    def __init__(
        self,
        model: PreTrainedModel,
        baseline_args: BaselineArguments,
    ):
        super().__init__()
        baseline_type = baseline_args.baseline_type.lower()
        if baseline_type not in SUPPORTED_BASELINE_TYPES:
            raise ValueError(
                f"Unsupported baseline_type: {baseline_type}. Supported types: {sorted(SUPPORTED_BASELINE_TYPES)}"
            )

        self.model = model
        self.config = self.model.config
        self.baseline_args = baseline_args

    def encode(self, model_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return pool_last_token_embedding(
            self.model(**model_inputs).last_hidden_state,
            model_inputs["attention_mask"],
            normalize=True,
        )

    def forward(
        self,
        query: Dict[str, torch.Tensor] = None,
        positive_document: Dict[str, torch.Tensor] = None,
        negative_document: Dict[str, torch.Tensor] = None,
        relevance_labels: torch.Tensor = None,
    ) -> BaselineModelOutput:
        if query is None:
            raise ValueError("query inputs are required for baseline training")
        if positive_document is None:
            raise ValueError("positive document inputs are required for baseline training")
        if negative_document is None:
            raise ValueError("negative document inputs are required for baseline training")
        if relevance_labels is None:
            raise ValueError("relevance_labels are required for baseline training")

        batch_size, slate_length = relevance_labels.shape
        query_embeddings = self.encode(query)
        document = {
            key: torch.cat((positive_document[key], negative_document[key]), dim=0)
            for key in positive_document
        }
        document_embeddings = self.encode(document).reshape(batch_size, slate_length, -1)
        scores = torch.matmul(document_embeddings, query_embeddings.unsqueeze(-1)).squeeze(-1)

        per_sample_loss = compute_baseline_loss(
            scores=scores,
            relevance_labels=relevance_labels,
            baseline_type=self.baseline_args.baseline_type,
            ndcg_k=self.baseline_args.baseline_ndcg_k,
            ranknet_sigma=self.baseline_args.ranknet_sigma,
            lambdaloss_sigma=self.baseline_args.lambdaloss_sigma,
            approxndcg_alpha=self.baseline_args.approxndcg_alpha,
            neuralndcg_temperature=self.baseline_args.neuralndcg_temperature,
            softrank_sigma=self.baseline_args.softrank_sigma,
            infonce_temperature=self.baseline_args.infonce_temperature,
        )
        ndcg = compute_ndcg_at_k(
            scores=scores,
            relevance_labels=relevance_labels,
            k=self.baseline_args.baseline_ndcg_k,
        )
        mrr = compute_mrr_at_k(
            scores=scores,
            relevance_labels=relevance_labels,
            k=self.baseline_args.baseline_ndcg_k,
        )
        return BaselineModelOutput(
            loss=per_sample_loss.mean(),
            ndcg=ndcg.mean(),
            mrr=mrr.mean(),
        )

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.model.gradient_checkpointing_enable(*args, **kwargs)

    def enable_input_require_grads(self):
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
