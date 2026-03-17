from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import PreTrainedModel
from transformers.file_utils import ModelOutput

from rewards import build_relevance_labels, compute_ndcg_reward


def pool_last_token_embedding(
    last_hidden_states: Tensor,
    attention_mask: Tensor,
    normalize: bool = True,
) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        embeddings = last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        embeddings = last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]

    if normalize:
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1, p=2)
    return embeddings


@dataclass
class GRPOModelOutput(ModelOutput):
    loss: Optional[Tensor] = None
    query_loss: Optional[Tensor] = None
    listwise_loss: Optional[Tensor] = None
    query_reward: Optional[Tensor] = None
    listwise_reward: Optional[Tensor] = None
    query_sigma: Optional[Tensor] = None
    listwise_sigma: Optional[Tensor] = None


class GRPO(nn.Module):
    def __init__(
        self,
        group_size: int = 8,
        sigma: float = 0.05,
        sigma_learnable: bool = False,
        reward_ndcg_k: int = 10,
        advantage_norm: bool = True,
        relevance_scheme: str = "graded",
    ):
        super().__init__()
        if group_size < 2:
            raise ValueError("group_size must be at least 2 for GRPO")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if relevance_scheme not in {"graded", "binary"}:
            raise ValueError(f"Unsupported relevance scheme: {relevance_scheme}")

        self.group_size = group_size
        self.reward_ndcg_k = reward_ndcg_k
        self.advantage_norm = advantage_norm
        self.relevance_scheme = relevance_scheme
        self.sigma_learnable = sigma_learnable

        if sigma_learnable:
            self.log_sigma = nn.Parameter(torch.log(torch.tensor(float(sigma), dtype=torch.float32)))
        else:
            self.register_buffer("fixed_sigma", torch.tensor(float(sigma), dtype=torch.float32))

    def current_sigma(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        sigma = torch.exp(self.log_sigma) if self.sigma_learnable else self.fixed_sigma
        return sigma.to(device=device, dtype=dtype)

    def forward(
        self,
        policy_embeddings: torch.Tensor,
        rollout_embeddings: torch.Tensor,
        document_embeddings: torch.Tensor,
        ranking: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if ranking is None:
            raise ValueError("ranking is required for GRPO training")
        if document_embeddings.dim() != 3:
            raise ValueError(
                f"document_embeddings must be [batch, slate, dim], got shape {tuple(document_embeddings.shape)}"
            )

        policy_embeddings = F.normalize(policy_embeddings, dim=-1)
        rollout_embeddings = F.normalize(rollout_embeddings, dim=-1)
        document_embeddings = F.normalize(document_embeddings, dim=-1)

        sigma = self.current_sigma(device=policy_embeddings.device, dtype=policy_embeddings.dtype)
        relevance_labels = build_relevance_labels(ranking, scheme=self.relevance_scheme)

        noise = torch.randn(
            self.group_size,
            *rollout_embeddings.shape,
            device=rollout_embeddings.device,
            dtype=rollout_embeddings.dtype,
        )
        sampled_embeddings = F.normalize(
            rollout_embeddings.unsqueeze(0) + sigma.detach() * noise,
            dim=-1,
        )

        rewards = []
        for sample_idx in range(self.group_size):
            rewards.append(
                compute_ndcg_reward(
                    query_embeddings=sampled_embeddings[sample_idx],
                    candidate_embeddings=document_embeddings,
                    relevance_labels=relevance_labels,
                    k=self.reward_ndcg_k,
                )
            )
        rewards = torch.stack(rewards, dim=0)

        advantages = rewards - rewards.mean(dim=0, keepdim=True)
        if self.advantage_norm:
            advantages = advantages / (advantages.std(dim=0, keepdim=True, unbiased=False) + 1e-8)

        squared_distance = (sampled_embeddings.detach() - policy_embeddings.unsqueeze(0)).pow(2).sum(dim=-1)
        log_prob = -0.5 * squared_distance / sigma.pow(2) - policy_embeddings.size(-1) * torch.log(sigma)
        loss = -(advantages.detach() * log_prob).mean()
        return loss, rewards.mean(), sigma.detach()


class GRPOModel(nn.Module):
    def __init__(
        self,
        model: PreTrainedModel,
        rl_mode: str = "dual",
        group_size: int = 8,
        sigma: float = 0.05,
        sigma_learnable: bool = False,
        query_reward_ndcg_k: int = 10,
        listwise_reward_ndcg_k: int = 16,
        listwise_loss_weight: float = 1.0,
        advantage_norm: bool = True,
        query_relevance_scheme: str = "binary",
        listwise_relevance_scheme: str = "graded",
    ):
        super().__init__()
        if rl_mode not in {"query_only", "listwise_only", "dual"}:
            raise ValueError(f"Unsupported rl_mode: {rl_mode}")

        self.model = model
        self.config = self.model.config
        self.listwise_loss_weight = listwise_loss_weight
        self.use_query_branch = rl_mode in {"query_only", "dual"}
        self.use_listwise_branch = rl_mode in {"listwise_only", "dual"}

        self.query_grpo = (
            GRPO(
                group_size=group_size,
                sigma=sigma,
                sigma_learnable=sigma_learnable,
                reward_ndcg_k=query_reward_ndcg_k,
                advantage_norm=advantage_norm,
                relevance_scheme=query_relevance_scheme,
            )
            if self.use_query_branch
            else None
        )
        self.listwise_grpo = (
            GRPO(
                group_size=group_size,
                sigma=sigma,
                sigma_learnable=sigma_learnable,
                reward_ndcg_k=listwise_reward_ndcg_k,
                advantage_norm=advantage_norm,
                relevance_scheme=listwise_relevance_scheme,
            )
            if self.use_listwise_branch
            else None
        )

    def encode(self, model_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return pool_last_token_embedding(
            self.model(**model_inputs).last_hidden_state,
            model_inputs["attention_mask"],
            normalize=True,
        )

    def forward(
        self,
        query: Dict[str, torch.Tensor] = None,
        document: Dict[str, torch.Tensor] = None,
        pseudo_query: Dict[str, torch.Tensor] = None,
        ranking: torch.Tensor = None,
    ) -> GRPOModelOutput:
        if ranking is None:
            raise ValueError("ranking is required for RL training")
        if self.use_query_branch and query is None:
            raise ValueError("query inputs are required when query-side GRPO is enabled")
        if self.use_listwise_branch and pseudo_query is None:
            raise ValueError("pseudo_query inputs are required when listwise-side GRPO is enabled")

        batch_size, slate_length = ranking.shape
        with torch.no_grad():
            rollout_document_embeddings = self.encode(document).reshape(batch_size, slate_length, -1)
            rollout_query_embeddings = self.encode(query) if self.use_query_branch else None
            rollout_listwise_embeddings = self.encode(pseudo_query) if self.use_listwise_branch else None

        query_loss = query_reward = query_sigma = None
        if self.use_query_branch:
            query_embeddings = self.encode(query)
            query_loss, query_reward, query_sigma = self.query_grpo(
                policy_embeddings=query_embeddings,
                rollout_embeddings=rollout_query_embeddings,
                document_embeddings=rollout_document_embeddings,
                ranking=ranking,
            )

        listwise_loss = listwise_reward = listwise_sigma = None
        if self.use_listwise_branch:
            listwise_embeddings = self.encode(pseudo_query)
            listwise_loss, listwise_reward, listwise_sigma = self.listwise_grpo(
                policy_embeddings=listwise_embeddings,
                rollout_embeddings=rollout_listwise_embeddings,
                document_embeddings=rollout_document_embeddings,
                ranking=ranking,
            )

        loss_terms = []
        if query_loss is not None:
            loss_terms.append(query_loss)
        if listwise_loss is not None:
            loss_terms.append(self.listwise_loss_weight * listwise_loss)

        total_loss = loss_terms[0]
        for loss_term in loss_terms[1:]:
            total_loss = total_loss + loss_term

        return GRPOModelOutput(
            loss=total_loss,
            query_loss=query_loss,
            listwise_loss=listwise_loss,
            query_reward=query_reward,
            listwise_reward=listwise_reward,
            query_sigma=query_sigma,
            listwise_sigma=listwise_sigma,
        )

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.model.gradient_checkpointing_enable(*args, **kwargs)

    def enable_input_require_grads(self):
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
