from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import PreTrainedModel
from transformers.file_utils import ModelOutput

from config import RLArguments
from rewards import (
    SUPPORTED_REWARD_TYPES,
    build_relevance_labels,
    compute_reward,
)


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
    reward: Optional[Tensor] = None
    reward_mean: Optional[Tensor] = None
    reward_std: Optional[Tensor] = None
    reward_min: Optional[Tensor] = None
    reward_max: Optional[Tensor] = None
    advantages_mean: Optional[Tensor] = None
    advantages_std: Optional[Tensor] = None
    advantages_min: Optional[Tensor] = None
    advantages_max: Optional[Tensor] = None
    sigma: Optional[Tensor] = None


class GRPO(nn.Module):
    def __init__(
        self,
        group_size: int = 8,
        sigma: float = 0.05,
        sigma_learnable: bool = False,
        reward_type: str = "ndcg",
        reward_ndcg_k: int = 10,
        mixed_contrastive_weight: float = 1.0,
        mixed_ndcg_weight: float = 1.0,
        contrastive_use_in_batch_negatives: bool = False,
        contrastive_temperature: float = 0.03,
        advantage_norm: bool = True,
        relevance_scheme: str = "graded",
    ):
        super().__init__()
        reward_type = reward_type.lower()
        if group_size < 2:
            raise ValueError("group_size must be at least 2 for GRPO")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if relevance_scheme not in {"graded", "binary"}:
            raise ValueError(f"Unsupported relevance scheme: {relevance_scheme}")
        if reward_type not in SUPPORTED_REWARD_TYPES:
            raise ValueError(
                f"Unsupported reward type: {reward_type}. Supported types: {sorted(SUPPORTED_REWARD_TYPES)}"
            )
        if contrastive_temperature <= 0:
            raise ValueError(f"contrastive_temperature must be positive, got {contrastive_temperature}")

        self.group_size = group_size
        self.reward_type = reward_type
        self.reward_ndcg_k = reward_ndcg_k
        self.mixed_contrastive_weight = mixed_contrastive_weight
        self.mixed_ndcg_weight = mixed_ndcg_weight
        self.contrastive_use_in_batch_negatives = contrastive_use_in_batch_negatives
        self.contrastive_temperature = contrastive_temperature
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

    @staticmethod
    def summarize_tensor(values: torch.Tensor, prefix: str) -> dict[str, torch.Tensor]:
        flattened_values = values.detach().reshape(-1).to(dtype=torch.float32)
        return {
            f"{prefix}_mean": flattened_values.mean(),
            f"{prefix}_std": flattened_values.std(unbiased=False),
            f"{prefix}_min": flattened_values.min(),
            f"{prefix}_max": flattened_values.max(),
        }

    def forward(
        self,
        policy_embeddings: torch.Tensor,
        rollout_embeddings: torch.Tensor,
        document_embeddings: torch.Tensor,
        ranking: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
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
                compute_reward(
                    query_embeddings=sampled_embeddings[sample_idx],
                    candidate_embeddings=document_embeddings,
                    relevance_labels=relevance_labels,
                    reward_type=self.reward_type,
                    k=self.reward_ndcg_k,
                    mixed_contrastive_weight=self.mixed_contrastive_weight,
                    mixed_ndcg_weight=self.mixed_ndcg_weight,
                    contrastive_use_in_batch_negatives=self.contrastive_use_in_batch_negatives,
                    contrastive_temperature=self.contrastive_temperature,
                )
            )
        rewards = torch.stack(rewards, dim=0)
        reward_stats = self.summarize_tensor(rewards, prefix="reward")

        advantages = rewards - rewards.mean(dim=0, keepdim=True)
        if self.advantage_norm:
            advantages = advantages / (advantages.std(dim=0, keepdim=True, unbiased=False) + 1e-8)
        advantage_stats = self.summarize_tensor(advantages, prefix="advantages")

        squared_distance = (sampled_embeddings.detach() - policy_embeddings.unsqueeze(0)).pow(2).sum(dim=-1)
        log_prob = -0.5 * squared_distance / sigma.pow(2) - policy_embeddings.size(-1) * torch.log(sigma)
        loss = -(advantages.detach() * log_prob).mean()
        return loss, reward_stats, advantage_stats, sigma.detach()


class GRPOModel(nn.Module):
    def __init__(
        self,
        model: PreTrainedModel,
        rl_args: RLArguments,
    ):
        super().__init__()
        self.model = model
        self.config = self.model.config
        self.grpo = GRPO(
            group_size=rl_args.group_size,
            sigma=rl_args.sigma,
            sigma_learnable=rl_args.sigma_learnable,
            reward_type=rl_args.reward_type,
            reward_ndcg_k=rl_args.reward_ndcg_k,
            mixed_contrastive_weight=rl_args.mixed_contrastive_weight,
            mixed_ndcg_weight=rl_args.mixed_ndcg_weight,
            contrastive_use_in_batch_negatives=rl_args.contrastive_use_in_batch_negatives,
            contrastive_temperature=rl_args.contrastive_temperature,
            advantage_norm=rl_args.advantage_norm,
            relevance_scheme=rl_args.relevance_scheme,
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
        ranking: torch.Tensor = None,
    ) -> GRPOModelOutput:
        if ranking is None:
            raise ValueError("ranking is required for RL training")
        if query is None:
            raise ValueError("query inputs are required for GRPO training")

        batch_size, slate_length = ranking.shape
        with torch.no_grad():
            rollout_document_embeddings = self.encode(document).reshape(batch_size, slate_length, -1)
            rollout_embeddings = self.encode(query)

        policy_embeddings = self.encode(query)
        loss, reward_stats, advantage_stats, sigma = self.grpo(
            policy_embeddings=policy_embeddings,
            rollout_embeddings=rollout_embeddings,
            document_embeddings=rollout_document_embeddings,
            ranking=ranking,
        )
        reward = reward_stats["reward_mean"]

        return GRPOModelOutput(
            loss=loss,
            reward=reward,
            reward_mean=reward_stats["reward_mean"],
            reward_std=reward_stats["reward_std"],
            reward_min=reward_stats["reward_min"],
            reward_max=reward_stats["reward_max"],
            advantages_mean=advantage_stats["advantages_mean"],
            advantages_std=advantage_stats["advantages_std"],
            advantages_min=advantage_stats["advantages_min"],
            advantages_max=advantage_stats["advantages_max"],
            sigma=sigma,
        )

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.model.gradient_checkpointing_enable(*args, **kwargs)

    def enable_input_require_grads(self):
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
