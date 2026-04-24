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
    compute_rollout_reward,
)


SUPPORTED_GRPO_MODES = {"query_only", "diagonal", "grid"}


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
        grpo_mode: str = "query_only",
        group_size: int = 8,
        sigma: float = 0.05,
        sigma_learnable: bool = False,
        perturb_negatives: bool = True,
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
        grpo_mode = grpo_mode.lower()
        reward_type = reward_type.lower()
        if grpo_mode not in SUPPORTED_GRPO_MODES:
            raise ValueError(
                f"Unsupported GRPO mode: {grpo_mode}. Supported modes: {sorted(SUPPORTED_GRPO_MODES)}"
            )
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

        self.grpo_mode = grpo_mode
        self.group_size = group_size
        self.perturb_negatives = perturb_negatives
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

    def _compute_advantages(self, rewards: torch.Tensor, sample_dims: tuple[int, ...]) -> torch.Tensor:
        advantages = rewards - rewards.mean(dim=sample_dims, keepdim=True)
        if self.advantage_norm:
            advantages = advantages / (advantages.std(dim=sample_dims, keepdim=True, unbiased=False) + 1e-8)
        return advantages

    def _compute_rewards(
        self,
        query_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        relevance_labels: torch.Tensor,
    ) -> torch.Tensor:
        return compute_rollout_reward(
            query_embeddings=query_embeddings,
            candidate_embeddings=candidate_embeddings,
            relevance_labels=relevance_labels,
            reward_type=self.reward_type,
            k=self.reward_ndcg_k,
            mixed_contrastive_weight=self.mixed_contrastive_weight,
            mixed_ndcg_weight=self.mixed_ndcg_weight,
            contrastive_use_in_batch_negatives=self.contrastive_use_in_batch_negatives,
            contrastive_temperature=self.contrastive_temperature,
            relevance_scheme=self.relevance_scheme,
        )

    def _sample_query_embeddings(
        self,
        rollout_embeddings: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        noise = torch.randn(
            rollout_embeddings.size(0),
            self.group_size,
            rollout_embeddings.size(-1),
            device=rollout_embeddings.device,
            dtype=rollout_embeddings.dtype,
        )
        return F.normalize(
            rollout_embeddings.detach().unsqueeze(1) + sigma.detach() * noise,
            dim=-1,
        )

    def _build_document_perturb_mask(
        self,
        relevance_labels: torch.Tensor,
        slate_length: int,
    ) -> torch.Tensor:
        if self.perturb_negatives:
            return torch.ones(
                relevance_labels.size(0),
                slate_length,
                device=relevance_labels.device,
                dtype=torch.bool,
            )

        positive_indices = relevance_labels.argmax(dim=-1, keepdim=True)
        perturb_mask = torch.zeros(
            relevance_labels.size(0),
            slate_length,
            device=relevance_labels.device,
            dtype=torch.bool,
        )
        perturb_mask.scatter_(dim=1, index=positive_indices, value=True)
        return perturb_mask

    def _sample_document_embeddings(
        self,
        rollout_document_embeddings: torch.Tensor,
        sigma: torch.Tensor,
        perturb_mask: torch.Tensor,
    ) -> torch.Tensor:
        noise = torch.randn(
            rollout_document_embeddings.size(0),
            self.group_size,
            rollout_document_embeddings.size(1),
            rollout_document_embeddings.size(-1),
            device=rollout_document_embeddings.device,
            dtype=rollout_document_embeddings.dtype,
        )
        base_document_embeddings = rollout_document_embeddings.detach().unsqueeze(1)
        sampled_document_embeddings = F.normalize(
            base_document_embeddings + sigma.detach() * noise,
            dim=-1,
        )
        stable_document_embeddings = base_document_embeddings.expand_as(sampled_document_embeddings)
        return torch.where(
            perturb_mask.unsqueeze(1).unsqueeze(-1),
            sampled_document_embeddings,
            stable_document_embeddings,
        )

    @staticmethod
    def _gaussian_log_prob(
        policy_embeddings: torch.Tensor,
        sampled_embeddings: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        squared_distance = (sampled_embeddings.detach() - policy_embeddings.unsqueeze(1)).pow(2).sum(dim=-1)
        return -0.5 * squared_distance / sigma.pow(2) - policy_embeddings.size(-1) * torch.log(sigma)

    @staticmethod
    def _document_gaussian_log_prob(
        policy_document_embeddings: torch.Tensor,
        sampled_document_embeddings: torch.Tensor,
        sigma: torch.Tensor,
        perturb_mask: torch.Tensor,
    ) -> torch.Tensor:
        squared_distance = (
            sampled_document_embeddings.detach() - policy_document_embeddings.unsqueeze(1)
        ).pow(2).sum(dim=-1)
        log_prob = -0.5 * squared_distance / sigma.pow(2) - policy_document_embeddings.size(-1) * torch.log(sigma)
        mask = perturb_mask.unsqueeze(1).to(dtype=log_prob.dtype)
        num_perturbed = mask.sum(dim=-1).clamp_min(1.0)
        return (log_prob * mask).sum(dim=-1) / num_perturbed

    def _compute_query_only_loss(
        self,
        policy_embeddings: torch.Tensor,
        sampled_query_embeddings: torch.Tensor,
        document_embeddings: torch.Tensor,
        relevance_labels: torch.Tensor,
        sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rewards = self._compute_rewards(
            query_embeddings=sampled_query_embeddings,
            candidate_embeddings=document_embeddings.detach(),
            relevance_labels=relevance_labels,
        )
        advantages = self._compute_advantages(rewards, sample_dims=(1,))
        log_prob = self._gaussian_log_prob(
            policy_embeddings=policy_embeddings,
            sampled_embeddings=sampled_query_embeddings,
            sigma=sigma,
        )
        loss = -(advantages.detach() * log_prob).mean()
        return loss, rewards, advantages

    def _compute_diagonal_loss(
        self,
        policy_embeddings: torch.Tensor,
        policy_document_embeddings: torch.Tensor,
        sampled_query_embeddings: torch.Tensor,
        sampled_document_embeddings: torch.Tensor,
        relevance_labels: torch.Tensor,
        sigma: torch.Tensor,
        perturb_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rewards = self._compute_rewards(
            query_embeddings=sampled_query_embeddings,
            candidate_embeddings=sampled_document_embeddings,
            relevance_labels=relevance_labels,
        )
        advantages = self._compute_advantages(rewards, sample_dims=(1,))
        query_log_prob = self._gaussian_log_prob(
            policy_embeddings=policy_embeddings,
            sampled_embeddings=sampled_query_embeddings,
            sigma=sigma,
        )
        document_log_prob = self._document_gaussian_log_prob(
            policy_document_embeddings=policy_document_embeddings,
            sampled_document_embeddings=sampled_document_embeddings,
            sigma=sigma,
            perturb_mask=perturb_mask,
        )
        loss = -(advantages.detach() * (query_log_prob + document_log_prob)).mean()
        return loss, rewards, advantages

    def _compute_grid_loss(
        self,
        policy_embeddings: torch.Tensor,
        policy_document_embeddings: torch.Tensor,
        sampled_query_embeddings: torch.Tensor,
        sampled_document_embeddings: torch.Tensor,
        relevance_labels: torch.Tensor,
        sigma: torch.Tensor,
        perturb_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, group_size = sampled_query_embeddings.shape[:2]
        query_grid = sampled_query_embeddings.unsqueeze(2).expand(
            batch_size,
            group_size,
            group_size,
            sampled_query_embeddings.size(-1),
        )
        document_grid = sampled_document_embeddings.unsqueeze(1).expand(
            batch_size,
            group_size,
            group_size,
            sampled_document_embeddings.size(-2),
            sampled_document_embeddings.size(-1),
        )
        rewards = self._compute_rewards(
            query_embeddings=query_grid,
            candidate_embeddings=document_grid,
            relevance_labels=relevance_labels,
        )

        query_advantages = self._compute_advantages(rewards.mean(dim=2), sample_dims=(1,))
        document_advantages = self._compute_advantages(rewards.mean(dim=1), sample_dims=(1,))
        query_log_prob = self._gaussian_log_prob(
            policy_embeddings=policy_embeddings,
            sampled_embeddings=sampled_query_embeddings,
            sigma=sigma,
        )
        document_log_prob = self._document_gaussian_log_prob(
            policy_document_embeddings=policy_document_embeddings,
            sampled_document_embeddings=sampled_document_embeddings,
            sigma=sigma,
            perturb_mask=perturb_mask,
        )
        query_loss = -(query_advantages.detach() * query_log_prob).mean()
        document_loss = -(document_advantages.detach() * document_log_prob).mean()
        return query_loss + document_loss, rewards, torch.cat((query_advantages, document_advantages), dim=1)

    def forward(
        self,
        policy_embeddings: torch.Tensor,
        rollout_embeddings: torch.Tensor,
        document_embeddings: torch.Tensor,
        ranking: torch.Tensor,
        policy_document_embeddings: torch.Tensor | None = None,
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
        if policy_document_embeddings is not None:
            if policy_document_embeddings.shape != document_embeddings.shape:
                raise ValueError(
                    "policy_document_embeddings shape must match document_embeddings, "
                    f"got policy={tuple(policy_document_embeddings.shape)} rollout={tuple(document_embeddings.shape)}"
                )
            policy_document_embeddings = F.normalize(policy_document_embeddings, dim=-1)

        sigma = self.current_sigma(device=policy_embeddings.device, dtype=policy_embeddings.dtype)
        relevance_labels = build_relevance_labels(ranking, scheme=self.relevance_scheme)
        sampled_query_embeddings = self._sample_query_embeddings(
            rollout_embeddings=rollout_embeddings,
            sigma=sigma,
        )

        if self.grpo_mode == "query_only":
            loss, rewards, advantages = self._compute_query_only_loss(
                policy_embeddings=policy_embeddings,
                sampled_query_embeddings=sampled_query_embeddings,
                document_embeddings=document_embeddings,
                relevance_labels=relevance_labels,
                sigma=sigma,
            )
        else:
            if policy_document_embeddings is None:
                raise ValueError(f"policy_document_embeddings is required for GRPO mode {self.grpo_mode}")
            perturb_mask = self._build_document_perturb_mask(
                relevance_labels=relevance_labels,
                slate_length=document_embeddings.size(1),
            )
            sampled_document_embeddings = self._sample_document_embeddings(
                rollout_document_embeddings=document_embeddings,
                sigma=sigma,
                perturb_mask=perturb_mask,
            )
            if self.grpo_mode == "diagonal":
                loss, rewards, advantages = self._compute_diagonal_loss(
                    policy_embeddings=policy_embeddings,
                    policy_document_embeddings=policy_document_embeddings,
                    sampled_query_embeddings=sampled_query_embeddings,
                    sampled_document_embeddings=sampled_document_embeddings,
                    relevance_labels=relevance_labels,
                    sigma=sigma,
                    perturb_mask=perturb_mask,
                )
            else:
                loss, rewards, advantages = self._compute_grid_loss(
                    policy_embeddings=policy_embeddings,
                    policy_document_embeddings=policy_document_embeddings,
                    sampled_query_embeddings=sampled_query_embeddings,
                    sampled_document_embeddings=sampled_document_embeddings,
                    relevance_labels=relevance_labels,
                    sigma=sigma,
                    perturb_mask=perturb_mask,
                )

        reward_stats = self.summarize_tensor(rewards, prefix="reward")
        advantage_stats = self.summarize_tensor(advantages, prefix="advantages")
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
            grpo_mode=rl_args.grpo_mode,
            group_size=rl_args.group_size,
            sigma=rl_args.sigma,
            sigma_learnable=rl_args.sigma_learnable,
            perturb_negatives=rl_args.perturb_negatives,
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
        if document is None:
            raise ValueError("document inputs are required for GRPO training")

        batch_size, slate_length = ranking.shape
        policy_embeddings = self.encode(query)
        rollout_embeddings = policy_embeddings.detach()

        policy_document_embeddings = None
        if self.grpo.grpo_mode != "query_only":
            policy_document_embeddings = self.encode(document).reshape(batch_size, slate_length, -1)
            rollout_document_embeddings = policy_document_embeddings.detach()
        else:
            with torch.no_grad():
                rollout_document_embeddings = self.encode(document).reshape(batch_size, slate_length, -1)

        loss, reward_stats, advantage_stats, sigma = self.grpo(
            policy_embeddings=policy_embeddings,
            rollout_embeddings=rollout_embeddings,
            document_embeddings=rollout_document_embeddings,
            ranking=ranking,
            policy_document_embeddings=policy_document_embeddings,
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
