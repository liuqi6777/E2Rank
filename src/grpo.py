from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import PreTrainedModel
from transformers.file_utils import ModelOutput

from config import RLArguments, normalize_action_components
from rewards import (
    SUPPORTED_REWARD_TYPES,
    compute_rollout_reward,
)


@dataclass
class _ActionComponent:
    role: str
    rollout_embeddings: torch.Tensor
    policy_embeddings: torch.Tensor | None = None
    sampled_embeddings: torch.Tensor | None = None
    sigma: torch.Tensor | None = None

    @property
    def is_active(self) -> bool:
        return self.sampled_embeddings is not None


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
        action_components=(("query",),),
        group_size: int = 8,
        sigma: float = 0.05,
        sigma_learnable: bool = False,
        reward_type: str = "ndcg",
        reward_ndcg_k: int = 10,
        ndcg_in_batch_include_negatives: bool = False,
        contrastive_use_in_batch_negatives: bool = False,
        contrastive_temperature: float = 0.03,
        advantage_norm: bool = True,
    ):
        super().__init__()
        reward_type = reward_type.lower()
        action_components = normalize_action_components(action_components)
        if group_size < 2:
            raise ValueError("group_size must be at least 2 for GRPO")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if reward_type not in SUPPORTED_REWARD_TYPES:
            raise ValueError(
                f"Unsupported reward type: {reward_type}. Supported types: {sorted(SUPPORTED_REWARD_TYPES)}"
            )
        if contrastive_temperature <= 0:
            raise ValueError(f"contrastive_temperature must be positive, got {contrastive_temperature}")

        self.action_components = action_components
        self.sample_query = any(group == ("query",) for group in action_components)
        self.sample_positive = any("positive" in group for group in action_components)
        self.sample_negative = any("negative" in group for group in action_components)
        self.group_size = group_size
        self.reward_type = reward_type
        self.reward_ndcg_k = reward_ndcg_k
        self.ndcg_in_batch_include_negatives = ndcg_in_batch_include_negatives
        self.contrastive_use_in_batch_negatives = contrastive_use_in_batch_negatives
        self.contrastive_temperature = contrastive_temperature
        self.advantage_norm = advantage_norm
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
            ndcg_in_batch_include_negatives=self.ndcg_in_batch_include_negatives,
            contrastive_use_in_batch_negatives=self.contrastive_use_in_batch_negatives,
            contrastive_temperature=self.contrastive_temperature,
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

    def _sample_document_embeddings(
        self,
        rollout_document_embeddings: torch.Tensor,
        sigma: torch.Tensor,
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
        return F.normalize(
            base_document_embeddings + sigma.detach() * noise,
            dim=-1,
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
        document_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if policy_document_embeddings.size(1) == 0:
            return torch.zeros(
                sampled_document_embeddings.size(0),
                sampled_document_embeddings.size(1),
                device=sampled_document_embeddings.device,
                dtype=sampled_document_embeddings.dtype,
            )

        squared_distance = (
            sampled_document_embeddings.detach() - policy_document_embeddings.unsqueeze(1)
        ).pow(2).sum(dim=-1)
        log_prob = -0.5 * squared_distance / sigma.pow(2) - policy_document_embeddings.size(-1) * torch.log(sigma)
        if document_mask is None:
            return log_prob.mean(dim=-1)

        mask = document_mask.unsqueeze(1).to(dtype=log_prob.dtype)
        num_documents = mask.sum(dim=-1).clamp_min(1.0)
        return (log_prob * mask).sum(dim=-1) / num_documents

    @staticmethod
    def _get_component_group_size(
        active_components: Sequence[_ActionComponent],
    ) -> int:
        if active_components:
            return active_components[0].sampled_embeddings.size(1)
        raise ValueError("At least one action component is required")

    @staticmethod
    def _validate_component_group_sizes(
        group_size: int,
        active_components: Sequence[_ActionComponent],
    ) -> None:
        for component in active_components:
            if component.sampled_embeddings.size(1) != group_size:
                raise ValueError("action component group sizes must match")

    @staticmethod
    def _expand_query_samples(
        sampled_embeddings: torch.Tensor,
        component_index: int,
        num_components: int,
    ) -> torch.Tensor:
        batch_size, group_size, embedding_dim = sampled_embeddings.shape
        before = [1] * component_index
        after = [1] * (num_components - component_index - 1)
        view_shape = [batch_size, *before, group_size, *after, embedding_dim]
        expand_shape = [batch_size, *([group_size] * num_components), embedding_dim]
        return sampled_embeddings.reshape(view_shape).expand(expand_shape)

    @staticmethod
    def _expand_fixed_query(
        fixed_query_embeddings: torch.Tensor,
        group_size: int,
        num_components: int,
    ) -> torch.Tensor:
        batch_size, embedding_dim = fixed_query_embeddings.shape
        view_shape = [batch_size, *([1] * num_components), embedding_dim]
        expand_shape = [batch_size, *([group_size] * num_components), embedding_dim]
        return fixed_query_embeddings.detach().reshape(view_shape).expand(expand_shape)

    @staticmethod
    def _expand_document_samples(
        sampled_document_embeddings: torch.Tensor,
        component_index: int,
        num_components: int,
    ) -> torch.Tensor:
        batch_size, group_size, slate_length, embedding_dim = sampled_document_embeddings.shape
        before = [1] * component_index
        after = [1] * (num_components - component_index - 1)
        view_shape = [batch_size, *before, group_size, *after, slate_length, embedding_dim]
        expand_shape = [batch_size, *([group_size] * num_components), slate_length, embedding_dim]
        return sampled_document_embeddings.reshape(view_shape).expand(expand_shape)

    @staticmethod
    def _expand_fixed_documents(
        fixed_document_embeddings: torch.Tensor,
        group_size: int,
        num_components: int,
    ) -> torch.Tensor:
        batch_size, slate_length, embedding_dim = fixed_document_embeddings.shape
        view_shape = [batch_size, *([1] * num_components), slate_length, embedding_dim]
        expand_shape = [batch_size, *([group_size] * num_components), slate_length, embedding_dim]
        return fixed_document_embeddings.detach().reshape(view_shape).expand(expand_shape)

    def _compute_component_loss(
        self,
        relevance_labels: torch.Tensor,
        components: Sequence[_ActionComponent],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        active_components = tuple(component for component in components if component.is_active)
        query_components = [component for component in components if component.role == "query"]
        document_components = [component for component in components if component.role == "document"]
        if len(query_components) != 1:
            raise ValueError("Exactly one query component is required")
        if not document_components:
            raise ValueError("At least one document component is required")

        num_components = len(active_components)
        group_size = self._get_component_group_size(
            active_components=active_components,
        )
        self._validate_component_group_sizes(
            group_size=group_size,
            active_components=active_components,
        )

        log_probs = []
        active_index_by_id = {
            id(component): component_index
            for component_index, component in enumerate(active_components)
        }
        query_component = query_components[0]
        if query_component.is_active:
            query_grid = self._expand_query_samples(
                sampled_embeddings=query_component.sampled_embeddings,
                component_index=active_index_by_id[id(query_component)],
                num_components=num_components,
            )
        else:
            query_grid = self._expand_fixed_query(
                fixed_query_embeddings=query_component.rollout_embeddings,
                group_size=group_size,
                num_components=num_components,
            )

        document_grids = []
        for document_component in document_components:
            if document_component.is_active:
                document_grids.append(
                    self._expand_document_samples(
                        sampled_document_embeddings=document_component.sampled_embeddings,
                        component_index=active_index_by_id[id(document_component)],
                        num_components=num_components,
                    )
                )
            else:
                document_grids.append(
                    self._expand_fixed_documents(
                        fixed_document_embeddings=document_component.rollout_embeddings,
                        group_size=group_size,
                        num_components=num_components,
                    )
                )
        document_grid = torch.cat(document_grids, dim=-2)

        for component in active_components:
            if component.role == "query":
                log_probs.append(
                    self._gaussian_log_prob(
                        policy_embeddings=component.policy_embeddings,
                        sampled_embeddings=component.sampled_embeddings,
                        sigma=component.sigma,
                    )
                )
            elif component.role == "document":
                log_probs.append(
                    self._document_gaussian_log_prob(
                        policy_document_embeddings=component.policy_embeddings,
                        sampled_document_embeddings=component.sampled_embeddings,
                        sigma=component.sigma,
                    )
                )
            else:
                raise ValueError(f"Unsupported action component role: {component.role}")

        rewards = self._compute_rewards(
            query_embeddings=query_grid,
            candidate_embeddings=document_grid,
            relevance_labels=relevance_labels,
        )

        sample_dims = tuple(range(1, 1 + num_components))
        losses = []
        advantages = []
        for component_index, log_prob in enumerate(log_probs):
            other_dims = tuple(dim for dim in sample_dims if dim != component_index + 1)
            component_rewards = rewards.mean(dim=other_dims) if other_dims else rewards
            component_advantages = self._compute_advantages(component_rewards, sample_dims=(1,))
            losses.append(-(component_advantages.detach() * log_prob).mean())
            advantages.append(component_advantages)

        return sum(losses), rewards, torch.cat(advantages, dim=1)

    def forward(
        self,
        rollout_query_embeddings: torch.Tensor,
        rollout_positive_document_embeddings: torch.Tensor,
        rollout_negative_document_embeddings: torch.Tensor,
        relevance_labels: torch.Tensor,
        policy_query_embeddings: torch.Tensor | None = None,
        policy_positive_document_embeddings: torch.Tensor | None = None,
        policy_negative_document_embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
        if relevance_labels is None:
            raise ValueError("relevance_labels are required for GRPO training")
        if rollout_positive_document_embeddings.dim() != 3:
            raise ValueError(
                "positive_document_embeddings must be [batch, 1, dim], "
                f"got shape {tuple(rollout_positive_document_embeddings.shape)}"
            )
        if rollout_positive_document_embeddings.size(1) != 1:
            raise ValueError("positive_document_embeddings must contain exactly one document per sample")
        if rollout_negative_document_embeddings.dim() != 3:
            raise ValueError(
                "negative_document_embeddings must be [batch, negatives, dim], "
                f"got shape {tuple(rollout_negative_document_embeddings.shape)}"
            )
        if rollout_negative_document_embeddings.size(0) != rollout_positive_document_embeddings.size(0):
            raise ValueError("positive and negative document batch sizes must match")
        if rollout_negative_document_embeddings.size(-1) != rollout_positive_document_embeddings.size(-1):
            raise ValueError("positive and negative document embedding dims must match")

        document_embeddings = torch.cat((rollout_positive_document_embeddings, rollout_negative_document_embeddings), dim=1)
        if relevance_labels.shape != document_embeddings.shape[:2]:
            raise ValueError(
                "relevance_labels shape must match [batch, slate], "
                f"got labels={tuple(relevance_labels.shape)} documents={tuple(document_embeddings.shape)}"
            )

        rollout_query_embeddings = F.normalize(rollout_query_embeddings, dim=-1)
        if policy_query_embeddings is not None:
            if policy_query_embeddings.shape != rollout_query_embeddings.shape:
                raise ValueError(
                    "policy_query_embeddings shape must match rollout_query_embeddings, "
                    f"got policy={tuple(policy_query_embeddings.shape)} "
                    f"rollout={tuple(rollout_query_embeddings.shape)}"
                )
            policy_query_embeddings = F.normalize(policy_query_embeddings, dim=-1)
        rollout_positive_document_embeddings = F.normalize(rollout_positive_document_embeddings, dim=-1)
        rollout_negative_document_embeddings = F.normalize(rollout_negative_document_embeddings, dim=-1)
        document_embeddings = F.normalize(document_embeddings, dim=-1)
        if policy_positive_document_embeddings is not None:
            if policy_positive_document_embeddings.shape != rollout_positive_document_embeddings.shape:
                raise ValueError(
                    "policy_positive_document_embeddings shape must match positive_document_embeddings, "
                    f"got policy={tuple(policy_positive_document_embeddings.shape)} "
                    f"rollout={tuple(rollout_positive_document_embeddings.shape)}"
                )
            policy_positive_document_embeddings = F.normalize(policy_positive_document_embeddings, dim=-1)
        if policy_negative_document_embeddings is not None:
            if policy_negative_document_embeddings.shape != rollout_negative_document_embeddings.shape:
                raise ValueError(
                    "policy_negative_document_embeddings shape must match negative_document_embeddings, "
                    f"got policy={tuple(policy_negative_document_embeddings.shape)} "
                    f"rollout={tuple(rollout_negative_document_embeddings.shape)}"
                )
            policy_negative_document_embeddings = F.normalize(policy_negative_document_embeddings, dim=-1)

        sigma = self.current_sigma(device=rollout_query_embeddings.device, dtype=rollout_query_embeddings.dtype)
        components = []
        if self.sample_query:
            if policy_query_embeddings is None:
                raise ValueError("policy_query_embeddings are required when query is sampled")
            sampled_query_embeddings = self._sample_query_embeddings(
                rollout_embeddings=rollout_query_embeddings,
                sigma=sigma,
            )
            components.append(_ActionComponent(
                role="query",
                rollout_embeddings=rollout_query_embeddings,
                policy_embeddings=policy_query_embeddings,
                sampled_embeddings=sampled_query_embeddings,
                sigma=sigma,
            ))
        else:
            components.append(_ActionComponent(
                role="query",
                rollout_embeddings=rollout_query_embeddings,
            ))

        joint_document_group = ("positive", "negative") in self.action_components or \
          ("negative", "positive") in self.action_components
        if joint_document_group:
            if policy_positive_document_embeddings is None:
                raise ValueError("policy_positive_document_embeddings are required when positive is sampled")
            if policy_negative_document_embeddings is None:
                raise ValueError("policy_negative_document_embeddings are required when negative is sampled")
            policy_document_embeddings = torch.cat(
                (policy_positive_document_embeddings, policy_negative_document_embeddings),
                dim=1,
            )
            sampled_document_embeddings = self._sample_document_embeddings(
                rollout_document_embeddings=document_embeddings,
                sigma=sigma,
            )
            components.append(_ActionComponent(
                role="document",
                rollout_embeddings=document_embeddings,
                policy_embeddings=policy_document_embeddings,
                sampled_embeddings=sampled_document_embeddings,
                sigma=sigma,
            ))
        elif self.sample_positive:
            if policy_positive_document_embeddings is None:
                raise ValueError("policy_positive_document_embeddings are required when positive is sampled")
            sampled_positive_document_embeddings = self._sample_document_embeddings(
                rollout_document_embeddings=rollout_positive_document_embeddings,
                sigma=sigma,
            )
            components.append(_ActionComponent(
                role="document",
                rollout_embeddings=rollout_positive_document_embeddings,
                policy_embeddings=policy_positive_document_embeddings,
                sampled_embeddings=sampled_positive_document_embeddings,
                sigma=sigma,
            ))
        else:
            components.append(_ActionComponent(
                role="document",
                rollout_embeddings=rollout_positive_document_embeddings,
            ))

        if not joint_document_group and self.sample_negative:
            if policy_negative_document_embeddings is None:
                raise ValueError("policy_negative_document_embeddings are required when negative is sampled")
            sampled_negative_document_embeddings = self._sample_document_embeddings(
                rollout_document_embeddings=rollout_negative_document_embeddings,
                sigma=sigma,
            )
            components.append(_ActionComponent(
                role="document",
                rollout_embeddings=rollout_negative_document_embeddings,
                policy_embeddings=policy_negative_document_embeddings,
                sampled_embeddings=sampled_negative_document_embeddings,
                sigma=sigma,
            ))
        elif not joint_document_group:
            components.append(_ActionComponent(
                role="document",
                rollout_embeddings=rollout_negative_document_embeddings,
            ))

        loss, rewards, advantages = self._compute_component_loss(
            relevance_labels=relevance_labels,
            components=tuple(components),
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
            action_components=rl_args.action_components,
            group_size=rl_args.group_size,
            sigma=rl_args.sigma,
            sigma_learnable=rl_args.sigma_learnable,
            reward_type=rl_args.reward_type,
            reward_ndcg_k=rl_args.reward_ndcg_k,
            ndcg_in_batch_include_negatives=rl_args.ndcg_in_batch_include_negatives,
            contrastive_use_in_batch_negatives=rl_args.contrastive_use_in_batch_negatives,
            contrastive_temperature=rl_args.contrastive_temperature,
            advantage_norm=rl_args.advantage_norm,
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
        positive_document: Dict[str, torch.Tensor] = None,
        negative_document: Dict[str, torch.Tensor] = None,
        relevance_labels: torch.Tensor = None,
    ) -> GRPOModelOutput:
        if query is None:
            raise ValueError("query inputs are required for GRPO training")
        if positive_document is None:
            raise ValueError("positive document inputs are required for GRPO training")
        if negative_document is None:
            raise ValueError("negative document inputs are required for GRPO training")
        if relevance_labels is None:
            raise ValueError("relevance_labels are required for GRPO training")

        batch_size, slate_length = relevance_labels.shape
        if self.grpo.sample_query:
            policy_query_embeddings = self.encode(query)
            rollout_query_embeddings = policy_query_embeddings.detach()
        else:
            with torch.no_grad():
                rollout_query_embeddings = self.encode(query)
            rollout_query_embeddings = rollout_query_embeddings.detach()
            policy_query_embeddings = None

        num_negatives = slate_length - 1
        document_inputs = {}
        for key in positive_document:
            positive_value = positive_document[key]
            negative_value = negative_document[key].reshape(batch_size, num_negatives, -1)
            slate_value = torch.cat((positive_value.unsqueeze(1), negative_value), dim=1)
            document_inputs[key] = slate_value.reshape(batch_size * slate_length, -1)

        sample_document = self.grpo.sample_positive or self.grpo.sample_negative
        if sample_document:
            encoded_document_embeddings = self.encode(document_inputs)
            policy_document_embeddings = encoded_document_embeddings.reshape(batch_size, slate_length, -1)
            rollout_document_embeddings = policy_document_embeddings.detach()
            policy_positive_document_embeddings = (
                policy_document_embeddings[:, :1] if self.grpo.sample_positive else None
            )
            policy_negative_document_embeddings = (
                policy_document_embeddings[:, 1:] if self.grpo.sample_negative else None
            )
        else:
            with torch.no_grad():
                encoded_document_embeddings = self.encode(document_inputs)
            rollout_document_embeddings = encoded_document_embeddings.detach().reshape(batch_size, slate_length, -1)
            policy_positive_document_embeddings = None
            policy_negative_document_embeddings = None

        rollout_positive_document_embeddings = rollout_document_embeddings[:, :1]
        rollout_negative_document_embeddings = rollout_document_embeddings[:, 1:]

        loss, reward_stats, advantage_stats, sigma = self.grpo(
            rollout_query_embeddings=rollout_query_embeddings,
            rollout_positive_document_embeddings=rollout_positive_document_embeddings,
            rollout_negative_document_embeddings=rollout_negative_document_embeddings,
            relevance_labels=relevance_labels,
            policy_query_embeddings=policy_query_embeddings,
            policy_positive_document_embeddings=policy_positive_document_embeddings,
            policy_negative_document_embeddings=policy_negative_document_embeddings,
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
