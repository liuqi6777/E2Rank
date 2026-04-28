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

    def _compute_rewards_from_scores(
        self,
        scores: torch.Tensor,
        relevance_labels: torch.Tensor,
        in_batch_positive_scores: torch.Tensor | None = None,
        in_batch_candidate_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if scores.dim() < 3:
            raise ValueError(f"scores must be [batch, *rollout, slate], got shape {tuple(scores.shape)}")
        batch_size = scores.size(0)
        rollout_shape = scores.shape[1:-1]
        slate_length = scores.size(-1)
        rollout_count = 1
        for rollout_dim in rollout_shape:
            rollout_count *= rollout_dim
        scores = scores.reshape(batch_size, rollout_count, slate_length)
        if in_batch_positive_scores is not None:
            in_batch_positive_scores = in_batch_positive_scores.reshape(batch_size, rollout_count, -1)
        if in_batch_candidate_scores is not None:
            in_batch_candidate_scores = in_batch_candidate_scores.reshape(batch_size, rollout_count, -1)
        expanded_labels = relevance_labels.unsqueeze(1).expand(batch_size, rollout_count, slate_length)

        if self.reward_type in {"ndcg", "ndcg_in_batch"}:
            ranking_scores = scores
            ranking_labels = expanded_labels
            if self.reward_type == "ndcg_in_batch":
                extra_scores = in_batch_candidate_scores if self.ndcg_in_batch_include_negatives else in_batch_positive_scores
                if extra_scores is not None:
                    ranking_scores = torch.cat((ranking_scores, extra_scores), dim=-1)
                    ranking_labels = torch.cat((ranking_labels, torch.zeros_like(extra_scores)), dim=-1)

            cutoff = min(self.reward_ndcg_k, ranking_scores.size(-1))
            if cutoff <= 0:
                return torch.zeros(batch_size, *rollout_shape, device=scores.device, dtype=scores.dtype)

            topk_indices = ranking_scores.topk(k=cutoff, dim=-1).indices
            topk_relevance = ranking_labels.gather(dim=-1, index=topk_indices)
            discounts = 1.0 / torch.log2(
                torch.arange(2, cutoff + 2, device=scores.device, dtype=scores.dtype)
            )
            dcg = (((2.0 ** topk_relevance) - 1.0) * discounts).sum(dim=-1)
            ideal_relevance = ranking_labels.topk(k=cutoff, dim=-1).values
            idcg = (((2.0 ** ideal_relevance) - 1.0) * discounts).sum(dim=-1)
            return torch.where(idcg > 0, dcg / idcg, torch.zeros_like(dcg)).reshape(batch_size, *rollout_shape)

        if self.reward_type == "mrr":
            cutoff = min(self.reward_ndcg_k, slate_length)
            if cutoff <= 0:
                return torch.zeros(batch_size, *rollout_shape, device=scores.device, dtype=scores.dtype)
            ranked_indices = scores.topk(k=cutoff, dim=-1).indices
            ranked_relevance = expanded_labels.gather(dim=-1, index=ranked_indices)
            relevant_mask = (
                ranked_relevance >= 2.0
                if bool((relevance_labels > 1).any().item())
                else ranked_relevance > 0.0
            )
            reciprocal_ranks = relevant_mask.to(scores.dtype) / torch.arange(
                1,
                cutoff + 1,
                device=scores.device,
                dtype=scores.dtype,
            )
            return reciprocal_ranks.max(dim=-1).values.reshape(batch_size, *rollout_shape)

        positive_indices = relevance_labels.argmax(dim=-1)
        arange_b = torch.arange(batch_size, device=scores.device)
        positive_scores = scores[arange_b, :, positive_indices]
        positive_one_hot = F.one_hot(positive_indices, num_classes=slate_length).bool()
        negative_scores = scores.masked_fill(positive_one_hot.unsqueeze(1), float("-inf"))
        if self.contrastive_use_in_batch_negatives and in_batch_positive_scores is not None:
            negative_scores = torch.cat((negative_scores, in_batch_positive_scores), dim=-1)

        temperature = torch.as_tensor(
            self.contrastive_temperature,
            device=scores.device,
            dtype=scores.dtype,
        )
        if self.reward_type == "contrastive":
            return (positive_scores - temperature * torch.logsumexp(negative_scores / temperature, dim=-1)).reshape(
                batch_size,
                *rollout_shape,
            )
        if self.reward_type == "infonce":
            partition_scores = torch.cat((positive_scores.unsqueeze(-1), negative_scores), dim=-1)
            return (positive_scores - temperature * torch.logsumexp(partition_scores / temperature, dim=-1)).reshape(
                batch_size,
                *rollout_shape,
            )
        raise AssertionError(f"Unhandled reward type: {self.reward_type}")

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
    def _compute_score_table(
        query_component: _ActionComponent,
        document_component: _ActionComponent,
    ) -> torch.Tensor:
        query_embeddings = (
            query_component.sampled_embeddings
            if query_component.is_active
            else query_component.rollout_embeddings.detach()
        )
        document_embeddings = (
            document_component.sampled_embeddings
            if document_component.is_active
            else document_component.rollout_embeddings.detach()
        )
        query_embeddings = F.normalize(query_embeddings, dim=-1)
        document_embeddings = F.normalize(document_embeddings, dim=-1)

        if query_component.is_active and document_component.is_active:
            return torch.einsum("bqd,bksd->bqks", query_embeddings, document_embeddings)
        if query_component.is_active:
            return torch.einsum("bqd,bsd->bqs", query_embeddings, document_embeddings)
        if document_component.is_active:
            return torch.einsum("bd,bksd->bks", query_embeddings, document_embeddings)
        return torch.einsum("bd,bsd->bs", query_embeddings, document_embeddings)

    @staticmethod
    def _compute_cross_score_table(
        query_component: _ActionComponent,
        document_component: _ActionComponent,
    ) -> torch.Tensor:
        query_embeddings = (
            query_component.sampled_embeddings
            if query_component.is_active
            else query_component.rollout_embeddings.detach()
        )
        document_embeddings = (
            document_component.sampled_embeddings
            if document_component.is_active
            else document_component.rollout_embeddings.detach()
        )
        query_embeddings = F.normalize(query_embeddings, dim=-1)
        document_embeddings = F.normalize(document_embeddings, dim=-1)

        if query_component.is_active and document_component.is_active:
            return torch.einsum("bqd,cksd->bcqks", query_embeddings, document_embeddings)
        if query_component.is_active:
            return torch.einsum("bqd,csd->bcqs", query_embeddings, document_embeddings)
        if document_component.is_active:
            return torch.einsum("bd,cksd->bcks", query_embeddings, document_embeddings)
        return torch.einsum("bd,csd->bcs", query_embeddings, document_embeddings)

    @staticmethod
    def _expand_score_table(
        score_table: torch.Tensor,
        query_component: _ActionComponent,
        document_component: _ActionComponent,
        active_index_by_id: dict[int, int],
        num_components: int,
        group_size: int,
    ) -> torch.Tensor:
        batch_size = score_table.size(0)
        slate_length = score_table.size(-1)
        if query_component.is_active and document_component.is_active:
            view_shape = [batch_size, *([1] * num_components), slate_length]
            view_shape[1 + active_index_by_id[id(query_component)]] = group_size
            view_shape[1 + active_index_by_id[id(document_component)]] = group_size
            return score_table.reshape(view_shape).expand(batch_size, *([group_size] * num_components), slate_length)
        if query_component.is_active:
            view_shape = [batch_size, *([1] * num_components), slate_length]
            view_shape[1 + active_index_by_id[id(query_component)]] = group_size
            return score_table.reshape(view_shape).expand(batch_size, *([group_size] * num_components), slate_length)
        if document_component.is_active:
            view_shape = [batch_size, *([1] * num_components), slate_length]
            view_shape[1 + active_index_by_id[id(document_component)]] = group_size
            return score_table.reshape(view_shape).expand(batch_size, *([group_size] * num_components), slate_length)
        return score_table.reshape(batch_size, *([1] * num_components), slate_length).expand(
            batch_size,
            *([group_size] * num_components),
            slate_length,
        )

    @staticmethod
    def _expand_cross_score_table(
        score_table: torch.Tensor,
        query_component: _ActionComponent,
        document_component: _ActionComponent,
        active_index_by_id: dict[int, int],
        num_components: int,
        group_size: int,
    ) -> torch.Tensor:
        batch_size, candidate_batch_size = score_table.shape[:2]
        slate_length = score_table.size(-1)
        view_shape = [batch_size, candidate_batch_size, *([1] * num_components), slate_length]
        if query_component.is_active:
            view_shape[2 + active_index_by_id[id(query_component)]] = group_size
        if document_component.is_active:
            view_shape[2 + active_index_by_id[id(document_component)]] = group_size
        expanded = score_table.reshape(view_shape).expand(
            batch_size,
            candidate_batch_size,
            *([group_size] * num_components),
            slate_length,
        )
        return expanded.permute(0, *range(2, 2 + num_components), 1, 2 + num_components)

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
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
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

        query_component = query_components[0]
        batch_size = relevance_labels.size(0)
        score_grids = []
        for document_component in document_components:
            score_table = self._compute_score_table(
                query_component=query_component,
                document_component=document_component,
            )
            score_grids.append(
                self._expand_score_table(
                    score_table=score_table,
                    query_component=query_component,
                    document_component=document_component,
                    active_index_by_id=active_index_by_id,
                    num_components=num_components,
                    group_size=group_size,
                )
            )
        scores = torch.cat(score_grids, dim=-1)

        in_batch_positive_scores = None
        in_batch_candidate_scores = None
        uses_positive_in_batch = self.reward_type == "ndcg_in_batch" and not self.ndcg_in_batch_include_negatives
        uses_positive_in_batch = uses_positive_in_batch or (
            self.reward_type in {"contrastive", "infonce"} and self.contrastive_use_in_batch_negatives
        )
        if batch_size > 1 and uses_positive_in_batch:
            positive_component = document_components[0]
            positive_cross_scores = self._expand_cross_score_table(
                score_table=self._compute_cross_score_table(
                    query_component=query_component,
                    document_component=positive_component,
                ),
                query_component=query_component,
                document_component=positive_component,
                active_index_by_id=active_index_by_id,
                num_components=num_components,
                group_size=group_size,
            )[..., 0]
            diagonal_mask = torch.eye(batch_size, device=scores.device, dtype=torch.bool)
            diagonal_mask = diagonal_mask.reshape(batch_size, *([1] * num_components), batch_size)
            in_batch_positive_scores = positive_cross_scores.masked_fill(diagonal_mask, float("-inf"))

        if batch_size > 1 and self.reward_type == "ndcg_in_batch" and self.ndcg_in_batch_include_negatives:
            cross_candidate_scores = []
            diagonal_mask = torch.eye(batch_size, device=scores.device, dtype=torch.bool)
            for document_component in document_components:
                expanded_cross_scores = self._expand_cross_score_table(
                    score_table=self._compute_cross_score_table(
                        query_component=query_component,
                        document_component=document_component,
                    ),
                    query_component=query_component,
                    document_component=document_component,
                    active_index_by_id=active_index_by_id,
                    num_components=num_components,
                    group_size=group_size,
                )
                mask_shape = [batch_size, *([1] * num_components), batch_size, 1]
                expanded_cross_scores = expanded_cross_scores.masked_fill(
                    diagonal_mask.reshape(mask_shape),
                    float("-inf"),
                )
                cross_candidate_scores.append(
                    expanded_cross_scores.reshape(batch_size, *([group_size] * num_components), -1)
                )
            in_batch_candidate_scores = torch.cat(cross_candidate_scores, dim=-1)

        rewards = self._compute_rewards_from_scores(
            scores=scores,
            relevance_labels=relevance_labels,
            in_batch_positive_scores=in_batch_positive_scores,
            in_batch_candidate_scores=in_batch_candidate_scores,
        )

        losses = []
        advantages = []
        for component_index, log_prob in enumerate(log_probs):
            sample_dims = tuple(range(1, 1 + num_components))
            other_dims = tuple(dim for dim in sample_dims if dim != component_index + 1)
            component_rewards = rewards.mean(dim=other_dims) if other_dims else rewards
            component_advantages = self._compute_advantages(component_rewards, sample_dims=(1,))
            losses.append(-(component_advantages.detach() * log_prob).mean())
            advantages.append(component_advantages)

        reward_stats = self.summarize_tensor(rewards, prefix="reward")
        return sum(losses), reward_stats, torch.cat(advantages, dim=1)

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

        loss, reward_stats, advantages = self._compute_component_loss(
            relevance_labels=relevance_labels,
            components=tuple(components),
        )

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
