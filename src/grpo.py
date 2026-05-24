from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import PreTrainedModel
from transformers.file_utils import ModelOutput

from config import RLArguments, normalize_action_components
from rewards import SUPPORTED_REWARD_TYPES, compute_reward_from_scores


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
    advantages_degenerate_frac: Optional[Tensor] = None
    sigma: Optional[Tensor] = None
    kl: Optional[Tensor] = None


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
        kl_coef: float = 0.0,
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
        if kl_coef < 0:
            raise ValueError(f"kl_coef must be non-negative, got {kl_coef}")

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
        self.kl_coef = float(kl_coef)

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

    def _compute_advantages(
        self,
        rewards: torch.Tensor,
        sample_dims: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Promote to fp32: bf16 round-off in mean/std introduces a systematic, distribution-
        # dependent bias in the normalized advantages (especially after marginalization).
        rewards = rewards.float()
        advantages = rewards - rewards.mean(dim=sample_dims, keepdim=True)
        batch_size = rewards.size(0)
        if self.advantage_norm:
            std = advantages.std(dim=sample_dims, keepdim=True, unbiased=False)
            is_degenerate = std <= 1e-4
            # Degenerate groups (all rollouts gave the same reward) carry no learning signal;
            # zero them out instead of dividing by a near-zero std and amplifying noise.
            advantages = torch.where(~is_degenerate, advantages / std, torch.zeros_like(advantages))
            degenerate_mask = is_degenerate.reshape(batch_size, -1).any(dim=-1)
        else:
            degenerate_mask = torch.zeros(batch_size, device=rewards.device, dtype=torch.bool)
        return advantages, degenerate_mask

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
        cross: bool = False,
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

        # query: 'bqd' if active else 'bd'; document (same-batch): 'bksd' if active else 'bsd';
        # document (cross-batch): swaps the leading 'b' for 'c' to pair every query with every other batch's docs.
        query_spec = "bqd" if query_component.is_active else "bd"
        doc_batch = "c" if cross else "b"
        doc_spec = f"{doc_batch}ksd" if document_component.is_active else f"{doc_batch}sd"
        out_spec = "b"
        if cross:
            out_spec += "c"
        if query_component.is_active:
            out_spec += "q"
        if document_component.is_active:
            out_spec += "k"
        out_spec += "s"
        return torch.einsum(f"{query_spec},{doc_spec}->{out_spec}", query_embeddings, document_embeddings)

    @staticmethod
    def _expand_score_table(
        score_table: torch.Tensor,
        query_component: _ActionComponent,
        document_component: _ActionComponent,
        active_index_by_id: dict[int, int],
        num_components: int,
        group_size: int,
        cross: bool = False,
    ) -> torch.Tensor:
        # _compute_score_table emits dims in the order (query, document); the reshape below relies on
        # active_index_by_id[query] < active_index_by_id[document] so the non-singleton slots line up.
        if query_component.is_active and document_component.is_active:
            assert active_index_by_id[id(query_component)] < active_index_by_id[id(document_component)], (
                "query component must precede document component in active_index_by_id"
            )

        slate_length = score_table.size(-1)
        if cross:
            batch_size, candidate_batch_size = score_table.shape[:2]
            leading = [batch_size, candidate_batch_size]
            offset = 2
        else:
            batch_size = score_table.size(0)
            leading = [batch_size]
            offset = 1

        view_shape = [*leading, *([1] * num_components), slate_length]
        if query_component.is_active:
            view_shape[offset + active_index_by_id[id(query_component)]] = group_size
        if document_component.is_active:
            view_shape[offset + active_index_by_id[id(document_component)]] = group_size
        expanded = score_table.reshape(view_shape).expand(
            *leading,
            *([group_size] * num_components),
            slate_length,
        )
        if cross:
            return expanded.permute(0, *range(2, 2 + num_components), 1, 2 + num_components)
        return expanded

    def _compute_component_loss(
        self,
        relevance_labels: torch.Tensor,
        components: Sequence[_ActionComponent],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
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
            positive_cross_scores = self._expand_score_table(
                score_table=self._compute_score_table(
                    query_component=query_component,
                    document_component=positive_component,
                    cross=True,
                ),
                query_component=query_component,
                document_component=positive_component,
                active_index_by_id=active_index_by_id,
                num_components=num_components,
                group_size=group_size,
                cross=True,
            )[..., 0]
            diagonal_mask = torch.eye(batch_size, device=scores.device, dtype=torch.bool)
            diagonal_mask = diagonal_mask.reshape(batch_size, *([1] * num_components), batch_size)
            in_batch_positive_scores = positive_cross_scores.masked_fill(diagonal_mask, float("-inf"))

        if batch_size > 1 and self.reward_type == "ndcg_in_batch" and self.ndcg_in_batch_include_negatives:
            cross_candidate_scores = []
            diagonal_mask = torch.eye(batch_size, device=scores.device, dtype=torch.bool)
            for document_component in document_components:
                expanded_cross_scores = self._expand_score_table(
                    score_table=self._compute_score_table(
                        query_component=query_component,
                        document_component=document_component,
                        cross=True,
                    ),
                    query_component=query_component,
                    document_component=document_component,
                    active_index_by_id=active_index_by_id,
                    num_components=num_components,
                    group_size=group_size,
                    cross=True,
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

        rewards = compute_reward_from_scores(
            scores=scores,
            relevance_labels=relevance_labels,
            reward_type=self.reward_type,
            k=self.reward_ndcg_k,
            ndcg_in_batch_include_negatives=self.ndcg_in_batch_include_negatives,
            contrastive_use_in_batch_negatives=self.contrastive_use_in_batch_negatives,
            contrastive_temperature=self.contrastive_temperature,
            in_batch_positive_scores=in_batch_positive_scores,
            in_batch_candidate_scores=in_batch_candidate_scores,
        ).float()

        losses = []
        advantages = []
        degenerate_masks = []
        for component_index, log_prob in enumerate(log_probs):
            sample_dims = tuple(range(1, 1 + num_components))
            other_dims = tuple(dim for dim in sample_dims if dim != component_index + 1)
            component_rewards = rewards.mean(dim=other_dims) if other_dims else rewards
            component_advantages, component_degenerate = self._compute_advantages(
                component_rewards, sample_dims=(1,)
            )
            losses.append(-(component_advantages.detach() * log_prob).mean())
            advantages.append(component_advantages)
            degenerate_masks.append(component_degenerate)

        # Fraction of (batch-element, component) rows whose reward variance collapsed.
        # These rows received zero advantage and contributed no learning signal.
        degenerate_frac = torch.cat(degenerate_masks).float().mean()
        reward_stats = self.summarize_tensor(rewards, prefix="reward")
        return sum(losses), reward_stats, torch.cat(advantages, dim=1), degenerate_frac

    @staticmethod
    def _kl_term(
        policy_embeddings: torch.Tensor,
        reference_embeddings: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        # KL between two isotropic Gaussians with the same sigma:
        #   KL(N(μ_pol, σ²I) || N(μ_ref, σ²I)) = ||μ_pol - μ_ref||² / (2σ²)
        # Averaged over batch (and slate, for document components).
        reference_embeddings = F.normalize(reference_embeddings, dim=-1).detach()
        squared_distance = (policy_embeddings - reference_embeddings).pow(2).sum(dim=-1)
        return squared_distance.float().mean() / (2.0 * sigma.float().pow(2))

    def forward(
        self,
        rollout_query_embeddings: torch.Tensor,
        rollout_positive_document_embeddings: torch.Tensor,
        rollout_negative_document_embeddings: torch.Tensor,
        relevance_labels: torch.Tensor,
        policy_query_embeddings: torch.Tensor | None = None,
        policy_positive_document_embeddings: torch.Tensor | None = None,
        policy_negative_document_embeddings: torch.Tensor | None = None,
        reference_query_embeddings: torch.Tensor | None = None,
        reference_positive_document_embeddings: torch.Tensor | None = None,
        reference_negative_document_embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
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

        loss, reward_stats, advantages, degenerate_frac = self._compute_component_loss(
            relevance_labels=relevance_labels,
            components=tuple(components),
        )

        kl = torch.zeros((), device=loss.device, dtype=torch.float32)
        if self.kl_coef > 0:
            kl_terms = []
            if self.sample_query:
                if reference_query_embeddings is None:
                    raise ValueError("reference_query_embeddings are required when kl_coef > 0 and query is sampled")
                kl_terms.append(self._kl_term(policy_query_embeddings, reference_query_embeddings, sigma))
            if joint_document_group:
                if reference_positive_document_embeddings is None or reference_negative_document_embeddings is None:
                    raise ValueError(
                        "reference_positive_document_embeddings and reference_negative_document_embeddings "
                        "are required when kl_coef > 0 and the joint document group is sampled"
                    )
                reference_document_embeddings = torch.cat(
                    (reference_positive_document_embeddings, reference_negative_document_embeddings),
                    dim=1,
                )
                kl_terms.append(self._kl_term(policy_document_embeddings, reference_document_embeddings, sigma))
            else:
                if self.sample_positive:
                    if reference_positive_document_embeddings is None:
                        raise ValueError("reference_positive_document_embeddings required when kl_coef > 0 and positive is sampled")
                    kl_terms.append(self._kl_term(policy_positive_document_embeddings, reference_positive_document_embeddings, sigma))
                if self.sample_negative:
                    if reference_negative_document_embeddings is None:
                        raise ValueError("reference_negative_document_embeddings required when kl_coef > 0 and negative is sampled")
                    kl_terms.append(self._kl_term(policy_negative_document_embeddings, reference_negative_document_embeddings, sigma))
            if kl_terms:
                kl = torch.stack(kl_terms).sum()
                loss = loss + self.kl_coef * kl

        advantage_stats = self.summarize_tensor(advantages, prefix="advantages")
        advantage_stats["advantages_degenerate_frac"] = degenerate_frac.detach()
        return loss, reward_stats, advantage_stats, sigma.detach(), kl.detach()


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
            kl_coef=rl_args.kl_coef,
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

        reference_query_embeddings = None
        reference_positive_document_embeddings = None
        reference_negative_document_embeddings = None
        if self.grpo.kl_coef > 0:
            if not hasattr(self.model, "disable_adapter"):
                raise RuntimeError(
                    "kl_coef > 0 requires a PEFT/LoRA model exposing .disable_adapter(); "
                    "either enable LoRA or set kl_coef=0."
                )
            was_training = self.model.training
            self.model.eval()
            try:
                with torch.no_grad():
                    with self.model.disable_adapter():
                        if self.grpo.sample_query:
                            reference_query_embeddings = self.encode(query)
                        if sample_document:
                            encoded_reference_documents = self.encode(document_inputs).reshape(
                                batch_size, slate_length, -1
                            )
                            if self.grpo.sample_positive:
                                reference_positive_document_embeddings = encoded_reference_documents[:, :1]
                            if self.grpo.sample_negative:
                                reference_negative_document_embeddings = encoded_reference_documents[:, 1:]
            finally:
                self.model.train(was_training)

        loss, reward_stats, advantage_stats, sigma, kl = self.grpo(
            rollout_query_embeddings=rollout_query_embeddings,
            rollout_positive_document_embeddings=rollout_positive_document_embeddings,
            rollout_negative_document_embeddings=rollout_negative_document_embeddings,
            relevance_labels=relevance_labels,
            policy_query_embeddings=policy_query_embeddings,
            policy_positive_document_embeddings=policy_positive_document_embeddings,
            policy_negative_document_embeddings=policy_negative_document_embeddings,
            reference_query_embeddings=reference_query_embeddings,
            reference_positive_document_embeddings=reference_positive_document_embeddings,
            reference_negative_document_embeddings=reference_negative_document_embeddings,
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
            advantages_degenerate_frac=advantage_stats["advantages_degenerate_frac"],
            sigma=sigma,
            kl=kl,
        )

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.model.gradient_checkpointing_enable(*args, **kwargs)

    def enable_input_require_grads(self):
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
