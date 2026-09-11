"""Provider-injected query-only score-function policy."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from grpo import sample_vmf
from embedding_protocol import pool_embeddings


@dataclass
class QueryPolicyOutput:
    loss: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor
    degenerate_fraction: torch.Tensor


class QueryPolicyHead(nn.Module):
    """Score-function policy head for externally evaluated query actions."""

    def __init__(self, group_size: int = 32, kappa: float = 755.0, normalize_advantages: bool = True):
        super().__init__()
        if group_size <= 1 or kappa <= 0:
            raise ValueError("group_size must exceed one and kappa must be positive")
        self.group_size = int(group_size)
        self.kappa = float(kappa)
        self.normalize_advantages = bool(normalize_advantages)

    def sample(self, mean_embeddings: torch.Tensor) -> torch.Tensor:
        means = F.normalize(mean_embeddings.float(), dim=-1)
        return sample_vmf(means.detach(), self.kappa, self.group_size).to(mean_embeddings.dtype)

    def compute_advantages(self, rewards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if rewards.dim() != 2 or rewards.size(1) != self.group_size:
            raise ValueError(
                f"rewards must be [batch, {self.group_size}], got {tuple(rewards.shape)}"
            )
        rewards = rewards.float()
        centered = rewards - rewards.mean(dim=1, keepdim=True)
        spread = centered.std(dim=1, keepdim=True, unbiased=False)
        tolerance = 1e-4 * rewards.abs().mean(dim=1, keepdim=True).clamp_min(1.0)
        degenerate = spread <= tolerance
        if self.normalize_advantages:
            centered = torch.where(~degenerate, centered / spread.clamp_min(tolerance), torch.zeros_like(centered))
        return centered, degenerate.squeeze(1)

    def loss(
        self,
        mean_embeddings: torch.Tensor,
        sampled_embeddings: torch.Tensor,
        rewards: torch.Tensor,
    ) -> QueryPolicyOutput:
        means = F.normalize(mean_embeddings.float(), dim=-1)
        if sampled_embeddings.shape != (means.size(0), self.group_size, means.size(1)):
            raise ValueError("sampled_embeddings shape does not match means/group_size")
        advantages, degenerate = self.compute_advantages(rewards)
        log_prob = self.kappa * (
            sampled_embeddings.detach().float() * means.unsqueeze(1)
        ).sum(dim=-1)
        loss = -(advantages.detach() * log_prob).mean()
        return QueryPolicyOutput(
            loss=loss,
            rewards=rewards.detach(),
            advantages=advantages.detach(),
            degenerate_fraction=degenerate.float().mean(),
        )


class QueryOnlyRLWrapper(nn.Module):
    """Shared query encoding, action sampling and provider-injected policy loss."""

    def __init__(
        self,
        model: nn.Module,
        reward_provider,
        *,
        group_size: int = 32,
        kappa: float = 755.0,
        pooling_method: str = "last",
        normalize_advantages: bool = True,
    ):
        super().__init__()
        self.model = model
        self.config = model.config
        self.reward_provider = reward_provider
        self.pooling_method = pooling_method
        self.policy = QueryPolicyHead(group_size, kappa, normalize_advantages)

    def encode_query(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return pool_embeddings(
            self.model(**inputs).last_hidden_state,
            inputs["attention_mask"],
            pooling_method=self.pooling_method,
            normalize=True,
        )

    def policy_step(self, query: dict[str, torch.Tensor], **reward_inputs) -> QueryPolicyOutput:
        means = self.encode_query(query)
        actions = self.policy.sample(means)
        rewards = self.reward_provider(actions, **reward_inputs)
        return self.policy.loss(means, actions, rewards)

    def forward(self, query: dict[str, torch.Tensor], **reward_inputs) -> QueryPolicyOutput:
        return self.policy_step(query, **reward_inputs)

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.model.gradient_checkpointing_enable(*args, **kwargs)

    def enable_input_require_grads(self):
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
