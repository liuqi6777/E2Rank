"""Provider-injected query-only score-function policy."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from grpo import sample_vmf
from policy_math import ExplorationSchedule, group_advantages, mean_alignment
from embedding_protocol import pool_embeddings


@dataclass
class QueryPolicyOutput:
    loss: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor
    degenerate_fraction: torch.Tensor


class QueryPolicyHead(nn.Module):
    """Score-function policy head for externally evaluated query actions."""

    def __init__(self, group_size=32, kappa=755.0, normalize_advantages=None, *,
                 advantage_baseline="leave_one_out", advantage_norm="none",
                 advantage_baseline_momentum=0.99, target_alignment=None,
                 final_alignment=None, exploration_schedule="fixed"):
        super().__init__()
        if group_size <= 1:
            raise ValueError("group_size must exceed one")
        if advantage_baseline not in {"group", "leave_one_out", "ema"}:
            raise ValueError("Unsupported advantage baseline")
        if advantage_norm not in {"none", "shared", "per_component"}:
            raise ValueError("Unsupported advantage normalization")
        if not 0 <= advantage_baseline_momentum < 1:
            raise ValueError("Baseline momentum must lie in [0, 1)")
        self.group_size = int(group_size)
        self.kappa = float(kappa)
        self.exploration = ExplorationSchedule(kappa, target_alignment, final_alignment, exploration_schedule)
        self.advantage_baseline = advantage_baseline
        self.advantage_norm = advantage_norm if normalize_advantages is None else (
            "per_component" if normalize_advantages else "none")
        self.advantage_baseline_momentum = advantage_baseline_momentum
        self.sigma_learnable = False
        self.register_buffer("reward_baseline", torch.zeros(()))
        self.register_buffer("reward_baseline_initialized", torch.tensor(False))
        self.dimension = None

    def sample(self, mean_embeddings: torch.Tensor) -> torch.Tensor:
        means = F.normalize(mean_embeddings.float(), dim=-1)
        self.dimension = means.size(-1)
        self.kappa = self.exploration.resolve(self.dimension)
        return sample_vmf(means.detach(), self.kappa, self.group_size).to(mean_embeddings.dtype)

    def compute_advantages(self, rewards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if rewards.dim() != 2 or rewards.size(1) != self.group_size:
            raise ValueError(f"rewards must be [batch, {self.group_size}]")
        baseline = None
        if self.advantage_baseline == "ema":
            baseline = self.reward_baseline.clone()
            if self.training:
                mean = rewards.detach().float().mean()
                if not bool(self.reward_baseline_initialized):
                    self.reward_baseline.copy_(mean)
                    self.reward_baseline_initialized.fill_(True)
                else:
                    self.reward_baseline.lerp_(mean, 1-self.advantage_baseline_momentum)
        return group_advantages(rewards, self.advantage_baseline, self.advantage_norm,
                                external_baseline=baseline)

    def exploration_metrics(self):
        return {"exploration/kappa": self.reward_baseline.new_tensor(self.kappa),
                "exploration/mean_alignment": self.reward_baseline.new_tensor(
                    mean_alignment(self.dimension, self.kappa))}

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
        normalize_advantages: bool | None = None,
        advantage_baseline: str = "leave_one_out",
        advantage_norm: str = "none",
        advantage_baseline_momentum: float = 0.99,
        target_alignment: float | None = None,
        final_alignment: float | None = None,
        exploration_schedule: str = "fixed",
    ):
        super().__init__()
        self.model = model
        self.config = model.config
        self.reward_provider = reward_provider
        self.pooling_method = pooling_method
        self.policy = QueryPolicyHead(group_size, kappa, normalize_advantages,
            advantage_baseline=advantage_baseline, advantage_norm=advantage_norm,
            advantage_baseline_momentum=advantage_baseline_momentum,
            target_alignment=target_alignment, final_alignment=final_alignment,
            exploration_schedule=exploration_schedule)

    @property
    def grpo(self):
        return self.policy

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
