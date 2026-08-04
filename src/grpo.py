from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import PreTrainedModel
from transformers.file_utils import ModelOutput

from config import (
    SUPPORTED_ADVANTAGE_BASELINES,
    SUPPORTED_ROLLOUTS,
    SUPPORTED_SAMPLING_LAWS,
    RLArguments,
    normalize_action_components,
    normalize_advantage_norm_mode,
)
from rewards import (
    SUPPORTED_REWARD_TYPES,
    compute_reward_terms,
    normalize_reward_combine_mode,
    normalize_reward_terms,
    reward_terms_need_in_batch_candidates,
    reward_terms_need_in_batch_positives,
)


def bessel_ratio(nu: float, kappa: torch.Tensor, num_iters: int | None = None) -> torch.Tensor:
    """Modified-Bessel ratio I_nu(kappa) / I_{nu-1}(kappa).

    For nu = d/2 this is the vMF mean resultant length A_d(kappa) = E[mu^T e].
    Evaluated with the backward recurrence 1/R_m = 2m/kappa + R_{m+1} from an
    asymptotic tail estimate; differentiable w.r.t. kappa. The recurrence only
    contracts once the order exceeds ~kappa/2, so the tail must start beyond that.
    """
    kappa = torch.as_tensor(kappa).double()
    if num_iters is None:
        num_iters = max(32, int(float(kappa.detach().max()) / 2.0 - nu) + 64)
    tail_order = nu + num_iters
    ratio = kappa / (tail_order + torch.sqrt(kappa.pow(2) + tail_order**2))
    for step in range(num_iters - 1, -1, -1):
        order = nu + step
        ratio = 1.0 / (2.0 * order / kappa + ratio)
    return ratio.float()


def sample_vmf(
    mean_directions: torch.Tensor,
    kappa: float,
    num_samples: int,
    max_rejection_rounds: int = 256,
) -> torch.Tensor:
    """Draw exact vMF(mean, kappa) samples with Wood's (1994) rejection sampler.

    mean_directions: [n, d] unit vectors. Returns [n, num_samples, d] in the input dtype.
    The w-marginal rejection step runs in float64; per-round acceptance is >~50%, so the
    loop converges in a handful of rounds independent of n and d.
    """
    if mean_directions.dim() != 2:
        raise ValueError(f"mean_directions must be [n, d], got shape {tuple(mean_directions.shape)}")
    kappa = float(kappa)
    if kappa <= 0:
        raise ValueError(f"kappa must be positive, got {kappa}")
    n, dim = mean_directions.shape
    if dim < 3:
        raise ValueError(f"sample_vmf requires embedding dim >= 3, got {dim}")
    device = mean_directions.device

    # Envelope parameters; b is written in rationalized form to avoid the
    # catastrophic cancellation of (-2k + sqrt(4k^2 + (d-1)^2)) at large kappa.
    b = (dim - 1) / (2.0 * kappa + math.sqrt(4.0 * kappa**2 + (dim - 1) ** 2))
    x0 = (1.0 - b) / (1.0 + b)
    log_c = kappa * x0 + (dim - 1) * math.log(1.0 - x0 * x0)

    total = n * num_samples
    w = torch.empty(total, dtype=torch.float64, device=device)
    pending = torch.ones(total, dtype=torch.bool, device=device)
    half = torch.tensor((dim - 1) / 2.0, dtype=torch.float64, device=device)
    beta = torch.distributions.Beta(half, half)
    for _ in range(max_rejection_rounds):
        num_pending = int(pending.sum())
        if num_pending == 0:
            break
        z = beta.sample((num_pending,))
        candidate = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
        uniform = torch.rand(num_pending, dtype=torch.float64, device=device)
        accept = kappa * candidate + (dim - 1) * torch.log1p(-x0 * candidate) - log_c >= torch.log(uniform)
        accepted_indices = pending.nonzero(as_tuple=True)[0][accept]
        w[accepted_indices] = candidate[accept]
        pending[accepted_indices] = False
    if pending.any():
        raise RuntimeError("sample_vmf rejection sampling did not converge")

    means = F.normalize(mean_directions.float(), dim=-1)
    tangent = torch.randn(n, num_samples, dim, dtype=torch.float32, device=device)
    tangent = tangent - (tangent * means.unsqueeze(1)).sum(dim=-1, keepdim=True) * means.unsqueeze(1)
    tangent = F.normalize(tangent, dim=-1)
    w = w.reshape(n, num_samples, 1).to(torch.float32)
    samples = w * means.unsqueeze(1) + torch.sqrt((1.0 - w.pow(2)).clamp_min(0.0)) * tangent
    return samples.to(mean_directions.dtype)


def sample_projected_gaussian(
    mean_directions: torch.Tensor,
    sigma: float,
    num_samples: int,
) -> torch.Tensor:
    """Draw e = normalize(mu + sigma * eps), eps ~ N(0, I_d) -- the projected-Gaussian shortcut.

    Provided only as the sampling-fidelity ablation: the samples follow the projected
    (angular) Gaussian while the surrogate scores them with the vMF log-density, so the
    score-function estimator is evaluated under a law the samples were not drawn from.
    The two laws agree only when sigma*sqrt(d) << 1. Returns [n, num_samples, d].
    """
    if mean_directions.dim() != 2:
        raise ValueError(f"mean_directions must be [n, d], got shape {tuple(mean_directions.shape)}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    means = F.normalize(mean_directions.float(), dim=-1)
    n, dim = means.shape
    noise = torch.randn(n, num_samples, dim, dtype=torch.float32, device=means.device)
    samples = F.normalize(means.unsqueeze(1) + float(sigma) * noise, dim=-1)
    return samples.to(mean_directions.dtype)


def projected_gaussian_mean_alignment(sigma: float, dim: int) -> float:
    """E[mu^T e] for e = normalize(mu + sigma*eps), from ||mu + sigma*eps||^2 -> 1 + sigma^2 d.

    The projected-Gaussian counterpart of the vMF mean resultant length A_d(kappa); used to
    put the frozen-candidate rescaling on the law actually being sampled from.
    """
    return 1.0 / math.sqrt(1.0 + float(sigma) ** 2 * int(dim))


@dataclass
class _ActionComponent:
    role: str
    rollout_embeddings: torch.Tensor
    policy_embeddings: torch.Tensor | None = None
    sampled_embeddings: torch.Tensor | None = None
    kappa: torch.Tensor | None = None

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
    # Per-reward-term scalars, already namespaced ("reward/<term>/mean"). Config-driven, so the
    # key set is identical on every rank -- which the trainer's cross-rank reduce relies on.
    reward_terms: Optional[Dict[str, Tensor]] = None


class GRPO(nn.Module):
    def __init__(
        self,
        action_components=(("query",),),
        group_size: int = 8,
        sigma: float = 0.05,
        kappa: float | None = None,
        sigma_learnable: bool = False,
        sigma_min: float = 1e-3,
        sigma_max: float = 0.5,
        reward_type: str = "ndcg",
        reward_terms=None,
        reward_combine: str = "sum",
        reward_ndcg_k: int = 10,
        ndcg_in_batch_include_negatives: bool = False,
        contrastive_use_in_batch_negatives: bool = False,
        contrastive_temperature: float = 0.03,
        advantage_norm: str | bool = "per_component",
        sampling_law: str = "vmf",
        rollout: str = "product",
        frozen_doc_rescale: bool = True,
        advantage_baseline: str = "group",
        advantage_baseline_momentum: float = 0.99,
        in_batch_use_sampled_documents: bool = False,
        kl_coef: float = 0.0,
    ):
        super().__init__()
        reward_type = reward_type.lower()
        action_components = normalize_action_components(action_components)
        if advantage_baseline not in SUPPORTED_ADVANTAGE_BASELINES:
            raise ValueError(
                f"Unsupported advantage_baseline: {advantage_baseline!r}. "
                f"Expected one of {SUPPORTED_ADVANTAGE_BASELINES}."
            )
        if sampling_law not in SUPPORTED_SAMPLING_LAWS:
            raise ValueError(
                f"Unsupported sampling_law: {sampling_law!r}. Expected one of {SUPPORTED_SAMPLING_LAWS}."
            )
        if rollout not in SUPPORTED_ROLLOUTS:
            raise ValueError(f"Unsupported rollout: {rollout!r}. Expected one of {SUPPORTED_ROLLOUTS}.")
        if group_size < 2:
            raise ValueError("group_size must be at least 2 for GRPO")
        if kappa is not None:
            if kappa <= 0:
                raise ValueError(f"kappa must be positive, got {kappa}")
            sigma = kappa ** -0.5
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if reward_type not in SUPPORTED_REWARD_TYPES:
            raise ValueError(
                f"Unsupported reward type: {reward_type}. Supported types: {sorted(SUPPORTED_REWARD_TYPES)}"
            )
        if contrastive_temperature <= 0:
            raise ValueError(f"contrastive_temperature must be positive, got {contrastive_temperature}")
        reward_combine = normalize_reward_combine_mode(reward_combine)
        reward_terms = normalize_reward_terms(
            reward_terms if reward_terms else reward_type,
            default_k=reward_ndcg_k,
            default_temperature=contrastive_temperature,
            default_ndcg_in_batch_include_negatives=ndcg_in_batch_include_negatives,
            default_contrastive_use_in_batch_negatives=contrastive_use_in_batch_negatives,
        )
        if advantage_baseline == "ema" and reward_combine == "normalized_sum" and len(reward_terms) > 1:
            # The EMA baseline is one global scalar; it cannot track several reward scales at
            # once, so per-term standardization would baseline every term against the same
            # running mean of a different quantity.
            raise ValueError(
                "advantage_baseline='ema' is incompatible with reward_combine='normalized_sum' "
                "for multiple reward terms."
            )
        if kl_coef < 0:
            raise ValueError(f"kl_coef must be non-negative, got {kl_coef}")
        if sigma_learnable:
            if not 0 < sigma_min < sigma_max:
                raise ValueError(f"Expected 0 < sigma_min < sigma_max, got [{sigma_min}, {sigma_max}]")
            if not sigma_min <= sigma <= sigma_max:
                raise ValueError(
                    f"Initial sigma {sigma} must lie within the learnable bounds [{sigma_min}, {sigma_max}]"
                )

        self.action_components = action_components
        self.sample_query = any(group == ("query",) for group in action_components)
        self.sample_positive = any("positive" in group for group in action_components)
        self.sample_negative = any("negative" in group for group in action_components)
        self.group_size = group_size
        self.reward_type = reward_type
        self.reward_terms = reward_terms
        self.reward_combine = reward_combine
        self.reward_ndcg_k = reward_ndcg_k
        self.ndcg_in_batch_include_negatives = ndcg_in_batch_include_negatives
        self.contrastive_use_in_batch_negatives = contrastive_use_in_batch_negatives
        self.contrastive_temperature = contrastive_temperature
        self.advantage_norm = normalize_advantage_norm_mode(advantage_norm)
        self.sampling_law = sampling_law
        self.rollout = rollout
        self.frozen_doc_rescale = bool(frozen_doc_rescale)
        self.advantage_baseline = advantage_baseline
        self.advantage_baseline_momentum = float(advantage_baseline_momentum)
        self.in_batch_use_sampled_documents = in_batch_use_sampled_documents
        self.sigma_learnable = sigma_learnable
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.kl_coef = float(kl_coef)

        if sigma_learnable:
            if advantage_baseline == "ema":
                # log C_d(kappa) is dropped from the surrogate on the grounds that it cancels
                # against advantages that sum to zero within each group. An EMA baseline does
                # not center within the group, so that cancellation fails and the recovered
                # d/d(log sigma) gradient would be missing its -A_d(kappa) term.
                raise ValueError(
                    "advantage_baseline='ema' is incompatible with sigma_learnable=True: the vMF "
                    "normalizer only cancels under group-centered advantages."
                )
            self.log_sigma = nn.Parameter(torch.log(torch.tensor(float(sigma), dtype=torch.float32)))
        else:
            self.register_buffer("fixed_sigma", torch.tensor(float(sigma), dtype=torch.float32))

        if advantage_baseline == "ema":
            self.register_buffer("reward_baseline", torch.zeros((), dtype=torch.float32))
            self.register_buffer("reward_baseline_initialized", torch.zeros((), dtype=torch.bool))

    def current_sigma(self, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if self.sigma_learnable:
            # Hard bounds: REINFORCE on the exploration scale has no restoring force (the
            # normalizer term cancels under centered advantages), so an unbounded sigma can
            # collapse exploration entirely.
            log_sigma = self.log_sigma.clamp(math.log(self.sigma_min), math.log(self.sigma_max))
            sigma = torch.exp(log_sigma)
        else:
            sigma = self.fixed_sigma
        return sigma.to(device=device, dtype=dtype)

    @staticmethod
    def summarize_tensor(
        values: torch.Tensor,
        prefix: str,
        separator: str = "_",
    ) -> dict[str, torch.Tensor]:
        flattened_values = values.detach().reshape(-1).to(dtype=torch.float32)
        return {
            f"{prefix}{separator}mean": flattened_values.mean(),
            f"{prefix}{separator}std": flattened_values.std(unbiased=False),
            f"{prefix}{separator}min": flattened_values.min(),
            f"{prefix}{separator}max": flattened_values.max(),
        }

    def _current_reward_baseline(self, component_rewards: torch.Tensor) -> torch.Tensor:
        """Global running baseline for the REINFORCE ablation (advantage_baseline='ema').

        Returns the baseline to subtract *before* updating it, so the baseline is
        independent of the rewards it baselines and the estimator stays unbiased.
        The update uses the local-rank batch mean; ranks therefore hold slightly
        different baselines, which is the usual REINFORCE practice and is why this
        exists only as a comparison point for the group baseline.
        """
        batch_mean = component_rewards.detach().mean()
        if not bool(self.reward_baseline_initialized):
            self.reward_baseline.copy_(batch_mean)
            self.reward_baseline_initialized.fill_(True)
            return self.reward_baseline.clone()

        baseline = self.reward_baseline.clone()
        if self.training:
            momentum = self.advantage_baseline_momentum
            self.reward_baseline.mul_(momentum).add_(batch_mean, alpha=1.0 - momentum)
        return baseline

    def _compute_advantages(
        self,
        component_rewards: torch.Tensor,
        shared_std: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Promote to fp32: bf16 round-off in mean/std introduces a systematic, distribution-
        # dependent bias in the normalized advantages (especially after marginalization).
        component_rewards = component_rewards.float()
        if self.advantage_baseline == "ema":
            advantages = component_rewards - self._current_reward_baseline(component_rewards)
        else:
            advantages = component_rewards - component_rewards.mean(dim=1, keepdim=True)
        batch_size = component_rewards.size(0)
        if self.advantage_norm == "none":
            degenerate_mask = torch.zeros(batch_size, device=component_rewards.device, dtype=torch.bool)
            return advantages, degenerate_mask

        # 'per_component' rescales each component's group to unit std, which equalizes gradient
        # magnitudes across components regardless of their true effect sizes; 'shared' divides all
        # components by the per-sample std of the raw reward tensor instead, preserving the
        # relative first-order effects of query vs. document perturbations.
        std = shared_std if shared_std is not None else advantages.std(dim=1, keepdim=True, unbiased=False)
        # Scale-aware threshold: rewards range from nDCG in [0, 1] to contrastive margins of a
        # very different magnitude, so a fixed absolute cutoff means something different for each.
        # clamp_min(1) keeps this identical to the old absolute 1e-4 for bounded rewards.
        reward_scale = component_rewards.abs().mean(dim=1, keepdim=True).clamp_min(1.0)
        is_degenerate = std <= 1e-4 * reward_scale
        if self.advantage_baseline == "ema":
            # Under a global baseline a zero-spread group is NOT signal-free: every rollout can
            # be uniformly better than the running baseline. Zeroing it would discard a real
            # gradient, so only guard the division.
            advantages = advantages / std.clamp_min(1e-4 * reward_scale)
        else:
            # Degenerate groups (all rollouts gave the same reward) carry no learning signal;
            # zero them out instead of dividing by a near-zero std and amplifying noise.
            advantages = torch.where(~is_degenerate, advantages / std, torch.zeros_like(advantages))
        degenerate_mask = is_degenerate.reshape(batch_size, -1).any(dim=-1)
        return advantages, degenerate_mask

    def _draw(self, mean_directions: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        if self.sampling_law == "gaussian":
            return sample_projected_gaussian(
                mean_directions,
                sigma=float(kappa.detach()) ** -0.5,
                num_samples=self.group_size,
            )
        return sample_vmf(
            mean_directions,
            kappa=float(kappa.detach()),
            num_samples=self.group_size,
        )

    def _sample_document_embeddings(
        self,
        rollout_document_embeddings: torch.Tensor,
        kappa: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, slate_length, dim = rollout_document_embeddings.shape
        samples = self._draw(
            rollout_document_embeddings.detach().reshape(batch_size * slate_length, dim),
            kappa,
        )
        return samples.reshape(batch_size, slate_length, self.group_size, dim).permute(0, 2, 1, 3)

    def _document_component(
        self,
        rollout_embeddings: torch.Tensor,
        policy_embeddings: torch.Tensor | None,
        kappa: torch.Tensor,
        sample: bool,
        role_name: str,
    ) -> _ActionComponent:
        """One document-side action component, sampled or frozen at its rollout mean."""
        if not sample:
            return _ActionComponent(role="document", rollout_embeddings=rollout_embeddings)
        if policy_embeddings is None:
            raise ValueError(
                f"policy_{role_name}_document_embeddings are required when {role_name} is sampled"
            )
        return _ActionComponent(
            role="document",
            rollout_embeddings=rollout_embeddings,
            policy_embeddings=policy_embeddings,
            sampled_embeddings=self._sample_document_embeddings(rollout_embeddings, kappa),
            kappa=kappa,
        )

    @staticmethod
    def _vmf_log_prob(
        policy_embeddings: torch.Tensor,
        sampled_embeddings: torch.Tensor,
        kappa: torch.Tensor,
    ) -> torch.Tensor:
        # log pi(e | h) = kappa * h^T e + log C_d(kappa). The normalizer is constant within each
        # group, so it cancels exactly against group-centered advantages — both in the policy-mean
        # gradient and in the learnable-kappa gradient — and is omitted. Cosines are computed in
        # fp32: kappa is large, so bf16 round-off in h^T e would dominate the log-prob differences.
        cosine = (sampled_embeddings.detach().float() * policy_embeddings.float().unsqueeze(1)).sum(dim=-1)
        return kappa.float() * cosine

    @staticmethod
    def _document_vmf_log_prob(
        policy_document_embeddings: torch.Tensor,
        sampled_document_embeddings: torch.Tensor,
        kappa: torch.Tensor,
        document_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if policy_document_embeddings.size(1) == 0:
            return torch.zeros(
                sampled_document_embeddings.size(0),
                sampled_document_embeddings.size(1),
                device=sampled_document_embeddings.device,
                dtype=torch.float32,
            )

        cosine = (
            sampled_document_embeddings.detach().float() * policy_document_embeddings.float().unsqueeze(1)
        ).sum(dim=-1)
        log_prob = kappa.float() * cosine
        # The exact product-policy log-prob is the SUM over slate documents; we average instead,
        # which rescales the slate-side gradient by 1/slate_length and keeps its magnitude
        # comparable across slate sizes.
        if document_mask is None:
            return log_prob.mean(dim=-1)

        mask = document_mask.unsqueeze(1).to(dtype=log_prob.dtype)
        num_documents = mask.sum(dim=-1).clamp_min(1.0)
        return (log_prob * mask).sum(dim=-1) / num_documents

    @staticmethod
    def _resolve_group_size(active_components: Sequence[_ActionComponent]) -> int:
        if not active_components:
            raise ValueError("At least one action component is required")
        group_size = active_components[0].sampled_embeddings.size(1)
        for component in active_components:
            if component.sampled_embeddings.size(1) != group_size:
                raise ValueError("action component group sizes must match")
        return group_size

    @staticmethod
    def _compute_score_table(
        query_component: _ActionComponent,
        document_component: _ActionComponent,
        cross: bool = False,
        diagonal: bool = False,
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
        # fp32: the reward ranks candidates by these scores, and bf16 resolves cosines only
        # to ~4e-3 near 1.0 — coarser than the score gaps inside a hard-negative slate. The
        # resulting ties are broken by topk toward the lowest index, which the collator
        # always fills with the gold positive, so the round-off biases the reward upward
        # instead of just adding noise.
        query_embeddings = F.normalize(query_embeddings.float(), dim=-1)
        document_embeddings = F.normalize(document_embeddings.float(), dim=-1)

        # query: 'bqd' if active else 'bd'; document (same-batch): 'bksd' if active else 'bsd';
        # document (cross-batch): swaps the leading 'b' for 'c' to pair every query with every other batch's docs.
        # Under the diagonal rollout the two sides share one group index, so the contraction
        # emits only the paired entries r^(g,g) instead of the full G x G grid.
        document_index = "q" if diagonal else "k"
        query_spec = "bqd" if query_component.is_active else "bd"
        doc_batch = "c" if cross else "b"
        doc_spec = f"{doc_batch}{document_index}sd" if document_component.is_active else f"{doc_batch}sd"
        out_spec = "b"
        if cross:
            out_spec += "c"
        if query_component.is_active or (diagonal and document_component.is_active):
            out_spec += "q"
        if document_component.is_active and not diagonal:
            out_spec += "k"
        out_spec += "s"
        return torch.einsum(f"{query_spec},{doc_spec}->{out_spec}", query_embeddings, document_embeddings)

    @staticmethod
    def _mask_cross_batch_diagonal(score_table: torch.Tensor, batch_size: int) -> torch.Tensor:
        # Drop each sample's own documents from its cross-batch distractor pool. Masking the
        # compact [batch, candidate_batch, ...] table here rather than the expanded grid avoids
        # materializing a second copy of a tensor that carries one axis per action component.
        diagonal_mask = torch.eye(batch_size, device=score_table.device, dtype=torch.bool)
        trailing_dims = [1] * (score_table.dim() - 2)
        return score_table.masked_fill(
            diagonal_mask.reshape(batch_size, batch_size, *trailing_dims),
            float("-inf"),
        )

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
            assert active_index_by_id[id(query_component)] <= active_index_by_id[id(document_component)], (
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

    def _cross_batch_component(self, component: _ActionComponent) -> _ActionComponent:
        # In-batch candidates from OTHER samples enter each sample's slate as distractors. Scoring
        # them with their sampled embeddings would leak every other sample's perturbation into this
        # sample's advantages through the shared group index (cross-sample credit contamination),
        # so by default they are represented by their detached mean embeddings.
        if self.in_batch_use_sampled_documents:
            return component
        return _ActionComponent(role=component.role, rollout_embeddings=component.rollout_embeddings)

    def _frozen_doc_scale(self, document_components: Sequence[_ActionComponent]) -> torch.Tensor | None:
        # A sampled vMF embedding is attenuated toward the origin in expectation: E[e] = A_d(kappa) mu.
        # Any candidate scored at its frozen mean embedding (an unsampled slate component, or in-batch
        # candidates under the default in_batch_use_sampled_documents=False) therefore lands on a score
        # scale 1/A_d(kappa) above the sampled documents it competes with (~3x at kappa=400, d=1024)
        # and systematically outranks them, collapsing ranking rewards to a constant. Whenever at
        # least one document component is sampled, every frozen-document score table is rescaled by
        # A_d(kappa) so all candidates share the same expected score scale.
        if not self.frozen_doc_rescale:
            return None
        sampled = [component for component in document_components if component.is_active]
        if not sampled:
            return None
        dim = sampled[0].rollout_embeddings.size(-1)
        kappa = sampled[0].kappa.detach()
        if self.sampling_law == "gaussian":
            # Match the law actually sampled from, so the ablation compares sampling fidelity
            # rather than an incidental miscalibration of the frozen candidates.
            alignment = projected_gaussian_mean_alignment(float(kappa) ** -0.5, dim)
            return torch.as_tensor(alignment, dtype=torch.float32, device=kappa.device)
        return bessel_ratio(dim / 2.0, kappa)

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

        diagonal = self.rollout == "diagonal"
        num_components = 1 if diagonal else len(active_components)
        group_size = self._resolve_group_size(active_components)

        log_probs = []
        # Diagonal rollout: every component writes into the single shared group axis.
        active_index_by_id = {
            id(component): (0 if diagonal else component_index)
            for component_index, component in enumerate(active_components)
        }
        for component in active_components:
            if component.role == "query":
                log_probs.append(
                    self._vmf_log_prob(
                        policy_embeddings=component.policy_embeddings,
                        sampled_embeddings=component.sampled_embeddings,
                        kappa=component.kappa,
                    )
                )
            elif component.role == "document":
                log_probs.append(
                    self._document_vmf_log_prob(
                        policy_document_embeddings=component.policy_embeddings,
                        sampled_document_embeddings=component.sampled_embeddings,
                        kappa=component.kappa,
                    )
                )
            else:
                raise ValueError(f"Unsupported action component role: {component.role}")

        query_component = query_components[0]
        batch_size = relevance_labels.size(0)
        frozen_doc_scale = self._frozen_doc_scale(document_components)

        def score_grid(document_component: _ActionComponent, cross: bool = False) -> torch.Tensor:
            """Query-vs-document scores, rescaled if frozen, broadcast onto the rollout grid."""
            if cross:
                document_component = self._cross_batch_component(document_component)
            score_table = self._compute_score_table(
                query_component=query_component,
                document_component=document_component,
                cross=cross,
                diagonal=diagonal,
            )
            if frozen_doc_scale is not None and not document_component.is_active:
                score_table = score_table * frozen_doc_scale.to(score_table.dtype)
            if cross:
                score_table = self._mask_cross_batch_diagonal(score_table, batch_size)
            return self._expand_score_table(
                score_table=score_table,
                query_component=query_component,
                document_component=document_component,
                active_index_by_id=active_index_by_id,
                num_components=num_components,
                group_size=group_size,
                cross=cross,
            )

        scores = torch.cat(
            [score_grid(document_component) for document_component in document_components],
            dim=-1,
        )

        in_batch_positive_scores = None
        in_batch_candidate_scores = None
        # The cross-batch tables do not depend on the reward term, so they are built once from
        # the UNION of what the terms ask for and shared by all of them.
        if batch_size > 1 and reward_terms_need_in_batch_positives(self.reward_terms):
            in_batch_positive_scores = score_grid(document_components[0], cross=True)[..., 0]

        if batch_size > 1 and reward_terms_need_in_batch_candidates(self.reward_terms):
            in_batch_candidate_scores = torch.cat(
                [
                    score_grid(document_component, cross=True).reshape(
                        batch_size, *([group_size] * num_components), -1
                    )
                    for document_component in document_components
                ],
                dim=-1,
            )

        term_rewards = {
            name: reward.float()
            for name, reward in compute_reward_terms(
                self.reward_terms,
                scores=scores,
                relevance_labels=relevance_labels,
                in_batch_positive_scores=in_batch_positive_scores,
                in_batch_candidate_scores=in_batch_candidate_scores,
            ).items()
        }

        rewards = None
        for term in self.reward_terms:
            contribution = term_rewards[term.name] * term.weight
            rewards = contribution if rewards is None else rewards + contribution

        # 'sum' takes one advantage over the combined reward, so each term enters the gradient
        # in proportion to weight x its own within-group spread. 'normalized_sum' standardizes
        # every term's advantage on its own first; because the surrogate is linear in the
        # advantage, the weighted sum of advantages IS a reward combination -- just one whose
        # weights are scale-free, which is what mixing bounded and unbounded rewards needs.
        if self.reward_combine == "sum" or len(self.reward_terms) == 1:
            advantage_inputs: tuple[tuple[float, torch.Tensor], ...] = ((1.0, rewards),)
        else:
            advantage_inputs = tuple(
                (term.weight, term_rewards[term.name])
                for term in self.reward_terms
                if term.weight != 0.0
            )

        shared_stds: list[torch.Tensor | None] = []
        for _, reward_tensor in advantage_inputs:
            if self.advantage_norm == "shared":
                shared_stds.append(
                    reward_tensor.reshape(batch_size, -1).std(dim=-1, unbiased=False, keepdim=True)
                )
            else:
                shared_stds.append(None)

        losses = []
        advantages = []
        degenerate_masks = []
        for component_index, log_prob in enumerate(log_probs):
            sample_dims = tuple(range(1, 1 + num_components))
            # Diagonal rollout has a single shared axis, so no axis is "the other side" and
            # every component receives the same advantage vector -- the interaction is
            # attributed to both sides at once, which is exactly the trade-off it makes.
            axis = 0 if diagonal else component_index
            other_dims = tuple(dim for dim in sample_dims if dim != axis + 1)
            component_advantages = None
            for (weight, reward_tensor), shared_std in zip(advantage_inputs, shared_stds):
                component_rewards = reward_tensor.mean(dim=other_dims) if other_dims else reward_tensor
                term_advantages, term_degenerate = self._compute_advantages(
                    component_rewards, shared_std=shared_std
                )
                if weight != 1.0:
                    term_advantages = term_advantages * weight
                component_advantages = (
                    term_advantages
                    if component_advantages is None
                    else component_advantages + term_advantages
                )
                degenerate_masks.append(term_degenerate)
            losses.append(-(component_advantages.detach() * log_prob).mean())
            advantages.append(component_advantages)

        # Fraction of (batch-element, component, reward-term) rows whose reward variance
        # collapsed. These rows received zero advantage and contributed no learning signal.
        degenerate_frac = torch.cat(degenerate_masks).float().mean()
        reward_stats = self.summarize_tensor(rewards, prefix="reward")
        if len(self.reward_terms) > 1:
            # Per-term diagnostics. 'group_std' is the mean within-sample spread and is the
            # number that actually decides how much a term contributes under reward_combine=
            # 'sum'; the plain 'std' below is the spread across the whole batch.
            for term in self.reward_terms:
                term_reward = term_rewards[term.name].detach()
                reward_stats.update(
                    self.summarize_tensor(term_reward, prefix=f"reward/{term.name}", separator="/")
                )
                reward_stats[f"reward/{term.name}/group_std"] = (
                    term_reward.reshape(batch_size, -1).std(dim=-1, unbiased=False).mean()
                )
        return sum(losses), reward_stats, torch.cat(advantages, dim=1), degenerate_frac

    @staticmethod
    def _kl_term(
        policy_embeddings: torch.Tensor,
        reference_embeddings: torch.Tensor,
        kappa: torch.Tensor,
    ) -> torch.Tensor:
        # KL between two vMF distributions with the same concentration:
        #   KL(vMF(μ_pol, κ) || vMF(μ_ref, κ)) = κ · A_d(κ) · (1 - μ_pol^T μ_ref),
        # where A_d(κ) = I_{d/2}(κ)/I_{d/2-1}(κ) is the mean resultant length.
        # Averaged over batch (and slate, for document components).
        reference_embeddings = F.normalize(reference_embeddings.float(), dim=-1).detach()
        alignment = (policy_embeddings.float() * reference_embeddings).sum(dim=-1)
        kappa = kappa.float()
        mean_resultant_length = bessel_ratio(policy_embeddings.size(-1) / 2.0, kappa)
        return kappa * mean_resultant_length * (1.0 - alignment).mean()

    @staticmethod
    def _normalize_policy(
        policy_embeddings: torch.Tensor | None,
        rollout_embeddings: torch.Tensor,
        policy_name: str,
        rollout_name: str,
    ) -> torch.Tensor | None:
        """Validate a policy tensor against its rollout twin and unit-normalize it."""
        if policy_embeddings is None:
            return None
        if policy_embeddings.shape != rollout_embeddings.shape:
            raise ValueError(
                f"{policy_name} shape must match {rollout_name}, "
                f"got policy={tuple(policy_embeddings.shape)} "
                f"rollout={tuple(rollout_embeddings.shape)}"
            )
        return F.normalize(policy_embeddings, dim=-1)

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
        policy_query_embeddings = self._normalize_policy(
            policy_query_embeddings,
            rollout_query_embeddings,
            "policy_query_embeddings",
            "rollout_query_embeddings",
        )
        rollout_positive_document_embeddings = F.normalize(rollout_positive_document_embeddings, dim=-1)
        rollout_negative_document_embeddings = F.normalize(rollout_negative_document_embeddings, dim=-1)
        document_embeddings = F.normalize(document_embeddings, dim=-1)
        policy_positive_document_embeddings = self._normalize_policy(
            policy_positive_document_embeddings,
            rollout_positive_document_embeddings,
            "policy_positive_document_embeddings",
            "positive_document_embeddings",
        )
        policy_negative_document_embeddings = self._normalize_policy(
            policy_negative_document_embeddings,
            rollout_negative_document_embeddings,
            "policy_negative_document_embeddings",
            "negative_document_embeddings",
        )

        # sigma stays in fp32; kappa is a differentiable function of log_sigma when learnable,
        # so the vMF log-prob term kappa * h^T e carries the exploration-scale gradient.
        sigma = self.current_sigma(device=rollout_query_embeddings.device, dtype=torch.float32)
        kappa = sigma.pow(-2)
        components = []
        if self.sample_query:
            if policy_query_embeddings is None:
                raise ValueError("policy_query_embeddings are required when query is sampled")
            components.append(_ActionComponent(
                role="query",
                rollout_embeddings=rollout_query_embeddings,
                policy_embeddings=policy_query_embeddings,
                sampled_embeddings=self._draw(rollout_query_embeddings.detach(), kappa),
                kappa=kappa,
            ))
        else:
            components.append(_ActionComponent(
                role="query",
                rollout_embeddings=rollout_query_embeddings,
            ))

        joint_document_group = ("positive", "negative") in self.action_components or \
          ("negative", "positive") in self.action_components
        if joint_document_group:
            # One sigma noise term covers the whole slate, so both policy sides must be present
            # before they are concatenated into a single document component.
            for role_name, policy_embeddings in (
                ("positive", policy_positive_document_embeddings),
                ("negative", policy_negative_document_embeddings),
            ):
                if policy_embeddings is None:
                    raise ValueError(
                        f"policy_{role_name}_document_embeddings are required when {role_name} is sampled"
                    )
            policy_document_embeddings = torch.cat(
                (policy_positive_document_embeddings, policy_negative_document_embeddings),
                dim=1,
            )
            components.append(self._document_component(
                rollout_embeddings=document_embeddings,
                policy_embeddings=policy_document_embeddings,
                kappa=kappa,
                sample=True,
                role_name="positive",
            ))
        else:
            components.append(self._document_component(
                rollout_embeddings=rollout_positive_document_embeddings,
                policy_embeddings=policy_positive_document_embeddings,
                kappa=kappa,
                sample=self.sample_positive,
                role_name="positive",
            ))
            components.append(self._document_component(
                rollout_embeddings=rollout_negative_document_embeddings,
                policy_embeddings=policy_negative_document_embeddings,
                kappa=kappa,
                sample=self.sample_negative,
                role_name="negative",
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
                    raise ValueError(
                        "reference_query_embeddings are required when kl_coef > 0 and query is sampled"
                    )
                kl_terms.append(self._kl_term(policy_query_embeddings, reference_query_embeddings, kappa))
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
                kl_terms.append(
                    self._kl_term(policy_document_embeddings, reference_document_embeddings, kappa)
                )
            else:
                for role_name, sampled, policy_embeddings, reference_embeddings in (
                    (
                        "positive",
                        self.sample_positive,
                        policy_positive_document_embeddings,
                        reference_positive_document_embeddings,
                    ),
                    (
                        "negative",
                        self.sample_negative,
                        policy_negative_document_embeddings,
                        reference_negative_document_embeddings,
                    ),
                ):
                    if not sampled:
                        continue
                    if reference_embeddings is None:
                        raise ValueError(
                            f"reference_{role_name}_document_embeddings required when kl_coef > 0 "
                            f"and {role_name} is sampled"
                        )
                    kl_terms.append(self._kl_term(policy_embeddings, reference_embeddings, kappa))
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
            kappa=rl_args.kappa,
            sigma_learnable=rl_args.sigma_learnable,
            sigma_min=rl_args.sigma_min,
            sigma_max=rl_args.sigma_max,
            reward_type=rl_args.reward_type,
            reward_terms=rl_args.reward_terms,
            reward_combine=rl_args.reward_combine,
            reward_ndcg_k=rl_args.reward_ndcg_k,
            ndcg_in_batch_include_negatives=rl_args.ndcg_in_batch_include_negatives,
            contrastive_use_in_batch_negatives=rl_args.contrastive_use_in_batch_negatives,
            contrastive_temperature=rl_args.contrastive_temperature,
            advantage_norm=rl_args.advantage_norm,
            sampling_law=rl_args.sampling_law,
            rollout=rl_args.rollout,
            frozen_doc_rescale=rl_args.frozen_doc_rescale,
            advantage_baseline=rl_args.advantage_baseline,
            advantage_baseline_momentum=rl_args.advantage_baseline_momentum,
            in_batch_use_sampled_documents=rl_args.in_batch_use_sampled_documents,
            kl_coef=rl_args.kl_coef,
        )

    def encode(self, model_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return pool_last_token_embedding(
            self.model(**model_inputs).last_hidden_state,
            model_inputs["attention_mask"],
            normalize=True,
        )

    @torch.no_grad()
    def eval_ranking_metrics(
        self,
        query: Dict[str, torch.Tensor] = None,
        positive_document: Dict[str, torch.Tensor] = None,
        negative_document: Dict[str, torch.Tensor] = None,
        relevance_labels: torch.Tensor = None,
        **_: object,
    ) -> Dict[str, torch.Tensor]:
        """Deterministic dev metrics: the policy MEAN, no sampling.

        This is what deployment computes, and unlike the policy-gradient surrogate its
        value is comparable across configurations -- which is what model selection needs.
        """
        from ranking_eval import ranking_eval_metrics, score_slate_deterministically

        slate_length = relevance_labels.size(1)
        scores = score_slate_deterministically(
            self.encode, query, positive_document, negative_document, slate_length
        )
        return ranking_eval_metrics(scores.float(), relevance_labels, k=self.grpo.reward_ndcg_k)

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
        # Namespaced keys are the per-term diagnostics; the flat ones (reward_mean/std/min/max
        # and advantages_*) are named exactly like the GRPOModelOutput fields they fill.
        term_metrics = {key: value for key, value in reward_stats.items() if "/" in key}
        aggregate_stats = {key: value for key, value in reward_stats.items() if "/" not in key}

        return GRPOModelOutput(
            loss=loss,
            reward=reward_stats["reward_mean"],
            **aggregate_stats,
            **advantage_stats,
            sigma=sigma,
            kl=kl,
            reward_terms=term_metrics or None,
        )

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.model.gradient_checkpointing_enable(*args, **kwargs)

    def enable_input_require_grads(self):
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
