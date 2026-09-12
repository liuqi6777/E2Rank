"""Shared bandit baselines and dimension-aware, externally scheduled vMF exploration."""
from __future__ import annotations

import math
from functools import lru_cache

import torch


@lru_cache(maxsize=4096)
def mean_alignment(dim: int, kappa: float) -> float:
    """I_(d/2)(kappa) / I_(d/2-1)(kappa), via a convergent continued fraction."""
    if dim < 3 or not math.isfinite(kappa) or kappa <= 0:
        raise ValueError("vMF requires dimension >= 3 and finite positive kappa")
    value = c = float(dim)
    inverse = 0.0
    for j in range(1, 10001):
        b = dim + 2.0 * j
        inverse = 1.0 / (b + kappa * kappa * inverse)
        c = b + kappa * kappa / c
        factor = c * inverse
        value *= factor
        if abs(factor - 1.0) < 2e-15:
            return kappa / value
    raise RuntimeError("vMF mean-alignment continued fraction did not converge")


@lru_cache(maxsize=4096)
def kappa_for_alignment(dim: int, alignment: float) -> float:
    if dim < 3 or not math.isfinite(alignment) or not 0 < alignment < 1:
        raise ValueError("dimension >= 3 and target alignment strictly between 0 and 1 required")
    low, high = 0.0, max(1.0, dim * alignment / (1.0 - alignment**2))
    while mean_alignment(dim, high) < alignment:
        high *= 2.0
    for _ in range(48):
        middle = (low + high) / 2.0
        if mean_alignment(dim, middle) < alignment:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def score_gap_variance(dim: int, kappa: float, gap, perpendicular_norm):
    """Exact vMF projected-score variance, including longitudinal fluctuations."""
    a = mean_alignment(dim, float(kappa))
    derivative = max(0.0, 1.0 - a*a - (dim - 1)*a/kappa)
    return (a / kappa) * perpendicular_norm**2 + derivative * gap**2


def validate_exploration(target_alignment, final_alignment, schedule):
    if schedule not in {"fixed", "linear"}:
        raise ValueError("exploration_schedule must be fixed or linear")
    for value in (target_alignment, final_alignment):
        if value is not None and (not math.isfinite(value) or not 0 < value < 1):
            raise ValueError("exploration alignments must be finite and strictly between 0 and 1")
    if schedule == "linear":
        if target_alignment is None or final_alignment is None:
            raise ValueError("linear exploration requires target_alignment and final_alignment")
        if final_alignment < target_alignment:
            raise ValueError("linear exploration must maintain or increase alignment")
    elif final_alignment is not None:
        raise ValueError("final_alignment requires a linear exploration schedule")


class ExplorationSchedule:
    """A predetermined schedule indexed by completed optimizer steps, never reward feedback.

    target_alignment overrides kappa; linear interpolation ends at the last update.
    HF Trainer restores global_step/max_steps before the first resumed forward pass.
    """
    def __init__(self, kappa=755.0, target_alignment=None, final_alignment=None, schedule="fixed"):
        validate_exploration(target_alignment, final_alignment, schedule)
        if not math.isfinite(kappa) or kappa <= 0:
            raise ValueError("kappa must be finite and positive")
        self.kappa = float(kappa)
        self.target_alignment = target_alignment
        self.final_alignment = final_alignment
        self.schedule = schedule
        self.step = 0
        self.total_steps = 0

    def set_step(self, step: int, total_steps: int):
        if step < 0 or total_steps < 0:
            raise ValueError("training progress cannot be negative")
        self.step, self.total_steps = int(step), int(total_steps)

    def resolve(self, dim: int) -> float:
        if self.target_alignment is None:
            return self.kappa
        alignment = self.target_alignment
        if self.schedule == "linear":
            if self.total_steps <= 0:
                raise ValueError("linear exploration requires total optimizer steps before sampling")
            progress = min(self.step / max(self.total_steps - 1, 1), 1.0)
            alignment += progress * (self.final_alignment - alignment)
        return kappa_for_alignment(dim, float(alignment))

    def state_dict(self):
        return dict(kappa=self.kappa, target_alignment=self.target_alignment,
                    final_alignment=self.final_alignment, schedule=self.schedule,
                    step=self.step, total_steps=self.total_steps)

    def load_state_dict(self, state):
        for key in ("kappa", "target_alignment", "final_alignment", "schedule"):
            if state[key] != getattr(self, key):
                raise ValueError(f"Checkpoint exploration setting differs: {key}")
        self.set_step(state["step"], state["total_steps"])


def group_advantages(rewards, baseline="group", normalization="none", shared_std=None,
                     external_baseline=None):
    """Center without hiding degeneracy when normalization is disabled.

    LOO uses the same opposite-role samples for every marginal. Only unnormalized
    group/LOO estimators have the stated constant-factor/unbiased guarantees.
    """
    rewards = rewards.float()
    if rewards.ndim != 2 or rewards.size(1) < 2:
        raise ValueError("Rewards must have shape [batch, group >= 2]")
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    spread = centered.std(dim=1, keepdim=True, unbiased=False)
    tolerance = 1e-4 * rewards.abs().mean(dim=1, keepdim=True).clamp_min(1.0)
    degenerate = spread <= tolerance
    if baseline == "leave_one_out":
        advantages = centered * (rewards.size(1) / (rewards.size(1) - 1))
    elif baseline == "group":
        advantages = centered
    elif baseline == "ema" and external_baseline is not None:
        advantages = rewards - external_baseline
    else:
        raise ValueError("Unsupported or missing advantage baseline")
    if normalization == "none":
        return advantages, degenerate.squeeze(1)
    if normalization not in {"per_component", "shared"}:
        raise ValueError("Unsupported advantage normalization")
    # Preserve legacy group scaling; LOO changes only the baseline, not this divisor.
    divisor = shared_std if normalization == "shared" and shared_std is not None else spread
    if baseline == "ema":
        advantages = advantages / divisor.clamp_min(tolerance)
    else:
        advantages = torch.where(divisor > tolerance,
                                 advantages / divisor.clamp_min(tolerance), torch.zeros_like(advantages))
    return advantages, degenerate.squeeze(1)
