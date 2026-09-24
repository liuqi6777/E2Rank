#!/usr/bin/env python3
"""Evaluate the 0.6B main-table checkpoints with BRIGHT GPT-reasoning queries."""

from __future__ import annotations

from experiments import iclr2027
from experiments.bright_gpt_reasoning import EvalSpec, main

ROOT = iclr2027.ROOT
SPEC = EvalSpec(
    suite=ROOT / "configs/experiments/iclr2027/suite_iclr2027_final_k0.yaml",
    settings=ROOT / "configs/experiments_iclr2027_final.yaml",
    model="Qwen/Qwen3-Embedding-0.6B",
    revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    lora=False,
    run_stems={
        "e0": "G1-R2-E0",
        "infonce": "G1-R2-CL-Strong",
        "lambdaloss": "G1-R2-LL-Graded",
        "reler": "G1-R2-RL-GradedNDCG64-CP-Align070-ShortlistUniform-K0-T1-Pairwise050",
    },
    reler_contract={
        "learning_rate": 5e-6,
        "reward_shortlist_count": 1,
        "reward_shortlist_size": 0,
        "reward_shortlist_pairwise_coef": 0.5,
        "gradient_estimator": "conditional_projection",
        "group_size": 64,
        "target_alignment": 0.70,
    },
)


if __name__ == "__main__":
    raise SystemExit(main(SPEC))
