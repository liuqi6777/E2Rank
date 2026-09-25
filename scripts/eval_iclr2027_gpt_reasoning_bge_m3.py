#!/usr/bin/env python3
"""Evaluate BGE-M3 main-table checkpoints with BRIGHT GPT-reasoning queries."""

from __future__ import annotations

from experiments import iclr2027
from experiments.bright_gpt_reasoning import EvalSpec, main

ROOT = iclr2027.ROOT
SPEC = EvalSpec(
    suite=ROOT / "configs/experiments/iclr2027/suite_g1_bge_m3_main.yaml",
    settings=ROOT / "configs/experiments_iclr2027_bge_m3_main.yaml",
    model="BAAI/bge-m3",
    revision="5617a9f61b028005a4858fdac845db406aefb181",
    lora=False,
    run_stems={
        "e0": "G1-BGE-M3-E0",
        "infonce": "G1-BGE-M3-INFONCE",
        "lambdaloss": "G1-BGE-M3-LAMBDALOSS",
        "reler": "G1-BGE-M3-RELER",
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
