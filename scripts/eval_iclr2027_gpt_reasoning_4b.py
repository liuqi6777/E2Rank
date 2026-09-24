#!/usr/bin/env python3
"""Evaluate the 4B main-table checkpoints with BRIGHT GPT-reasoning queries."""

from __future__ import annotations

from experiments import iclr2027
from experiments.bright_gpt_reasoning import EvalSpec, main

ROOT = iclr2027.ROOT
SPEC = EvalSpec(
    suite=ROOT / "configs/experiments/iclr2027/suite_qwen3_embedding_4b_lora.yaml",
    settings=ROOT / "configs/experiments_qwen3_embedding_4b_lora.yaml",
    model="Qwen/Qwen3-Embedding-4B",
    revision="5cf2132abc99cad020ac570b19d031efec650f2b",
    lora=True,
    run_stems={
        "e0": "Q4B-LORA-E0",
        "infonce": "Q4B-LORA-InfoNCE",
        "lambdaloss": "Q4B-LORA-LambdaLoss",
        "reler": "Q4B-LORA-RELER-K0",
    },
    reler_contract={
        "learning_rate": 2e-4,
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
