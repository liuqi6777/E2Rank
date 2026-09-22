#!/usr/bin/env python3
"""Calibrate LR, shortlist K and target alignment for the full 4B RELER recipe."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from experiments import iclr2027

ROOT = iclr2027.ROOT
SUITE = ROOT / 'configs/experiments/iclr2027/suite_qwen3_embedding_4b_lora_calibration.yaml'
SETTINGS = ROOT / 'configs/experiments_qwen3_embedding_4b_lora_calibration.yaml'
SEEDS = (42, 3407, 2026)
RECIPES = {
    'lr050': 'Q4B-LORA-RELER-Cal-LR050',
    'lr200': 'Q4B-LORA-RELER-Cal-LR200',
    'k3': 'Q4B-LORA-RELER-Cal-K3',
    'k15': 'Q4B-LORA-RELER-Cal-K15',
    'align065': 'Q4B-LORA-RELER-Cal-Align065',
    'align080': 'Q4B-LORA-RELER-Cal-Align080',
}


def selected_runs(recipes: list[str], seeds: list[int]) -> list[str]:
    return [RECIPES[recipe] + (f'-Seed{seed}' if seed != 42 else '')
            for recipe in dict.fromkeys(recipes) for seed in dict.fromkeys(seeds)]


def validate_contracts(run_ids: list[str], settings: Path) -> None:
    suite = iclr2027.apply_settings(iclr2027.load_suite(SUITE), settings)
    for name in run_ids:
        row = iclr2027.resolve_run(suite, SUITE, name, nproc=8)
        config = row['config']
        expected = {
            'model_name_or_path': 'Qwen/Qwen3-Embedding-4B',
            'model_revision': '5cf2132abc99cad020ac570b19d031efec650f2b',
            'lora_enabled': True,
            'lora_r': 16,
            'lora_alpha': 32,
            'max_steps': 113,
            'per_device_train_batch_size': 8,
            'gradient_accumulation_steps': 2,
            'gradient_estimator': 'conditional_projection',
            'group_size': 64,
            'reward_shortlist_pairwise_coef': 0.5,
            'frozen_doc_rescale': True,
        }
        mismatch = {key: (config.get(key), value) for key, value in expected.items()
                    if config.get(key) != value}
        if mismatch:
            raise ValueError(f'{name}: incompatible 4B calibration contract: {mismatch}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', default='check', choices=['check', 'train', 'eval'])
    parser.add_argument('--recipes', nargs='+', choices=list(RECIPES), default=list(RECIPES),
                        help='Calibration recipes (default: all six)')
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=[42],
                        help='Default is the seed-42 pilot; confirm a winner with 3407 2026')
    parser.add_argument('--gpus', type=int, choices=[8], default=8)
    parser.add_argument('--config', type=Path, default=SETTINGS)
    args = parser.parse_args()

    runs = selected_runs(args.recipes, args.seeds)
    validate_contracts(runs, args.config)
    if args.action == 'train':
        import torch
        if torch.cuda.device_count() != 8 or not torch.cuda.is_bf16_supported():
            raise ValueError('Expose exactly eight CUDA GPUs with BF16 support (CUDA_VISIBLE_DEVICES)')
    iclr2027.run_simple_baselines(SUITE, args.config, runs, args.action, nproc=args.gpus)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        raise SystemExit(2)
