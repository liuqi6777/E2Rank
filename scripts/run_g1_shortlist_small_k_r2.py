#!/usr/bin/env python3
"""G1-R2 small-K controls: K0/K1/K3, T1, alignment 0.70, pure graded reward."""
import argparse
from pathlib import Path
import subprocess
import sys

from experiments.iclr2027 import ROOT, run_simple_baselines


RECIPES = {f'uniform-k{k}-t1': ('070', f'Uniform-K{k}-T1') for k in (0, 1, 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', default='check', choices=['check', 'train', 'eval'])
    parser.add_argument('--recipes', nargs='+', choices=list(RECIPES), default=list(RECIPES),
                        help='Recipes to run (default: all three new small-K controls)')
    parser.add_argument('--seeds', nargs='+', type=int, choices=[42, 3407, 2026], default=[42, 3407, 2026],
                        help='Training/data/rollout seeds (default: 42 3407 2026)')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/experiments_r2.yaml')
    args = parser.parse_args()
    runs = [f'G1-R2-RL-GradedNDCG64-CP-Align{RECIPES[recipe][0]}-Shortlist{RECIPES[recipe][1]}'
            + (f'-Seed{seed}' if seed != 42 else '')
            for recipe in dict.fromkeys(args.recipes) for seed in dict.fromkeys(args.seeds)]
    if args.action == 'train':
        import torch
        if torch.cuda.device_count() != 8 or not torch.cuda.is_bf16_supported():
            raise ValueError('Expose exactly eight CUDA GPUs with BF16 support (CUDA_VISIBLE_DEVICES)')
    run_simple_baselines(ROOT / 'configs/experiments/iclr2027/suite_g1_shortlist_small_k.yaml',
                         args.config, runs, args.action)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        raise SystemExit(2)
