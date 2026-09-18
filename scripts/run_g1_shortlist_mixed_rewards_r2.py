#!/usr/bin/env python3
"""G1-R2 graded/binary shortlist rewards: alpha 0.25/0.50, three seeds, 113 steps."""
import argparse
from pathlib import Path
import subprocess
import sys

from experiments.iclr2027 import ROOT, run_simple_baselines


BINARY_WEIGHTS = {0.25: '025', 0.50: '050'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', default='check', choices=['check', 'train', 'eval'])
    parser.add_argument('--binary-weights', nargs='+', type=float, choices=list(BINARY_WEIGHTS),
                        default=list(BINARY_WEIGHTS),
                        help='Binary reward weights (default: 0.25 0.50)')
    parser.add_argument('--seeds', nargs='+', type=int, choices=[42, 3407, 2026], default=[42, 3407, 2026],
                        help='Training/data/rollout seeds (default: 42 3407 2026)')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/experiments_r2.yaml')
    args = parser.parse_args()
    runs = [f'G1-R2-RL-GradedNDCG64-CP-Align070-ShortlistUniform-K15-T1-Binary{BINARY_WEIGHTS[weight]}'
            + (f'-Seed{seed}' if seed != 42 else '')
            for weight in dict.fromkeys(args.binary_weights) for seed in dict.fromkeys(args.seeds)]
    if args.action == 'train':
        import torch
        if torch.cuda.device_count() != 8 or not torch.cuda.is_bf16_supported():
            raise ValueError('Expose exactly eight CUDA GPUs with BF16 support (CUDA_VISIBLE_DEVICES)')
    run_simple_baselines(ROOT / 'configs/experiments/iclr2027/suite_g1_shortlist_mixed_rewards.yaml',
                         args.config, runs, args.action)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        raise SystemExit(2)
