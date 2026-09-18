#!/usr/bin/env python3
"""G1-R2 negative-distribution controls: two recipes x three seeds, Uniform K15/T1."""
import argparse
from pathlib import Path
import subprocess
import sys

from experiments.iclr2027 import ROOT, run_simple_baselines


RECIPES = {
    'local-all': 'Uniform-LocalAll-K15-T1',
    'cross-device-representatives': 'Uniform-CrossDeviceRepresentatives-K15-T1',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', default='check', choices=['check', 'train', 'eval'])
    parser.add_argument('--recipes', nargs='+', choices=list(RECIPES), default=list(RECIPES),
                        help='Recipes to run (default: both, in the listed order)')
    parser.add_argument('--seeds', nargs='+', type=int, choices=[42, 3407, 2026], default=[42, 3407, 2026],
                        help='Training/data/rollout seeds (default: 42 3407 2026)')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/experiments_r2.yaml')
    args = parser.parse_args()
    runs = [f'G1-R2-RL-GradedNDCG64-CP-Align080-Shortlist{RECIPES[recipe]}'
            + (f'-Seed{seed}' if seed != 42 else '')
            for recipe in dict.fromkeys(args.recipes) for seed in dict.fromkeys(args.seeds)]
    if args.action == 'train':
        import torch
        if torch.cuda.device_count() != 8 or not torch.cuda.is_bf16_supported():
            raise ValueError('Expose exactly eight CUDA GPUs with BF16 support (CUDA_VISIBLE_DEVICES)')
    run_simple_baselines(ROOT / 'configs/experiments/iclr2027/suite_g1_shortlist_distribution.yaml',
                         args.config, runs, args.action)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        raise SystemExit(2)
