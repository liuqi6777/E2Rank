#!/usr/bin/env python3
"""G1-R2 shared document policies: local/large pools, CP/G64, alignment 0.90, three seeds."""
import argparse
from pathlib import Path
import subprocess
import sys

from experiments.iclr2027 import ROOT, run_simple_baselines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', default='train', choices=['check', 'train', 'eval'])
    parser.add_argument('--pool', choices=['local', 'large', 'both'], default='both',
                        help='Candidate pool; both runs local then large (default)')
    parser.add_argument('--seeds', nargs='+', type=int, choices=[42, 3407, 2026], default=[42, 3407, 2026],
                        help='Training/data/rollout seeds (default: 42 3407 2026)')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/experiments_r2.yaml')
    args = parser.parse_args()
    base = 'G1-R2-RL-GradedNDCG64-CP-CrossQuery'
    bases = []
    if args.pool in ('local', 'both'):
        bases.append(base)
    if args.pool in ('large', 'both'):
        bases.append(base + '-LargePool')
    runs = [name + (f'-Seed{seed}' if seed != 42 else '')
            for name in bases for seed in dict.fromkeys(args.seeds)]
    if args.action == 'train':
        import torch
        if torch.cuda.device_count() != 8 or not torch.cuda.is_bf16_supported():
            raise ValueError('Expose exactly eight CUDA GPUs with BF16 support (CUDA_VISIBLE_DEVICES)')
    run_simple_baselines(ROOT / 'configs/experiments/iclr2027/suite_g1_cross_query_policy.yaml',
                         args.config, runs, args.action)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        raise SystemExit(2)
