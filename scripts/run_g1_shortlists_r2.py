#!/usr/bin/env python3
"""G1-R2 multi-shortlist RL: Mixed/Uniform x T8/T16 x three seeds, CP/G64/alignment 0.80."""
import argparse
from pathlib import Path
import subprocess
import sys

from experiments.iclr2027 import ROOT, run_simple_baselines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', default='check', choices=['check', 'train', 'eval'])
    parser.add_argument('--sampling', choices=['mixed', 'uniform', 'both'], default='both')
    parser.add_argument('--groups', nargs='+', type=int, choices=[8, 16], default=[8, 16])
    parser.add_argument('--seeds', nargs='+', type=int, choices=[42, 3407, 2026], default=[42, 3407, 2026],
                        help='Training/data/rollout seeds (default: 42 3407 2026)')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/experiments_r2.yaml')
    args = parser.parse_args()
    sampling = ['mixed', 'uniform'] if args.sampling == 'both' else [args.sampling]
    runs = [f'G1-R2-RL-GradedNDCG64-CP-Align080-Shortlist{mode.title()}-T{count}'
            + (f'-Seed{seed}' if seed != 42 else '')
            for mode in sampling for count in dict.fromkeys(args.groups)
            for seed in dict.fromkeys(args.seeds)]
    if args.action == 'train':
        import torch
        if torch.cuda.device_count() != 8 or not torch.cuda.is_bf16_supported():
            raise ValueError('Expose exactly eight CUDA GPUs with BF16 support (CUDA_VISIBLE_DEVICES)')
    run_simple_baselines(ROOT / 'configs/experiments/iclr2027/suite_g1_shortlists.yaml',
                         args.config, runs, args.action)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        raise SystemExit(2)
