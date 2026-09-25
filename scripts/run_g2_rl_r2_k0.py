#!/usr/bin/env python3
"""Train or evaluate direct Qwen3-0.6B RL with the current G1 K=0 recipe."""
import argparse
from pathlib import Path
import subprocess
import sys

from experiments.iclr2027 import ROOT, run_simple_baselines


RUN = 'G2-R2-D-RL-Align070-ShortlistUniform-K0-T1-Pairwise050'
SUITE = ROOT / 'configs/experiments/iclr2027/suite_g2_rl_r2_k0.yaml'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', default='check', choices=['check', 'train', 'eval'])
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/experiments_g2_rl_r2.yaml')
    args = parser.parse_args()
    if args.action == 'train':
        import torch
        if torch.cuda.device_count() != 8 or not torch.cuda.is_bf16_supported():
            raise ValueError('Expose exactly eight CUDA GPUs with BF16 support (CUDA_VISIBLE_DEVICES)')
    run_simple_baselines(SUITE, args.config, [RUN], args.action)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        raise SystemExit(2)
