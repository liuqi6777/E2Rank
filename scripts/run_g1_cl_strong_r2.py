#!/usr/bin/env python3
"""Run G1-R2 Strong CL directly, without the overnight queue's hash contracts."""
import argparse
from pathlib import Path
import sys

from experiments.iclr2027 import ROOT, run_simple_baselines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', default='train', choices=['check', 'train', 'eval'])
    parser.add_argument('--seeds', nargs='+', type=int, choices=[42, 3407, 2026], default=[42, 3407, 2026])
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/experiments_r2.yaml')
    args = parser.parse_args()
    runs = [f'G1-R2-CL-Strong' + (f'-Seed{seed}' if seed != 42 else '') for seed in dict.fromkeys(args.seeds)]
    run_simple_baselines(ROOT / 'configs/experiments/iclr2027/suite_r2.yaml', args.config, runs, args.action)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        raise SystemExit(2)
