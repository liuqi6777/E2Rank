#!/usr/bin/env python3
"""Run G2-R2 Strong CL D/E/W directly, without hash contracts or launch receipts."""
import argparse
from pathlib import Path
import sys

from experiments.iclr2027 import ROOT, run_simple_baselines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', default='train', choices=['check', 'train', 'eval'])
    parser.add_argument('--branches', nargs='+', choices=['D', 'E', 'W'], default=['D', 'E', 'W'])
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/experiments_g2_cl_r2.yaml')
    args = parser.parse_args()
    runs = [f'G2-R2-{branch}-CL-Strong' for branch in dict.fromkeys(args.branches)]
    run_simple_baselines(ROOT / 'configs/experiments/iclr2027/suite_g2_cl_r2.yaml', args.config, runs, args.action)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        raise SystemExit(2)
