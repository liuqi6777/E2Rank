#!/usr/bin/env python3
"""Train/evaluate ReasonEmbed Strong CL D/E/W for one epoch per stage."""
import argparse
from pathlib import Path
import sys

from experiments.iclr2027 import ROOT, run_simple_baselines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', default='check', choices=['check', 'train', 'eval'])
    parser.add_argument('--branches', nargs='+', choices=['D', 'E', 'W'], default=['D', 'E', 'W'])
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/experiments_g2_reasonembed_cl.yaml')
    args = parser.parse_args()
    runs = [f'G2-ReasonEmbed-{branch}-CL-Strong' for branch in dict.fromkeys(args.branches)]
    run_simple_baselines(
        ROOT / 'configs/experiments/iclr2027/suite_g2_reasonembed_cl.yaml',
        args.config, runs, args.action,
    )


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        raise SystemExit(2)
