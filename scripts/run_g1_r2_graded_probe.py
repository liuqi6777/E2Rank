#!/usr/bin/env python3
"""Run the supplementary G1-R2 graded nDCG SF/CP probe at E0 on one GPU.

Uses the registered training recipe and the original MRR probe's batches/draws.
No training, evaluation, or existing overnight-queue contracts are modified.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

from run_g1_r2 import ROOT, SETTINGS, SUITE, resolve_matrix, run_id, verify_data

ROLLOUT_SEEDS = (42, 3407, 2026, *range(13))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=SETTINGS)
    parser.add_argument('--output', type=Path,
                        help='New JSON path; existing results are never overwritten')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--check', action='store_true',
                        help='Validate config/data and print command without loading a model')
    args = parser.parse_args(argv)
    settings = args.config.resolve()
    _, matrix = resolve_matrix(settings, seeds=(42,))
    row = next(r for r in matrix if r['run_id'] == run_id('RL-GradedNDCG64-SF'))
    config = row['config']
    if config['relevance_scheme'] != 'graded' or config['reward_type'] != 'ndcg_in_batch':
        raise ValueError('The registered probe must use graded nDCG')
    data_hash = verify_data(matrix)
    output = args.output or (
        Path(config['output_dir']).parent / '.r2_batch' /
        'gradient_probe_graded' / 'attempt-1.json'
    )
    output = output.resolve()
    if output.exists():
        raise ValueError(f'Output already exists: {output}; select a new --output path')
    command = [
        sys.executable, '-u', str(ROOT / 'scripts/diagnose_rollout_gradients.py'),
        '--suite', str(SUITE), '--config', str(settings), '--run', row['run_id'],
        '--compare-gradient-estimators', '--batch-indices', '0', '100', '200',
        '--rollout-seeds', *map(str, ROLLOUT_SEEDS),
        '--precision', 'bf16', '--device', args.device, '--output', str(output),
    ]
    print(f'Settings: {settings}\nTraining data SHA256: {data_hash}\n'
          f'Model: E0 ({config["model_name_or_path"]})\n'
          f'Reward: graded nDCG@{config["reward_ndcg_k"]}; G={config["group_size"]}\n'
          f'Batches: 0/100/200; draws per batch: {len(ROLLOUT_SEEDS)}\n'
          f'Output: {output}\nCommand: {shlex.join(command)}', flush=True)
    if args.check:
        print('CPU preflight passed; no model loaded or GPU job started.')
        return 0
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        raise SystemExit(2)
