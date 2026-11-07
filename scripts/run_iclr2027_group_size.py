#!/usr/bin/env python3
"""Rebuttal E3: train/evaluate G=16/32/128; reuse the final-suite G=64 control."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

from experiments import iclr2027 as experiments
from run_iclr2027_final_k0 import RECIPES, SEEDS, SUITE as CONTROL_SUITE
from run_iclr2027_final_k0 import SETTINGS as CONTROL_SETTINGS, run_name, validate_complete
from run_g1_r2 import validate_final_model
from run_g1_stability import read_bright, SUBSETS

ROOT = experiments.ROOT
SUITE = ROOT / 'configs/experiments/iclr2027/suite_iclr2027_group_size.yaml'
SETTINGS = ROOT / 'configs/experiments_iclr2027_group_size.yaml'
GROUPS = (16, 32, 128)


def resolve_matrix(settings, control_settings):
    suite = experiments.apply_settings(experiments.load_suite(SUITE), settings)
    control = experiments.apply_settings(experiments.load_suite(CONTROL_SUITE), control_settings)
    matrix = {}
    for seed in SEEDS:
        name = run_name(RECIPES['main'], seed)
        reference = experiments.resolve_run(control, CONTROL_SUITE, name, nproc=8)
        matrix[64, seed] = reference
        for group in GROUPS:
            variant = name.replace('GradedNDCG64', f'GradedNDCG{group}')
            row = experiments.resolve_run(suite, SUITE, variant, nproc=8)
            a, b = row['config'], reference['config']
            ignored = {'group_size', 'output_dir', 'run_name'}
            differences = {key: (a.get(key), b.get(key)) for key in set(a) | set(b)
                           if key not in ignored and a.get(key) != b.get(key)}
            if differences or a['group_size'] != group or b['group_size'] != 64:
                raise ValueError(f'{variant}: not a group-size-only ablation: {differences}')
            matrix[group, seed] = row
    if len(suite['runs']) != 9:
        raise ValueError('Expected exactly nine new runs')
    return matrix


def summarize(matrix, destination):
    runs, grouped = [], []
    for group in (16, 32, 64, 128):
        values = []
        for seed in SEEDS:
            row = matrix[group, seed]
            scores, source = read_bright(Path(row['config']['output_dir']))
            avg = statistics.mean(scores.values()) if scores else None
            if avg is not None:
                values.append(avg)
            runs.append(dict(group_size=group, seed=seed, run=row['run_id'],
                             avg=avg, source=source, **(scores or {})))
        complete = len(values) == len(SEEDS)
        grouped.append(dict(group_size=group, completed=len(values),
                            mean=statistics.mean(values) if complete else None,
                            sample_sd=statistics.stdev(values) if complete else None))
    destination.mkdir(parents=True, exist_ok=True)
    for name, rows, fields in (
        ('per_run.csv', runs, ['group_size', 'seed', 'run', 'avg', 'source', *SUBSETS]),
        ('grouped.csv', grouped, ['group_size', 'completed', 'mean', 'sample_sd']),
    ):
        with (destination / name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    (destination / 'results.json').write_text(json.dumps(dict(
        score_units='nDCG@10 x 100', seeds=SEEDS, groups=grouped, runs=runs,
    ), indent=2, allow_nan=False) + '\n')
    print(f'Results: {destination}; aggregate statistics require all three seeds.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'train', 'eval', 'summary'))
    parser.add_argument('--config', type=Path, default=SETTINGS)
    parser.add_argument('--control-config', type=Path, default=CONTROL_SETTINGS)
    parser.add_argument('--groups', type=int, nargs='+', choices=GROUPS, default=list(GROUPS))
    parser.add_argument('--seeds', type=int, nargs='+', choices=SEEDS, default=list(SEEDS))
    parser.add_argument('--summary-dir', type=Path, default=ROOT / 'outputs/iclr2027_group_size')
    args = parser.parse_args(argv)
    if len(set(args.groups)) != len(args.groups) or len(set(args.seeds)) != len(args.seeds):
        parser.error('Groups and seeds must be distinct')
    matrix = resolve_matrix(args.config, args.control_config)
    if args.action == 'summary':
        summarize(matrix, args.summary_dir)
        return 0
    rows = [matrix[g, s] for g in args.groups for s in args.seeds]
    if args.action == 'check':
        for row in rows:
            cfg = row['config']
            print(f'G={cfg["group_size"]} seed={cfg["seed"]} -> {cfg["output_dir"]}')
        experiments.run_simple_baselines(SUITE, args.config, [r['run_id'] for r in rows], 'check')
        print('Resolved configurations match the paired G=64 controls except group_size and run/output names.')
        return 0
    if args.action == 'train':
        import torch
        if torch.cuda.device_count() != 8 or not torch.cuda.is_bf16_supported():
            raise ValueError('Training requires exactly eight visible BF16 CUDA GPUs')
    for row in rows:
        out = Path(row['config']['output_dir'])
        if args.action == 'train' and out.exists() and any(out.iterdir()):
            validate_complete(row)
            print(f'{row["run_id"]}: already complete; skipping', flush=True)
            continue
        if args.action == 'eval':
            validate_final_model(row)
        experiments.run_simple_baselines(SUITE, args.config, [row['run_id']], args.action)
        validate_complete(row)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        raise SystemExit(2)
