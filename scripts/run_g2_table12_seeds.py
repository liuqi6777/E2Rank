#!/usr/bin/env python3
"""Rebuttal Table 12: repeat direct InfoNCE/RELER at seeds 3407/2026; reuse 42."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from experiments import iclr2027 as experiments
from experiments.g2_mteb_summary import METRICS, SEEDS, SUMMARY_DIR, TYPES, summarize

ROOT = experiments.ROOT
SETTINGS = ROOT / 'configs/experiments_g2_rl_r2.yaml'
RECIPES = {
    'infonce': ('suite_g2_cl_r2.yaml', 'G2-R2-D-CL-Strong'),
    'reler': ('suite_g2_rl_r2_k0.yaml',
              'G2-R2-D-RL-Align070-ShortlistUniform-K0-T1-Pairwise050'),
}


def resolve_matrix(settings=SETTINGS):
    matrix = {}
    for recipe, (filename, base) in RECIPES.items():
        path = ROOT / 'configs/experiments/iclr2027' / filename
        suite = experiments.apply_settings(experiments.load_suite(path), settings)
        original = experiments.resolve_run(suite, path, base, nproc=8)
        for seed in SEEDS:
            name = base if seed == 42 else f'{base}-Seed{seed}'
            row = experiments.resolve_run(suite, path, name, nproc=8)
            cfg, reference = row['config'], original['config']
            ignored = {'seed', 'data_seed', 'rollout_seed', 'run_name', 'output_dir'}
            changes = {k for k in cfg.keys() | reference.keys()
                       if k not in ignored and cfg.get(k) != reference.get(k)}
            if (changes or cfg['seed'] != seed or cfg['data_seed'] != seed
                    or (recipe == 'reler' and cfg['rollout_seed'] != seed)
                    or row['protocol'] != original['protocol']
                    or row['final_evaluation'] != ['mteb_eng_v2']):
                raise ValueError(f'{name}: seed repeat changed the original recipe: {changes}')
            matrix[recipe, seed] = (path, row)
    if len({r['config']['output_dir'] for _, r in matrix.values()}) != 6:
        raise ValueError('Table 12 runs must have six distinct output directories')
    return matrix


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', default='check',
                        choices=('show', 'check', 'train', 'eval', 'summary'))
    parser.add_argument('--config', type=Path, default=SETTINGS)
    parser.add_argument('--methods', nargs='+', choices=RECIPES, default=list(RECIPES))
    parser.add_argument('--seeds', type=int, nargs='+', choices=SEEDS, default=list(SEEDS[1:]))
    parser.add_argument('--summary-dir', type=Path, default=SUMMARY_DIR)
    parser.add_argument('--results-root', type=Path,
                        help='Summary only: staged final results under <root>/<run-directory>/')
    args = parser.parse_args(argv)
    matrix = resolve_matrix(args.config)
    if args.action == 'summary':
        summarize(matrix, args.summary_dir, results_root=args.results_root)
        return 0
    rows = [matrix[m, s] for s in dict.fromkeys(args.seeds) for m in dict.fromkeys(args.methods)]
    for _, row in rows:
        cfg = row['config']
        print(f'{row["run_id"]}: seed={cfg["seed"]} data_seed={cfg["data_seed"]} '
              f'rollout_seed={cfg.get("rollout_seed")} steps={cfg["max_steps"]} '
              f'-> {cfg["output_dir"]}', flush=True)
    if args.action == 'show':
        return 0
    if args.action == 'train':
        import torch
        if torch.cuda.device_count() != 8 or not torch.cuda.is_bf16_supported():
            raise ValueError('Expose exactly eight BF16 CUDA GPUs (CUDA_VISIBLE_DEVICES)')
    # Preflight the entire selected queue before launching the first training job.
    if args.action in ('check', 'train'):
        for path, row in rows:
            experiments.run_simple_baselines(path, args.config, [row['run_id']], 'check')
    if args.action in ('train', 'eval'):
        for path, row in rows:
            experiments.run_simple_baselines(path, args.config, [row['run_id']], args.action)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        raise SystemExit(2)
