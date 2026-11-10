#!/usr/bin/env python3
"""Continue three Strong CL checkpoints with Strong CL; evaluate MTEB English v2."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from experiments import iclr2027 as experiments
from experiments.g2_mteb_summary import SUMMARY_DIR, summarize
from run_g2_cl_to_reler import preflight, require_training_gpus
from run_g2_cl_to_reler import resolve_matrix as resolve_reler_matrix
from run_g2_table12_seeds import ROOT, SEEDS, SETTINGS
from run_g2_table12_seeds import resolve_matrix as resolve_original_matrix

SUITE = ROOT / 'configs/experiments/iclr2027/suite_g2_cl_to_cl.yaml'
RUN = 'G2-R2-CLStrong-CL'
METHODS = ('cl', 'reler', 'cl_reler', 'cl_cl')


def resolve_matrix(settings=SETTINGS):
    originals = resolve_original_matrix(settings)
    matrix = resolve_reler_matrix(settings)
    suite = experiments.apply_settings(experiments.load_suite(SUITE), settings)
    for seed in SEEDS:
        name = RUN if seed == 42 else f'{RUN}-Seed{seed}'
        row = experiments.resolve_run(suite, SUITE, name, nproc=8)
        baseline = originals['infonce', seed][1]
        reler = matrix['cl_reler', seed][1]
        cfg, direct = row['config'], baseline['config']
        identity = {'model_name_or_path', 'run_name', 'output_dir'}
        changes = {k for k in cfg.keys() | direct.keys()
                   if k not in identity and cfg.get(k) != direct.get(k)}
        if (changes or row['protocol'] != baseline['protocol']
                or row['entrypoint'] != baseline['entrypoint']
                or row['dependency'] != baseline['config']['output_dir']
                or cfg['model_name_or_path'] != row['dependency']
                or row['final_evaluation'] != ['mteb_eng_v2']):
            raise ValueError(f'{name}: continuation changed the original Strong CL recipe: {changes}')

        # Same initialization, data, optimizer settings, budget and evaluation as
        # CL-to-RELER. The loss-specific fields and binary/graded labels differ.
        allowed = {'run_name', 'output_dir', 'index_cache_dir', 'relevance_scheme'}
        paired = reler['config']
        changes = {k for k in cfg.keys() & paired.keys()
                   if k not in allowed and cfg[k] != paired[k]}
        protocol = {k: v for k, v in row['protocol'].items() if k != 'labels'}
        paired_protocol = {k: v for k, v in reler['protocol'].items() if k != 'labels'}
        if changes or protocol != paired_protocol or row['dependency'] != reler['dependency']:
            raise ValueError(f'{name}: CL-to-RELER continuation budget/protocol differs: {changes}')
        matrix['reler', seed] = originals['reler', seed]
        matrix['cl_cl', seed] = (SUITE, row)
    if len({r['config']['output_dir'] for _, r in matrix.values()}) != len(METHODS) * len(SEEDS):
        raise ValueError('All G2 recipes/seeds must have distinct model directories')
    return matrix


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', default='check',
                        choices=('show', 'check', 'train', 'eval', 'summary'))
    parser.add_argument('--config', type=Path, default=SETTINGS)
    parser.add_argument('--seeds', type=int, nargs='+', choices=SEEDS, default=list(SEEDS))
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=['cl_cl'],
                        help='For eval: select existing controls or cl_cl (default); train only cl_cl')
    parser.add_argument('--summary-dir', type=Path, default=SUMMARY_DIR)
    parser.add_argument('--results-root', type=Path,
                        help='Summary only: staged final results under <root>/<run-directory>/')
    args = parser.parse_args(argv)
    if args.action in ('check', 'train') and args.methods != ['cl_cl']:
        parser.error('check/train only support --methods cl_cl; earlier checkpoints are reused')
    matrix = resolve_matrix(args.config)
    if args.action == 'summary':
        summarize(matrix, args.summary_dir, results_root=args.results_root)
        return 0
    seeds, methods = list(dict.fromkeys(args.seeds)), list(dict.fromkeys(args.methods))
    for seed in seeds:
        for method in methods:
            row = matrix[method, seed][1]
            cfg = row['config']
            print(f'{row["run_id"]}: seed={seed} steps={cfg["max_steps"]} '
                  f'lr={cfg["learning_rate"]} init={cfg["model_name_or_path"]}\n'
                  f'  data={cfg["data_path"]} -> {cfg["output_dir"]}', flush=True)
    if args.action == 'show':
        return 0
    preflight(matrix, seeds, args.action, methods, training_method='cl_cl')
    if args.action == 'train':
        require_training_gpus()
    if args.action in ('train', 'eval'):
        for seed in seeds:
            for method in methods:
                path, row = matrix[method, seed]
                experiments.run_simple_baselines(path, args.config, [row['run_id']], args.action)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        raise SystemExit(2)
