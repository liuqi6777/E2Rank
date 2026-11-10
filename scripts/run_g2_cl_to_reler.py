#!/usr/bin/env python3
"""Continue three existing Strong CL seeds with RELER; evaluate MTEB English v2."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from experiments import iclr2027 as experiments
from experiments.g2_mteb_summary import SUMMARY_DIR, summarize, write_csv
from run_g2_table12_seeds import METRICS, ROOT, SEEDS, SETTINGS, TYPES
from run_g2_table12_seeds import resolve_matrix as resolve_original_matrix

SUITE = ROOT / 'configs/experiments/iclr2027/suite_g2_cl_to_reler.yaml'
RUN = 'G2-R2-CLStrong-RL-Align070-ShortlistUniform-K0-T1-Pairwise050'
METHODS = ('cl', 'cl_reler')


def resolve_matrix(settings=SETTINGS):
    originals = resolve_original_matrix(settings)
    suite = experiments.apply_settings(experiments.load_suite(SUITE), settings)
    matrix = {}
    for seed in SEEDS:
        baseline_path, baseline = originals['infonce', seed]
        name = RUN if seed == 42 else f'{RUN}-Seed{seed}'
        row = experiments.resolve_run(suite, SUITE, name, nproc=8)
        reference = originals['reler', seed][1]
        ignored = {'model_name_or_path', 'run_name', 'output_dir'}
        cfg, direct = row['config'], reference['config']
        changes = {k for k in cfg.keys() | direct.keys()
                   if k not in ignored and cfg.get(k) != direct.get(k)}
        if (changes or row['protocol'] != reference['protocol']
                or row['dependency'] != baseline['config']['output_dir']
                or cfg['model_name_or_path'] != row['dependency']
                or row['final_evaluation'] != ['mteb_eng_v2']):
            raise ValueError(f'{name}: continuation changed the paired K0 recipe: {changes}')
        matrix['cl', seed] = (baseline_path, baseline)
        matrix['cl_reler', seed] = (SUITE, row)
    if len({r['config']['output_dir'] for _, r in matrix.values()}) != 6:
        raise ValueError('CL and CL-to-RELER must have six distinct model directories')
    return matrix


def check_model(path, config):
    path = Path(path)
    if not (path / 'config.json').is_file() or not (
        list(path.glob('*.safetensors')) or list(path.glob('pytorch_model*.bin'))
    ):
        raise ValueError(f'Missing final checkpoint: {path}')
    protocol_path = path / 'embedding_protocol.json'
    if not protocol_path.is_file():
        raise ValueError(f'Missing checkpoint embedding protocol: {protocol_path}')
    protocol = experiments.read_mapping(protocol_path)
    expected = {k: config[k] for k in ('pooling_method', 'padding_side', 'append_token',
                                     'query_prompt_template', 'document_prompt_template')}
    expected.update(tokenization_version=2, pooling_compute_dtype='float32',
                    max_length=config['embedding_max_length'], normalize=True)
    mismatched = {k for k, v in expected.items() if protocol.get(k) != v}
    if mismatched:
        raise ValueError(f'{path}: embedding protocol differs: {sorted(mismatched)}')


def preflight(matrix, seeds, action, methods, training_method='cl_reler'):
    # Check the entire selected queue before starting any expensive work.
    for seed in seeds:
        for method in methods:
            row = matrix[method, seed][1]
            cfg = row['config']
            output = Path(cfg['output_dir'])
            if action == 'eval':
                check_model(output, cfg)
                continue
            if method != training_method:
                raise ValueError(f'This entry point only trains {training_method}; earlier checkpoints are reused')
            if not experiments.path_at_root(cfg['data_path']).is_file():
                raise ValueError(f'Missing training data: {cfg["data_path"]}')
            check_model(row['dependency'], cfg)
            if output.exists() and (not output.is_dir() or any(output.iterdir())):
                raise ValueError(f'Training output already exists: {output}; use eval to retry evaluation')


def require_training_gpus():
    import torch
    if torch.cuda.device_count() != 8 or not torch.cuda.is_bf16_supported():
        raise ValueError('Expose exactly eight BF16 CUDA GPUs (CUDA_VISIBLE_DEVICES)')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', default='check',
                        choices=('show', 'check', 'train', 'eval', 'summary'))
    parser.add_argument('--config', type=Path, default=SETTINGS)
    parser.add_argument('--seeds', type=int, nargs='+', choices=SEEDS, default=list(SEEDS))
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=['cl_reler'],
                        help='For eval: cl_reler (default), cl, or both; training always continues RELER only')
    parser.add_argument('--summary-dir', type=Path, default=SUMMARY_DIR)
    parser.add_argument('--results-root', type=Path,
                        help='Summary only: staged final results under <root>/<run-directory>/')
    args = parser.parse_args(argv)
    if args.action in ('check', 'train') and args.methods != ['cl_reler']:
        parser.error('check/train only support --methods cl_reler; CL checkpoints are reused')
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
                  f'init={cfg["model_name_or_path"]}\n'
                  f'  data={cfg["data_path"]} -> {cfg["output_dir"]}', flush=True)
    if args.action == 'show':
        return 0
    preflight(matrix, seeds, args.action, methods)
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
