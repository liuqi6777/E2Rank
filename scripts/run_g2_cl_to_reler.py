#!/usr/bin/env python3
"""Continue three existing Strong CL seeds with RELER; evaluate MTEB English v2."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

from experiments import iclr2027 as experiments
from run_g2_table12_seeds import METRICS, ROOT, SEEDS, SETTINGS, TYPES
from run_g2_table12_seeds import resolve_matrix as resolve_original_matrix

SUITE = ROOT / 'configs/experiments/iclr2027/suite_g2_cl_to_reler.yaml'
RUN = 'G2-R2-CLStrong-RL-Align070-ShortlistUniform-K0-T1-Pairwise050'
METHODS = ('cl', 'cl_reler')
SUMMARY_DIR = ROOT / 'paper/_summary/g2_r2_mteb_v2'


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


def preflight(matrix, seeds, action, methods):
    # Check the entire selected queue before starting any expensive work.
    for seed in seeds:
        for method in methods:
            row = matrix[method, seed][1]
            cfg = row['config']
            output = Path(cfg['output_dir'])
            if action == 'eval':
                check_model(output, cfg)
                continue
            if method != 'cl_reler':
                raise ValueError('This entry point only trains CL-to-RELER; CL is reused')
            if not experiments.path_at_root(cfg['data_path']).is_file():
                raise ValueError(f'Missing training data: {cfg["data_path"]}')
            check_model(row['dependency'], cfg)
            if output.exists() and (not output.is_dir() or any(output.iterdir())):
                raise ValueError(f'Training output already exists: {output}; use eval to retry evaluation')


def require_training_gpus():
    import torch
    if torch.cuda.device_count() != 8 or not torch.cuda.is_bf16_supported():
        raise ValueError('Expose exactly eight BF16 CUDA GPUs (CUDA_VISIBLE_DEVICES)')


def merge_csv_rows(path, incoming):
    """Replace complete incoming runs while retaining every unrelated existing row."""
    fields = list(incoming[0])
    previous = []
    if path.is_file():
        with path.open(newline='') as handle:
            reader = csv.DictReader(handle)
            if set(reader.fieldnames or []) != set(fields):
                raise ValueError(f'{path}: existing CSV columns differ from the MTEB view')
            fields = reader.fieldnames
            previous = list(reader)
    replaced = {row['run'] for row in incoming}
    return fields, [r for r in previous if r['run'] not in replaced] + incoming


def write_csv(path, fields, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(matrix, destination, catalog=None, results_root=None):
    sys.path.insert(0, str(ROOT))
    from eval_mteb import summary as mteb_summary

    destination = Path(destination)
    catalog = catalog if catalog is not None else mteb_summary.load_task_catalog(experiments.MTEB_ENG_V2)
    if len(catalog) != 41 or set(catalog.values()) != set(TYPES):
        raise ValueError('Expected the original 41 tasks in seven MTEB English v2 types')
    complete_results, scores, runs = [], {}, []
    for (method, seed), (_, row) in matrix.items():
        output = Path(row['config']['output_dir'])
        source = Path(results_root) / output.name if results_root else output / 'mteb_eval/final'
        directories = mteb_summary.discover_result_dirs(source, catalog)
        if len(directories) > 1:
            raise ValueError(f'{source}: ambiguous final MTEB results: {directories}')
        result = None
        if directories:
            result = mteb_summary.summarize_result_dir(
                directories[0], output.parent, experiments.MTEB_ENG_V2, catalog)
            if result.errors or any(not math.isfinite(value)
                                    for subsets in result.task_subset_scores.values()
                                    for value in subsets.values()):
                raise ValueError(f'{source}: invalid MTEB results: {result.errors}')
            result.run_id = output.name
        missing = sorted(set(catalog) - set(result.task_scores if result else {}))
        metrics = {}
        if result is not None and not missing:
            complete_results.append(result)
            metrics = {t: result.type_scores[t] * 100 for t in TYPES}
            metrics.update(mean_task_score=result.task_mean_pct, mean_type_score=result.type_mean_pct)
            scores[method, seed] = metrics
        runs.append(dict(method=method, seed=seed, run=output.name,
                         init_from=row['dependency'], complete=not missing,
                         missing_tasks=missing, result_dir=str(directories[0]) if directories else None,
                         metrics=metrics,
                         task_scores={t: v * 100 for t, v in result.task_scores.items()} if result else {}))

    # Build all four merges before writing; incomplete runs never replace a saved result.
    views = {}
    if complete_results:
        for view, rows in (
            ('run', mteb_summary.make_run_rows(complete_results, catalog)),
            ('type', mteb_summary.make_type_pivot_rows(complete_results)),
            ('task', mteb_summary.make_task_rows(complete_results)),
            ('retrieval', mteb_summary.make_retrieval_rows(complete_results)),
        ):
            path = destination / f'{view}_summary.csv'
            views[path] = merge_csv_rows(path, rows)

    groups, paired = [], []
    for metric in METRICS:
        for method in METHODS:
            values = [scores[method, s][metric] for s in SEEDS if (method, s) in scores]
            ready = len(values) == len(SEEDS)
            groups.append(dict(method=method, metric=metric, completed=len(values),
                               mean=statistics.mean(values) if ready else None,
                               sample_sd=statistics.stdev(values) if ready else None))
        deltas = []
        for seed in SEEDS:
            delta = None
            if ('cl', seed) in scores and ('cl_reler', seed) in scores:
                delta = scores['cl_reler', seed][metric] - scores['cl', seed][metric]
                deltas.append(delta)
            paired.append(dict(metric=metric, seed=seed, cl_reler_minus_cl=delta))
        ready = len(deltas) == len(SEEDS)
        paired.append(dict(metric=metric, seed='all', completed=len(deltas),
                           mean=statistics.mean(deltas) if ready else None,
                           sample_sd=statistics.stdev(deltas) if ready else None))

    destination.mkdir(parents=True, exist_ok=True)
    for path, (fields, rows) in views.items():
        write_csv(path, fields, rows)
    write_csv(destination / 'cl_to_reler_grouped.csv',
              ['method', 'metric', 'completed', 'mean', 'sample_sd'], groups)
    write_csv(destination / 'cl_to_reler_paired.csv',
              ['metric', 'seed', 'cl_reler_minus_cl', 'completed', 'mean', 'sample_sd'], paired)
    payload = dict(benchmark=experiments.MTEB_ENG_V2, score_units='main_score x 100',
                   seeds=SEEDS, runs=runs, groups=groups, paired=paired)
    (destination / 'cl_to_reler_results.json').write_text(
        json.dumps(payload, indent=2, allow_nan=False) + '\n')
    print(f'{len(complete_results)}/6 complete runs merged into {destination}; '
          'three-seed means/SD stay empty until all three seeds are complete.')


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
