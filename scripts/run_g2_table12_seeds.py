#!/usr/bin/env python3
"""Rebuttal Table 12: repeat direct InfoNCE/RELER at seeds 3407/2026; reuse 42."""
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

ROOT = experiments.ROOT
SEEDS = (42, 3407, 2026)
SETTINGS = ROOT / 'configs/experiments_g2_rl_r2.yaml'
RECIPES = {
    'infonce': ('suite_g2_cl_r2.yaml', 'G2-R2-D-CL-Strong'),
    'reler': ('suite_g2_rl_r2_k0.yaml',
              'G2-R2-D-RL-Align070-ShortlistUniform-K0-T1-Pairwise050'),
}
TYPES = ('Classification', 'Clustering', 'PairClassification', 'Reranking',
         'Retrieval', 'STS', 'Summarization')
METRICS = (*TYPES, 'mean_task_score', 'mean_type_score')


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


def summarize(matrix, destination, catalog=None):
    # Reuse the repository's main-score/subset averaging and benchmark catalog.
    sys.path.insert(0, str(ROOT))
    from eval_mteb import summary as mteb_summary

    if catalog is None:
        catalog = mteb_summary.load_task_catalog(experiments.MTEB_ENG_V2)
    if len(catalog) != 41 or set(catalog.values()) != set(TYPES):
        raise ValueError('Table 12 requires the original 41 tasks in seven MTEB English v2 types')
    runs, tasks, grouped, paired = [], [], [], []
    scores = {}
    for (recipe, seed), (_, row) in matrix.items():
        out = Path(row['config']['output_dir'])
        directories = mteb_summary.discover_result_dirs(out / 'mteb_eval/final', catalog)
        if len(directories) > 1:
            raise ValueError(f'{out}: ambiguous final MTEB results: {directories}')
        result = None
        if directories:
            result = mteb_summary.summarize_result_dir(
                directories[0], out.parent, experiments.MTEB_ENG_V2, catalog)
            if result.errors or any(not math.isfinite(v) for v in result.task_scores.values()):
                raise ValueError(f'{out}: invalid MTEB results: {result.errors}')
        complete = result is not None and not mteb_summary.missing_tasks(result, catalog)
        metrics = {}
        if complete:
            metrics = {t: result.type_scores[t] * 100 for t in TYPES}
            metrics.update(mean_task_score=result.task_mean_pct,
                           mean_type_score=result.type_mean_pct)
            scores[recipe, seed] = metrics
        runs.append(dict(recipe=recipe, seed=seed, run=row['run_id'],
                         tasks_found=result.task_count if result else 0,
                         source=str(directories[0]) if directories else '', **metrics))
        for task, task_type in catalog.items():
            value = result.task_scores.get(task) if result else None
            tasks.append(dict(recipe=recipe, seed=seed, task=task, type=task_type,
                              score=value * 100 if value is not None else None))
    for recipe in RECIPES:
        complete_seeds = [s for s in SEEDS if (recipe, s) in scores]
        for metric in METRICS:
            values = [scores[recipe, s][metric] for s in complete_seeds]
            complete = len(values) == len(SEEDS)
            grouped.append(dict(recipe=recipe, metric=metric, completed=len(values),
                                mean=statistics.mean(values) if complete else None,
                                sample_sd=statistics.stdev(values) if complete else None))
    for metric in METRICS:
        deltas = []
        for seed in SEEDS:
            delta = None
            if ('infonce', seed) in scores and ('reler', seed) in scores:
                delta = scores['reler', seed][metric] - scores['infonce', seed][metric]
                deltas.append(delta)
            paired.append(dict(metric=metric, seed=seed, reler_minus_infonce=delta))
        complete = len(deltas) == len(SEEDS)
        paired.append(dict(metric=metric, seed='all', completed=len(deltas),
                           mean=statistics.mean(deltas) if complete else None,
                           sample_sd=statistics.stdev(deltas) if complete else None))
    destination.mkdir(parents=True, exist_ok=True)
    for filename, rows, fields in (
        ('per_run.csv', runs, ['recipe', 'seed', 'run', 'tasks_found', 'source', *METRICS]),
        ('per_task.csv', tasks, ['recipe', 'seed', 'task', 'type', 'score']),
        ('grouped.csv', grouped, ['recipe', 'metric', 'completed', 'mean', 'sample_sd']),
        ('paired.csv', paired, ['metric', 'seed', 'reler_minus_infonce', 'completed', 'mean', 'sample_sd']),
    ):
        with (destination / filename).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    (destination / 'results.json').write_text(json.dumps(dict(
        score_units='MTEB main_score x 100', seeds=SEEDS, runs=runs,
        groups=grouped, paired=paired, tasks=tasks,
    ), indent=2, allow_nan=False) + '\n')
    print(f'Results: {destination}; means/SD require all three complete seeds.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', default='check',
                        choices=('show', 'check', 'train', 'eval', 'summary'))
    parser.add_argument('--config', type=Path, default=SETTINGS)
    parser.add_argument('--methods', nargs='+', choices=RECIPES, default=list(RECIPES))
    parser.add_argument('--seeds', type=int, nargs='+', choices=SEEDS, default=list(SEEDS[1:]))
    parser.add_argument('--summary-dir', type=Path, default=ROOT / 'outputs/g2_table12_seeds')
    args = parser.parse_args(argv)
    matrix = resolve_matrix(args.config)
    if args.action == 'summary':
        summarize(matrix, args.summary_dir)
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
