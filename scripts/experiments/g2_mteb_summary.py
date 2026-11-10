"""Shared, lossless G2 MTEB exports for direct training and CL continuations."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import statistics
import sys

from experiments import iclr2027 as experiments

ROOT = experiments.ROOT
SUMMARY_DIR = ROOT / 'paper/_summary/g2_r2_mteb_v2'
SEEDS = (42, 3407, 2026)
TYPES = ('Classification', 'Clustering', 'PairClassification', 'Reranking',
         'Retrieval', 'STS', 'Summarization')
METRICS = (*TYPES, 'mean_task_score', 'mean_type_score')
RECIPES = ('infonce', 'reler', 'cl_reler', 'cl_cl')
COMPARISONS = (('reler', 'infonce'), ('cl_reler', 'infonce'), ('cl_reler', 'reler'),
               ('cl_cl', 'infonce'), ('cl_reler', 'cl_cl'))


def write_csv(path, fields, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def merge_csv_rows(path, incoming):
    """Replace incoming run views without removing unrelated historical runs."""
    fields, previous = list(incoming[0]), []
    if path.is_file():
        with path.open(newline='') as handle:
            reader = csv.DictReader(handle)
            if set(reader.fieldnames or []) != set(fields):
                raise ValueError(f'{path}: existing CSV columns differ from the MTEB view')
            fields, previous = reader.fieldnames, list(reader)
    replaced = {row['run'] for row in incoming}
    rows = [r for r in previous if r['run'] not in replaced] + incoming
    keys = [(r['run'], r.get('task')) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError(f'{path}: duplicate run/task rows')
    return fields, rows


def validate_records(runs, tasks, catalog):
    if len(catalog) != 41 or set(catalog.values()) != set(TYPES):
        raise ValueError('Expected 41 tasks in seven MTEB English v2 types')
    indexed = {(r['recipe'], r['seed']): r for r in runs}
    if len(indexed) != len(runs) or len({r['run'] for r in runs}) != len(runs):
        raise ValueError('Duplicate G2 run identity')
    task_map = {}
    for t in tasks:
        key = (t['recipe'], t['seed'])
        if key not in indexed or t['task'] not in catalog or t['type'] != catalog[t['task']]:
            raise ValueError(f'Unexpected G2 task: {t}')
        values = task_map.setdefault(key, {})
        if t['task'] in values or (t['score'] is not None and not math.isfinite(t['score'])):
            raise ValueError(f'Duplicate or nonfinite G2 task: {t}')
        values[t['task']] = t['score']
    for key, run in indexed.items():
        if key[0] not in RECIPES or key[1] not in SEEDS:
            raise ValueError(f'Unknown G2 recipe/seed: {key}')
        values = task_map.get(key, {})
        found = {t: v for t, v in values.items() if v is not None}
        if set(values) != set(catalog) or run['tasks_found'] != len(found):
            raise ValueError(f'G2 task coverage mismatch: {key}')
        if len(found) < len(catalog):
            if any(run.get(m) is not None for m in METRICS):
                raise ValueError(f'Incomplete G2 run has aggregate scores: {key}')
            continue
        expected = {kind: statistics.mean(v for t, v in found.items() if catalog[t] == kind)
                    for kind in TYPES}
        expected.update(mean_task_score=statistics.mean(found.values()),
                        mean_type_score=statistics.mean(expected.values()))
        for metric, value in expected.items():
            actual = run.get(metric)
            if actual is None or not math.isfinite(actual) or not math.isclose(
                    actual, value, rel_tol=0, abs_tol=1e-9):
                raise ValueError(f'G2 aggregate mismatch: {key}/{metric}')
    return indexed


def write_summary(destination, runs, tasks, catalog):
    """Merge precise records, then derive all CSV views and paired comparisons."""
    destination = Path(destination)
    indexed = validate_records(runs, tasks, catalog)
    saved_path = destination / 'seeds/results.json'
    if saved_path.is_file():
        saved = json.loads(saved_path.read_text())
        if saved['benchmark'] != experiments.MTEB_ENG_V2 or saved['task_types'] != catalog:
            raise ValueError(f'{saved_path}: benchmark/catalog changed')
        previous = validate_records(saved['runs'], saved['tasks'], catalog)
        # A local summary with unavailable raw results must not erase archived scores.
        retained = {key for key, row in previous.items() if key not in indexed or (
            row['tasks_found'] == 41 and indexed[key]['tasks_found'] < 41)}
        runs = [r for r in runs if (r['recipe'], r['seed']) not in retained]
        tasks = [t for t in tasks if (t['recipe'], t['seed']) not in retained]
        runs += [r for key, r in previous.items() if key in retained]
        tasks += [t for t in saved['tasks'] if (t['recipe'], t['seed']) in retained]
        if retained & indexed.keys():
            print(f'Reused {len(retained & indexed.keys())} archived complete runs; raw results unavailable/incomplete.')
    runs = sorted(runs, key=lambda r: (RECIPES.index(r['recipe']), SEEDS.index(r['seed'])))
    tasks = sorted(tasks, key=lambda t: (RECIPES.index(t['recipe']), SEEDS.index(t['seed']), t['task']))
    validate_records(runs, tasks, catalog)
    scores = {(r['recipe'], r['seed']): r for r in runs if r['tasks_found'] == 41}
    groups, paired = [], []
    for recipe in RECIPES:
        for metric in METRICS:
            values = [scores[recipe, s][metric] for s in SEEDS if (recipe, s) in scores]
            groups.append(dict(recipe=recipe, metric=metric, completed=len(values),
                               mean=statistics.mean(values) if len(values) == 3 else None,
                               sample_sd=statistics.stdev(values) if len(values) == 3 else None))
    for treatment, reference in COMPARISONS:
        comparison = f'{treatment}_minus_{reference}'
        for metric in METRICS:
            deltas = []
            for seed in SEEDS:
                delta = None
                if (treatment, seed) in scores and (reference, seed) in scores:
                    delta = scores[treatment, seed][metric] - scores[reference, seed][metric]
                    deltas.append(delta)
                paired.append(dict(comparison=comparison, metric=metric, seed=seed, delta=delta))
            paired.append(dict(comparison=comparison, metric=metric, seed='all', completed=len(deltas),
                               mean=statistics.mean(deltas) if len(deltas) == 3 else None,
                               sample_sd=statistics.stdev(deltas) if len(deltas) == 3 else None))

    # Keep the existing four rounded views and their schemas for paper consumers.
    sys.path.insert(0, str(ROOT))
    from eval_mteb import summary as mteb_summary
    complete = []
    for key, row in scores.items():
        values = {t['task']: t['score'] / 100 for t in tasks
                  if (t['recipe'], t['seed']) == key}
        complete.append(mteb_summary.RunSummary(row['run'], Path(row['source']),
                        experiments.MTEB_ENG_V2, catalog, values, {}, []))
    views = {}
    if complete:
        for name, rows in (
            ('run', mteb_summary.make_run_rows(complete, catalog)),
            ('type', mteb_summary.make_type_pivot_rows(complete)),
            ('task', mteb_summary.make_task_rows(complete)),
            ('retrieval', mteb_summary.make_retrieval_rows(complete)),
        ):
            path = destination / f'{name}_summary.csv'
            views[path] = merge_csv_rows(path, rows)
    exports = (
        ('per_run.csv', runs, ['recipe', 'seed', 'run', 'init_from', 'tasks_found', 'source', *METRICS]),
        ('per_task.csv', tasks, ['recipe', 'seed', 'task', 'type', 'score']),
        ('grouped.csv', groups, ['recipe', 'metric', 'completed', 'mean', 'sample_sd']),
        ('paired.csv', paired, ['comparison', 'metric', 'seed', 'delta', 'completed', 'mean', 'sample_sd']),
    )
    payload = dict(schema_version=1, benchmark=experiments.MTEB_ENG_V2,
                   score_units='MTEB main_score x 100', seeds=SEEDS, task_types=catalog,
                   runs=runs, tasks=tasks, groups=groups, paired=paired)
    serialized = json.dumps(payload, indent=2, allow_nan=False) + '\n'
    saved_path.parent.mkdir(parents=True, exist_ok=True)
    for path, (fields, rows) in views.items():
        write_csv(path, fields, rows)
    for filename, rows, fields in exports:
        write_csv(saved_path.parent / filename, fields, rows)
    saved_path.write_text(serialized)
    print(f'{len(scores)}/{len(RECIPES) * len(SEEDS)} complete seed runs in {destination}; '
          'means/SD require three complete seeds.')
    return payload


def summarize(matrix, destination, catalog=None, results_root=None):
    sys.path.insert(0, str(ROOT))
    from eval_mteb import summary as mteb_summary

    if catalog is None:
        catalog = mteb_summary.load_task_catalog(experiments.MTEB_ENG_V2)
    runs, tasks = [], []
    for (method, seed), (_, row) in matrix.items():
        recipe = 'infonce' if method == 'cl' else method
        out = Path(row['config']['output_dir'])
        source = Path(results_root) / out.name if results_root else out / 'mteb_eval/final'
        directories = mteb_summary.discover_result_dirs(source, catalog)
        if len(directories) > 1:
            raise ValueError(f'{source}: ambiguous final MTEB results: {directories}')
        result = None
        if directories:
            result = mteb_summary.summarize_result_dir(
                directories[0], out.parent, experiments.MTEB_ENG_V2, catalog)
            if result.errors or any(not math.isfinite(v)
                                    for subsets in result.task_subset_scores.values()
                                    for v in subsets.values()):
                raise ValueError(f'{source}: invalid MTEB results: {result.errors}')
        metrics = {}
        if result is not None and not mteb_summary.missing_tasks(result, catalog):
            metrics = {t: result.type_scores[t] * 100 for t in TYPES}
            metrics.update(mean_task_score=result.task_mean_pct, mean_type_score=result.type_mean_pct)
        runs.append(dict(recipe=recipe, seed=seed, run=out.name, init_from=row['dependency'],
                         tasks_found=result.task_count if result else 0,
                         source=str(directories[0]) if directories else '', **metrics))
        for task, kind in catalog.items():
            value = result.task_scores.get(task) if result else None
            tasks.append(dict(recipe=recipe, seed=seed, task=task, type=kind,
                              score=value * 100 if value is not None else None))
    return write_summary(destination, runs, tasks, catalog)
