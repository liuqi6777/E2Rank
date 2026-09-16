#!/usr/bin/env python3
"""Post-train reasoning E0s: CL, graded LambdaLoss and graded nDCG64-CP.

Independent output/receipts; reuse G1-R2 data, optimizer and final BRIGHT protocol.
check never loads models. Default: two E0 evaluations and 18 trainings.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
from pathlib import Path
import statistics
import sys

import run_g1_r2 as night
import run_g1_r2_ablations as receipts

ROOT = night.ROOT
SETTINGS = ROOT / 'configs/experiments_reasoning_e0.yaml'
MODELS = {'diver': ROOT / 'configs/model/diver_0.6b.yaml',
          'reasonembed': ROOT / 'configs/model/reason_embed_4b.yaml'}
METHODS = {'CL': 'CL', 'LL-Graded': 'LL-Graded',
           'RL-Graded': 'RL-GradedNDCG64-CP'}


def resolve(settings=SETTINGS, models=tuple(MODELS), seeds=night.SEEDS):
    models = tuple(models)
    if not models or len(set(models)) != len(models) or set(models) - MODELS.keys():
        raise ValueError('Select distinct registered reasoning E0s')
    seeds = night.selected_seeds(seeds)
    suite, base = night.resolve_matrix(settings, seeds=night.SEEDS)
    output = Path(base[0]['config']['output_dir']).parent
    rows = []
    for model in models:
        preset = night.experiments.resolve_config(MODELS[model])
        selections = [('E0', 42)] if 42 in seeds else []
        selections += [(method, seed) for seed in seeds for method in METHODS]
        for method, seed in selections:
            original = 'G1-R2-E0' if method == 'E0' else night.run_id(METHODS[method], seed)
            row = copy.deepcopy(next(r for r in base if r['run_id'] == original))
            name = f'G1-Reasoning-{model}-{method}' + (f'-Seed{seed}' if seed != 42 else '')
            row.update(run_id=name, model_key=model, variant=f'{model}/{method}',
                       execution_stage=None, execution_stage_name='Reasoning E0 core comparison',
                       control=None, priority='core')
            row['config'].update(preset)
            row['protocol'].update(preset)
            row['config'].update(run_name=name, output_dir=str(output / f'{name}-s{seed}'),
                                 index_cache_dir=f'.cache/dataset_index/reasoning-e0/{model}')
            rows.append(row)
    return suite, rows, output / '.reasoning_e0'


def sources():
    result = night.source_fingerprint()
    for path in [Path(__file__), Path(receipts.__file__), *MODELS.values()]:
        result[str(path.relative_to(ROOT))] = night.experiments.digest(path)
    return result


def contract(row, data_hash, source_hashes):
    return night.contract_for(row, data_hash, source_hashes)


def preflight(suite, rows, directory, contracts):
    training = [r for r in rows if r['kind'] == 'train']
    receipts.preflight(training, directory, contracts)
    problems = [f'{r["run_id"]}: {p}' for r in training
                for p in night.experiments.blockers(suite, r)]
    for row in rows:
        if row['kind'] != 'evaluation':
            continue
        folder = directory / row['run_id']
        state_path = folder / 'state.json'
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state['contract_sha256'] != night.fingerprint(contracts[row['run_id']]):
                problems.append(f'{row["run_id"]}: existing E0 contract mismatch')
            if state['evaluation_complete'] and night.read_bright(row['config']['output_dir'])[0] is None:
                problems.append(f'{row["run_id"]}: completed E0 results missing')
        elif Path(row['config']['output_dir']).exists():
            problems.append(f'{row["run_id"]}: unowned E0 output exists')
    if problems:
        raise ValueError('\n'.join(problems))


def summarize(rows, directory, output):
    results = [receipts.read_result(r, directory) for r in rows]
    hashes = {r['data_sha256'] for r in results if r['data_sha256']}
    source_sets = {night.fingerprint(r['training_source_sha256']) for r in results
                   if r['evaluation_complete'] and r['training_source_sha256']}
    comparable = len(hashes) <= 1 and len(source_sets) <= 1 and all(
        r['training_source_sha256'] for r in results if r['evaluation_complete'])
    groups = []
    for variant in dict.fromkeys(r['variant'] for r in rows):
        members = [r for r in results if r['variant'] == variant]
        values = [r['mean'] for r in members if r['evaluation_complete']]
        complete = comparable and len(values) == len(members)
        groups.append(dict(variant=variant, completed=len(values), expected=len(members),
                           mean=statistics.mean(values) if complete else None,
                           sample_sd=statistics.stdev(values) if complete and len(values) > 1 else None,
                           worst=min(values) if complete else None))
    def fmt(value):
        return '—' if value is None else f'{value:.3f}'
    lines = ['# Reasoning E0 core results', '', f'Data/code comparable: {comparable}', '',
             '| E0 / Method | Complete | Mean | Sample SD | Worst |',
             '|---|---:|---:|---:|---:|']
    for group in groups:
        lines.append(f'| {group["variant"]} | {group["completed"]}/{group["expected"]} | '
                     f'{fmt(group["mean"])} | {fmt(group["sample_sd"])} | {fmt(group["worst"])} |')
    lines += ['', '| Run | Mean | ' + ' | '.join(night.SUBSETS) + ' |',
              '|---|---:|' + '---:|' * len(night.SUBSETS)]
    for row in results:
        lines.append(f'| {row["run"]} | {fmt(row["mean"])} | ' +
                     ' | '.join(fmt((row['scores'] or {}).get(k)) for k in night.SUBSETS) + ' |')
    lines += ['', 'Missing / failed:', ''] + [f'- {r["run"]}: {r["error"]}' for r in results if r['error']]
    report = dict(comparable=comparable, data_sha256=sorted(hashes), groups=groups, runs=results)
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.summary.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        night.write_json(output / 'summary.json', report)
        (output / 'summary.md').write_text('\n'.join(lines) + '\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', default='run', choices=['run', 'check', 'summary'])
    parser.add_argument('--config', type=Path, default=SETTINGS)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--seeds', '--seed', nargs='+', type=int, choices=night.SEEDS, default=list(night.SEEDS))
    args = parser.parse_args(argv)
    suite, rows, directory = resolve(args.config, args.models, args.seeds)
    if args.action == 'summary':
        report = summarize(rows, directory, directory)
        print(directory / 'summary.md')
        return int(not report['comparable'] or not all(r['evaluation_complete'] for r in report['runs']))
    data_hash = night.verify_data(rows)
    source_hashes = sources()
    contracts = {r['run_id']: contract(r, data_hash, source_hashes) for r in rows}
    for row in rows:
        print(f'{row["run_id"]}: {row["config"]["model_name_or_path"]}; {row["config"]["output_dir"]}', flush=True)
    if args.action == 'check':
        preflight(suite, rows, directory, contracts)
        print(f'CPU preflight passed: {sum(r["kind"] == "train" for r in rows)} trainings, '
              f'{sum(r["kind"] == "evaluation" for r in rows)} E0 evaluations; data SHA256={data_hash}. '
              'No model loaded or GPU job started.')
        return 0
    import torch
    import deepspeed
    if torch.cuda.device_count() < 8 or not torch.cuda.is_bf16_supported():
        raise ValueError('Need eight visible CUDA GPUs with BF16 support')
    runtime = dict(torch_version=torch.__version__, cuda_version=torch.version.cuda,
                   deepspeed_version=deepspeed.__version__, nproc=8,
                   gpus=[dict(name=torch.cuda.get_device_properties(i).name,
                              total_memory=torch.cuda.get_device_properties(i).total_memory) for i in range(8)])
    with receipts.run_locks(directory, [r['run_id'] for r in rows]):
        preflight(suite, rows, directory, contracts)
        lookup = {r['run_id']: r for r in rows}
        output = directory / 'queues' / '+'.join(sorted(args.models)) / ('seeds-' + '-'.join(map(str, sorted(args.seeds))))
        def execute(name):
            print(f'START {name}; logs: {directory / name}', flush=True)
            return receipts.execute(lookup[name], contracts[name], directory / name, runtime=runtime)
        def record(event):
            print(json.dumps(event), flush=True)
            folder = directory / event['job']
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / 'events.jsonl').open('a') as handle:
                handle.write(json.dumps(event) + '\n')
            summarize(rows, directory, output)
        failures = night.execute_queue(list(lookup), execute, record)
        print(f'Done; failed={failures}; summary={output / "summary.md"}', flush=True)
        return int(bool(failures))


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        raise SystemExit(2)
