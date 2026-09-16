#!/usr/bin/env python3
"""Independent G1-R2 ablations; default queue is SF32, CP32 and SF64 paired.

Run/check never launch existing G64 controls. Optional variants are explicit.
Each run has its own lock, receipt, event log, training/evaluation log and timing.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import copy
import fcntl
import json
from pathlib import Path
import statistics
import sys
import time

import run_g1_r2 as night

ROOT = night.ROOT
SPEC = ROOT / 'configs/experiments/iclr2027/g1_r2_ablations.yaml'
CORE = ('g32_sf', 'g32_cp', 'paired')


def definitions():
    return night.experiments.read_mapping(SPEC)['variants']


def resolve(settings=night.SETTINGS, variants=CORE, seeds=night.SEEDS, output_root=None,
            reference_root=None):
    variants = tuple(variants)
    specs = definitions()
    if not variants or len(set(variants)) != len(variants) or set(variants) - specs.keys():
        raise ValueError('Select distinct registered variants')
    _, base = night.resolve_matrix(settings, seeds=seeds)
    original_root = Path(base[0]['config']['output_dir']).parent
    output_root = Path(output_root).resolve() if output_root else original_root
    reference_root = Path(reference_root).resolve() if reference_root else original_root
    rows, controls = [], []
    for seed in night.selected_seeds(seeds):
        for estimator in ('SF', 'CP'):
            row = copy.deepcopy(next(r for r in base if r['run_id'] == night.run_id(
                f'RL-GradedNDCG64-{estimator}', seed)))
            row['variant'] = f'g64_{estimator.lower()}'
            row['config']['output_dir'] = str(reference_root / Path(row['config']['output_dir']).name)
            controls.append(row)
        for variant in variants:
            spec = specs[variant]
            control = next(r for r in controls if r['config']['seed'] == seed and
                           r['variant'] == f'g64_{spec["estimator"].lower()}')
            row = copy.deepcopy(control)
            row.update(run_id=night.run_id(spec['method'], seed), variant=variant,
                       control=control['run_id'], priority='core' if variant in CORE else 'optional',
                       execution_stage=None, execution_stage_name='G1-R2 ablations')
            if spec.get('control_variant'):
                row['control'] = night.run_id(specs[spec['control_variant']]['method'], seed)
            row['config'].update(copy.deepcopy(spec['overrides']))
            row['config'].update(run_name=row['run_id'],
                                 output_dir=str(output_root / f'{row["run_id"]}-s{seed}'))
            # The inherited scope remains joint even for a one-sided policy.
            rows.append(row)
    return rows, controls, output_root / '.r2_ablations', reference_root / '.r2_batch'


def sources():
    result = night.source_fingerprint()
    paths = [SPEC, Path(__file__), ROOT / 'scripts/run_g1_r2_mechanisms.py',
             ROOT / 'scripts/run_g1_r2_exploration.py']
    result.update({str(p.relative_to(ROOT)): night.experiments.digest(p) for p in paths})
    return result


@contextmanager
def run_locks(directory, names):
    directory = directory / 'locks'
    directory.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for name in sorted(names):
            handle = stack.enter_context((directory / f'{name}.lock').open('a'))
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError(f'Another task owns {name}') from exc
        yield


def preflight(rows, directory, contracts):
    for row in rows:
        folder = directory / row['run_id']
        state_path = folder / 'state.json'
        out = Path(row['config']['output_dir'])
        if not state_path.exists():
            if out.exists() or (out.parent / '.launches' / out.name).exists():
                raise ValueError(f'Unowned output exists: {out}')
            continue
        state = json.loads(state_path.read_text())
        if state['contract_sha256'] != night.fingerprint(contracts[row['run_id']]):
            raise ValueError(f'{row["run_id"]}: config/data/code changed; use a new --output-root')
        if state['train_started'] and not state['train_complete']:
            raise ValueError(f'{row["run_id"]}: incomplete training preserved; use a new --output-root')
        if state['train_complete']:
            night.validate_final_model(row)
        if state['evaluation_complete'] and night.read_bright(out)[0] is None:
            raise ValueError(f'{row["run_id"]}: completed evaluation is missing')


def execute(row, contract, folder, call=night.invoke, runtime=None):
    def timed(command, log):
        if log.stem == 'train' and runtime is not None:
            night.write_json(folder / 'runtime.json', runtime)
        started = time.monotonic()
        code = None
        try:
            code = call(command, log)
            return code
        finally:
            elapsed = time.monotonic() - started
            with (folder / 'timings.jsonl').open('a') as handle:
                handle.write(json.dumps(dict(phase=log.stem, seconds=elapsed, exit_code=code)) + '\n')
    return night.execute_job(row, contract, folder, timed)


def read_result(row, directory):
    result = dict(run=row['run_id'], variant=row['variant'], seed=row['config']['seed'],
                  mean=None, scores=None, data_sha256=None, error=None, train_seconds=None,
                  train_complete=False, evaluation_complete=False, training_source_sha256=None)
    folder = directory / row['run_id']
    try:
        state = night.experiments.read_mapping(folder / 'state.json')
        saved = night.experiments.read_mapping(folder / 'contract.json')
        if state['contract_sha256'] != night.fingerprint(saved):
            raise ValueError('Completion receipt does not match contract')
        actual = saved['resolved']
        # Host paths can change when collecting artifacts from independent machines.
        paths = {'output_dir', 'data_path', 'index_cache_dir', 'deepspeed'}
        if actual['run_id'] != row['run_id'] or any(
            actual['config'].get(k) != v for k, v in row['config'].items() if k not in paths
        ):
            raise ValueError('Saved configuration does not match this experiment')
        result['data_sha256'] = saved['data_sha256']
        result['training_source_sha256'] = {
            k: v for k, v in saved.get('source_sha256', {}).items() if k.startswith('src/')
        }
        if not result['data_sha256']:
            raise ValueError('Missing training data hash')
        result['train_complete'] = state['train_complete']
        if not state['evaluation_complete']:
            raise ValueError(state.get('last_error') or 'Evaluation not complete')
        scores, source = night.read_bright(row['config']['output_dir'])
        if scores is None:
            raise ValueError('Missing complete 12-subset BRIGHT results')
        result.update(scores=scores, mean=statistics.mean(scores.values()),
                      evaluation_complete=True, source=source)
        timing_path = folder / 'timings.jsonl'
        if timing_path.exists():
            timings = [json.loads(line) for line in timing_path.read_text().splitlines()]
            times = [x['seconds'] for x in timings if x['phase'] == 'train' and x['exit_code'] == 0]
            result['train_seconds'] = sum(times) if times else None
    except (ValueError, OSError, KeyError, TypeError) as exc:
        result['error'] = str(exc)
    return result


def summarize(rows, controls, directory, reference_directory, output):
    results = [read_result(r, directory) for r in rows] + [read_result(r, reference_directory) for r in controls]
    seeds = sorted({r['config']['seed'] for r in rows})
    hashes = sorted({r['data_sha256'] for r in results if r['data_sha256']})
    source_sets = {night.fingerprint(r['training_source_sha256']) for r in results
                   if r['evaluation_complete'] and r['training_source_sha256']}
    source_consistent = len(source_sets) <= 1 and all(
        r['training_source_sha256'] for r in results if r['evaluation_complete'])
    consistent = len(hashes) <= 1 and source_consistent
    variants = list(dict.fromkeys(r['variant'] for r in results))
    groups = []
    for variant in variants:
        members = [r for r in results if r['variant'] == variant]
        values = [r['mean'] for r in members if r['mean'] is not None]
        complete = len(values) == len(seeds) and consistent
        times = [r['train_seconds'] for r in members]
        groups.append(dict(variant=variant, completed=len(values), expected=len(seeds),
                           mean=statistics.mean(values) if complete else None,
                           sample_sd=statistics.stdev(values) if complete and len(values)>1 else None,
                           worst=min(values) if complete else None,
                           mean_train_seconds=statistics.mean(times) if all(t is not None for t in times) else None))
    specs = definitions()
    pairs = [(v, specs[v].get('control_variant', f'g64_{specs[v]["estimator"].lower()}'))
             for v in variants if v in specs]
    pairs += [('g32_cp', 'g32_sf'), ('g32_cp', 'g64_sf'), ('no_rescale_cp', 'no_rescale_sf'),
              ('anneal_cp', 'anneal_sf'), ('anneal_sf', 'align080_sf'), ('anneal_cp', 'align080_cp')]
    pairs += [(f'align{point}_cp', f'align{point}_sf') for point in ('040', '053', '065', '080', '095', '098')]
    comparisons = []
    for left, right in pairs:
        if left not in variants or right not in variants:
            continue
        per_seed = []
        for seed in seeds:
            a, b = [next(r for r in results if r['variant'] == v and r['seed'] == seed) for v in (left, right)]
            valid = consistent and a['mean'] is not None and b['mean'] is not None and a['data_sha256'] == b['data_sha256']
            per_seed.append(dict(seed=seed, delta=a['mean']-b['mean'] if valid else None))
        values = [r['delta'] for r in per_seed if r['delta'] is not None]
        comparisons.append(dict(left=left, right=right, per_seed=per_seed,
                                mean_delta=statistics.mean(values) if len(values)==len(seeds) else None))
    report = dict(seeds=seeds, data_consistent=len(hashes)<=1, source_consistent=source_consistent,
                  comparable=consistent, data_sha256=hashes,
                  groups=groups, comparisons=comparisons, runs=results)
    def fmt(x):
        return '—' if x is None else f'{x:.3f}'
    lines = ['# G1-R2 ablation results', '', f'Seeds: {seeds}; data/code comparable: {consistent}.',
             'Missing controls/contracts are reported, never replaced by historical CSV scores.', '',
             '| Variant | Complete | Mean | Sample SD | Worst | Mean train seconds |',
             '|---|---:|---:|---:|---:|---:|']
    for g in groups:
        lines.append(f'| {g["variant"]} | {g["completed"]}/{g["expected"]} | {fmt(g["mean"])} | '
                     f'{fmt(g["sample_sd"])} | {fmt(g["worst"])} | {fmt(g["mean_train_seconds"])} |')
    lines += ['', '| Comparison | ' + ' | '.join(str(s) for s in seeds) + ' | Mean delta |',
              '|---|' + '---:|'*(len(seeds)+1)]
    for c in comparisons:
        lines.append(f'| {c["left"]} − {c["right"]} | ' + ' | '.join(fmt(r['delta']) for r in c['per_seed']) + f' | {fmt(c["mean_delta"])} |')
    lines += ['', '| Run | Mean | ' + ' | '.join(night.SUBSETS) + ' |', '|---|---:|'+'---:|'*len(night.SUBSETS)]
    for r in results:
        lines.append(f'| {r["run"]} | {fmt(r["mean"])} | '+' | '.join(fmt((r['scores'] or {}).get(k)) for k in night.SUBSETS)+' |')
    lines += ['', '## Missing / failed', ''] + [f'- {r["run"]}: {r["error"]}' for r in results if r['error']]
    output.mkdir(parents=True, exist_ok=True)
    # Explicit summary and running tasks may read the same receipts concurrently.
    with (output / '.summary.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        night.write_json(output / 'summary.json', report)
        (output / 'summary.md').write_text('\n'.join(lines)+'\n')
    return report


def main(argv=None, defaults=CORE):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', choices=['run', 'check', 'summary'], default='run')
    parser.add_argument('--variants', nargs='+', choices=[*definitions(), 'all'], default=list(defaults))
    parser.add_argument('--seeds', '--seed', nargs='+', type=int, choices=night.SEEDS, default=list(night.SEEDS))
    parser.add_argument('--config', type=Path, default=night.SETTINGS)
    parser.add_argument('--output-root', type=Path, help='New runs only; default is settings output_dir')
    parser.add_argument('--reference-root', type=Path, help='Existing G64 runs/receipts; default is settings output_dir')
    args = parser.parse_args(argv)
    variants = list(definitions()) if args.variants == ['all'] else args.variants
    rows, controls, directory, refdir = resolve(args.config, variants, args.seeds, args.output_root, args.reference_root)
    if args.action == 'summary':
        report = summarize(rows, controls, directory, refdir, directory)
        print(directory / 'summary.md')
        return 0 if report['comparable'] and all(r['evaluation_complete'] for r in report['runs']) else 1
    data_hash = night.verify_data(rows)
    source_hashes = sources()
    contracts = {r['run_id']: night.contract_for(r, data_hash, source_hashes) for r in rows}
    print(f'Training data SHA256: {data_hash}', flush=True)
    for row in rows:
        print(f'{row["run_id"]}: {row["config"]["output_dir"]}', flush=True)
    if args.action == 'check':
        preflight(rows, directory, contracts)
        print(f'CPU preflight passed: {len(rows)} trainings; 8 GPUs/run; no model loaded.')
        return 0
    import torch
    import deepspeed  # Validate shared dependencies before marking any training started.
    if torch.cuda.device_count() < 8 or not torch.cuda.is_bf16_supported():
        raise ValueError('Each training task needs eight visible CUDA GPUs with BF16 support')
    runtime = dict(torch_version=torch.__version__, cuda_version=torch.version.cuda,
                   deepspeed_version=deepspeed.__version__, nproc=8,
                   gpus=[dict(name=torch.cuda.get_device_properties(i).name,
                              total_memory=torch.cuda.get_device_properties(i).total_memory) for i in range(8)])
    with run_locks(directory, [r['run_id'] for r in rows]):
        lookup = {r['run_id']: r for r in rows}
        output = directory / 'queues' / ('+'.join(sorted(variants))) / ('seeds-'+'-'.join(map(str, sorted(args.seeds))))
        def run(name):
            print(f'START {name}; logs: {directory / name}', flush=True)
            return execute(lookup[name], contracts[name], directory / name, runtime=runtime)
        def record(event):
            folder = directory / event['job']
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / 'events.jsonl').open('a') as handle:
                handle.write(json.dumps(event)+'\n')
            print(json.dumps(event), flush=True)
            summarize(rows, controls, directory, refdir, output)
        failures = night.execute_queue(list(lookup), run, record)
        print(f'Done; failed={failures}; summary={output / "summary.md"}', flush=True)
        return int(bool(failures))


def cli(defaults=CORE):
    try:
        raise SystemExit(main(defaults=defaults))
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        raise SystemExit(2)


if __name__ == '__main__':
    cli()
