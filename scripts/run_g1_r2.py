#!/usr/bin/env python3
"""Run the new-protocol G1 night: E0, paired gradient probe, 9 methods x 3 seeds.

Default action is run. `check` resolves everything without importing torch or
downloading models. Re-running resumes the queue, never a partial training job.
Use --seeds 42, --seeds 3407, or --seeds 2026 on separate machines.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from experiments import iclr2027 as experiments
from run_g1_stability import read_bright, SUBSETS

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / 'configs/experiments/iclr2027/suite_r2.yaml'
SETTINGS = ROOT / 'configs/experiments_r2.yaml'
SEEDS = (42, 3407, 2026)
METHODS = ('CL', 'LL-Binary', 'LL-Graded', 'RL-MRR64-SF', 'RL-MRR64-CP',
           'RL-GradedNDCG64-SF', 'RL-GradedNDCG64-CP',
           'RL-BinaryNDCG64-SF', 'RL-BinaryNDCG64-CP')
REVISION = '97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3'


def run_id(method, seed=42):
    return f'G1-R2-{method}' + (f'-Seed{seed}' if seed != 42 else '')


def selected_seeds(seeds):
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(s not in SEEDS for s in seeds):
        raise ValueError(f'Select distinct registered seeds from {SEEDS}')
    return tuple(seed for seed in SEEDS if seed in seeds)


def queue_directory(directory, seeds):
    seeds = selected_seeds(seeds)
    return directory if seeds == SEEDS else directory / 'queues' / ('seeds-' + '-'.join(map(str, seeds)))


@contextmanager
def seed_locks(directory, seeds):
    """Disjoint seeds may run together; overlapping queues cannot share a run."""
    lock_dir = directory / 'locks'
    lock_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for seed in selected_seeds(seeds):
            lock = stack.enter_context((lock_dir / f'seed-{seed}.lock').open('a'))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError(f'Another R2 queue owns seed {seed} in this output root') from exc
        yield


def build_jobs(matrix, skip_probe=False):
    jobs = [row['run_id'] for row in matrix]
    if not skip_probe and 'G1-R2-E0' in jobs:
        jobs.insert(0, 'gradient_probe')
    if not skip_probe and run_id('LL-Binary') in jobs:
        jobs.insert(jobs.index(run_id('LL-Binary')) + 1, 'gradient_probe_ll25')
    return jobs


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def resolve_matrix(settings=SETTINGS, gpus=8, seeds=SEEDS):
    seeds = selected_seeds(seeds)
    if gpus != 8:
        raise ValueError('The matched overnight recipe requires 8 GPUs (16 queries per device)')
    suite = experiments.apply_settings(experiments.load_suite(SUITE), settings)
    matrix = [experiments.resolve_run(suite, SUITE, name, nproc=gpus)
              for name in experiments.ordered_run_ids(suite)]
    expected_ids = ['G1-R2-E0'] + [run_id(method, seed) for seed in SEEDS for method in METHODS]
    if [row['run_id'] for row in matrix] != expected_ids:
        raise ValueError('R2 must contain E0 followed by nine methods for each of the three seeds')
    for row in matrix:
        cfg = row['config']
        expected = dict(model_name_or_path='Qwen/Qwen3-Embedding-0.6B', model_revision=REVISION,
                        dev_samples_per_source=0, max_steps=113, learning_rate=5e-6,
                        per_device_train_batch_size=16, gradient_accumulation_steps=1,
                        max_grad_norm=0, save_steps=25, save_only_model=True, load_best_model_at_end=False,
                        q_max_len=512, d_max_len=1024, embedding_max_length=8192,
                        lora_enabled=False, overwrite_output_dir=False)
        if row['objective'] == 'rl':
            expected.update(group_size=64, target_alignment=.9, final_alignment=None,
                            exploration_schedule='fixed', advantage_baseline='leave_one_out',
                            advantage_norm='none', document_log_prob_reduction='sum',
                            document_advantage_baseline='shared', rollout='product',
                            sampling_law='vmf', frozen_doc_rescale=True, kl_coef=0,
                            sigma_learnable=False, in_batch_use_sampled_documents=False,
                            reward_combine='sum', aux_infonce_coef=0,
                            rollout_seed=cfg['seed'], action_components=[['query'], ['positive', 'negative']])
        mismatch = {k: (cfg.get(k), v) for k, v in expected.items() if cfg.get(k) != v}
        if mismatch or row['dependency'] or row['reuse']:
            raise ValueError(f'{row["run_id"]}: incompatible R2 recipe: {mismatch}')
        if cfg.get('mteb_eval_tasks') or cfg.get('mteb_eval_benchmark'):
            raise ValueError('Only final BRIGHT evaluation is scheduled, no checkpoint MTEB callback')
        ds = experiments.read_mapping(experiments.path_at_root(cfg['deepspeed']))
        if ds.get('gradient_clipping') != 0:
            raise ValueError('DeepSpeed must explicitly disable gradient clipping')
    for seed in SEEDS:
        for reward in ('MRR64', 'BinaryNDCG64', 'GradedNDCG64'):
            pair = [next(r['config'] for r in matrix if r['run_id'] == run_id(f'RL-{reward}-{est}', seed))
                    for est in ('SF', 'CP')]
            difference = {k for k in set(pair[0]) | set(pair[1]) if pair[0].get(k) != pair[1].get(k)}
            if difference != {'run_name', 'output_dir', 'gradient_estimator'}:
                raise ValueError(f'{reward}, seed {seed}: SF/CP differ beyond the estimator: {difference}')
    return suite, [row for row in matrix if (
        row['kind'] == 'evaluation' and 42 in seeds
        or row['kind'] == 'train' and row['config']['seed'] in seeds
    )]


def verify_data(matrix):
    """Check the selected data against its own manifest, independent of host paths."""
    paths = {experiments.path_at_root(row['config']['data_path']) for row in matrix}
    if len(paths) != 1:
        raise ValueError(f'All R2 methods must consume the same prepared data: {sorted(map(str, paths))}')
    path = paths.pop()
    manifest_path = path.parent / 'manifest.json'
    manifest = experiments.read_mapping(manifest_path)
    if (manifest.get('version') != 1
            or manifest.get('selection') != 'fixed_budget_final_checkpoint'
            or manifest.get('positive_selection_seed') != 42):
        raise ValueError(f'R2 needs a no-dev dataset prepared with positive selection seed 42: {manifest_path}')
    artifacts = [item for item in manifest.get('artifacts', []) if item.get('role') == 'train']
    if len(artifacts) != 1 or Path(artifacts[0].get('path', '')).name != path.name:
        raise ValueError(f'Manifest must identify one prepared train artifact named {path.name}: {manifest_path}')
    # Artifact paths describe the preparation host. Use the configured local file;
    # only its content fingerprint and artifact name need to survive relocation.
    expected = artifacts[0].get('sha256')
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError(f'Manifest train artifact needs a valid SHA256: {manifest_path}')
    actual = experiments.digest(path)
    if actual != expected:
        raise ValueError(
            f'R2 prepared training data hash mismatch:\n'
            f'  data: {path}\n  manifest: {manifest_path}\n'
            f'  expected (manifest): {expected}\n  actual: {actual}\n'
            'Check G1.data in the selected settings file and copy the matching data/manifest pair.'
        )
    return actual


def evaluation_command(row):
    cfg = row['config']
    reference = row['kind'] == 'evaluation'
    kwargs = dict(max_length=cfg['embedding_max_length'], pooler_type=cfg['pooling_method'],
                  padding_side=cfg['padding_side'], append_token=cfg['append_token'], do_norm=True,
                  use_instruction='{task_description}' in cfg['query_prompt_template'],
                  query_prompt_template=cfg['query_prompt_template'],
                  document_prompt_template=cfg['document_prompt_template'],
                  instruction_dict_path=str(ROOT / 'eval_mteb/scripts/task_prompts.json'))
    if reference:
        kwargs.update(revision=cfg['model_revision'], model_name='G1-R2-E0')
    return [sys.executable, str(ROOT / 'eval_mteb/run_mteb.py'),
            '--model', cfg['model_name_or_path'] if reference else cfg['output_dir'],
            '--precision', cfg['mteb_eval_precision'], '--model_kwargs', json.dumps(kwargs),
            '--output_dir', str(Path(cfg['output_dir']) / 'mteb_eval/bright'),
            '--batch_size', '16', '--langs', 'eng', '--benchmark', 'BRIGHT',
            '--run_kwargs', json.dumps({'overwrite_results': True}), '--fail_on_task_error']


def validate_final_model(row):
    out = Path(row['config']['output_dir'])
    if not (out / 'config.json').is_file():
        raise ValueError(f'Missing final config: {out}')
    index = out / 'model.safetensors.index.json'
    if index.is_file():
        weights = set(json.loads(index.read_text())['weight_map'].values())
    else:
        weights = {'model.safetensors'}
    if not weights or any(not (out / p).is_file() or (out / p).stat().st_size == 0 for p in weights):
        raise ValueError(f'Missing final model weights: {out}')
    protocol = experiments.read_mapping(out / 'embedding_protocol.json')
    cfg = row['config']
    expected = dict(tokenization_version=2, pooling_compute_dtype='float32',
                    pooling_method=cfg['pooling_method'], padding_side=cfg['padding_side'],
                    append_token=cfg['append_token'], max_length=cfg['embedding_max_length'],
                    query_prompt_template=cfg['query_prompt_template'],
                    document_prompt_template=cfg['document_prompt_template'])
    if any(protocol.get(k) != v for k, v in expected.items()):
        raise ValueError(f'Final embedding protocol mismatch: {out}')


def source_fingerprint():
    paths = sorted((ROOT / 'src').rglob('*.py')) + sorted((ROOT / 'eval_mteb').rglob('*.py'))
    paths += [Path(__file__), SUITE, ROOT / 'scripts/experiments/iclr2027.py',
              ROOT / 'scripts/diagnose_rollout_gradients.py', ROOT / 'eval_mteb/scripts/task_prompts.json']
    return {str(p.relative_to(ROOT)): experiments.digest(p) for p in paths}


def contract_for(row, data_hash, sources):
    ds_path = experiments.path_at_root(row['config']['deepspeed'])
    return dict(resolved=row, data_sha256=data_hash, source_sha256=sources,
                deepspeed=experiments.read_mapping(ds_path), evaluation_command=evaluation_command(row))


def invoke(command, log):
    env = os.environ.copy()
    # Unattended runs must not pause at the W&B login prompt. Explicit user mode wins.
    env.setdefault('WANDB_MODE', 'offline')
    env.setdefault('TOKENIZERS_PARALLELISM', 'false')
    with log.open('a') as handle:
        handle.write('\n' + json.dumps(command) + '\n')
        handle.flush()
        return subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT).returncode


def execute_job(row, contract, folder, call=invoke):
    """Persist train/eval separately so an eval failure never retrains a model."""
    state_path = folder / 'state.json'
    out = Path(row['config']['output_dir'])
    reference = row['kind'] == 'evaluation'
    identity = fingerprint(contract)
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state['contract_sha256'] != identity:
            raise ValueError('Existing run belongs to a different config/data/code; choose a new output_dir')
    else:
        if out.exists() or (out.parent / '.launches' / out.name).exists():
            raise ValueError(f'Unowned output/launch receipt exists; refusing to resume or overwrite: {out}')
        state = dict(contract_sha256=identity, train_started=False, train_complete=reference,
                     evaluation_complete=False)
        write_json(folder / 'contract.json', contract)
        write_json(state_path, state)
    if state['train_started'] and not state['train_complete']:
        raise ValueError('Previous training failed or was interrupted; preserved without implicit checkpoint resume')
    if state['evaluation_complete']:
        if not reference:
            validate_final_model(row)
        if read_bright(out)[0] is None:
            raise ValueError('Completed run is missing its final BRIGHT results')
        return 'skipped_complete'
    if not state['train_complete']:
        runtime = folder / 'config.json'
        write_json(runtime, row['config'])
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                   '--nproc_per_node=8', str(ROOT / row['entrypoint']), str(runtime)]
        state['train_started'] = True
        write_json(state_path, state)
        code = call(command, folder / 'train.log')
        if code:
            raise ValueError(f'Training failed (exit {code}); see {folder / "train.log"}')
        validate_final_model(row)
        state['train_complete'] = True
        write_json(state_path, state)
    if not reference:
        validate_final_model(row)
    code = call(contract['evaluation_command'], folder / 'eval.log')
    if code:
        raise ValueError(f'Final evaluation failed (exit {code}); re-run the queue to retry only evaluation')
    if read_bright(out)[0] is None:
        raise ValueError('Evaluation returned success without complete 12-subset BRIGHT results')
    state['evaluation_complete'] = True
    write_json(state_path, state)
    return 'complete'


def summarize(matrix, directory, output_dir=None):
    seeds = selected_seeds(dict.fromkeys(row['config']['seed'] for row in matrix if row['kind'] == 'train'))
    expected_count = len(seeds)
    output_dir = output_dir if output_dir is not None else queue_directory(directory, seeds)
    rows = []
    for row in matrix:
        name = row['run_id']
        state_path = directory / name / 'state.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        scores, source, error, data_hash = None, None, None, None
        if state.get('evaluation_complete') and not state.get('last_error'):
            try:
                contract = experiments.read_mapping(directory / name / 'contract.json')
                if fingerprint(contract) != state.get('contract_sha256'):
                    raise ValueError('Saved run contract does not match the completion receipt')
                data_hash = contract.get('data_sha256')
                if not data_hash:
                    raise ValueError('Completed run contract is missing the training data SHA256')
                scores, source = read_bright(row['config']['output_dir'])
            except (ValueError, OSError, KeyError, TypeError) as exc:
                error = str(exc)
        rows.append(dict(run=name, kind=row['kind'], seed=row['config']['seed'], scores=scores, source=source, data_sha256=data_hash,
                         mean=statistics.mean(scores.values()) if scores else None,
                         train_complete=state.get('train_complete', False),
                         evaluation_complete=bool(scores), error=error or state.get('last_error')))
    data_hashes = sorted({row['data_sha256'] for row in rows
                          if row['evaluation_complete'] and row['kind'] == 'train'})
    data_consistent = len(data_hashes) <= 1
    groups = []
    for method in METHODS:
        members = [next(r for r in rows if r['run'] == run_id(method, seed)) for seed in seeds]
        values = [r['mean'] for r in members if r['mean'] is not None]
        complete = len(values) == expected_count and data_consistent
        groups.append(dict(method=method, completed=len(values),
                           mean=statistics.mean(values) if complete else None,
                           sample_sd=statistics.stdev(values) if complete and expected_count > 1 else None,
                           worst=min(values) if complete else None))
    paired = []
    for reward in ('MRR64', 'BinaryNDCG64', 'GradedNDCG64'):
        differences = []
        for seed in seeds:
            sf, cp = [next(r for r in rows if r['run'] == run_id(f'RL-{reward}-{est}', seed))
                      for est in ('SF', 'CP')]
            differences.append(dict(seed=seed, delta=cp['mean']-sf['mean']
                                    if cp['mean'] is not None and sf['mean'] is not None
                                    and cp['data_sha256'] == sf['data_sha256'] else None))
        values = [x['delta'] for x in differences if x['delta'] is not None]
        paired.append(dict(reward=reward, per_seed=differences,
                           mean_delta=statistics.mean(values) if len(values) == expected_count and data_consistent else None))
    auxiliary = {}
    event_paths = [output_dir / 'events.jsonl']
    if seeds == SEEDS:
        event_paths += sorted((directory / 'queues').glob('*/events.jsonl'))
    for events_path in event_paths:
        if not events_path.exists():
            continue
        for line in events_path.read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # Preserve readable events after an interrupted final append.
            if event['job'].startswith('gradient_probe') and event['time'] >= auxiliary.get(event['job'], {}).get('time', ''):
                auxiliary[event['job']] = event
    report = dict(units='BRIGHT nDCG@10 percentage points', seeds=list(seeds), groups=groups, paired_cp_minus_sf=paired,
                  runs=rows, gradient_probes=auxiliary, data_sha256=data_hashes, data_consistent=data_consistent)
    write_json(output_dir / 'summary.json', report)
    def fmt(value):
        return '—' if value is None else f'{value:.3f}'
    lines = ['# G1 R2 overnight results', '', f'Selected seeds: {", ".join(map(str, seeds))}.',
             'Statistics require all selected seeds with identical training data; a single seed has no sample SD.', '']
    if not data_consistent:
        lines += ['Training data hashes differ between runs. Aggregate statistics are withheld; see per-run hashes below.', '']
    lines += ['| Method | Complete | Mean | Sample SD | Worst |', '|---|---:|---:|---:|---:|']
    for g in groups:
        lines.append(f'| {g["method"]} | {g["completed"]}/{expected_count} | {fmt(g["mean"])} | {fmt(g["sample_sd"])} | {fmt(g["worst"])} |')
    lines += ['', '| Reward (CP − SF) | ' + ' | '.join(f'Seed {s}' for s in seeds) + ' | Mean difference |',
              '|---|' + '---:|' * (expected_count + 1)]
    for p in paired:
        lines.append(f'| {p["reward"]} | ' + ' | '.join(fmt(d['delta']) for d in p['per_seed']) + f' | {fmt(p["mean_delta"])} |')
    lines += ['', '| Run | Mean | ' + ' | '.join(SUBSETS) + ' |', '|---|---:|' + '---:|'*len(SUBSETS)]
    for r in rows:
        lines.append(f'| {r["run"]} | {fmt(r["mean"])} | ' + ' | '.join(fmt((r['scores'] or {}).get(s)) for s in SUBSETS) + ' |')
    lines += ['', '## Training data SHA256', '']
    lines += [f'- {r["run"]}: `{r["data_sha256"]}`' for r in rows
              if r['evaluation_complete'] and r['kind'] == 'train']
    lines += ['', '## Pending / failed', '']
    lines += [f'- {r["run"]}: {r["error"] or "not complete"}' for r in rows if not r['evaluation_complete']]
    lines += ['', '## Gradient probes', '']
    lines += [f'- {name}: {event["status"]}; {event["error"] or "see probe JSON"}'
              for name, event in auxiliary.items()]
    (output_dir / 'summary.md').write_text('\n'.join(lines) + '\n')
    return report


def run_probe(settings, directory, sources, data_hash, call=invoke, checkpoint=None):
    folder = directory / ('gradient_probe_ll25' if checkpoint else 'gradient_probe')
    folder.mkdir(parents=True, exist_ok=True)
    _, matrix = resolve_matrix(settings)
    config = next(r['config'] for r in matrix if r['run_id'] == run_id('RL-MRR64-SF'))
    weights = sorted(checkpoint.glob('*.safetensors')) if checkpoint else []
    if checkpoint and not weights:
        raise ValueError(f'No new LL-Binary step-25 weights: {checkpoint}')
    identity = fingerprint(dict(config=config, sources=sources, data=data_hash,
                                checkpoint_weights={p.name: experiments.digest(p) for p in weights}))
    marker = folder / 'complete.json'
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved['contract_sha256'] != identity:
            raise ValueError('Existing probe used different code/data/config')
        result = json.loads(Path(saved['result']).read_text())
        if len(result.get('probes', [])) != 3:
            raise ValueError('Completed probe result is missing batches')
        return 'skipped_complete'
    # Each retry preserves an interrupted probe's partial statistics.
    attempt = len(list(folder.glob('attempt-*.log'))) + 1
    output = folder / f'attempt-{attempt}.json'
    command = [sys.executable, str(ROOT / 'scripts/diagnose_rollout_gradients.py'),
               '--suite', str(SUITE), '--config', str(settings.resolve()),
               '--run', run_id('RL-MRR64-SF'), '--compare-gradient-estimators',
               '--batch-indices', '0', '100', '200', '--precision', 'bf16', '--device', 'cuda:0',
               '--output', str(output)]
    if checkpoint:
        command += ['--checkpoint', str(checkpoint), '--step', '25']
    if call(command, folder / f'attempt-{attempt}.log'):
        raise ValueError(f'Paired gradient probe failed; see {folder}')
    payload = json.loads(output.read_text())
    if payload.get('comparison') != 'paired_gradient_estimators' or len(payload.get('probes', [])) != 3:
        raise ValueError('Paired probe did not complete all three fixed batches')
    write_json(marker, dict(contract_sha256=identity, result=str(output)))
    return 'complete'


def execute_queue(jobs, execute, record):
    """Failure in an independent job cannot suppress the remaining comparisons."""
    failures = []
    for job in jobs:
        started = time.monotonic()
        try:
            status, error = execute(job), None
        except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
            status, error = 'failed', str(exc)
            failures.append(job)
        record(dict(job=job, status=status, error=error, seconds=time.monotonic()-started,
                    time=datetime.now(timezone.utc).isoformat()))
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', default='run', choices=['run', 'check', 'summary'])
    parser.add_argument('--config', type=Path, default=SETTINGS)
    parser.add_argument('--gpus', type=int, choices=[8], default=8)
    parser.add_argument('--seeds', '--seed', nargs='+', type=int, choices=SEEDS, default=list(SEEDS),
                        help='Seeds to run on this machine; E0 and both shared probes belong to seed 42')
    parser.add_argument('--skip-probe', action='store_true', help='Skip the standalone gradient probes assigned to seed 42')
    args = parser.parse_args(argv)
    seeds = selected_seeds(args.seeds)
    suite, matrix = resolve_matrix(args.config, args.gpus, seeds)
    directory = Path(matrix[0]['config']['output_dir']).parent / '.r2_batch'
    queue_dir = queue_directory(directory, seeds)
    if args.action == 'summary':
        report = summarize(matrix, directory)
        print(queue_dir / 'summary.md')
        return 0 if report['data_consistent'] else 1
    data_path = experiments.path_at_root(matrix[0]['config']['data_path'])
    print(f'Settings: {args.config.resolve()}\nTraining data: {data_path}\n'
          f'Data manifest: {data_path.parent / "manifest.json"}', flush=True)
    data_hash = verify_data(matrix)
    print(f'Training data SHA256: {data_hash}', flush=True)
    sources = source_fingerprint()
    contracts = {row['run_id']: contract_for(row, data_hash, sources) for row in matrix}
    for row in matrix:
        print(f'{row["run_id"]}: {row["config"]["output_dir"]}', flush=True)
    if args.action == 'check':
        problems = []
        for row in matrix:
            state = directory / row['run_id'] / 'state.json'
            if state.exists():
                saved = json.loads(state.read_text())
                if saved['contract_sha256'] != fingerprint(contracts[row['run_id']]):
                    problems.append(f'{row["run_id"]}: existing contract mismatch')
                elif saved['train_started'] and not saved['train_complete']:
                    problems.append(f'{row["run_id"]}: incomplete training is preserved and will not resume')
                elif saved['evaluation_complete']:
                    if row['kind'] == 'train':
                        validate_final_model(row)
                    if read_bright(row['config']['output_dir'])[0] is None:
                        problems.append(f'{row["run_id"]}: completed evaluation is missing')
            elif row['kind'] == 'train':
                problems.extend(f'{row["run_id"]}: {p}' for p in experiments.blockers(suite, row))
            elif Path(row['config']['output_dir']).exists():
                problems.append(f'{row["run_id"]}: unowned evaluation output exists')
        if problems:
            raise ValueError('\n'.join(problems))
        print(f'Validated: {len(METHODS)*len(seeds)} trainings, {int(42 in seeds)} E0 evaluation; '
              f'seeds={seeds}; data hash, G64 reward pairs, 113 steps, no dev, clipping=0.')
        print('CPU preflight only; no models downloaded or GPU jobs started.')
        return 0
    import torch
    import deepspeed  # Fail once on a missing shared dependency, before queuing jobs.
    if torch.cuda.device_count() < args.gpus or not torch.cuda.is_bf16_supported():
        raise ValueError('Need eight visible CUDA GPUs with BF16 support')
    with seed_locks(directory, seeds):
        queue_dir.mkdir(parents=True, exist_ok=True)
        rows = {row['run_id']: row for row in matrix}
        jobs = build_jobs(matrix, args.skip_probe)
        def execute(job):
            print(f'START {job}; logs: {directory / job}', flush=True)
            if job == 'gradient_probe':
                return run_probe(args.config, directory, sources, data_hash)
            if job == 'gradient_probe_ll25':
                ll = rows[run_id('LL-Binary')]
                state = json.loads((directory / ll['run_id'] / 'state.json').read_text())
                if state['contract_sha256'] != fingerprint(contracts[ll['run_id']]) or not state['train_complete']:
                    raise ValueError('Step-25 diagnostic needs the matching newly completed LL-Binary training')
                return run_probe(args.config, directory, sources, data_hash,
                                 checkpoint=Path(ll['config']['output_dir']) / 'checkpoint-25')
            return execute_job(rows[job], contracts[job], directory / job)
        def record(event):
            print(json.dumps(event, ensure_ascii=False), flush=True)
            with (queue_dir / 'events.jsonl').open('a') as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + '\n')
            state_path = directory / event['job'] / 'state.json'
            if state_path.exists():
                state = json.loads(state_path.read_text())
                state['last_error'] = event['error']
                write_json(state_path, state)
            summarize(matrix, directory)
        failures = execute_queue(jobs, execute, record)
        print(f'Done; failed={failures}; summary={queue_dir / "summary.md"}', flush=True)
        return 1 if failures else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        raise SystemExit(2)
