#!/usr/bin/env python3
"""Run the 21 new G1 stability jobs; finish BRIGHT after every training run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

from experiments import iclr2027 as experiments

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 3407, 2026)
# Ordered by complete recipe, so a partial night still yields multi-seed groups.
RECIPES = (
    ('MRR32', 'mrr_in_batch', 'binary', 32, 5e-6),
    ('GradedNDCG64', 'ndcg_in_batch', 'graded', 64, 5e-6),
    ('MRR64', 'mrr_in_batch', 'binary', 64, 5e-6),
    ('GradedNDCG32', 'ndcg_in_batch', 'graded', 32, 5e-6),
    ('BinaryNDCG32', 'ndcg_in_batch', 'binary', 32, 5e-6),
    ('GradedNDCG64LRHalf', 'ndcg_in_batch', 'graded', 64, 2.5e-6),
    ('MRR64LRHalf', 'mrr_in_batch', 'binary', 64, 2.5e-6),
)
SUBSETS = ('biology', 'earth_science', 'economics', 'psychology', 'robotics',
           'stackoverflow', 'sustainable_living', 'pony', 'leetcode', 'aops',
           'theoremqa_theorems', 'theoremqa_questions')


def run_id(recipe, seed):
    return f'G1-S-{recipe}-Seed{seed}'


def resolve_matrix(suite, gpus=8):
    matrix = []
    for recipe, reward, labels, group, lr in RECIPES:
        for seed in SEEDS:
            name = run_id(recipe, seed)
            row = experiments.resolve_run(suite, experiments.DEFAULT_SUITE, name, nproc=gpus)
            cfg = row['config']
            expected = dict(seed=seed, data_seed=seed, rollout_seed=seed, reward_type=reward,
                            relevance_scheme=labels, group_size=group, learning_rate=lr,
                            max_steps=113, target_alignment=.9, final_alignment=None,
                            exploration_schedule='fixed', advantage_baseline='leave_one_out',
                            advantage_norm='none', document_log_prob_reduction='sum',
                            rollout='product', sampling_law='vmf', frozen_doc_rescale=True,
                            in_batch_use_sampled_documents=False, sigma_learnable=False,
                            kl_coef=0, lora_enabled=False, document_encoder_mode='joint',
                            action_components=[['query'], ['positive', 'negative']],
                            model_name_or_path='Qwen/Qwen3-Embedding-0.6B',
                            per_device_train_batch_size=16)
            mismatches = {key: (cfg.get(key), value) for key, value in expected.items() if cfg.get(key) != value}
            if cfg.get('document_advantage_baseline', 'shared') != 'shared':
                mismatches['document_advantage_baseline'] = 'must be shared'
            if cfg['per_device_train_batch_size'] * cfg['gradient_accumulation_steps'] * gpus != 128:
                mismatches['global_batch'] = 'must be 128'
            if row['protocol']['learning_rate'] != lr or row['dependency'] is not None:
                mismatches['protocol'] = 'must match recipe LR and start fresh from E0'
            if mismatches:
                raise ValueError(f'{name}: stability contract mismatch: {mismatches}')
            matrix.append(dict(recipe=recipe, seed=seed, resolved=row))
    return matrix


def read_bright(output_dir):
    """Return complete 12-subset scores in percentage points, never a partial mean."""
    paths = list((Path(output_dir) / 'mteb_eval' / 'bright').rglob('BrightRetrieval.json'))
    if len(paths) > 1:
        raise ValueError(f'Ambiguous final BRIGHT results: {paths}')
    if paths:
        payload = json.loads(paths[0].read_text())
        entries = payload.get('scores', {}).get('standard', [])
        scores = {}
        for entry in entries:
            subset = entry.get('hf_subset')
            if subset in scores:
                raise ValueError(f'Duplicate BRIGHT subset: {subset}')
            if subset in SUBSETS:
                value = float(entry['main_score'])
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f'Invalid BRIGHT score: {subset}={value}')
                scores[subset] = value * 100
        if set(scores) != set(SUBSETS):
            raise ValueError(f'Incomplete BRIGHT result: {paths[0]} ({len(scores)}/12 subsets)')
        return scores, str(paths[0])
    return None, None


def summarize(matrix, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, groups = [], []
    for cell in matrix:
        name = cell['resolved']['run_id']
        try:
            scores, source = read_bright(cell['resolved']['config']['output_dir'])
            error = None
        except (ValueError, TypeError, KeyError, OSError) as exc:
            scores, source, error = None, None, str(exc)
        rows.append(dict(run=name, recipe=cell['recipe'], seed=cell['seed'],
                         scores=scores, mean=statistics.mean(scores.values()) if scores else None,
                         source=source, error=error))
    for recipe, *_ in RECIPES:
        values = [r['mean'] for r in rows if r['recipe'] == recipe and r['mean'] is not None]
        complete = len(values) == 3
        groups.append(dict(recipe=recipe, completed=len(values),
                           mean=statistics.mean(values) if complete else None,
                           sample_std=statistics.stdev(values) if complete else None,
                           worst=min(values) if complete else None, best=max(values) if complete else None))
    report = dict(score_units='BRIGHT nDCG@10 percentage points', seeds=SEEDS, groups=groups, runs=rows)
    (output_dir / 'results.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    def fmt(value):
        return '—' if value is None else f'{value:.3f}'
    lines = ['# G1 stability batch', '', 'Only complete three-seed recipes receive aggregate statistics.', '',
             '| Recipe | Complete | Mean | Sample SD | Worst | Best |', '|---|---:|---:|---:|---:|---:|']
    for group in groups:
        lines.append(f"| {group['recipe']} | {group['completed']}/3 | " + ' | '.join(fmt(group[k]) for k in ('mean','sample_std','worst','best')) + ' |')
    lines += ['', '| Run | Seed | Mean | ' + ' | '.join(SUBSETS) + ' |', '|---|---:|---:|' + '---:|' * len(SUBSETS)]
    for row in rows:
        lines.append(f"| {row['run']} | {row['seed']} | {fmt(row['mean'])} | " + ' | '.join(fmt((row['scores'] or {}).get(s)) for s in SUBSETS) + ' |')
    lines += ['', '## Sources / incomplete results', '']
    lines += [f"- {r['run']}: {r['error'] or r['source'] or 'not yet available'}" for r in rows]
    (output_dir / 'results.md').write_text('\n'.join(lines) + '\n')
    return report


def execute_queue(queue, invoke, record):
    """Attempt every selected job; never change recipes or retry failed outputs."""
    for position, cell in queue:
        started = time.monotonic()
        code = invoke(position, cell)
        record(dict(position=position, run=cell['resolved']['run_id'],
                    status='complete' if code == 0 else 'failed', returncode=code,
                    seconds=time.monotonic() - started))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'train', 'summary'])
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/experiments.yaml')
    parser.add_argument('--gpus', type=int, choices=[8], default=8)
    parser.add_argument('--start-at', type=int, default=1, help='Resume queue at this 1-based NEW-job position; never resumes model state')
    parser.add_argument('--log-dir', type=Path, default=ROOT / 'outputs/g1_stability_night')
    args = parser.parse_args()
    if not 1 <= args.start_at <= 21:
        parser.error('start-at must be in 1..21')
    suite = experiments.apply_settings(experiments.load_suite(), args.config)
    matrix = resolve_matrix(suite, args.gpus)
    summary_dir = args.log_dir / 'summary'
    if args.action == 'summary':
        summarize(matrix, summary_dir)
        print(summary_dir / 'results.md')
        return 0
    jobs = matrix
    queue = [(i, cell) for i, cell in enumerate(jobs, 1) if i >= args.start_at]
    data_path = experiments.path_at_root(matrix[0]['resolved']['config']['data_path'], ROOT)
    errors = []
    for position, cell in queue:
        problems = experiments.blockers(suite, cell['resolved'])
        print(f"{position:02d}/21 {cell['resolved']['run_id']}: {'BLOCKED' if problems else 'READY'}", flush=True)
        errors.extend(f"{cell['resolved']['run_id']}: {p}" for p in problems)
    if errors:
        raise ValueError('\n'.join(errors))
    if args.action == 'check':
        return 0
    # Fail once, before queuing 21 doomed launches on the wrong environment.
    subprocess.run([sys.executable, '-c', 'import torch; assert torch.cuda.device_count() >= 8, "8 visible CUDA devices required"'], check=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    session = args.log_dir / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    session.mkdir(exist_ok=False)
    (session / 'plan.json').write_text(json.dumps(dict(gpus=args.gpus,
        data_sha256=experiments.digest(data_path), matrix=matrix,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        git_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip())), indent=2) + '\n')
    failures = []
    def record(event):
        print(json.dumps(event), flush=True)
        with (session / 'status.jsonl').open('a') as handle:
            handle.write(json.dumps(event) + '\n')
        if event['status'] == 'failed':
            failures.append(event['run'])
    def invoke(position, cell):
        name = cell['resolved']['run_id']
        log = session / f'{position:02d}_{name}.log'
        print(f'START {position:02d}/21 {name}; log={log}', flush=True)
        command = [sys.executable, str(ROOT / 'scripts/experiment.py'), 'train', name,
                   '--gpus', str(args.gpus), '--config', str(args.config.resolve())]
        with log.open('w') as handle:
            code = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT).returncode
            if code == 0:
                try:
                    scores, _ = read_bright(cell['resolved']['config']['output_dir'])
                    if scores is None:
                        raise ValueError('Training command returned success without final BRIGHT results')
                except (ValueError, TypeError, KeyError, OSError) as exc:
                    handle.write(f'\nBatch validation failed: {exc}\n')
                    code = 2
        summarize(matrix, summary_dir)
        return code
    execute_queue(queue, invoke, record)
    summarize(matrix, summary_dir)
    print(f'Results: {summary_dir / "results.md"}; failed={failures}', flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        raise SystemExit(2)
