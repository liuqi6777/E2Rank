#!/usr/bin/env python3
"""Audit and summarize the completed R2/DIVER CSV exports without model access."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics as stats

PAPER = Path(__file__).resolve().parents[1] / 'paper'
OUTPUT = PAPER / 'g1_r2_results'
SEEDS = (42, 3407, 2026)
DOMAINS = ('biology', 'earth_science', 'economics', 'psychology', 'robotics',
           'stackoverflow', 'sustainable_living', 'pony', 'leetcode', 'aops',
           'theoremqa_theorems', 'theoremqa_questions')
FAMILIES = {'qwen': ('g1_r2_bright', 'G1-R2-'),
            'diver': ('g1_r2_bright_diver', 'G1-Reasoning-diver-')}


def write_csv(name, rows):
    with (OUTPUT / name).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def load(family, directory, prefix):
    root = PAPER / '_summary' / directory
    paths = [root / 'run_summary.csv', root / 'subset_summary.csv']
    with paths[0].open(newline='') as handle:
        runs = list(csv.DictReader(handle))
    with paths[1].open(newline='') as handle:
        subsets = list(csv.DictReader(handle))
    by_run, groups = {}, {}
    for row in runs:
        run = row['run']
        if run in by_run or not run.startswith(prefix):
            raise ValueError(f'Duplicate/unexpected run: {run}')
        if any(int(row[k]) != expected for k, expected in
               [('errors', 0), ('tasks_missing', 0), ('tasks_found', 1), ('types_found', 1)]):
            raise ValueError(f'Incomplete evaluation: {run}')
        if 'tokens-v2__pool-fp32' not in row['result_dir']:
            raise ValueError(f'Unexpected embedding protocol: {run}')
        stem = re.sub(r'-s\d+$', '', run.removeprefix(prefix))
        method = re.sub(r'-Seed\d+$', '', stem)
        suffix = re.search(r'-s(\d+)$', run)
        seed = int(suffix[1]) if suffix else None
        if method != 'E0' and seed not in SEEDS:
            raise ValueError(f'Unexpected seed: {run}')
        score = float(row['mean_task_score'])
        if not math.isfinite(score):
            raise ValueError(f'Nonfinite score: {run}')
        record = dict(family=family, method=method, seed=seed, score=score,
                      run=run, domains={}, result_dir=row['result_dir'])
        by_run[run] = record
        groups.setdefault(method, []).append(record)
    for row in subsets:
        record = by_run[row['run']]
        domain = row['subset']
        value = float(row['score'])
        if (row['task'] != 'BrightRetrieval' or domain not in DOMAINS or
                domain in record['domains'] or not math.isfinite(value)):
            raise ValueError(f'Invalid/duplicate subset: {row}')
        record['domains'][domain] = value
    max_rounding_gap = 0.0
    for record in by_run.values():
        if set(record['domains']) != set(DOMAINS):
            raise ValueError(f'Missing subsets: {record["run"]}')
        gap = abs(record['score'] - stats.mean(record['domains'].values()))
        max_rounding_gap = max(max_rounding_gap, gap)
        if gap > 0.0100001:
            raise ValueError(f'Inconsistent macro average: {record["run"]}: {gap}')
    for method, members in groups.items():
        if method == 'E0':
            if len(members) != 1:
                raise ValueError('E0 must be evaluated once')
        elif len(members) != 3 or {r['seed'] for r in members} != set(SEEDS):
            raise ValueError(f'Incomplete/duplicate seed group: {family}/{method}')
    return groups, dict(runs=len(runs), trained_runs=len(runs) - 1,
                        methods=len(groups) - 1, subset_rows=len(subsets),
                        max_macro_rounding_gap=max_rounding_gap,
                        input_sha256={str(p.relative_to(PAPER)):
                                      hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def main():
    OUTPUT.mkdir(exist_ok=True)
    families, audits, summaries, domains = {}, {}, [], []
    for family, (directory, prefix) in FAMILIES.items():
        groups, audit = load(family, directory, prefix)
        families[family], audits[family] = groups, audit
        for method, members in groups.items():
            values = [r['score'] for r in members]
            by_seed = {r['seed']: r['score'] for r in members}
            summaries.append(dict(family=family, method=method, n=len(values),
                                  seed42=by_seed.get(42) if method != 'E0' else None,
                                  seed3407=by_seed.get(3407), seed2026=by_seed.get(2026),
                                  mean=stats.mean(values),
                                  sample_sd=stats.stdev(values) if len(values) > 1 else None,
                                  worst=min(values),
                                  delta_e0=stats.mean(values) - groups['E0'][0]['score']))
            domains.append(dict(family=family, method=method,
                                **{d: stats.mean(r['domains'][d] for r in members) for d in DOMAINS}))
    pairs = []
    sf, cp = 'RL-GradedNDCG64-SF', 'RL-GradedNDCG64-CP'
    comparisons = [(cp + suffix, sf + suffix) for suffix in
                   ('', '-Align040', '-Align053', '-Align065', '-Align080',
                    '-Align095', '-Align098', '-Anneal080090', '-NoRescale')]
    comparisons += [('RL-MRR64-CP', 'RL-MRR64-SF'),
                    ('RL-BinaryNDCG64-CP', 'RL-BinaryNDCG64-SF'),
                    ('RL-GradedNDCG32-CP', 'RL-GradedNDCG32-SF'),
                    ('RL-GradedNDCG32-CP', cp), ('RL-GradedNDCG32-SF', sf),
                    (cp, 'LL-Graded'), ('RL-GradedNDCG32-CP', 'LL-Graded')]
    comparisons += [(cp + '-Align080', baseline) for baseline in ('E0', 'LL-Graded', 'CL')]
    comparisons += [(sf + suffix, sf) for suffix in
                    ('-Paired', '-QPolicy', '-DPolicy', '-Norm', '-DocMean',
                     '-NormDocMean', '-NoRescale', '-Anneal080090')]
    comparisons += [(cp + suffix, cp) for suffix in ('-NoRescale', '-Anneal080090', '-Align080')]
    comparisons += [(sf + '-GaussianMismatch755', sf + '-Kappa755'),
                    ('DR-GradedNDCG64-SF', 'E0'),
                    ('DR-GradedNDCG64-SF', sf + '-QPolicy')]
    for family, tests in [('qwen', comparisons),
                          ('diver', [('RL-Graded', 'E0'), ('LL-Graded', 'E0'), ('CL', 'E0'),
                                     ('RL-Graded', 'LL-Graded'), ('RL-Graded', 'CL')])]:
        groups = families[family]
        for left, right in tests:
            a = {r['seed']: r['score'] for r in groups[left]}
            b = {r['seed']: r['score'] for r in groups[right]}
            differences = [a[s] - (groups[right][0]['score'] if right == 'E0' else b[s]) for s in SEEDS]
            domain_diffs = [stats.mean(r['domains'][d] for r in groups[left]) -
                            stats.mean(r['domains'][d] for r in groups[right]) for d in DOMAINS]
            pairs.append(dict(family=family, left=left, right=right,
                              seed42=differences[0], seed3407=differences[1], seed2026=differences[2],
                              mean_delta=stats.mean(differences), sample_sd_delta=stats.stdev(differences),
                              positive_seeds=sum(x > 0 for x in differences),
                              positive_domains=sum(x > 0 for x in domain_diffs)))
    write_csv('method_summary.csv', summaries)
    write_csv('domain_means.csv', domains)
    write_csv('paired_deltas.csv', pairs)
    (OUTPUT / 'audit.json').write_text(json.dumps(audits, ensure_ascii=False, indent=2) + '\n')
    lines = ['# G1-R2 与 DIVER 完整结果附表', '',
             '由 `python scripts/analyze_g1_r2_results.py` 从现有 CSV 生成。', '',
             '分数采用 run_summary；SD 为三训练 seed 样本标准差（ddof=1）。E0 仅评测一次。', '']
    for family in FAMILIES:
        lines += [f'## {family}：方法汇总', '',
                  '| 方法 | 42 | 3407 | 2026 | 均值 ± SD | 最差 seed | 相对自身 E0 |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for r in sorted((r for r in summaries if r['family'] == family), key=lambda r: -r['mean']):
            seeds = ['—' if r[k] is None else f'{r[k]:.2f}' for k in ('seed42', 'seed3407', 'seed2026')]
            mean = f'{r["mean"]:.2f}' + (f' ± {r["sample_sd"]:.2f}' if r['sample_sd'] is not None else '')
            worst = f'{r["worst"]:.2f}' if r['n'] > 1 else '—'
            delta = f'{r["delta_e0"]:+.2f}' if r['method'] != 'E0' else '—'
            lines.append(f'| {r["method"]} | ' + ' | '.join(seeds) +
                         f' | {mean} | {worst} | {delta} |')
        lines += ['', f'## {family}：逐 run 领域分数', '',
                  '| Run | 宏平均 | ' + ' | '.join(DOMAINS) + ' |',
                  '|---|---:|' + '---:|' * len(DOMAINS)]
        for method in sorted(families[family]):
            for r in sorted(families[family][method], key=lambda r: r['seed'] or 0):
                lines.append(f'| {r["run"]} | {r["score"]:.2f} | ' +
                             ' | '.join(f'{r["domains"][d]:.2f}' for d in DOMAINS) + ' |')
        lines.append('')
    (OUTPUT / 'all_runs.md').write_text('\n'.join(lines))
    print(json.dumps(audits, ensure_ascii=False, indent=2))
    print(f'Wrote supporting tables to {OUTPUT}')


if __name__ == '__main__':
    main()
