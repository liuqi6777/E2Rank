#!/usr/bin/env python3
"""Reanalyze saved gradient moments without loading a model or changing raw JSONs.

The signal-square estimate is unbiased under iid fixed-state draws. Its square
root and the resulting noise/signal ratio are NOT unbiased, and may be unresolved.
No confidence interval can be recovered from the stored aggregate moments alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def corrected_moments(summary, draws):
    n = summary['draws']
    if n < 2 or n != len(draws):
        raise ValueError('Expected at least two draws and matching summary count')
    mean_norm = summary['mean_gradient_norm']
    variance = summary['noise_rms'] ** 2
    norm_squares = math.fsum(d['gradient_norm'] ** 2 for d in draws)
    reconstructed = max(0.0, (norm_squares - n * mean_norm ** 2) / (n - 1))
    if not math.isclose(variance, reconstructed, rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError('Stored variance disagrees with per-draw norms and mean norm')
    signal_square = mean_norm ** 2 - variance / n
    return dict(
        n=n, noise_variance=variance, noise_rms=summary['noise_rms'],
        sample_mean_norm=mean_norm, original_ratio=summary['noise_to_mean_ratio'],
        mean_noise_square=variance / n,
        signal_square_estimate=signal_square,
        signal_resolved=signal_square > 0,
        corrected_ratio=math.sqrt(variance / signal_square) if signal_square > 0 else None,
        pairwise_cosine=summary['mean_pairwise_cosine'],
    )


def batch_identity(probe):
    return [{k: v for k, v in batch.items() if k != 'tensor_sha256'} for batch in probe['batches']]


def comparison(label, left, right, allowed_config_changes=()):
    a, b = left['raw'], right['raw']
    metadata_keys = ('step', 'precision', 'training_seed', 'rollout_seeds',
                     'checkpoint_weights_sha256', 'model_commit_hash', 'data_sha256',
                     'torch_version', 'gradient_space', 'dropout', 'microbatch_size',
                     'microbatches_per_probe', 'parameters')
    mismatches = [key for key in metadata_keys if a.get(key) != b.get(key)]
    if len(a['probes']) != len(b['probes']):
        raise ValueError(f'{label}: mismatched probe counts')
    if any(batch_identity(x) != batch_identity(y) for x, y in zip(a['probes'], b['probes'])):
        mismatches.append('batch_identity')
    configs = [dict(d['config'], document_advantage_baseline=d['config'].get(
        'document_advantage_baseline', 'shared')) for d in (a, b)]
    ignored = {'run_name', 'output_dir', *allowed_config_changes}
    differences = {key: [configs[0].get(key), configs[1].get(key)]
                   for key in sorted(configs[0].keys() | configs[1].keys())
                   if configs[0].get(key) != configs[1].get(key)}
    unexpected = set(differences) - ignored
    if mismatches or unexpected:
        raise ValueError(f'{label}: metadata={mismatches}; unexpected config changes={unexpected}')
    tensor_hashes_equal = all(
        [v['tensor_sha256'] for v in x['batches']] == [v['tensor_sha256'] for v in y['batches']]
        for x, y in zip(a['probes'], b['probes']))
    if 'relevance_scheme' not in allowed_config_changes and not tensor_hashes_equal:
        raise ValueError(f'{label}: identical-label comparison has different batch tensors')
    reward_gap = max(abs(x['reward_mean'] - y['reward_mean'])
                     for p, q in zip(a['probes'], b['probes'])
                     for x, y in zip(p['draws'], q['draws']))
    rows = []
    for x, y in zip(left['probes'], right['probes']):
        rows.append(dict(probe=x['probe'], variance_ratio=y['noise_variance'] / x['noise_variance'],
                         noise_rms_ratio=y['noise_rms'] / x['noise_rms'],
                         corrected_ratio_left=x['corrected_ratio'],
                         corrected_ratio_right=y['corrected_ratio']))
    return dict(label=label, left=left['file'], right=right['file'],
                config_differences=differences, tensor_hashes_equal=tensor_hashes_equal,
                git_commits_equal=a['git_commit'] == b['git_commit'],
                either_git_dirty=a['git_dirty'] or b['git_dirty'],
                max_reward_mean_gap=reward_gap, probes=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path('outputs/rollout_gradients'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/rollout_gradient_reanalysis'))
    args = parser.parse_args()
    records = {}
    for path in sorted(args.input_dir.rglob('*.json')):
        raw = json.loads(path.read_text())
        if 'probes' not in raw or 'config' not in raw:
            continue
        if path.stem in records:
            raise ValueError(f'Duplicate input stem: {path.stem}')
        probes = []
        for i, probe in enumerate(raw['probes']):
            if [d['rollout_seed'] for d in probe['draws']] != raw['rollout_seeds']:
                raise ValueError(f'{path}: inconsistent draw seeds')
            if len(set(raw['rollout_seeds'])) != len(raw['rollout_seeds']):
                raise ValueError(f'{path}: repeated seeds')
            probes.append(dict(probe=i, sources=sorted({s for batch in probe['batches'] for s in batch['sources']}),
                               **corrected_moments(probe['summary'], probe['draws'])))
        records[path.stem] = dict(file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                 raw=raw, probes=probes)
    if not records:
        raise ValueError('No gradient diagnostic JSONs found')
    comparisons = []
    for state in ('e0', 'final'):
        pairs = [
            ('G16 → G32', f'{state}_g16', f'{state}_R0_mrr_shared', ('group_size',)),
            ('G32 → G64', f'{state}_R0_mrr_shared', f'{state}_g64', ('group_size',)),
            ('MRR shared → counterfactual', f'{state}_R0_mrr_shared', f'{state}_C0_mrr_counterfactual', ('document_advantage_baseline',)),
            ('graded shared → counterfactual', f'{state}_R2_ndcggraded_shared', f'{state}_C2_ndcggraded_counterfactual', ('document_advantage_baseline',)),
            ('MRR → binary nDCG', f'{state}_R0_mrr_shared', f'{state}_R1_ndcgbin_shared', ('reward_type',)),
            ('MRR → graded nDCG', f'{state}_R0_mrr_shared', f'{state}_R2_ndcggraded_shared', ('reward_type', 'relevance_scheme')),
        ]
        for label, left, right, allowed in pairs:
            if left in records and right in records:
                comparisons.append(comparison(f'{state}: {label}', records[left], records[right], allowed))
    batch128_pairs = [
        ('MRR → binary nDCG', 'e0_batch128', 'e0_R1_ndcgbin_shared_batch128', ('reward_type',)),
        ('MRR → graded nDCG', 'e0_batch128', 'e0_R2_ndcggraded_shared_batch128', ('reward_type', 'relevance_scheme')),
        ('MRR shared → counterfactual', 'e0_batch128', 'e0_C0_mrr_counterfactual_batch128', ('document_advantage_baseline',)),
        ('graded shared → counterfactual', 'e0_R2_ndcggraded_shared_batch128', 'e0_C2_ndcggraded_counterfactual_batch128', ('document_advantage_baseline',)),
    ]
    for label, left, right, allowed in batch128_pairs:
        if left in records and right in records:
            comparisons.append(comparison(f'e0 batch128: {label}', records[left], records[right], allowed))
    public_records = [{key: value for key, value in rec.items() if key != 'raw'} for rec in records.values()]
    result = dict(signal_square_formula='mean_gradient_norm**2 - noise_rms**2 / N',
                  caveat='corrected ratio is a noisy plug-in estimate, not an unbiased SNR or a confidence interval',
                  records=public_records, comparisons=comparisons)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    lines = ['# 原始梯度 JSON 逐 probe 重分析', '',
             '修正比值是 plug-in 估计；负信号平方保留，记为未分辨，不截断后求比值。', '',
             '| 文件 | probe | noise RMS | 信号平方估计 | 原比值 | 修正比值 |',
             '|---|---:|---:|---:|---:|---:|']
    for rec in public_records:
        for row in rec['probes']:
            corrected = '未分辨' if row['corrected_ratio'] is None else f"{row['corrected_ratio']:.3f}"
            lines.append(f"| {Path(rec['file']).stem} | {row['probe']} | {row['noise_rms']:.3f} | {row['signal_square_estimate']:.3f} | {row['original_ratio']:.3f} | {corrected} |")
    lines += ['', '## 配对条件的方差比（右 / 左）', '', '| 比较 | probe 0 | probe 1 | probe 2 |', '|---|---:|---:|---:|']
    for pair in comparisons:
        lines.append('| ' + pair['label'] + ' | ' + ' | '.join(f"{p['variance_ratio']:.4f}" for p in pair['probes']) + ' |')
    (args.output_dir / 'per_probe.md').write_text('\n'.join(lines) + '\n')
    unresolved = [(Path(r['file']).stem, p['probe']) for r in public_records for p in r['probes'] if not p['signal_resolved']]
    print(json.dumps(dict(files=len(records), probes=sum(len(r['probes']) for r in public_records),
                          paired_comparisons=len(comparisons), unresolved=unresolved,
                          output_dir=str(args.output_dir)), ensure_ascii=False))


if __name__ == '__main__':
    main()
