#!/usr/bin/env python3
"""Manage the 0.6B K=0 paper suite and import completed R2 runs.

Actions: check, status, import, train, eval. Import copies only complete old
runs after exact resolved-config comparison; it never writes into the old root.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from experiments import iclr2027 as experiments
from run_g1_r2 import read_bright, validate_final_model

ROOT = experiments.ROOT
SUITE = ROOT / 'configs/experiments/iclr2027/suite_iclr2027_final_k0.yaml'
SETTINGS = ROOT / 'configs/experiments_iclr2027_final.yaml'
SOURCE_SETTINGS = ROOT / 'configs/experiments_r2.yaml'
SEEDS = (42, 3407, 2026)
BASE = 'G1-R2-RL-GradedNDCG64-CP-Align070-ShortlistUniform-K0-T1'
RECIPES = {
    'main': BASE + '-Pairwise050',
    'no_cmp': BASE.replace('-CP-', '-SF-') + '-Pairwise050',
    'lambda1': BASE + '-Pairwise100',
    'binary_ndcg': BASE + '-BinaryNDCG',
    'binary_mrr': BASE + '-BinaryMRR',
    'rloo_product': BASE.replace('-CP-', '-SF-'),
    'paired': 'G1-R2-RL-GradedNDCG64-SF-Align070-OwnCandidates-Paired',
    'query_only': 'G1-R2-RL-GradedNDCG64-SF-Align070-OwnCandidates-QueryOnly',
    'doc_only': 'G1-R2-RL-GradedNDCG64-SF-Align070-OwnCandidates-DocOnly',
    'k1_pairwise': BASE.replace('K0', 'K1') + '-Pairwise050',
    'align060': BASE.replace('Align070', 'Align060') + '-Pairwise050',
    'align080': BASE.replace('Align070', 'Align080') + '-Pairwise050',
    'align090': BASE.replace('Align070', 'Align090') + '-Pairwise050',
}


def run_name(stem: str, seed: int) -> str:
    return stem + (f'-Seed{seed}' if seed != 42 else '')


def resolved_suite(settings: Path):
    suite = experiments.apply_settings(experiments.load_suite(SUITE), settings)
    if len(suite['runs']) != 67 or len(suite['imports']) != 28:
        raise ValueError('Final suite must contain 28 imports and 39 new trainings')
    expected_new = {run_name(stem, seed) for stem in RECIPES.values() for seed in SEEDS}
    if set(suite['runs']) - set(suite['imports']) != expected_new:
        raise ValueError('Final suite recipes differ from the 13 x 3 plan')
    return suite


def old_row(import_entry: dict, source_settings: Path, source_root: Path | None):
    path = ROOT / import_entry['suite']
    suite = experiments.apply_settings(experiments.load_suite(path), source_settings)
    if source_root is not None:
        suite['output_root'] = str(source_root)
    row = experiments.resolve_run(suite, path, import_entry['source_run'], nproc=8)
    if directory := import_entry.get('source_directory'):
        if Path(directory).name != directory:
            raise ValueError(f'Invalid source directory name: {directory}')
        root = experiments.path_at_root(suite['output_root'])
        row['config']['output_dir'] = str(root / directory)
    return row


def ensure_same_training_config(new: dict, old: dict):
    ignored = {'output_dir', 'run_name'}
    a, b = new['config'], old['config']
    mismatch = {key: (a.get(key), b.get(key)) for key in set(a) | set(b)
                if key not in ignored and a.get(key) != b.get(key)}
    if new['kind'] != old['kind'] or new['objective'] != old['objective'] or mismatch:
        raise ValueError(f'{new["run_id"]}: legacy configuration differs: {mismatch}')


def validate_complete(row: dict):
    out = Path(row['config']['output_dir'])
    if row['kind'] == 'train':
        validate_final_model(row)
    scores, source = read_bright(out)
    if scores is None:
        raise ValueError(f'Missing complete 12-subset BRIGHT evaluation: {out}')
    return scores, source


def import_one(new: dict, old: dict, *, dry_run: bool = False):
    ensure_same_training_config(new, old)
    source = Path(old['config']['output_dir'])
    target = Path(new['config']['output_dir'])
    receipt_path = target.parent / '.imports' / f'{target.name}.json'
    if target.exists():
        validate_complete(new)
        if not receipt_path.is_file():
            raise ValueError(f'Complete target has no import receipt: {target}')
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get('source') != str(source) or receipt.get('destination') != str(target)
                or receipt.get('source_run') != old['run_id']):
            raise ValueError(f'Import receipt does not match this source: {receipt_path}')
        return 'already_complete'
    if not source.is_dir():
        return 'missing_source'
    source_scores, _ = validate_complete(old)
    if dry_run:
        return 'ready'
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{target.name}.import-', dir=target.parent) as temporary:
        staged = Path(temporary) / target.name
        shutil.copytree(source, staged, symlinks=False)
        staged_row = dict(new, config=dict(new['config'], output_dir=str(staged)))
        staged_scores, _ = validate_complete(staged_row)
        if staged_scores != source_scores:
            raise ValueError(f'Copied BRIGHT scores differ from source: {source}')
        if target.exists():
            raise ValueError(f'Target appeared during import: {target}')
        staged.rename(target)
    validate_complete(new)
    receipt = {
        'run': new['run_id'], 'source': str(source), 'destination': str(target),
        'source_run': old['run_id'],
        'copied_at_utc': datetime.now(timezone.utc).isoformat(),
        'bright_subsets': 12,
    }
    receipts = target.parent / '.imports'
    receipts.mkdir(exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    return 'copied'


def selected_new(suite: dict, recipes: list[str], seeds: list[int]):
    names = [run_name(RECIPES[recipe], seed) for recipe in recipes for seed in seeds]
    if len(names) != len(set(names)) or any(name not in suite['runs'] or name in suite['imports'] for name in names):
        raise ValueError('Invalid new-run selection')
    return names


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'status', 'import', 'train', 'eval'))
    parser.add_argument('--config', type=Path, default=SETTINGS, help='New-suite settings and output root')
    parser.add_argument('--source-config', type=Path, default=SOURCE_SETTINGS)
    parser.add_argument('--source-root', type=Path, help='Override the old checkpoint root for import')
    parser.add_argument('--recipes', nargs='+', choices=tuple(RECIPES), default=list(RECIPES))
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument('--dry-run', action='store_true', help='For import: verify without copying')
    args = parser.parse_args(argv)
    if len(set(args.recipes)) != len(args.recipes) or len(set(args.seeds)) != len(args.seeds):
        parser.error('Select distinct recipes and seeds')
    if args.dry_run and args.action != 'import':
        parser.error('--dry-run applies only to import')
    suite = resolved_suite(args.config)
    names = selected_new(suite, args.recipes, args.seeds)
    rows = {name: experiments.resolve_run(suite, SUITE, name, nproc=8)
            for name in suite['runs']}
    if args.action in {'check', 'status'}:
        imported, new_complete = 0, 0
        for name, row in rows.items():
            try:
                validate_complete(row)
            except (ValueError, OSError, KeyError, TypeError):
                continue
            if name in suite['imports']:
                out = Path(row['config']['output_dir'])
                imported += (out.parent / '.imports' / f'{out.name}.json').is_file()
            else:
                new_complete += 1
        print(f'Configured: {len(suite["imports"])} legacy imports, {len(names)} selected new trainings; '
              f'complete in new root: {imported} imports, {new_complete} new trainings')
        if args.action == 'check':
            for name, entry in suite['imports'].items():
                old = old_row(entry, args.source_config, args.source_root)
                ensure_same_training_config(rows[name], old)
            print('All legacy run configurations match the new suite (output path excluded).')
        return 0
    if args.action == 'import':
        counts = {'copied': 0, 'already_complete': 0, 'ready': 0, 'missing_source': 0}
        for name, entry in suite['imports'].items():
            old = old_row(entry, args.source_config, args.source_root)
            result = import_one(rows[name], old, dry_run=args.dry_run)
            counts[result] += 1
            print(f'{name}: {result}', flush=True)
        print(json.dumps(counts, sort_keys=True))
        return 0 if args.dry_run or counts['missing_source'] == 0 else 1
    if args.action == 'train':
        import torch
        if torch.cuda.device_count() != 8 or not torch.cuda.is_bf16_supported():
            raise ValueError('Training needs exactly eight visible BF16 CUDA GPUs')
    for name in names:
        row = rows[name]
        out = Path(row['config']['output_dir'])
        if args.action == 'train':
            if out.exists():
                validate_complete(row)
                print(f'{name}: already complete; skipping', flush=True)
                continue
            print(f'{name}: training -> {out}', flush=True)
            experiments.run_simple_baselines(SUITE, args.config, [name], 'train')
        else:
            print(f'{name}: evaluating -> {out}', flush=True)
            experiments.run_simple_baselines(SUITE, args.config, [name], 'eval')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        raise SystemExit(2)
