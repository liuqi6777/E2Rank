#!/usr/bin/env python3
"""Run Qwen3-Embedding-4B LoRA baselines, ablations, and the paper K=0 recipe."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from experiments import iclr2027

ROOT = iclr2027.ROOT
SUITE = ROOT / 'configs/experiments/iclr2027/suite_qwen3_embedding_4b_lora.yaml'
SETTINGS = ROOT / 'configs/experiments_qwen3_embedding_4b_lora.yaml'
SEEDS = (42, 3407, 2026)
MAIN_METHODS = ('InfoNCE', 'LambdaLoss', 'RELER')
ABLATION_METHODS = ('RELER-NoPairwise-CP', 'RELER-NoPairwise-RLOO', 'RELER-NoRescale')


def run_id(method: str, seed: int) -> str:
    return f'Q4B-LORA-{method}' + (f'-Seed{seed}' if seed != 42 else '')


def selected_runs(experiment_set: str, seeds: list[int], action: str) -> list[str]:
    if experiment_set == 'e0':
        if action == 'train':
            raise ValueError('E0 is evaluation-only; use action=eval')
        return ['Q4B-LORA-E0']
    methods = {
        'pilot': MAIN_METHODS + ABLATION_METHODS,
        'main': MAIN_METHODS,
        'ablations': ABLATION_METHODS,
        'all': MAIN_METHODS + ABLATION_METHODS,
        'k0': ('RELER-K0',),
    }[experiment_set]
    selected_seeds = (42,) if experiment_set == 'pilot' else tuple(dict.fromkeys(seeds))
    runs = [run_id(method, seed) for method in methods for seed in selected_seeds]
    if action == 'eval' and experiment_set == 'all':
        runs.insert(0, 'Q4B-LORA-E0')
    return runs


def validate_contracts(run_ids: list[str], settings: Path) -> None:
    suite = iclr2027.apply_settings(iclr2027.load_suite(SUITE), settings)
    for name in run_ids:
        row = iclr2027.resolve_run(suite, SUITE, name, nproc=8)
        config = row['config']
        if config['model_name_or_path'] != 'Qwen/Qwen3-Embedding-4B':
            raise ValueError(f'{name}: unexpected model {config["model_name_or_path"]}')
        if config.get('model_revision') != '5cf2132abc99cad020ac570b19d031efec650f2b':
            raise ValueError(f'{name}: Qwen3-Embedding-4B revision is not pinned')
        if row['kind'] == 'train':
            expected = {
                'lora_enabled': True,
                'lora_r': 16,
                'lora_alpha': 32,
                'max_steps': 113,
                'per_device_train_batch_size': 8,
                'gradient_accumulation_steps': 2,
            }
            mismatch = {key: (config.get(key), value) for key, value in expected.items()
                        if config.get(key) != value}
            if mismatch:
                raise ValueError(f'{name}: incompatible 4B LoRA contract: {mismatch}')
            global_batch = (
                config['per_device_train_batch_size']
                * config['gradient_accumulation_steps']
                * 8
            )
            if global_batch != 128:
                raise ValueError(f'{name}: global batch size is not 128')
            if name.startswith('Q4B-LORA-RELER-K0'):
                k0_expected = {
                    'learning_rate': 0.0002,
                    'gradient_estimator': 'conditional_projection',
                    'group_size': 64,
                    'target_alignment': 0.70,
                    'reward_shortlist_count': 1,
                    'reward_shortlist_size': 0,
                    'reward_shortlist_pairwise_coef': 0.5,
                }
                mismatch = {
                    key: (config.get(key), value)
                    for key, value in k0_expected.items()
                    if config.get(key) != value
                }
                if mismatch:
                    raise ValueError(
                        f'{name}: incompatible K=0 paper recipe: {mismatch}'
                    )
                old_name = run_id('RELER', config['seed'])
                old_config = iclr2027.resolve_run(
                    suite, SUITE, old_name, nproc=8
                )['config']
                changed = {
                    key for key in set(config) | set(old_config)
                    if config.get(key) != old_config.get(key)
                } - {'output_dir', 'run_name'}
                if changed != {'learning_rate', 'reward_shortlist_size'}:
                    raise ValueError(
                        f'{name}: K=0 differs from {old_name} in {sorted(changed)}'
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', default='check', choices=['check', 'train', 'eval'])
    parser.add_argument(
        '--set', dest='experiment_set',
        choices=['pilot', 'main', 'ablations', 'all', 'e0', 'k0'],
        default='pilot',
        help='k0 runs the three-seed K=0 + pairwise paper recipe',
    )
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument('--gpus', type=int, choices=[8], default=8)
    parser.add_argument('--config', type=Path, default=SETTINGS)
    args = parser.parse_args()

    runs = selected_runs(args.experiment_set, args.seeds, args.action)
    validate_contracts(runs, args.config)
    if args.action == 'train':
        import torch
        if torch.cuda.device_count() != 8 or not torch.cuda.is_bf16_supported():
            raise ValueError('Expose exactly eight CUDA GPUs with BF16 support (CUDA_VISIBLE_DEVICES)')
    iclr2027.run_simple_baselines(
        SUITE, args.config, runs, args.action, nproc=args.gpus
    )


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        raise SystemExit(2)
