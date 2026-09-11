#!/usr/bin/env python3
"""One entry point: prepare/encode data, inspect experiments, check or train."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from experiments import iclr2027

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'encode', 'list', 'show', 'check', 'train'])
    parser.add_argument('run', nargs='?')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/experiments.yaml')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--verbose', action='store_true', help='Show the full resolved config')
    args = parser.parse_args()
    if args.action == 'prepare':
        if args.run not in (None, 'G1'):
            parser.error('Data preparation is currently implemented for G1 only')
        settings = iclr2027.read_mapping(args.config)
        data = settings.get('G1', {}).get('data')
        if not data:
            parser.error('Set G1.data in the experiment config')
        command = [sys.executable, str(ROOT/'scripts/prepare_reasonrank.py'), '--output-dir', str(ROOT/data)]
        if (ROOT/data/'train.ready.jsonl').exists():
            print(f'Training data already prepared: {ROOT/data}')
            return 0
        if (ROOT/data/'train.jsonl').exists():
            command.append('--compile-only')
        return subprocess.call(command, cwd=ROOT)
    if args.action == 'encode':
        if args.run not in (None, 'G1'):
            parser.error('Frozen document encoding is currently implemented for G1 only')
        if args.gpus < 1:
            parser.error('--gpus must be positive')
        suite = iclr2027.apply_settings(iclr2027.load_suite(), args.config)
        resolved = iclr2027.resolve_run(
            suite, iclr2027.DEFAULT_SUITE, 'G1-J-CL', nproc=args.gpus
        )
        config = resolved['config']
        data_path = Path(config['data_path'])
        output_dir = data_path.parent / 'frozen_document_index'
        launcher = (
            [sys.executable]
            if args.gpus == 1
            else ['torchrun', '--standalone', f'--nproc_per_node={args.gpus}']
        )
        command = [
            *launcher, str(ROOT/'src/encode_frozen_documents.py'),
            '--input', str(ROOT/data_path),
            '--output-dir', str(ROOT/output_dir),
            '--model', str(config['model_name_or_path']),
            '--max-length', str(min(config['d_max_len'], config['embedding_max_length'])),
            '--pooling-method', str(config['pooling_method']),
            '--padding-side', str(config['padding_side']),
            '--append-token', str(config['append_token']),
            '--document-prompt-template', str(config['document_prompt_template']),
            '--query-prompt-template', str(config['query_prompt_template']),
            '--embedding-max-length', str(config['embedding_max_length']),
        ]
        return subprocess.call(command, cwd=ROOT)
    if args.action == 'show' and not args.verbose:
        suite = iclr2027.apply_settings(iclr2027.load_suite(), args.config)
        if args.run not in suite['runs']:
            parser.error('show requires an explicit run ID, e.g. G1-J-RL')
        resolved = iclr2027.resolve_run(suite, iclr2027.DEFAULT_SUITE, args.run, nproc=args.gpus)
        config = resolved['config']
        summary = {key: resolved[key] for key in ['run_id', 'scope', 'objective', 'selection']}
        summary.update(model=config.get('model_name_or_path'), data=config.get('data_path', config.get('rag_dataset_root')),
                       steps=resolved['protocol'].get('max_steps'), learning_rate=resolved['protocol'].get('learning_rate'),
                       micro_batch_size=config.get('per_device_train_batch_size'),
                       gradient_accumulation=config.get('gradient_accumulation_steps'),
                       global_batch_size=config.get('per_device_train_batch_size', 0)*config.get('gradient_accumulation_steps', 1)*args.gpus,
                       blockers=iclr2027.blockers(suite, resolved))
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    action = {'show':'resolve', 'train':'launch'}.get(args.action, args.action)
    argv = [action]
    if args.run in {'G1','G2','G3'} and action in {'list','check'}:
        argv += ['--group', args.run]
    elif args.run:
        argv.append(args.run)
    argv += ['--settings', str(args.config), '--nproc', str(args.gpus)]
    return iclr2027.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
