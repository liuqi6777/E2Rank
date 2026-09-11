#!/usr/bin/env python3
"""Resolve and preflight the three-group ICLR plan without importing torch.

Requires PyYAML. No model download, GPU allocation, or implicit dependency runs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUITE = ROOT / 'configs/experiments/iclr2027/suite.yaml'


def merge(left, right):
    result = copy.deepcopy(left)
    for key, value in right.items():
        result[key] = (merge(result[key], value)
                       if isinstance(result.get(key), dict) and isinstance(value, dict)
                       else copy.deepcopy(value))
    return result


def read_mapping(path):
    content = Path(path).read_text()
    value = json.loads(content) if Path(path).suffix == '.json' else yaml.safe_load(content)
    if not isinstance(value, dict):
        raise ValueError(f'Expected mapping: {path}')
    return value


def resolve_config(path, stack=()):
    """Same recursive _base_ semantics as src/utils; ignores model slot hint."""
    path = Path(path).resolve()
    if path in stack:
        raise ValueError(f'Config inheritance cycle: {path}')
    config = read_mapping(path)
    bases = config.pop('_base_', [])
    config.pop('_default_train_', None)
    if isinstance(bases, str):
        bases = [bases]
    result = {}
    for base in bases:
        result = merge(result, resolve_config(path.parent / base, (*stack, path)))
    return merge(result, config)


def load_suite(path=DEFAULT_SUITE):
    suite = read_mapping(path)
    if suite.get('version') != 1 or suite.get('seed') != 42:
        raise ValueError('Expected suite version=1 and shared seed=42')
    runs = suite['runs']
    for run_id, run in runs.items():
        if run['group'] not in suite['profiles']:
            raise ValueError(f'{run_id}: unknown group')
        if run.get('kind', 'train') not in {'train', 'reuse', 'evaluation'}:
            raise ValueError(f'{run_id}: invalid run kind')
        if run.get('kind', 'train') == 'train' and run.get('objective') not in {'infonce', 'ranknet', 'lambdaloss', 'rl'}:
            raise ValueError(f'{run_id}: invalid objective')
        if run.get('scope') not in {'joint', 'query_only', 'reference'}:
            raise ValueError(f'{run_id}: invalid update scope')
        for key in ('control', 'reuse', 'init_from'):
            if key in run and (run[key] not in runs or run[key] == run_id):
                raise ValueError(f'{run_id}: invalid {key} dependency')
        requirements = suite['profiles'][run['group']].get('requirements', []) + run.get('requirements', [])
        if any(name not in suite['pending'] for name in requirements):
            raise ValueError(f'{run_id}: requirement has no readiness explanation')
    # Detect cycles across both checkpoint and control links.
    def visit(run_id, stack):
        if run_id in stack:
            raise ValueError(f'Dependency cycle at {run_id}')
        for key in ('control', 'reuse', 'init_from'):
            if key in runs[run_id]:
                visit(runs[run_id][key], (*stack, run_id))
    for run_id in runs:
        visit(run_id, ())
    return suite


def apply_settings(suite, path):
    """Map user-facing settings to internal profiles; reject silent spelling mistakes."""
    settings = read_mapping(path)
    unknown = set(settings) - {'output_dir', 'G1', 'G2', 'G3'}
    if unknown:
        raise ValueError(f'Unknown experiment settings: {sorted(unknown)}')
    suite = copy.deepcopy(suite)
    if settings.get('output_dir'):
        suite['output_root'] = settings['output_dir']
    mapping = dict(model='model_name_or_path', learning_rate='learning_rate',
                   steps='max_steps', checkpoint_every='eval_steps',
                   batch_size='global_batch_size', micro_batch_size='micro_batch_size')
    for group in ('G1', 'G2', 'G3'):
        values = settings.get(group, {})
        unknown = set(values) - set(mapping) - {'data'}
        if unknown:
            raise ValueError(f'{group}: unknown settings {sorted(unknown)}')
        profile = suite['profiles'][group]
        for key, value in values.items():
            if key == 'data':
                if group == 'G3':
                    raise ValueError('G3 corpus/index/QA settings must use the advanced profile')
                profile['dataset_manifest'] = str(Path(value) / 'manifest.json')
                profile.setdefault('runtime_overrides', {})['data_path'] = str(Path(value) / ('train.ready.jsonl' if group == 'G1' else 'train.jsonl'))
            elif value is not None:
                if key == 'learning_rate':
                    value = float(value)
                if key in {'steps', 'checkpoint_every'} and (isinstance(value, bool) or not isinstance(value, int)):
                    raise ValueError(f'{group}.{key} must be an integer')
                profile['protocol'][mapping[key]] = value
    return suite


def path_at_root(path, root=ROOT):
    return (root / path).resolve()


def resolve_run(suite, suite_path, run_id, root=ROOT, nproc=1):
    run = suite['runs'][run_id]
    profile = suite['profiles'][run['group']]
    protocol = merge(profile['protocol'], run.get('protocol', {}))
    config = merge(resolve_config(Path(suite_path).parent / profile['config']), profile.get('runtime_overrides', {}))
    kind = run.get('kind', 'train')
    objective = run.get('objective')
    group = run['group']
    entrypoint = None
    if kind == 'train':
        if group == 'G3':
            entrypoint = 'src/train_rag.py'
            config['rag_objective'] = objective
        elif objective == 'rl':
            entrypoint = 'src/train.py'
            config = merge(config, resolve_config(root / 'configs/grpo/posttrain.yaml'))
            config.update(reward_type='ndcg_in_batch', reward_ndcg_k=10,
                          ndcg_in_batch_include_negatives=False)
            if run['scope'] == 'query_only':
                config['action_components'] = [['query']]
        else:
            entrypoint = 'src/train_baseline.py'
            config.update(baseline_loss=objective, baseline_temperature=0.03,
                          baseline_use_in_batch_negatives=True)
            if objective == 'lambdaloss':
                # Freeze the metric-aware control before any result is observed.
                config.update(baseline_ndcg_k=10, lambdaloss_sigma=1.0)
        if group == 'G1':
            config['document_encoder_mode'] = (
                'frozen_index' if run['scope'] == 'query_only' else 'joint'
            )
            if run['scope'] == 'query_only':
                data_parent = Path(config['data_path']).parent
                config['frozen_document_index_manifest'] = str(
                    data_parent / 'frozen_document_index' / 'index_manifest.json'
                )
                config['frozen_document_verify_hashes'] = True
    for key in ('model_name_or_path', 'pooling_method', 'append_token', 'padding_side',
                'query_prompt_template', 'document_prompt_template', 'learning_rate', 'max_steps'):
        if protocol.get(key) is not None:
            config[key] = protocol[key]
    if protocol.get('eval_steps') is not None:
        config['save_steps'] = protocol['eval_steps']
    config = merge(config, run.get('overrides', {}))
    if protocol.get('global_batch_size') is not None:
        total = protocol['global_batch_size']
        micro = protocol.get('micro_batch_size', config['per_device_train_batch_size'])
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in [total, micro, nproc]):
            raise ValueError('Batch sizes and GPU count must be positive integers')
        if total % (micro * nproc):
            raise ValueError('batch_size must be divisible by micro_batch_size * GPU count')
        config['per_device_train_batch_size'] = micro
        config['gradient_accumulation_steps'] = total // (micro * nproc)

    output = path_at_root(suite['output_root'], root) / f'{run_id}-s42'
    config.update(seed=42, data_seed=42, lora_enabled=False, overwrite_output_dir=False,
                  output_dir=str(output), run_name=run_id)
    dependency = None
    if run.get('init_from'):
        dependency = path_at_root(suite['output_root'], root) / f"{run['init_from']}-s42"
        config['model_name_or_path'] = str(dependency)
    return dict(run_id=run_id, group=group, kind=kind, scope=run['scope'],
                objective=objective, entrypoint=entrypoint, config=config, protocol=protocol,
                selection=profile['selection'], final_evaluation=profile['final_evaluation'],
                dataset_manifest=str(path_at_root(profile['dataset_manifest'], root)),
                dependency=str(dependency) if dependency else None,
                reuse=run.get('reuse'), control=run.get('control'),
                requirements=profile.get('requirements', []) + run.get('requirements', []))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def check_manifest(path, root=ROOT):
    """Validate immutable artifact references; trainer consumption is separate."""
    errors = []
    if not Path(path).is_file():
        return [f'Missing dataset/protocol manifest: {path}']
    try:
        manifest = read_mapping(path)
        if manifest.get('version') != 1 or not manifest.get('split_seed'):
            errors.append('Manifest needs version=1 and independent split_seed')
        if not manifest.get('artifacts'):
            errors.append('Manifest has no fingerprinted artifacts')
        roles = set()
        for item in manifest.get('artifacts', []):
            artifact = path_at_root(item['path'], root)
            roles.add(item['role'])
            if not artifact.is_file():
                errors.append(f'Missing artifact: {artifact}')
            elif digest(artifact) != item['sha256']:
                errors.append(f'Artifact hash mismatch: {artifact}')
        required_roles = {'train'} if manifest.get('selection') == 'fixed_budget_final_checkpoint' else {'train', 'dev'}
        if not required_roles.issubset(roles):
            errors.append('Manifest must fingerprint train and dev artifacts')
        if not manifest.get('evaluation_protocol'):
            errors.append('Manifest needs frozen evaluation_protocol')
        if not manifest.get('split_audit'):
            errors.append('Manifest needs split_audit evidence; hashes alone do not prove disjointness')
    except (ValueError, KeyError, TypeError) as exc:
        errors.append(f'Invalid manifest: {exc}')
    return errors


def blockers(suite, resolved, root=ROOT):
    errors = []
    if resolved['kind'] != 'train':
        return ['Evaluation is post-hoc; no automatic evaluator selection'] if resolved['kind'] == 'evaluation' else []
    errors.extend(f'Implementation: {suite["pending"][name]}' for name in resolved['requirements'])
    for key, value in resolved['protocol'].items():
        if value is None:
            errors.append(f'Unresolved protocol setting: {key}')
    cfg, protocol = resolved['config'], resolved['protocol']
    if not isinstance(protocol.get('max_steps'), int) or protocol['max_steps'] <= 0:
        errors.append('Freeze a positive total max_steps budget; one-epoch fallback is disabled')
    if protocol.get('learning_rate') is not None and protocol['learning_rate'] <= 0:
        errors.append('learning_rate must be positive')
    if protocol.get('eval_steps') is not None and (not isinstance(protocol['eval_steps'], int) or protocol['eval_steps'] <= 0):
        errors.append('eval_steps must be a positive integer')
    data_keys = ('rag_corpus_path', 'rag_candidate_manifest', 'rag_index_manifest') if resolved['group'] == 'G3' else ('data_path',)
    for key in data_keys:
        if not cfg.get(key) or not path_at_root(cfg[key], root).is_file():
            errors.append(f'Missing runtime input {key}: {cfg.get(key)}')
    if resolved['group'] == 'G1' and resolved['scope'] == 'query_only':
        errors.extend(check_frozen_document_index(cfg, root))
    if resolved['dependency']:
        dependency = Path(resolved['dependency'])
        if not (dependency / 'config.json').is_file() or not any(
            (dependency / name).is_file() for name in (
                'model.safetensors', 'model.safetensors.index.json',
                'pytorch_model.bin', 'pytorch_model.bin.index.json')):
            errors.append(f'Missing initialization model: {dependency}')
    if Path(cfg['output_dir']).exists():
        errors.append(f'Output already exists; no implicit skip/resume/overwrite: {cfg["output_dir"]}')
    output = Path(cfg['output_dir'])
    if (output.parent / '.launches' / output.name).exists():
        errors.append('Launch receipt already exists; explicit recovery is required')
    return errors


def check_frozen_document_index(config, root=ROOT):
    """Lightweight runner preflight without importing torch/model code."""
    value = config.get('frozen_document_index_manifest')
    if not value:
        return ['Missing frozen_document_index_manifest for query-only training']
    path = path_at_root(value, root)
    if not path.is_file():
        return [f'Missing frozen document index: {path}; run `python scripts/experiment.py encode G1 --gpus N`']
    errors = []
    try:
        manifest = read_mapping(path)
        required = {
            'format_version', 'dimension', 'count', 'shards', 'corpus_path',
            'corpus_offsets_path', 'document_key_to_ordinal_path',
        }
        missing = sorted(required - manifest.keys())
        if manifest.get('format_version') != 1 or missing:
            errors.append(f'Invalid frozen document index manifest; missing={missing}')
            return errors
        expected = {
            'model_name_or_path': config.get('model_name_or_path'),
            'pooling_method': config.get('pooling_method'),
            'padding_side': config.get('padding_side'),
            'append_token': config.get('append_token'),
            'document_prompt_template': config.get('document_prompt_template'),
            'document_max_length': min(config.get('d_max_len'), config.get('embedding_max_length')),
            'query_prompt_template': config.get('query_prompt_template'),
            'embedding_max_length': config.get('embedding_max_length'),
        }
        for key, actual in expected.items():
            if manifest.get(key) != actual:
                errors.append(
                    f'Frozen document protocol mismatch for {key}: '
                    f'{manifest.get(key)!r} != {actual!r}'
                )
        source_path = path_at_root(config['data_path'], root)
        if manifest.get('source_data_sha256') != digest(source_path):
            errors.append(f'Frozen document source hash mismatch: {source_path}')
        artifacts = [
            (manifest['corpus_path'], manifest.get('corpus_sha256')),
            (manifest['corpus_offsets_path'], manifest.get('corpus_offsets_sha256')),
            (manifest['document_key_to_ordinal_path'], manifest.get('document_key_to_ordinal_sha256')),
        ]
        artifacts.extend((item['path'], item.get('sha256')) for item in manifest['shards'])
        for artifact_path, expected_hash in artifacts:
            target = Path(artifact_path)
            if not target.is_absolute():
                target = path.parent / target
            if not target.is_file():
                errors.append(f'Missing frozen document artifact: {target}')
            elif not expected_hash:
                errors.append(f'Missing SHA256 for frozen document artifact: {target}')
            elif digest(target) != expected_hash:
                errors.append(f'Frozen document artifact hash mismatch: {target}')
        mapping_path = Path(manifest['document_key_to_ordinal_path'])
        if not mapping_path.is_absolute():
            mapping_path = path.parent / mapping_path
        if mapping_path.is_file():
            mapping = read_mapping(mapping_path)
            if len(mapping) != int(manifest['count']) or sorted(mapping.values()) != list(range(int(manifest['count']))):
                errors.append('Frozen document key mapping is not a complete ordinal permutation')
    except (ValueError, KeyError, TypeError, OSError) as exc:
        errors.append(f'Invalid frozen document index: {exc}')
    return errors


def command_for(resolved, runtime_path, nproc):
    if nproc < 1:
        raise ValueError('nproc must be positive')
    return ['torchrun', '--standalone', f'--nproc_per_node={nproc}', resolved['entrypoint'], str(runtime_path)]


def launch(suite, resolved, nproc, root=ROOT):
    if resolved['kind'] != 'train':
        raise ValueError('Only training rows can launch; reuse/evaluation never schedules a dependency')
    errors = blockers(suite, resolved, root)
    if errors:
        raise ValueError('\n'.join(errors))
    out = Path(resolved['config']['output_dir'])
    # Separate audit artifacts avoid polluting the trainer's initially empty output dir.
    artifacts = out.parent / '.launches' / out.name
    artifacts.mkdir(parents=True, exist_ok=False)
    runtime = artifacts / 'config.json'
    runtime.write_text(json.dumps(resolved['config'], indent=2) + '\n')
    evidence = copy.deepcopy(resolved)
    evidence['runtime_config_sha256'] = digest(runtime)
    evidence['command'] = command_for(resolved, runtime, nproc)
    (artifacts / 'launch.json').write_text(json.dumps(evidence, indent=2) + '\n')
    subprocess.run(evidence['command'], cwd=root, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['list', 'resolve', 'check', 'launch'])
    parser.add_argument('run_id', nargs='?')
    parser.add_argument('--group', choices=['G1', 'G2', 'G3'])
    parser.add_argument('--suite', type=Path, default=DEFAULT_SUITE)
    parser.add_argument('--nproc', type=int, default=1)
    parser.add_argument('--settings', type=Path, default=ROOT / 'configs/experiments.yaml')
    args = parser.parse_args(argv)
    try:
        suite = apply_settings(load_suite(args.suite), args.settings)
        if args.run_id and args.run_id not in suite['runs']:
            raise ValueError(f'Unknown run ID: {args.run_id}')
        selected = [args.run_id] if args.run_id else [k for k, r in suite['runs'].items() if not args.group or r['group'] == args.group]
        if args.action in {'resolve', 'launch'} and not args.run_id:
            raise ValueError(f'{args.action} requires one explicit run ID')
        failures = 0
        for run_id in selected:
            resolved = resolve_run(suite, args.suite, run_id, nproc=args.nproc)
            if args.action == 'launch':
                launch(suite, resolved, args.nproc)
                continue
            problems = blockers(suite, resolved)
            if args.action == 'resolve':
                resolved['blockers'] = problems
                resolved['launchable'] = not problems and resolved['kind'] == 'train'
                if resolved['kind'] == 'train':
                    resolved['command_preview'] = shlex.join(command_for(resolved, '<resolved-config.json>', args.nproc))
                print(json.dumps(resolved, indent=2, ensure_ascii=False))
            else:
                status = 'REUSE' if resolved['reuse'] else ('EVAL' if resolved['kind'] == 'evaluation' else ('BLOCKED' if problems else 'READY'))
                print(f'{run_id:16} {status:7} {resolved["scope"]:10} {resolved["objective"] or "-"}')
                if args.action == 'check':
                    for issue in problems:
                        print(f'  - {issue}')
                    failures += int(bool(problems) and resolved['kind'] == 'train')
        return 2 if args.action == 'check' and failures else 0
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
