#!/usr/bin/env python3
"""G1-R2 query-only dynamic retrieval: encode, check, run or summarize."""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import sys

import run_g1_r2 as night
import run_g1_r2_ablations as ab

ROOT = night.ROOT


def resolve(settings=night.SETTINGS, seeds=night.SEEDS, index_root=None, backend='faiss'):
    _, base = night.resolve_matrix(settings, seeds=seeds)
    data = night.experiments.path_at_root(base[0]['config']['data_path'])
    index_root = Path(index_root).resolve() if index_root else data.parent / 'reasonrank_frozen_document_indices_r2'
    rows = []
    for seed in night.selected_seeds(seeds):
        row = copy.deepcopy(next(r for r in base if r['run_id']==night.run_id('RL-GradedNDCG64-SF', seed)))
        name = night.run_id('DR-GradedNDCG64-SF', seed)
        root = Path(row['config']['output_dir']).parent
        row.update(run_id=name, scope='query_only', variant='dynamic_sf', control=None,
                   entrypoint='scripts/train_g1_r2_dynamic.py')
        row['config'].update(run_name=name, output_dir=str(root/f'{name}-s{seed}'),
            dynamic_retrieval=True, dynamic_retrieval_k=20, action_components=[['query']],
            document_encoder_mode='frozen_index', frozen_document_index_backend=backend,
            frozen_document_index_manifest=str(index_root/'index_router_manifest.json'),
            reward_type='ndcg', reward_terms='')
        row['protocol'].update(candidate_access='per_source_dynamic_full_corpus',
            retrieval_reward='known_qrels_graded_ndcg_at_10', unjudged_documents='zero_gain',
            document_encoder='frozen_E0_index', rollout_rng='independent_rank_step_v1')
        rows.append(row)
    return rows, index_root, root / '.r2_dynamic'


def evaluation_command(row, directory):
    command = night.evaluation_command(row)
    kwargs = json.loads(command[command.index('--model_kwargs')+1])
    command += ['--fixed_corpus_model', row['config']['model_name_or_path'],
                '--fixed_corpus_model_revision', night.REVISION,
                '--fixed_corpus_model_kwargs', json.dumps(kwargs),
                '--fixed_corpus_index_dir', str(directory/'bright_e0_index')]
    return command


def validate_index(row):
    cfg = row['config']
    path = Path(cfg['frozen_document_index_manifest'])
    if not path.is_file():
        raise ValueError(f'Missing R2 router: {path}; run python scripts/run_g1_r2_dynamic.py encode')
    errors = night.experiments.check_frozen_document_index(cfg)
    if errors:
        raise ValueError('\n'.join(errors)+'\nBuild the R2 index: python scripts/run_g1_r2_dynamic.py encode')
    manifest = night.experiments.read_mapping(path)
    if manifest.get('artifact_type') != 'frozen_document_index_router':
        raise ValueError('Dynamic R2 requires the canonical per-source ReasonRank router')
    for route in manifest['routes'].values():
        child = Path(route['manifest'])
        child = child if child.is_absolute() else path.parent/child
        saved = night.experiments.read_mapping(child)
        if saved.get('model_revision') != night.REVISION or saved.get('resolved_model_revision') != night.REVISION:
            raise ValueError(f'Index must use the fixed R2 E0 revision: {child}')
    return night.experiments.digest(path)


def encode(rows, index_root, directory, documents_dir, gpus, batch_size):
    sys.path.insert(0, str(ROOT/'src'))
    from fixed_corpus.router import (REASONRANK_TRAINING_SOURCES, REASONRANK_DATASET,
        REASONRANK_CONFIG, REASONRANK_REVISION, materialize_reasonrank_documents, write_reasonrank_index_router)
    missing = [s for s in REASONRANK_TRAINING_SOURCES if not (documents_dir/f'{s}.json').is_file()]
    if missing:
        raise ValueError(f'Missing canonical ReasonRank id_doc sources: {missing}. '
            'Run python scripts/download_reasonrank_audit.py --include-reasonrank-documents')
    cfg = rows[0]['config']
    data = night.experiments.path_at_root(cfg['data_path'])
    canonical = index_root.parent / (index_root.name+'_canonical')
    # Index lock is tied to the index location, including when different model
    # output roots share it. Completed routes validate and skip on retry.
    with ab.run_locks(index_root.parent / (index_root.name+'_build'), ['encode']):
        print(materialize_reasonrank_documents(data, documents_dir, canonical), flush=True)
        for source in sorted(REASONRANK_TRAINING_SOURCES):
            command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                f'--nproc_per_node={gpus}', str(ROOT/'src/encode_frozen_documents.py'),
                '--input', str(canonical/f'{source}.jsonl'), '--input-format', 'document_jsonl',
                '--output-dir', str(index_root/source), '--source-name', source,
                '--source-dataset', REASONRANK_DATASET, '--source-config', REASONRANK_CONFIG,
                '--source-revision', REASONRANK_REVISION, '--model', cfg['model_name_or_path'],
                '--revision', night.REVISION, '--batch-size', str(batch_size),
                '--max-length', str(min(cfg['d_max_len'], cfg['embedding_max_length'])),
                '--embedding-max-length', str(cfg['embedding_max_length']),
                '--pooling-method', cfg['pooling_method'], '--padding-side', cfg['padding_side'],
                '--append-token', cfg['append_token'], '--document-prompt-template', cfg['document_prompt_template'],
                '--query-prompt-template', cfg['query_prompt_template']]
            log = directory/'encode'/f'{source}.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            print(f'ENCODE {source}; log: {log}', flush=True)
            if night.invoke(command, log):
                raise ValueError(f'Encoding failed; see {log}; partial index build is preserved')
        print(write_reasonrank_index_router(index_root, data, revision=REASONRANK_REVISION))
        validate_index(rows[0])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', default='run', choices=['encode','check','run','summary'])
    parser.add_argument('--config', type=Path, default=night.SETTINGS)
    parser.add_argument('--seeds', '--seed', nargs='+', type=int, choices=night.SEEDS, default=list(night.SEEDS))
    parser.add_argument('--index-root', type=Path)
    parser.add_argument('--index-backend', choices=['faiss','torch'], default='faiss')
    parser.add_argument('--documents-dir', type=Path, default=ROOT/'data/audit_reasonrank_bright/reasonrank_documents/id_doc')
    parser.add_argument('--encode-gpus', type=int, default=8)
    parser.add_argument('--encode-batch-size', type=int, default=16)
    args = parser.parse_args(argv)
    if args.encode_gpus < 1 or args.encode_batch_size < 1:
        parser.error('Encoding GPU count and batch size must be positive')
    rows, index_root, directory = resolve(args.config, args.seeds, args.index_root, args.index_backend)
    if args.action=='summary':
        report = ab.summarize(rows, [], directory, directory, directory)
        print(directory/'summary.md')
        return int(not report['comparable'] or not all(r['evaluation_complete'] for r in report['runs']))
    data_hash = night.verify_data(rows)
    print(f'Training data SHA256: {data_hash}\nR2 frozen index: {index_root}', flush=True)
    if args.action=='encode':
        encode(rows, index_root, directory, args.documents_dir.resolve(), args.encode_gpus, args.encode_batch_size)
        return 0
    index_hash = validate_index(rows[0])
    sources = ab.sources()
    for filename in ('run_g1_r2_dynamic.py','train_g1_r2_dynamic.py'):
        sources[f'scripts/{filename}'] = night.experiments.digest(ROOT/'scripts'/filename)
    contracts = {}
    for row in rows:
        contract = night.contract_for(row, data_hash, sources)
        contract.update(frozen_index_router_sha256=index_hash, evaluation_command=evaluation_command(row,directory))
        contracts[row['run_id']] = contract
        print(f'{row["run_id"]}: {row["config"]["output_dir"]}', flush=True)
    if args.action=='check':
        ab.preflight(rows, directory, contracts)
        print(f'Validated {len(rows)} dynamic SF runs; G64, top20, graded nDCG@10; frozen E0 evaluation.')
        return 0
    import torch
    import deepspeed
    if torch.cuda.device_count()<8 or not torch.cuda.is_bf16_supported():
        raise ValueError('Training requires eight visible CUDA GPUs with BF16 support')
    if args.index_backend=='faiss':
        import faiss
    runtime = dict(nproc=8, torch_version=torch.__version__, cuda_version=torch.version.cuda,
        deepspeed_version=deepspeed.__version__, gpus=[torch.cuda.get_device_name(i) for i in range(8)])
    with ab.run_locks(directory, [r['run_id'] for r in rows]):
        lookup = {r['run_id']:r for r in rows}
        def call(command, log):
            if log.stem!='eval':
                return night.invoke(command, log)
            # The fixed-corpus evaluation cache does not implement its own lock.
            with (directory/'bright_e0_eval.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                return night.invoke(command, log)
        def execute(name):
            return ab.execute(lookup[name], contracts[name], directory/name, call, runtime)
        def record(event):
            folder = directory/event['job']
            folder.mkdir(parents=True, exist_ok=True)
            with (folder/'events.jsonl').open('a') as handle:
                handle.write(json.dumps(event)+'\n')
            print(json.dumps(event), flush=True)
            output = directory/'queues'/('seeds-'+'-'.join(map(str,sorted(args.seeds))))
            ab.summarize(rows, [], directory, directory, output)
        return int(bool(night.execute_queue(list(lookup), execute, record)))


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (ValueError,OSError,KeyError,TypeError,RuntimeError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        raise SystemExit(2)
