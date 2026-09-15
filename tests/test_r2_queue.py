"""Exercise unattended queue recovery, numerical recipe parsing and data pairing."""
import copy
from dataclasses import fields
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
import run_g1_r2 as night
from config import BaselineArguments, DataArguments, LoraArguments, ModelArguments, RLArguments, TrainingArguments, MTEBEvalArguments
from embedding_data import SingleSourceBatchSampler
from train import build_embedding_data, load_backbone_and_tokenizer


def parsed(cls, config):
    allowed = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in config.items() if k in allowed})


@pytest.fixture
def row(tmp_path):
    result = copy.deepcopy(night.resolve_matrix()[1][1])
    result['config']['output_dir'] = str(tmp_path / 'model')
    return result


def save_model(row):
    out = Path(row['config']['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text('{}')
    (out / 'model.safetensors').write_bytes(b'nonempty fake weights for queue-only checks')
    cfg = row['config']
    protocol = {k: cfg[k] for k in ('pooling_method', 'padding_side', 'append_token',
                                  'query_prompt_template', 'document_prompt_template')}
    protocol.update(tokenization_version=2, pooling_compute_dtype='float32', max_length=8192)
    (out / 'embedding_protocol.json').write_text(json.dumps(protocol))


def save_scores(row, value=.2):
    path = Path(row['config']['output_dir']) / 'mteb_eval/bright/BrightRetrieval.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'scores': {'standard': [
        {'hf_subset': s, 'main_score': value} for s in night.SUBSETS]}}))


def save_completion(row, receipts, data_hash='a' * 64):
    contract = {'data_sha256': data_hash}
    night.write_json(receipts / row['run_id'] / 'contract.json', contract)
    night.write_json(receipts / row['run_id'] / 'state.json',
                     {'evaluation_complete': True, 'contract_sha256': night.fingerprint(contract)})


@pytest.fixture
def prepared_data(row, tmp_path):
    directory = tmp_path / 'prepared'
    directory.mkdir()
    path = directory / 'train.ready.jsonl'
    path.write_text(json.dumps(dict(schema='embedding_candidates_v2', id='sample', source='biology',
                                    query='query', document=['positive', 'negative'], relevance=[1, 0])) + '\n')
    manifest = dict(version=1, selection='fixed_budget_final_checkpoint', positive_selection_seed=42,
                    artifacts=[dict(role='train', path='/original/preparation/host/train.ready.jsonl',
                                    sha256=night.experiments.digest(path))])
    night.write_json(directory / 'manifest.json', manifest)
    row['config']['data_path'] = str(path)
    row['dataset_manifest'] = str(directory / 'manifest.json')
    return row, path, manifest


def test_prepared_data_uses_local_manifest_and_resolves_equivalent_paths(prepared_data, monkeypatch, tmp_path):
    row, path, _ = prepared_data
    relative = copy.deepcopy(row)
    relative['config']['data_path'] = os.path.relpath(path, night.ROOT)
    monkeypatch.chdir(tmp_path)  # Relative data paths remain rooted in the project.
    assert night.verify_data([row, relative]) == night.experiments.digest(path)


def test_prepared_data_rejects_changed_content_and_different_files(prepared_data, tmp_path):
    row, path, manifest = prepared_data
    other = copy.deepcopy(row)
    other['config']['data_path'] = str(tmp_path / 'other.ready.jsonl')
    with pytest.raises(ValueError, match='same prepared data'):
        night.verify_data([row, other])
    path.write_text(path.read_text().replace('positive', 'changed positive'))
    with pytest.raises(ValueError, match='hash mismatch'):
        night.verify_data([row])
    # Even an updated manifest cannot silently switch data in an existing run.
    old_contract = night.contract_for(row, manifest['artifacts'][0]['sha256'], {})
    def call(command, log):
        save_model(row)
        save_scores(row)
        return 0
    receipts = tmp_path / 'receipt'
    night.execute_job(row, old_contract, receipts, call)
    manifest['artifacts'][0]['sha256'] = night.experiments.digest(path)
    night.write_json(path.parent / 'manifest.json', manifest)
    updated = night.contract_for(row, night.verify_data([row]), {})
    again = Mock()
    with pytest.raises(ValueError, match='different config/data/code'):
        night.execute_job(row, updated, receipts, again)
    again.assert_not_called()


@pytest.mark.parametrize('change', ['dev', 'seed', 'duplicate_train', 'public_train', 'no_hash'])
def test_prepared_data_requires_matching_preparation_metadata(prepared_data, change):
    row, path, manifest = prepared_data
    if change == 'dev':
        manifest['selection'] = 'independent_dev'
    elif change == 'seed':
        manifest['positive_selection_seed'] = 3407
    elif change == 'duplicate_train':
        manifest['artifacts'].append(dict(manifest['artifacts'][0]))
    elif change == 'public_train':
        manifest['artifacts'][0]['path'] = '/original/host/train.jsonl'
    else:
        del manifest['artifacts'][0]['sha256']
    night.write_json(path.parent / 'manifest.json', manifest)
    with pytest.raises(ValueError):
        night.verify_data([row])


def test_eval_failure_retries_only_eval_and_completion_skips(row, tmp_path):
    calls = []
    contract = {'evaluation_command': ['evaluation']}
    folder = tmp_path / 'receipt'
    def first(command, log):
        calls.append(command)
        if command == ['evaluation']:
            return 7
        save_model(row)
        return 0
    with pytest.raises(ValueError):
        night.execute_job(row, contract, folder, first)
    assert json.loads((folder / 'state.json').read_text())['train_complete']
    def retry(command, log):
        calls.append(command)
        assert command == ['evaluation']
        save_scores(row)
        return 0
    assert night.execute_job(row, contract, folder, retry) == 'complete'
    assert night.execute_job(row, contract, folder, retry) == 'skipped_complete'
    assert len(calls) == 3
    with pytest.raises(ValueError):
        night.execute_job(row, {**contract, 'data_sha256': 'changed'}, folder, retry)
    assert len(calls) == 3


def test_training_failure_never_resumes_partial_checkpoint(row, tmp_path):
    call = Mock(return_value=1)
    folder = tmp_path / 'receipt'
    contract = {'evaluation_command': ['evaluation']}
    with pytest.raises(ValueError):
        night.execute_job(row, contract, folder, call)
    with pytest.raises(ValueError):
        night.execute_job(row, contract, folder, call)
    assert call.call_count == 1
    assert not json.loads((folder / 'state.json').read_text())['train_complete']


def test_unowned_output_is_preserved(row, tmp_path):
    out = Path(row['config']['output_dir'])
    out.mkdir()
    (out / 'keep').write_bytes(b'previous result')
    call = Mock()
    with pytest.raises(ValueError):
        night.execute_job(row, {'evaluation_command': []}, tmp_path / 'receipt', call)
    call.assert_not_called()
    assert (out / 'keep').read_bytes() == b'previous result'


def test_completed_scores_cannot_hide_missing_weights(row, tmp_path):
    def call(command, log):
        save_model(row)
        save_scores(row)
        return 0
    contract = {'evaluation_command': ['evaluation']}
    folder = tmp_path / 'receipt'
    night.execute_job(row, contract, folder, call)
    (Path(row['config']['output_dir']) / 'model.safetensors').unlink()
    again = Mock()
    with pytest.raises(ValueError):
        night.execute_job(row, contract, folder, again)
    again.assert_not_called()


def test_queue_continues_after_failure_and_reports_nonzero_failures():
    visited, events = [], []
    def execute(job):
        visited.append(job)
        if job == 'broken':
            raise ValueError('training failed')
        return 'complete'
    failed = night.execute_queue(['first', 'broken', 'last'], execute, events.append)
    assert visited == ['first', 'broken', 'last']
    assert failed == ['broken']
    assert [e['status'] for e in events] == ['complete', 'failed', 'complete']


def test_matrix_parses_real_arguments_and_pairs_rewards():
    _, matrix = night.resolve_matrix()
    assert len({r['config']['output_dir'] for r in matrix}) == 28
    for row in matrix:
        cfg = row['config']
        classes = (ModelArguments, DataArguments, LoraArguments, TrainingArguments, MTEBEvalArguments,
                   RLArguments if row['objective'] == 'rl' else BaselineArguments)
        assert not set(cfg) - {f.name for cls in classes for f in fields(cls)}
        assert parsed(ModelArguments, cfg).model_revision == night.REVISION
        assert parsed(DataArguments, cfg).dev_samples_per_source == 0
        if row['objective'] == 'rl':
            rl = parsed(RLArguments, cfg)
            assert rl.group_size == 64
            assert rl.reward_terms[0].k == 10
            assert rl.rollout_seed == cfg['seed']
            assert rl.gradient_estimator == ('conditional_projection' if '-CP' in row['run_id'] else 'score_function')
            assert rl.reward_terms[0].type == ('mrr_in_batch' if 'MRR64' in row['run_id'] else 'ndcg_in_batch')
        elif row['objective']:
            base = parsed(BaselineArguments, cfg)
            assert base.baseline_use_in_batch_negatives
            if row['objective'] == 'lambdaloss':
                assert base.lambdaloss_sigma == pytest.approx(1 / .03)
        assert cfg['relevance_scheme'] == ('graded' if 'Graded' in row['run_id'] else 'binary') or row['kind'] == 'evaluation'


def test_e0_evaluation_pins_revision_and_does_not_load_checkpoint_output():
    row = night.resolve_matrix()[1][0]
    args = night.evaluation_command(row)
    kwargs = json.loads(args[args.index('--model_kwargs') + 1])
    assert args[args.index('--model') + 1] == row['config']['model_name_or_path']
    assert kwargs['revision'] == night.REVISION
    assert kwargs['max_length'] == 8192
    assert kwargs['use_instruction']


def test_summary_never_averages_missing_seeds(tmp_path):
    _, matrix = night.resolve_matrix()
    matrix = copy.deepcopy(matrix)
    receipts = tmp_path / 'receipts'
    for row in matrix:
        row['config']['output_dir'] = str(tmp_path / row['run_id'])
        if row['run_id'] in {night.run_id('RL-MRR64-SF', s) for s in night.SEEDS}:
            save_scores(row, .20)
            save_completion(row, receipts)
        if row['run_id'] in {night.run_id('RL-MRR64-CP', s) for s in night.SEEDS[:2]}:
            save_scores(row, .22)
            save_completion(row, receipts)
    report = night.summarize(matrix, receipts)
    groups = {g['method']: g for g in report['groups']}
    assert groups['RL-MRR64-SF']['mean'] == pytest.approx(20)
    assert groups['RL-MRR64-CP']['completed'] == 2
    assert groups['RL-MRR64-CP']['mean'] is None
    paired = report['paired_cp_minus_sf'][0]
    assert paired['mean_delta'] is None
    assert paired['per_seed'][0]['delta'] == pytest.approx(2)


def test_data_pairing_does_not_depend_on_model_loading_rng(tmp_path):
    path = tmp_path / 'train.ready.jsonl'
    records = [dict(id=str(i), schema='embedding_candidates_v2', source='biology', query=f'q {i}',
                    document=['a', 'b'], relevance=[1, 0], ranking=[1, 2]) for i in range(19)]
    path.write_text('\n'.join(json.dumps(r) for r in records) + '\n')
    data = DataArguments(data_path=str(path), index_cache_dir=str(tmp_path / 'cache'), per_dataset_max_samples=None)
    args = SimpleNamespace(per_device_train_batch_size=4, per_device_eval_batch_size=4, data_seed=42)
    orders = []
    for consumed in (5, 37):
        for _ in range(consumed):
            random.random()
        dataset, dev, _ = build_embedding_data(data, args, tokenizer=SimpleNamespace(pad_token='[PAD]'))
        try:
            assert dev is None
            assert len(dataset) == 16  # Existing tail policy remains in effect.
            positions = list(SingleSourceBatchSampler(dataset, 4, seed=42))
            orders.append([dataset.entries[p] for p in positions])
        finally:
            dataset.close()
    assert orders[0] == orders[1]


def test_model_revision_reaches_backbone_config_and_tokenizer(monkeypatch):
    import train
    loaders = [Mock() for _ in range(3)]
    for name, loader in zip(('AutoConfig', 'AutoModel', 'AutoTokenizer'), loaders):
        monkeypatch.setattr(train, name, loader)
    load_backbone_and_tokenizer(ModelArguments(model_name_or_path='example/model', model_revision='pinned'),
                                LoraArguments(lora_enabled=False))
    for loader in loaders:
        assert loader.from_pretrained.call_args.kwargs['revision'] == 'pinned'


def test_probe_checkpoint_is_explicit_and_retries_preserve_partial_results(tmp_path):
    checkpoint = tmp_path / 'checkpoint-25'
    checkpoint.mkdir()
    (checkpoint / 'model.safetensors').write_bytes(b'new weights')
    calls = []
    def partial(command, log):
        calls.append(command)
        log.touch()
        Path(command[command.index('--output') + 1]).write_text('{"probes": []}')
        return 1
    with pytest.raises(ValueError):
        night.run_probe(night.SETTINGS, tmp_path, {}, 'data', partial, checkpoint=checkpoint)
    def complete(command, log):
        calls.append(command)
        log.touch()
        assert Path(command[command.index('--checkpoint') + 1]) == checkpoint
        assert int(command[command.index('--step') + 1]) == 25
        night.write_json(Path(command[command.index('--output') + 1]),
                         {'comparison': 'paired_gradient_estimators', 'probes': [{}, {}, {}]})
        return 0
    assert night.run_probe(night.SETTINGS, tmp_path, {}, 'data', complete, checkpoint=checkpoint) == 'complete'
    assert night.run_probe(night.SETTINGS, tmp_path, {}, 'data', complete, checkpoint=checkpoint) == 'skipped_complete'
    folder = tmp_path / 'gradient_probe_ll25'
    assert len(calls) == 2
    assert json.loads((folder / 'attempt-1.json').read_text())['probes'] == []
    assert len(json.loads((folder / 'attempt-2.json').read_text())['probes']) == 3


def test_seed_partitions_cover_the_full_queue_once_without_changing_runs():
    _, full = night.resolve_matrix()
    expected = {row['run_id']: row for row in full}
    seen, jobs = [], []
    for seed in night.SEEDS:
        _, part = night.resolve_matrix(seeds=[seed])
        training = [row for row in part if row['kind'] == 'train']
        assert len(training) == len(night.METHODS)
        assert {row['config']['seed'] for row in training} == {seed}
        assert {row['config']['rollout_seed'] for row in training if row['objective'] == 'rl'} == {seed}
        assert all(row == expected[row['run_id']] for row in part)
        seen.extend(row['run_id'] for row in part)
        jobs.extend(night.build_jobs(part))
        if seed != 42:
            assert night.build_jobs(part) == [row['run_id'] for row in part]
    assert len(seen) == len(set(seen)) == len(full)
    assert set(seen) == set(expected)
    assert len(jobs) == len(set(jobs)) == len(full) + 2
    assert set(jobs) == set(night.build_jobs(full))
    assert night.build_jobs(full, skip_probe=True) == [row['run_id'] for row in full]


@pytest.mark.parametrize('seeds', [[], [42, 42], [1234]])
def test_invalid_seed_selection_is_rejected(seeds):
    with pytest.raises(ValueError):
        night.resolve_matrix(seeds=seeds)


def test_disjoint_seed_locks_allow_parallel_queues_and_overlap_releases_locks(tmp_path):
    with night.seed_locks(tmp_path, [3407]):
        with night.seed_locks(tmp_path, [42]):
            pass
        with pytest.raises(ValueError):
            # Acquires seed 42 before encountering the occupied seed 3407.
            with night.seed_locks(tmp_path, [42, 3407]):
                pytest.fail('Overlapping queue acquired the same seed')
        with night.seed_locks(tmp_path, [42, 2026]):
            pass  # The unsuccessful multi-seed lock released its seed-42 fd.
    with night.seed_locks(tmp_path, night.SEEDS):
        pass


def test_partition_summaries_keep_other_seeds_separate_then_combine(tmp_path):
    _, full = night.resolve_matrix()
    full = copy.deepcopy(full)
    receipts = tmp_path / 'receipts'
    for row in full:
        row['config']['output_dir'] = str(tmp_path / row['run_id'])
    for seed, base_score in zip(night.SEEDS, [.10, .20, .30]):
        part = [row for row in full if row['kind'] == 'train' and row['config']['seed'] == seed]
        for row in part:
            save_scores(row, base_score + (.02 if '-CP' in row['run_id'] else 0))
            save_completion(row, receipts)
        report = night.summarize(part, receipts)
        assert report['seeds'] == [seed]
        assert report['groups'][0]['completed'] == 1
        assert report['groups'][0]['mean'] == pytest.approx(base_score * 100)
        assert report['groups'][0]['sample_sd'] is None
        assert report['paired_cp_minus_sf'][0]['mean_delta'] == pytest.approx(2)
        assert not (receipts / 'summary.json').exists()
    reports_before = {seed: (night.queue_directory(receipts, [seed]) / 'summary.json').read_bytes()
                      for seed in night.SEEDS}
    combined = night.summarize(full, receipts)
    assert combined['seeds'] == list(night.SEEDS)
    assert combined['groups'][0]['completed'] == 3
    assert combined['groups'][0]['mean'] == pytest.approx(20)
    assert combined['groups'][0]['sample_sd'] == pytest.approx(10)
    assert combined['paired_cp_minus_sf'][0]['mean_delta'] == pytest.approx(2)
    for seed in night.SEEDS:
        assert (night.queue_directory(receipts, [seed]) / 'summary.json').read_bytes() == reports_before[seed]


@pytest.mark.parametrize('mismatch', ['seed', 'estimator'])
def test_summary_does_not_combine_different_training_data(tmp_path, mismatch):
    _, matrix = night.resolve_matrix()
    receipts = tmp_path / 'receipts'
    for row in matrix:
        row['config']['output_dir'] = str(tmp_path / row['run_id'])
        save_scores(row, .22 if '-CP' in row['run_id'] else .20)
        differs = (row['config']['seed'] == 3407 if mismatch == 'seed' else '-CP' in row['run_id'])
        save_completion(row, receipts, ('b' if differs else 'a') * 64)
    report = night.summarize(matrix, receipts)
    assert not report['data_consistent']
    assert report['data_sha256'] == ['a' * 64, 'b' * 64]
    assert all(group['completed'] == 3 and group['mean'] is None and group['sample_sd'] is None
               for group in report['groups'])
    assert all(pair['mean_delta'] is None for pair in report['paired_cp_minus_sf'])
    deltas = [item['delta'] for item in report['paired_cp_minus_sf'][0]['per_seed']]
    assert deltas == (pytest.approx([2, 2, 2]) if mismatch == 'seed' else [None, None, None])
    assert all(row['evaluation_complete'] for row in report['runs'])


def test_summary_requires_a_matching_saved_data_contract(row, tmp_path):
    receipts = tmp_path / 'receipts'
    save_scores(row)
    save_completion(row, receipts)
    night.write_json(receipts / row['run_id'] / 'contract.json', {'data_sha256': 'b' * 64})
    _, matrix = night.resolve_matrix(seeds=[42])
    matrix = [row if candidate['run_id'] == row['run_id'] else candidate for candidate in matrix]
    report = night.summarize(matrix, receipts)
    result = next(item for item in report['runs'] if item['run'] == row['run_id'])
    assert not result['evaluation_complete']
    assert result['mean'] is None
    assert result['error']


def test_e0_evaluation_does_not_constrain_training_data_identity(tmp_path):
    _, matrix = night.resolve_matrix(seeds=[42])
    receipts = tmp_path / 'receipts'
    for row in matrix:
        row['config']['output_dir'] = str(tmp_path / row['run_id'])
        save_scores(row, .20)
        save_completion(row, receipts, ('b' if row['kind'] == 'evaluation' else 'a') * 64)
    report = night.summarize(matrix, receipts)
    assert report['data_consistent']
    assert report['data_sha256'] == ['a' * 64]
    assert all(group['mean'] == pytest.approx(20) for group in report['groups'])
