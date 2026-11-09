"""CPU coverage for checkpoint pairing, safe queues, and shared MTEB result merges."""
import copy
import csv
import json
from pathlib import Path
import statistics
import sys
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))

from eval_mteb import summary as mteb_summary
import run_g2_cl_to_reler as driver


def test_continuation_pairs_checkpoints_and_preserves_the_direct_recipe():
    original = driver.resolve_original_matrix()
    matrix = driver.resolve_matrix()
    for seed in driver.SEEDS:
        baseline = matrix['cl', seed][1]
        row = matrix['cl_reler', seed][1]
        cfg = copy.deepcopy(row['config'])
        direct = original['reler', seed][1]['config']
        assert row['dependency'] == baseline['config']['output_dir']
        assert cfg['model_name_or_path'] == row['dependency']
        assert cfg['seed'] == cfg['data_seed'] == cfg['rollout_seed'] == seed
        assert not cfg['overwrite_output_dir']
        for key in ('model_name_or_path', 'output_dir', 'run_name'):
            cfg[key] = direct[key]
        assert cfg == direct
        assert matrix['cl', seed] == original['infonce', seed]
        commands = driver.experiments.post_train_commands_for(row)
        assert len(commands) == 1
        command = commands[0]
        assert command[command.index('--benchmark') + 1] == driver.experiments.MTEB_ENG_V2
        assert command[command.index('--model') + 1] == row['config']['output_dir']


def make_input_model(path, cfg):
    path.mkdir(parents=True)
    (path / 'config.json').write_text('{}')
    (path / 'model.safetensors').touch()
    protocol = {k: cfg[k] for k in ('pooling_method', 'padding_side', 'append_token',
                                   'query_prompt_template', 'document_prompt_template')}
    protocol.update(tokenization_version=2, pooling_compute_dtype='float32',
                    max_length=cfg['embedding_max_length'], normalize=True)
    (path / 'embedding_protocol.json').write_text(json.dumps(protocol))


def local_matrix(tmp_path):
    matrix = driver.resolve_matrix()
    data = tmp_path / 'train.jsonl'
    data.write_text('{}\n')
    for (method, seed), (_, row) in matrix.items():
        row['config']['output_dir'] = str(tmp_path / f'{method}-{seed}')
        row['config']['data_path'] = str(data)
    for seed in driver.SEEDS:
        row = matrix['cl_reler', seed][1]
        row['dependency'] = matrix['cl', seed][1]['config']['output_dir']
        row['config']['model_name_or_path'] = row['dependency']
        make_input_model(Path(row['dependency']), row['config'])
    return matrix


def test_whole_queue_is_checked_before_launch_and_cl_is_never_trained(tmp_path):
    matrix = local_matrix(tmp_path)
    last = Path(matrix['cl', 2026][1]['config']['output_dir']) / 'model.safetensors'
    last.unlink()
    with patch.object(driver, 'resolve_matrix', return_value=matrix), \
         patch.object(driver, 'require_training_gpus'), \
         patch.object(driver.experiments, 'run_simple_baselines') as launch:
        with pytest.raises(ValueError, match='Missing final checkpoint'):
            driver.main(['train'])
        launch.assert_not_called()
        last.touch()
        occupied = Path(matrix['cl_reler', 2026][1]['config']['output_dir'])
        occupied.mkdir()
        (occupied / 'trainer_state.json').write_text('{}')
        with pytest.raises(ValueError, match='Training output already exists'):
            driver.main(['train'])
        launch.assert_not_called()
        (occupied / 'trainer_state.json').unlink()
        assert driver.main(['train']) == 0
        assert [call.args[2] for call in launch.call_args_list] == [
            [matrix['cl_reler', seed][1]['run_id']] for seed in driver.SEEDS]
        assert all(call.args[3] == 'train' for call in launch.call_args_list)
        launch.reset_mock()
        assert driver.main(['eval', '--methods', 'cl', '--seeds', '3407']) == 0
        path, baseline = matrix['cl', 3407]
        launch.assert_called_once_with(path, driver.SETTINGS, [baseline['run_id']], 'eval')


def test_preflight_rejects_a_changed_embedding_protocol(tmp_path):
    matrix = local_matrix(tmp_path)
    protocol = Path(matrix['cl', 3407][1]['config']['output_dir']) / 'embedding_protocol.json'
    payload = json.loads(protocol.read_text())
    payload['pooling_method'] = 'mean'
    protocol.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='embedding protocol differs'):
        driver.preflight(matrix, driver.SEEDS, 'check', ['cl_reler'])


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def test_summary_merges_all_views_and_computes_unrounded_paired_statistics(tmp_path):
    matrix = driver.resolve_matrix()
    catalog = {f'Task{i}': driver.TYPES[min(i // 5, 6)] for i in range(41)}
    expected = {}
    staged = tmp_path / 'staged'
    for (method, seed), (_, row) in matrix.items():
        result = staged / Path(row['config']['output_dir']).name / 'model/revision'
        result.mkdir(parents=True)
        offset = (driver.SEEDS.index(seed) + 1) * (0.00113 if method == 'cl' else 0.00327)
        values = {}
        for i, task in enumerate(catalog):
            value = i / 100 + offset
            values[task] = value * 100
            (result / f'{task}.json').write_text(json.dumps({'scores': {'test': [
                {'hf_subset': 'a', 'main_score': value - 0.001},
                {'hf_subset': 'b', 'main_score': value + 0.001},
            ]}}))
        type_means = [statistics.mean(values[t] for t in catalog if catalog[t] == kind)
                      for kind in driver.TYPES]
        expected[method, seed] = (statistics.mean(values.values()), statistics.mean(type_means))

    destination = tmp_path / 'summary'
    destination.mkdir()
    # Start from the same four-view schema as the existing G2 summary directory.
    old = mteb_summary.RunSummary('existing-direct-run', tmp_path / 'old', 'MTEB(eng, v2)',
                                  catalog, {t: 0.1 for t in catalog}, {}, [])
    old_views = {
        'run': mteb_summary.make_run_rows([old], catalog),
        'type': mteb_summary.make_type_pivot_rows([old]),
        'task': mteb_summary.make_task_rows([old]),
        'retrieval': mteb_summary.make_retrieval_rows([old]),
    }
    for view, rows in old_views.items():
        driver.write_csv(destination / f'{view}_summary.csv', list(rows[0]), rows)
    previous = {view: read_csv(destination / f'{view}_summary.csv') for view in old_views}

    driver.summarize(matrix, destination, catalog, staged)
    payload = json.loads((destination / 'cl_to_reler_results.json').read_text())
    assert all(run['complete'] for run in payload['runs'])
    for run in payload['runs']:
        task_mean, type_mean = expected[run['method'], run['seed']]
        assert run['metrics']['mean_task_score'] == pytest.approx(task_mean)
        assert run['metrics']['mean_type_score'] == pytest.approx(type_mean)
        assert task_mean != pytest.approx(type_mean)
    aggregate = next(r for r in payload['groups']
                     if r['method'] == 'cl_reler' and r['metric'] == 'mean_task_score')
    assert aggregate['sample_sd'] == pytest.approx(
        statistics.stdev(expected['cl_reler', seed][0] for seed in driver.SEEDS))
    paired = next(r for r in payload['paired'] if r['seed'] == 'all' and r['metric'] == 'mean_task_score')
    assert paired['mean'] == pytest.approx(0.428)
    assert paired['sample_sd'] == pytest.approx(0.214)
    for view in old_views:
        rows = read_csv(destination / f'{view}_summary.csv')
        assert [r for r in rows if r['run'] == old.run_id] == previous[view]
        assert len(rows) == (7 * 41 if view == 'task' else 7)

    # An incomplete rerun retains the already saved score and withholds 3-seed claims.
    snapshots = {view: read_csv(destination / f'{view}_summary.csv') for view in old_views}
    last_name = Path(matrix['cl_reler', 2026][1]['config']['output_dir']).name
    last = staged / last_name / 'model/revision/Task0.json'
    last.unlink()
    driver.summarize(matrix, destination, catalog, staged)
    payload = json.loads((destination / 'cl_to_reler_results.json').read_text())
    assert all(r['mean'] is None and r['sample_sd'] is None
               for r in payload['groups'] if r['method'] == 'cl_reler')
    assert all(r['mean'] is None for r in payload['paired'] if r['seed'] == 'all')
    for view in old_views:
        rows = read_csv(destination / f'{view}_summary.csv')
        assert [r for r in rows if r['run'] == last_name] == [
            r for r in snapshots[view] if r['run'] == last_name]
        assert len(rows) == len(snapshots[view])
    last.write_text('{broken')
    with pytest.raises(ValueError, match='invalid MTEB results'):
        driver.summarize(matrix, destination, catalog, staged)
