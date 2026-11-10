"""CPU coverage for the matched Strong CL continuation and shared G2 exports."""
import copy
import csv
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))

import run_g2_cl_to_cl as driver
from experiments import g2_mteb_summary as summary


def make_model(path, config):
    path.mkdir(parents=True, exist_ok=True)
    (path / 'config.json').write_text('{}')
    (path / 'model.safetensors').touch()
    protocol = {k: config[k] for k in ('pooling_method', 'padding_side', 'append_token',
                                      'query_prompt_template', 'document_prompt_template')}
    protocol.update(tokenization_version=2, pooling_compute_dtype='float32',
                    max_length=config['embedding_max_length'], normalize=True)
    (path / 'embedding_protocol.json').write_text(json.dumps(protocol))


@pytest.fixture
def local_inputs(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'train.jsonl').write_text('{}\n')
    settings = tmp_path / 'settings.json'
    settings.write_text(json.dumps(dict(output_dir=str(tmp_path / 'checkpoints'),
                                       G2=dict(data=str(data)))))
    matrix = driver.resolve_matrix(settings)
    for seed in driver.SEEDS:
        row = matrix['cl', seed][1]
        make_model(Path(row['config']['output_dir']), row['config'])
    return settings, matrix


def test_continuation_preserves_strong_cl_and_matches_reler_budget(local_inputs):
    _, matrix = local_inputs
    assert len({r['config']['output_dir'] for _, r in matrix.values()}) == 12
    for seed in driver.SEEDS:
        baseline = matrix['cl', seed][1]
        reler = matrix['cl_reler', seed][1]
        row = matrix['cl_cl', seed][1]
        cfg = copy.deepcopy(row['config'])
        assert row['dependency'] == reler['dependency'] == baseline['config']['output_dir']
        assert cfg['model_name_or_path'] == row['dependency']
        assert cfg['seed'] == cfg['data_seed'] == seed
        assert cfg['baseline_in_batch_include_negatives']
        assert cfg['baseline_cross_device_negatives']
        assert not cfg['baseline_detach_in_batch_documents']
        assert cfg['max_steps'] == reler['config']['max_steps'] == 1200
        assert cfg['learning_rate'] == reler['config']['learning_rate'] == 5e-6
        assert cfg['per_device_train_batch_size'] * cfg['gradient_accumulation_steps'] * 8 == 128
        assert not cfg['overwrite_output_dir']
        assert not cfg['load_best_model_at_end']
        for key in ('model_name_or_path', 'output_dir', 'run_name'):
            cfg[key] = baseline['config'][key]
        assert cfg == baseline['config']
        assert row['protocol'] == baseline['protocol']
        commands = driver.experiments.post_train_commands_for(row)
        assert len(commands) == 1
        command = commands[0]
        assert command[command.index('--benchmark') + 1] == 'MTEB(eng, v2)'
        assert command[command.index('--model') + 1] == row['config']['output_dir']


def test_preflight_checks_all_seeds_before_only_new_cl_jobs_launch(local_inputs):
    settings, matrix = local_inputs
    last = Path(matrix['cl', 2026][1]['config']['output_dir']) / 'model.safetensors'
    last.unlink()
    train = ['train', '--config', str(settings)]
    with patch.object(driver, 'require_training_gpus'), \
         patch.object(driver.experiments, 'run_simple_baselines') as launch:
        with pytest.raises(ValueError, match='Missing final checkpoint'):
            driver.main(train)
        launch.assert_not_called()
        last.touch()
        occupied = Path(matrix['cl_cl', 2026][1]['config']['output_dir'])
        occupied.mkdir()
        (occupied / 'trainer_state.json').write_text('{}')
        with pytest.raises(ValueError, match='Training output already exists'):
            driver.main(train)
        launch.assert_not_called()
        (occupied / 'trainer_state.json').unlink()
        assert driver.main(train) == 0
        assert [call.args[2] for call in launch.call_args_list] == [
            [matrix['cl_cl', seed][1]['run_id']] for seed in driver.SEEDS]
        assert all(call.args[3] == 'train' for call in launch.call_args_list)
        launch.reset_mock()
        assert driver.main(['eval', '--config', str(settings), '--methods', 'cl', '--seeds', '3407']) == 0
        path, baseline = matrix['cl', 3407]
        launch.assert_called_once_with(path, settings, [baseline['run_id']], 'eval')
        launch.reset_mock()
        with pytest.raises(SystemExit):
            driver.main([*train, '--methods', 'cl_reler'])
        launch.assert_not_called()


def test_launcher_writes_fresh_stage_config_then_evaluates(local_inputs):
    settings, matrix = local_inputs
    source = Path(matrix['cl', 42][1]['config']['output_dir'])
    (source / 'optimizer.pt').touch()
    (source / 'trainer_state.json').write_text('{"global_step": 1200}')
    with patch.object(driver, 'require_training_gpus'), \
         patch.object(driver.experiments.subprocess, 'run') as launch:
        assert driver.main(['train', '--config', str(settings), '--seeds', '42']) == 0
        assert launch.call_count == 2
        train_command, eval_command = [call.args[0] for call in launch.call_args_list]
        row = matrix['cl_cl', 42][1]
        runtime = Path(train_command[-1])
        config = json.loads(runtime.read_text())
        assert config == row['config']
        assert train_command == driver.experiments.command_for(row, runtime, 8)
        assert row['entrypoint'] == 'src/train_baseline.py'
        assert eval_command == driver.experiments.post_train_commands_for(row)[0]
        assert runtime.parent == source.parent / '.cl_strong_configs'
        # The launch writes only the new config: no optimizer/trainer state is
        # copied into the output directory where the trainer detects resumption.
        assert not Path(config['output_dir']).exists()
        assert json.loads((source / 'trainer_state.json').read_text())['global_step'] == 1200


def test_summary_adds_cl_control_and_paired_deltas_without_changing_existing_scores(tmp_path):
    matrix = driver.resolve_matrix()
    catalog = {f'Task{i}': summary.TYPES[min(i // 5, 6)] for i in range(41)}
    staged = tmp_path / 'results'
    destination = tmp_path / 'summary'
    offsets = {'cl': 0.001, 'reler': 0.002, 'cl_reler': 0.004, 'cl_cl': 0.003}

    def write_run(method, seed):
        output = Path(matrix[method, seed][1]['config']['output_dir'])
        result = staged / output.name / 'model/revision'
        result.mkdir(parents=True)
        for i, task in enumerate(catalog):
            value = i / 100 + offsets[method] * (driver.SEEDS.index(seed) + 1)
            (result / f'{task}.json').write_text(json.dumps({'scores': {'test': [
                {'hf_subset': 'a', 'main_score': value - 0.001},
                {'hf_subset': 'b', 'main_score': value + 0.001},
            ]}}))

    for method in ('cl', 'reler', 'cl_reler'):
        for seed in driver.SEEDS:
            write_run(method, seed)
    original_matrix = {key: row for key, row in matrix.items() if key[0] != 'cl_cl'}
    previous = summary.summarize(original_matrix, destination, catalog, staged)
    assert len(previous['runs']) == 9
    write_run('cl_cl', 42)
    write_run('cl_cl', 3407)
    incomplete = summary.summarize(matrix, destination, catalog, staged)
    assert all(r['mean'] is None and r['sample_sd'] is None
               for r in incomplete['groups'] if r['recipe'] == 'cl_cl')
    assert all(r['mean'] is None for r in incomplete['paired']
               if r['seed'] == 'all' and r['comparison'] == 'cl_reler_minus_cl_cl')

    write_run('cl_cl', 2026)
    result = summary.summarize(matrix, destination, catalog, staged)
    assert len(result['runs']) == 12
    assert len(result['tasks']) == 492
    assert [r for r in result['runs'] if r['recipe'] != 'cl_cl'] == previous['runs']
    assert [t for t in result['tasks'] if t['recipe'] != 'cl_cl'] == previous['tasks']
    for comparison, mean, sd in [('cl_cl_minus_infonce', 0.4, 0.2),
                                  ('cl_reler_minus_cl_cl', 0.2, 0.1)]:
        paired = next(r for r in result['paired'] if r['comparison'] == comparison
                      and r['metric'] == 'Retrieval' and r['seed'] == 'all')
        assert paired['completed'] == 3
        assert paired['mean'] == pytest.approx(mean)
        assert paired['sample_sd'] == pytest.approx(sd)
    for view, count in [('run', 12), ('type', 12), ('retrieval', 12), ('task', 492)]:
        with (destination / f'{view}_summary.csv').open() as handle:
            assert len(list(csv.DictReader(handle))) == count
    # Refreshing the earlier entry point must also retain the new control.
    refreshed = summary.summarize(original_matrix, destination, catalog, staged)
    assert refreshed == result
