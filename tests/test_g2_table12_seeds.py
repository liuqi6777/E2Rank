"""CPU checks for the Table 12 queue and complete three-seed MTEB aggregation."""
import copy
import json
from pathlib import Path
import statistics
import sys
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import run_g2_table12_seeds as driver


def test_repeats_preserve_original_resolved_training_configs():
    matrix = driver.resolve_matrix()
    for method in driver.RECIPES:
        original = matrix[method, 42][1]
        for seed in driver.SEEDS:
            row = matrix[method, seed][1]
            actual = copy.deepcopy(row['config'])
            for key in ('seed', 'data_seed', 'run_name', 'output_dir'):
                actual[key] = original['config'][key]
            if method == 'reler':
                assert actual['rollout_seed'] == seed
                actual['rollout_seed'] = 42
            assert actual == original['config']
            assert row['protocol'] == original['protocol']
            assert row['dependency'] is None
            commands = driver.experiments.post_train_commands_for(row)
            assert len(commands) == 1
            command = commands[0]
            assert command[command.index('--benchmark') + 1] == 'MTEB(eng, v2)'
            assert command[command.index('--model') + 1] == row['config']['output_dir']


def test_default_queue_selects_only_four_new_runs_and_eval_can_select_one():
    with patch.object(driver.experiments, 'run_simple_baselines') as run:
        assert driver.main(['check']) == 0
        names = [call.args[2][0] for call in run.call_args_list]
        matrix = driver.resolve_matrix()
        assert names == [matrix[m, s][1]['run_id'] for s in (3407, 2026) for m in driver.RECIPES]
        assert all(call.args[3] == 'check' for call in run.call_args_list)
        run.reset_mock()
        assert driver.main(['eval', '--methods', 'reler', '--seeds', '2026']) == 0
        run.assert_called_once_with(matrix['reler', 2026][0], driver.SETTINGS,
                                    [matrix['reler', 2026][1]['run_id']], 'eval')


def test_summary_uses_task_and_type_weights_and_requires_three_seeds(tmp_path):
    matrix = driver.resolve_matrix()
    # Unequal task counts distinguish task-weighted and type-weighted averages.
    catalog = {f'Task{i}': driver.TYPES[min(i // 5, 6)] for i in range(41)}
    expected = {}
    for (method, seed), (_, row) in matrix.items():
        out = tmp_path / f'{method}-{seed}'
        row['config']['output_dir'] = str(out)
        result = out / 'mteb_eval/final/model/revision'
        result.mkdir(parents=True)
        offset = (driver.SEEDS.index(seed) + 1) * (0.001 if method == 'infonce' else 0.003)
        values = {}
        for i, (task, task_type) in enumerate(catalog.items()):
            value = i / 100 + offset
            values[task] = value * 100
            # The existing MTEB summarizer averages subsets inside each task first.
            (result / f'{task}.json').write_text(json.dumps({'scores': {'test': [
                {'hf_subset': 'a', 'main_score': value - 0.01},
                {'hf_subset': 'b', 'main_score': value + 0.01},
            ]}}))
        type_means = [statistics.mean(values[t] for t in catalog if catalog[t] == kind)
                      for kind in driver.TYPES]
        expected[method, seed] = (statistics.mean(values.values()), statistics.mean(type_means))
    destination = tmp_path / 'summary'
    driver.summarize(matrix, destination, catalog)
    data = json.loads((destination / 'results.json').read_text())
    assert len(data['tasks']) == 6 * 41
    for row in data['runs']:
        task_mean, type_mean = expected[row['recipe'], row['seed']]
        assert row['mean_task_score'] == pytest.approx(task_mean)
        assert row['mean_type_score'] == pytest.approx(type_mean)
        assert task_mean != pytest.approx(type_mean)
    aggregate = next(r for r in data['groups']
                     if r['recipe'] == 'reler' and r['metric'] == 'mean_task_score')
    assert aggregate['sample_sd'] == pytest.approx(statistics.stdev(expected['reler', s][0] for s in driver.SEEDS))
    paired = next(r for r in data['paired'] if r['seed'] == 'all' and r['metric'] == 'mean_task_score')
    assert paired['mean'] == pytest.approx(0.4)
    assert paired['sample_sd'] == pytest.approx(0.2)

    last = Path(matrix['reler', 2026][1]['config']['output_dir']) / 'mteb_eval/final/model/revision/Task0.json'
    last.unlink()
    driver.summarize(matrix, destination, catalog)
    data = json.loads((destination / 'results.json').read_text())
    assert all(r['mean'] is None and r['sample_sd'] is None
               for r in data['groups'] if r['recipe'] == 'reler')
    assert all(r['mean'] is None for r in data['paired'] if r['seed'] == 'all')
    last.write_text('{broken')
    with pytest.raises(ValueError, match='invalid MTEB results'):
        driver.summarize(matrix, destination, catalog)
