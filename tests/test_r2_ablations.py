"""Exercise real ablation recipes, isolated scheduling and result provenance."""
import copy
from dataclasses import fields
import json
from pathlib import Path
import sys

import pytest
import torch
from transformers import Qwen3Config, Qwen3Model

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
import run_g1_r2_ablations as ab
from config import RLArguments
from grpo import GRPOModel


@pytest.fixture(scope='module')
def recipes():
    return ab.resolve(variants=tuple(ab.definitions()))[0]


def test_distinct_outputs_and_matched_controls(recipes):
    assert len(recipes) == 78
    assert len({r['config']['output_dir'] for r in recipes}) == len(recipes)
    assert {r['config']['seed'] for r in recipes} == {42, 3407, 2026}
    for row in recipes:
        cfg = row['config']
        assert cfg['seed'] == cfg['data_seed'] == cfg['rollout_seed']
        assert cfg['max_steps'] == 113
        assert cfg['per_device_train_batch_size'] * cfg['gradient_accumulation_steps'] * 8 == 128
        args = RLArguments(**{f.name: cfg[f.name] for f in fields(RLArguments) if f.name in cfg})
        assert args.kl_coef == args.aux_infonce_coef == 0
        assert cfg['document_encoder_mode'] == 'joint'
    assert len(ab.resolve()[0]) == 9  # Optional experiments never join the default queue.
    by_variant = {r['variant']: r for r in recipes if r['config']['seed'] == 42}
    gaussian = by_variant['gaussian_k755']['config']
    vmf = by_variant['vmf_k755']['config']
    differences = {k for k in gaussian if gaussian[k] != vmf[k]}
    assert differences == {'sampling_law', 'output_dir', 'run_name'}
    for point in ('040', '053', '065', '080', '095', '098'):
        a, b = [by_variant[f'align{point}_{est}']['config'] for est in ('sf', 'cp')]
        assert {k for k in a if a[k] != b[k]} == {'gradient_estimator', 'run_name', 'output_dir'}


@pytest.mark.parametrize('variant', list(ab.definitions()))
def test_real_graded_forward_backward_for_each_variant(variant, recipes):
    """New combinations must execute, not merely pass a YAML/string check."""
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(18)
        cfg = next(r['config'] for r in recipes if r['variant'] == variant)
        args = RLArguments(**{f.name: cfg[f.name] for f in fields(RLArguments) if f.name in cfg})
        backbone = Qwen3Model(Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, head_dim=8,
            max_position_embeddings=32, attention_dropout=0., use_cache=False))
        model = GRPOModel(backbone, args)
        model.train()
        model.grpo.exploration.set_step(50, cfg['max_steps'])
        def tokens(n):
            ids = torch.randint(1, 32, (n, 5))
            return dict(input_ids=ids, attention_mask=torch.ones_like(ids))
        output = model(query=tokens(2), positive_document=tokens(2), negative_document=tokens(4),
                       relevance_labels=torch.tensor([[3., 1., 0.], [3., 2., 0.]]))
        assert torch.isfinite(output.loss)
        output.loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.double().square().sum().item() for g in grads) > 0
    finally:
        torch.set_num_threads(previous)


def test_independent_runs_lock_without_blocking_other_variants(tmp_path):
    with ab.run_locks(tmp_path, ['sf32-s42']):
        with ab.run_locks(tmp_path, ['cp32-s42', 'sf32-s3407']):
            with pytest.raises(ValueError):
                with ab.run_locks(tmp_path, ['sf32-s42']):
                    pytest.fail('overlapping run acquired lock')
    with ab.run_locks(tmp_path, ['sf32-s42']):
        pass


def save_model(row):
    out = Path(row['config']['output_dir'])
    out.mkdir(parents=True)
    (out / 'config.json').write_text('{}')
    (out / 'model.safetensors').write_bytes(b'queue test fixture')
    cfg = row['config']
    protocol = {k: cfg[k] for k in ('pooling_method', 'padding_side', 'append_token',
                                  'query_prompt_template', 'document_prompt_template')}
    protocol.update(tokenization_version=2, pooling_compute_dtype='float32', max_length=8192)
    ab.night.write_json(out / 'embedding_protocol.json', protocol)


def save_scores(row, score):
    path = Path(row['config']['output_dir']) / 'mteb_eval/bright/BrightRetrieval.json'
    ab.night.write_json(path, {'scores': {'standard': [
        {'hf_subset': s, 'main_score': score} for s in ab.night.SUBSETS]}})


def test_retry_eval_does_not_repeat_training_and_records_time(tmp_path):
    row = ab.resolve(variants=['g32_cp'], seeds=[42], output_root=tmp_path)[0][0]
    contract = ab.night.contract_for(row, 'a'*64, {'src/grpo.py': 'code'})
    folder = tmp_path / 'receipt'
    calls = []
    def failed_eval(command, log):
        calls.append(log.stem)
        if log.stem == 'train':
            save_model(row)
            return 0
        return 7
    with pytest.raises(ValueError):
        ab.execute(row, contract, folder, failed_eval, runtime={'nproc': 8})
    def retry(command, log):
        calls.append(log.stem)
        save_scores(row, .20)
        return 0
    assert ab.execute(row, contract, folder, retry) == 'complete'
    assert ab.execute(row, contract, folder, retry) == 'skipped_complete'
    assert calls == ['train', 'eval', 'eval']
    timings = [json.loads(x) for x in (folder / 'timings.jsonl').read_text().splitlines()]
    assert [x['exit_code'] for x in timings] == [0, 7, 0]
    assert all(x['seconds'] >= 0 for x in timings)
    assert json.loads((folder / 'runtime.json').read_text())['nproc'] == 8


def save_receipt(row, directory, score, data_hash='a'*64, source_hash='same'):
    contract = ab.night.contract_for(row, data_hash, {'src/grpo.py': source_hash})
    folder = directory / row['run_id']
    ab.night.write_json(folder / 'contract.json', contract)
    ab.night.write_json(folder / 'state.json', dict(contract_sha256=ab.night.fingerprint(contract),
                        train_started=True, train_complete=True, evaluation_complete=True))
    save_scores(row, score)


def test_summary_uses_complete_seeds_and_blocks_mismatched_provenance(tmp_path):
    rows, controls, directory, refdir = ab.resolve(variants=['g32_cp', 'g32_sf'],
        output_root=tmp_path/'new', reference_root=tmp_path/'old')
    for row in rows:
        save_receipt(row, directory, .21 if row['variant']=='g32_cp' else .18)
    for row in controls:
        save_receipt(row, refdir, .20)
    report = ab.summarize(rows, controls, directory, refdir, directory)
    pair = next(c for c in report['comparisons'] if (c['left'],c['right'])==('g32_cp','g32_sf'))
    assert pair['mean_delta'] == pytest.approx(3)
    assert report['comparable']
    for field, kwargs in [('data_consistent', {'data_hash':'b'*64}),
                          ('source_consistent', {'source_hash':'changed'})]:
        save_receipt(rows[0], directory, .21, **kwargs)
        report = ab.summarize(rows, controls, directory, refdir, directory)
        assert not report[field]
        assert all(g['mean'] is None for g in report['groups'])
        assert all(c['mean_delta'] is None for c in report['comparisons'])
    save_receipt(rows[0], directory, .21)
    (directory / rows[0]['run_id'] / 'state.json').unlink()
    report = ab.summarize(rows, controls, directory, refdir, directory)
    cp = next(g for g in report['groups'] if g['variant']=='g32_cp')
    assert cp['completed'] == 2 and cp['mean'] is None


def test_preflight_rejects_changed_recipe_without_touching_outputs(tmp_path):
    rows, _, directory, _ = ab.resolve(variants=['g32_sf'], seeds=[42], output_root=tmp_path)
    row = rows[0]
    save_model(row)
    save_receipt(row, directory, .19)
    contract = ab.night.contract_for(row, 'a'*64, {'src/grpo.py':'same'})
    ab.preflight(rows, directory, {row['run_id']:contract})
    changed = copy.deepcopy(contract)
    changed['resolved']['config']['group_size'] = 16
    with pytest.raises(ValueError):
        ab.preflight(rows, directory, {row['run_id']:changed})
    assert (Path(row['config']['output_dir'])/'model.safetensors').read_bytes() == b'queue test fixture'
