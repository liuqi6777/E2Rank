"""CPU checks for the paper suite and its completed-run import boundary."""

import dataclasses
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import run_iclr2027_final_k0 as final_suite
from experiments import iclr2027 as experiments
from run_g1_stability import SUBSETS

from config import RLArguments


def test_all_new_recipes_parse_and_legacy_configs_match():
    suite = final_suite.resolved_suite(final_suite.SETTINGS)
    fields = {field.name for field in dataclasses.fields(RLArguments)}
    new_runs = set(suite['runs']) - set(suite['imports'])
    assert len(new_runs) == 36
    for name in new_runs:
        row = experiments.resolve_run(suite, final_suite.SUITE, name, nproc=8)
        RLArguments(**{key: value for key, value in row['config'].items() if key in fields})
    for name, entry in suite['imports'].items():
        new = experiments.resolve_run(suite, final_suite.SUITE, name, nproc=8)
        old = final_suite.old_row(entry, final_suite.SOURCE_SETTINGS, None)
        final_suite.ensure_same_training_config(new, old)


def test_import_copies_only_complete_results_and_preserves_existing_target(tmp_path):
    suite = final_suite.resolved_suite(final_suite.SETTINGS)
    name = final_suite.BASE
    new = experiments.resolve_run(suite, final_suite.SUITE, name, nproc=8)
    old = final_suite.old_row(suite['imports'][name], final_suite.SOURCE_SETTINGS, None)
    old['config']['output_dir'] = str(tmp_path / 'old' / name)
    new['config']['output_dir'] = str(tmp_path / 'new' / name)
    source = Path(old['config']['output_dir'])
    source.mkdir(parents=True)
    (source / 'config.json').write_text('{}')
    (source / 'model.safetensors').write_bytes(b'example weights')
    cfg = old['config']
    protocol = {
        'tokenization_version': 2, 'pooling_compute_dtype': 'float32',
        'pooling_method': cfg['pooling_method'], 'padding_side': cfg['padding_side'],
        'append_token': cfg['append_token'], 'max_length': cfg['embedding_max_length'],
        'query_prompt_template': cfg['query_prompt_template'],
        'document_prompt_template': cfg['document_prompt_template'],
    }
    (source / 'embedding_protocol.json').write_text(json.dumps(protocol))
    with pytest.raises(ValueError):
        final_suite.import_one(new, old)
    result = source / 'mteb_eval' / 'bright' / 'run' / 'BrightRetrieval.json'
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps({'scores': {'standard': [
        {'hf_subset': subset, 'main_score': 0.2} for subset in SUBSETS
    ]}}))
    assert final_suite.import_one(new, old, dry_run=True) == 'ready'
    assert not Path(new['config']['output_dir']).exists()
    assert final_suite.import_one(new, old) == 'copied'
    target = Path(new['config']['output_dir'])
    assert (target / 'model.safetensors').read_bytes() == b'example weights'
    assert final_suite.import_one(new, old) == 'already_complete'
    (target.parent / '.imports' / f'{target.name}.json').unlink()
    with pytest.raises(ValueError):
        final_suite.import_one(new, old)
