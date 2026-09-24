"""Check main-table run selection and isolation of reasoning-query results."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_iclr2027_gpt_reasoning_06b as small
import eval_iclr2027_gpt_reasoning_4b as large
from experiments import bright_gpt_reasoning as reasoning
from run_g1_stability import SUBSETS


@pytest.mark.parametrize("spec", [small.SPEC, large.SPEC])
def test_main_table_selects_one_e0_and_three_seeds_per_method(spec):
    names = reasoning.selected_runs(spec, list(reasoning.METHODS), list(reasoning.SEEDS))
    rows = reasoning.resolve_rows(spec, spec.settings, names)
    assert len(rows) == len(set(names)) == 10
    assert sum(row["kind"] == "evaluation" for row in rows) == 1
    assert sum(row["kind"] == "train" for row in rows) == 9
    other_seed_names = reasoning.selected_runs(
        spec, list(reasoning.METHODS), [3407]
    )
    assert len(other_seed_names) == 3


def test_reasoning_result_is_separate_from_original_and_requires_all_subsets(tmp_path):
    row = {"run_id": "example", "config": {"output_dir": str(tmp_path / "run")}}
    original = tmp_path / "run/mteb_eval/bright/model/BrightRetrieval.json"
    original.parent.mkdir(parents=True)
    original.write_text("{}")
    assert reasoning.result_score(row) is None

    result = reasoning.result_root(row) / "model/BrightRetrieval.json"
    result.parent.mkdir(parents=True)
    entries = [{"hf_subset": subset, "main_score": 0.25} for subset in SUBSETS]
    result.write_text(json.dumps({"scores": {"standard": entries}}))
    assert reasoning.result_score(row) == pytest.approx(25.0)

    result.write_text(json.dumps({"scores": {"standard": entries[:-1]}}))
    with pytest.raises(ValueError):
        reasoning.result_score(row)
