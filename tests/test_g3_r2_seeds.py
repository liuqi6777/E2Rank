"""CPU checks for the rebuttal E1 seed matrix and unchanged RAG recipes."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_g3_r2 as driver
from experiments import iclr2027 as experiments


def test_e1_repeats_change_only_seed_and_run_identity():
    suite = experiments.apply_settings(experiments.load_suite(driver.SUITE), driver.SETTINGS)
    outputs = set()
    for base in driver.PAPER_RUNS:
        original = experiments.resolve_run(suite, driver.SUITE, base, nproc=8)
        reference = original["config"]
        assert reference["rag_anchor_coef"] == 0.5
        assert reference["rag_objective"] in {"infonce", "rl"}
        if reference["rag_objective"] == "rl":
            assert reference["rag_group_size"] == 32
            assert reference["rag_target_alignment"] == 0.80
        for seed in driver.SEEDS:
            name, = driver.select_runs([base], [seed])
            resolved = experiments.resolve_run(suite, driver.SUITE, name, nproc=8)
            config = resolved["config"]
            assert config["seed"] == config["data_seed"] == seed
            assert config["output_dir"] not in outputs
            outputs.add(config["output_dir"])
            ignored = {"seed", "data_seed", "output_dir", "run_name"}
            assert {k: v for k, v in config.items() if k not in ignored} == {
                k: v for k, v in reference.items() if k not in ignored
            }
            assert resolved["protocol"] == original["protocol"]
            assert resolved["entrypoint"] == "src/train_rag.py"
    assert len(outputs) == 12


def test_seed_selection_preserves_legacy_and_limits_e1_to_eight_new_runs():
    assert driver.select_runs() == driver.LEGACY_RUNS
    selected = driver.select_runs(seeds=[3407, 2026])
    assert len(selected) == len(set(selected)) == 8
    suite = experiments.load_suite(driver.SUITE)
    assert {experiments.run_seed(suite, name) for name in selected} == {3407, 2026}
    assert driver.select_runs([selected[0], selected[0]], [42, 42]) == [driver.PAPER_RUNS[0]]


def test_unregistered_repeat_is_rejected_before_launch():
    with pytest.raises(ValueError, match="No registered seed"):
        driver.select_runs(["G3-R2-CL-Binary"], [3407])
