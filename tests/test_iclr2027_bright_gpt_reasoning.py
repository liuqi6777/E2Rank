"""Check main-table run selection and isolation of reasoning-query results."""

import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

import eval_iclr2027_gpt_reasoning_06b as small
import eval_iclr2027_gpt_reasoning_4b as large
import eval_iclr2027_gpt_reasoning_bge_m3 as bge
from eval_mteb import run_mteb
from experiments import bright_gpt_reasoning as reasoning
from run_g1_stability import SUBSETS


@pytest.mark.parametrize("spec", [small.SPEC, large.SPEC, bge.SPEC])
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
    legacy = tmp_path / "run/mteb_eval/bright_gpt_reasoning/query-gpt-reasoning/model/BrightRetrieval.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(json.dumps({"scores": {"standard": [
        {"hf_subset": subset, "main_score": 0.99} for subset in SUBSETS]
    }}))
    assert reasoning.result_score(row) is None

    result = reasoning.result_root(row) / "model/BrightRetrieval.json"
    result.parent.mkdir(parents=True)
    entries = [{"hf_subset": subset, "main_score": 0.25} for subset in SUBSETS]
    result.write_text(json.dumps({"scores": {"standard": entries}}))
    assert reasoning.result_score(row) == pytest.approx(25.0)

    result.write_text(json.dumps({"scores": {"standard": entries[:-1]}}))
    with pytest.raises(ValueError):
        reasoning.result_score(row)


@pytest.fixture
def bright_data(monkeypatch):
    records = {
        "examples": [{"id": "q1", "query": "original", "reasoning": "gold gold"}],
        "gpt4_reason": [{"id": "q1", "query": "generated", "reasoning": "gold gold"}],
    }
    corpus = {"d1": {"text": "generated"}, "d2": {"text": "original gold"}}
    qrels = {"q1": {"d1": 1}}
    task = SimpleNamespace(metadata=SimpleNamespace(eval_langs=["biology"]))

    def load_base(**kwargs):
        return (
            {"biology": {"standard": corpus}},
            {"biology": {"standard": {r["id"]: r["query"] for r in records["examples"]}}},
            {"biology": {"standard": qrels}},
        )

    task.load_bright_data = load_base
    monkeypatch.setattr(run_mteb.datasets, "load_dataset",
                        lambda path, config, **kwargs: records[config])
    return task, records, corpus, qrels


@pytest.mark.parametrize("query_set", ["gpt4-reasoning", "gpt-reasoning"])
def test_official_generated_query_retrieves_without_gold_annotations(bright_data, query_set):
    task, _, corpus, qrels = bright_data
    args = run_mteb.EvalArguments(bright_query_set=query_set)
    run_mteb.load_official_bright(task, ["biology"], args)
    # A toy dense retriever distinguishes generated-query evidence from the
    # original question and gold rationale. The legacy input ranks d2 first.
    vocabulary = ("original", "generated", "gold")

    def encode(texts):
        return np.array([[Counter(text.split())[word] for word in vocabulary]
                         for text in texts])

    query_vectors = encode(task.queries["biology"]["standard"].values())
    document_vectors = encode(doc["text"] for doc in corpus.values())
    np.testing.assert_array_equal(query_vectors, [[0, 1, 0]])
    assert int((query_vectors @ document_vectors.T).argmax()) == 0
    assert task.corpus["biology"]["standard"] is corpus
    assert task.relevant_docs["biology"]["standard"] is qrels


@pytest.mark.parametrize("query", [None, "", "  ", "N/A", "empty"])
def test_missing_official_query_aborts_instead_of_falling_back(bright_data, query):
    task, records, _, _ = bright_data
    records["gpt4_reason"][0]["query"] = query
    with pytest.raises(ValueError):
        run_mteb.load_official_bright(
            task, ["biology"], run_mteb.EvalArguments(bright_query_set="gpt4-reasoning"))
    assert not getattr(task, "data_loaded", False)


@pytest.mark.parametrize("case", ["missing", "extra", "duplicate"])
def test_invalid_official_query_ids_abort(bright_data, case):
    task, records, _, _ = bright_data
    if case == "missing":
        records["gpt4_reason"].clear()
    else:
        extra = dict(records["gpt4_reason"][0])
        if case == "extra":
            extra["id"] = "unexpected"
        records["gpt4_reason"].append(extra)
    with pytest.raises((ValueError, RuntimeError)):
        run_mteb.load_official_bright(
            task, ["biology"], run_mteb.EvalArguments(bright_query_set="gpt4-reasoning"))
    assert not getattr(task, "data_loaded", False)


def test_legacy_cli_alias_evaluates_in_new_namespace(bright_data, tmp_path, monkeypatch):
    task, _, _, _ = bright_data
    row = {"run_id": "example", "config": {"output_dir": str(tmp_path)}}
    legacy = tmp_path / "mteb_eval/bright_gpt_reasoning/query-gpt-reasoning/model/BrightRetrieval.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(json.dumps({"scores": {"standard": [
        {"hf_subset": subset, "main_score": 0.99} for subset in SUBSETS]
    }}))

    class Evaluation:
        def __init__(self, **kwargs):
            pass

        def run(self, model, *, output_folder, **kwargs):
            path = Path(output_folder) / "model/BrightRetrieval.json"
            if path.exists():
                payload = json.loads(path.read_text())
            else:
                payload = {"scores": {"standard": [
                    {"hf_subset": subset, "main_score": 0.25} for subset in SUBSETS]}}
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(payload))
            return [SimpleNamespace(**payload)]

    monkeypatch.setattr(run_mteb.mteb, "MTEB", Evaluation)
    args = run_mteb.EvalArguments(
        output_dir=str(tmp_path / "mteb_eval" / reasoning.RESULT_NAME),
        bright_query_set="gpt-reasoning")
    run_mteb.run_bright(task, SimpleNamespace(mteb_model_meta=None), args)
    assert reasoning.result_score(row) == pytest.approx(25.0)
    assert json.loads(legacy.read_text())["scores"]["standard"][0]["main_score"] == 0.99
