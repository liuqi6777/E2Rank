"""Numerical aggregation and incomplete-result checks without model dependencies."""

import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import summarize_bright_repair as summary


@pytest.fixture
def results(tmp_path):
    args = SimpleNamespace(cost_4b=4.0, config_4b=None, config_06b=None, config_bge_m3=None)
    jobs = summary.launch.make_jobs(args)
    offsets = {42: 0.1, 3407: 0.2, 2026: 0.3}
    for job in jobs:
        job.row["config"]["output_dir"] = str(tmp_path / job.backbone / job.row["run_id"])
        path = summary.launch.result_root(job) / "model/revision/BrightRetrieval.json"
        path.parent.mkdir(parents=True)
        entries = [{"hf_subset": subset, "main_score": offsets[job.row["config"]["seed"]] + i * 0.01}
                   for i, subset in enumerate(summary.SUBSETS)]
        path.write_text(json.dumps({"task_name": "BrightRetrieval", "scores": {"standard": entries}}))
    return jobs, tmp_path / "logs"


def result_path(job):
    return next(summary.launch.result_root(job).rglob("BrightRetrieval.json"))


def test_complete_means_sample_sd_and_csv_roundtrip(results, tmp_path):
    jobs, logs = results
    before = {result_path(j): result_path(j).read_bytes() for j in jobs}
    report = summary.collect(jobs, logs)
    assert report["completed_runs"] == 30
    assert len(report["groups"]) == 12
    for group in report["groups"]:
        if group["method"] == "e0":
            assert group["mean"] == pytest.approx(15.5)
            assert group["sample_sd"] is None
        else:
            assert group["mean"] == pytest.approx(25.5)
            assert group["sample_sd"] == pytest.approx(10.0)
            assert list(group["subset_means"].values()) == pytest.approx([20 + i for i in range(12)])
            assert list(group["subset_sample_sds"].values()) == pytest.approx([10.0] * 12)
    out = tmp_path / "summary"
    summary.write_report(report, out)
    with (out / "subset_summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 360
    assert all(float(r["score"]) == pytest.approx(float(r["ndcg_at_10"]) * 100) for r in rows)
    with (out / "pending.csv").open() as stream:
        assert not list(csv.DictReader(stream))
    assert json.loads((out / "summary.json").read_text())["completed_runs"] == 30
    assert all(path.read_bytes() == data for path, data in before.items())


def test_partial_seed_does_not_produce_group_or_run_average(results, tmp_path):
    jobs, logs = results
    path = result_path(jobs[1])  # First trained method, seed 42.
    payload = json.loads(path.read_text())
    payload["scores"]["standard"].pop()
    path.write_text(json.dumps(payload))
    report = summary.collect(jobs, logs)
    assert report["completed_runs"] == 29
    assert report["runs"][1]["status"] == "partial"
    assert report["runs"][1]["completed_subsets"] == 11
    assert report["runs"][1]["avg_ndcg_at_10"] is None
    group = next(g for g in report["groups"] if g["backbone"] == jobs[1].backbone and g["method"] == "infonce")
    assert group["completed_runs"] == 2
    assert group["mean"] is group["sample_sd"] is None
    assert not group["subset_means"]
    out = tmp_path / "summary"
    summary.write_report(report, out)
    with (out / "per_run_summary.csv").open() as stream:
        row = list(csv.DictReader(stream))[1]
    assert row["avg_ndcg_at_10"] == ""


@pytest.mark.parametrize("case", ["duplicate_subset", "ambiguous_file", "nan", "out_of_range", "metric_mismatch", "corrupt_json"])
def test_invalid_result_is_reported_without_blocking_other_runs(results, case):
    jobs, logs = results
    path = result_path(jobs[0])
    payload = json.loads(path.read_text())
    entries = payload["scores"]["standard"]
    if case == "duplicate_subset":
        entries.append(dict(entries[0]))
    elif case == "ambiguous_file":
        other = path.parent / "duplicate/BrightRetrieval.json"
        other.parent.mkdir()
        other.write_text(json.dumps(payload))
    elif case == "nan":
        entries[0]["main_score"] = float("nan")
    elif case == "out_of_range":
        entries[0]["main_score"] = 1.1
    elif case == "metric_mismatch":
        entries[0]["ndcg_at_10"] = 0.9
    path.write_text("{" if case == "corrupt_json" else json.dumps(payload))
    report = summary.collect(jobs, logs)
    assert report["completed_runs"] == 29
    assert report["runs"][0]["status"] == "invalid"
    assert report["runs"][0]["error"]
    assert report["runs"][0]["avg_ndcg_at_10"] is None


def test_old_results_ignored_and_current_completion_supersedes_failure_log(results):
    jobs, logs = results
    for job in jobs[:2]:
        old = summary.launch.result_root(job).parent / "query-gpt-reasoning/model/BrightRetrieval.json"
        old.parent.mkdir(parents=True)
        old.write_bytes(result_path(job).read_bytes())
    result_path(jobs[0]).unlink()
    failure_path = logs / "slot-00/failures.json"
    failure_path.parent.mkdir(parents=True)
    failure_path.write_text(json.dumps([{"job": j.name, "error": "old failure"} for j in jobs[:2]]))
    report = summary.collect(jobs, logs)
    assert report["runs"][0]["status"] == "failed"
    assert report["runs"][0]["completed_subsets"] == 0
    assert report["runs"][0]["avg_ndcg_at_10"] is None
    assert report["runs"][1]["status"] == "complete"
    assert not report["runs"][1]["failures"]


def test_require_complete_exits_nonzero_but_still_exports_report(results, tmp_path, monkeypatch):
    jobs, logs = results
    monkeypatch.setattr(summary.launch, "make_jobs", lambda args: jobs)
    out = tmp_path / "summary"
    cmd = ["--output-dir", str(out), "--log-dir", str(logs), "--require-complete"]
    assert summary.main(cmd) == 0
    result_path(jobs[0]).unlink()
    assert summary.main(cmd) == 1
    assert json.loads((out / "summary.json").read_text())["completed_runs"] == 29
