"""Scheduling, result isolation, and restart checks without GPU jobs."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import launch_bright_repair as launch


def test_main_jobs_cover_thirty_checkpoints_once_and_balance_sixteen_nodes():
    args = SimpleNamespace(cost_4b=4.0, config_4b=None, config_06b=None, config_bge_m3=None)
    jobs = launch.make_jobs(args)
    slots, loads = launch.allocate(jobs, 16)
    assert len(jobs) == len({j.name for slot in slots for j in slot}) == 30
    assert all(len(slot) == 1 and slot[0].backbone == "4b" for slot in slots[:10])
    assert sorted(map(len, slots[10:])) == [3, 3, 3, 3, 4, 4]
    assert max(loads) == 4.0
    assert slots == launch.allocate(launch.make_jobs(args), 16)[0]


@pytest.fixture
def partial_job(tmp_path):
    job = launch.Job("06b", {"run_id": "test", "config": {"output_dir": str(tmp_path)}}, 1.0)
    for setting in ("bright", "bright_gpt_reasoning/query-gpt-reasoning"):
        legacy = tmp_path / "mteb_eval" / setting / "model/BrightRetrieval.json"
        legacy.parent.mkdir(parents=True)
        legacy.write_text(json.dumps({"scores": {"standard": [
            {"hf_subset": s, "main_score": 0.99} for s in launch.SUBSETS]}}))
    path = launch.result_root(job) / "model/BrightRetrieval.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"scores": {"standard": [
        {"hf_subset": s, "main_score": 0.25} for s in launch.SUBSETS[:9]]}}))
    return job, path


def test_resume_runs_only_missing_domains_then_skips_complete(partial_job, tmp_path, monkeypatch):
    job, path = partial_job
    calls = []
    legacy = {p: p.read_bytes() for p in tmp_path.rglob("BrightRetrieval.json") if p != path}

    def command(job, missing):
        calls.append(set(missing))
        return [sys.executable]

    def evaluate(*a, **kw):
        payload = json.loads(path.read_text())
        payload["scores"]["standard"].extend(
            {"hf_subset": s, "main_score": 0.25} for s in launch.SUBSETS[9:])
        path.write_text(json.dumps(payload))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launch, "command", command)
    monkeypatch.setattr(launch.subprocess, "run", evaluate)
    launch.run_job(job, tmp_path)
    launch.run_job(job, tmp_path)
    assert calls == [set(launch.SUBSETS[9:])]
    _, _, scores = launch.read_result(launch.result_root(job))
    assert len(scores) == 12
    assert list(e["main_score"] for e in scores.values()) == [0.25] * 12
    assert all(p.read_bytes() == data for p, data in legacy.items())


def test_duplicate_worker_cannot_write_same_checkpoint(partial_job):
    job, _ = partial_job
    with launch.job_lock(job):
        with pytest.raises(RuntimeError):
            with launch.job_lock(job):
                pytest.fail("A second worker acquired the same checkpoint")


def test_failed_evaluation_keeps_partial_result_for_retry(partial_job, tmp_path, monkeypatch):
    job, path = partial_job
    original = path.read_bytes()
    monkeypatch.setattr(launch, "command", lambda *a: [sys.executable])
    monkeypatch.setattr(launch.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1))
    with pytest.raises(RuntimeError):
        launch.run_job(job, tmp_path)
    assert path.read_bytes() == original
