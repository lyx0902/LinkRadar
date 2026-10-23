"""Evidence-preserving resume and process-tree deadline regressions."""
from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import psutil
import pytest

from parallel_bc import benchmark
from parallel_bc.brandes import brandes_serial
from parallel_bc.data import graph_from_edges, save_graph


@pytest.fixture
def mock_experiment(tmp_path, monkeypatch):
    """Use real tiny Brandes scores, fake only expensive supervision/export."""
    import parallel_bc.plotting as plotting
    import parallel_bc.reporting as reporting

    data_dir = tmp_path / "data"
    graph = graph_from_edges([(10, 20), (20, 30)], nodes=[10, 20, 30])
    save_graph(graph, data_dir / "processed" / "tiny")
    config = dict(seed=42, start_method="spawn", datasets=[dict(id="tiny")],
                  benchmark=dict(modes=["serial", "dynamic_batches"], processes=[2],
                      batch_sizes=[1], repeats=1, warmups=0, task_logging=True,
                      max_processes=2, max_case_seconds=30, max_total_seconds=120),
                  validation=dict(enabled=True, max_reference_nodes=1000),
                  robustness=dict(enabled=False))
    monkeypatch.setattr(benchmark, "source_fingerprint", lambda: "regression-source-fingerprint")
    monkeypatch.setattr(benchmark, "environment", lambda: {"test_fixture": True})
    monkeypatch.setattr(plotting, "plot_results", lambda root: [])
    monkeypatch.setattr(reporting, "generate_report", lambda root: Path(root) / "report.md")

    def successful_call(kind, candidate, params, seconds):
        assert seconds > 0
        result = brandes_serial(candidate.adj, batch_size=params.get("batch_size", 1),
                                task_logging=params.get("task_logging", False))
        if kind == "preflight":
            return "ok", (result, {"allclose": True, "finite": True}), None
        assert kind == "algorithm"
        return "ok", result, None

    monkeypatch.setattr(benchmark, "bounded_call", successful_call)
    return data_dir, config, successful_call


def _read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _read_tasks(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def test_resume_rejects_changed_graph_before_overwriting_original_evidence(tmp_path, mock_experiment):
    data_dir, config, _ = mock_experiment
    root = benchmark.run_benchmark(config, data_dir=data_dir, output_dir=tmp_path / "results")
    directory = root / "tiny"
    protected = {path.relative_to(directory): path.read_bytes()
                 for path in (directory / "graph").rglob("*") if path.is_file()}
    protected[Path("graph_metadata.json")] = (directory / "graph_metadata.json").read_bytes()
    changed = graph_from_edges([(10, 20), (20, 30), (10, 30)], nodes=[10, 20, 30])
    save_graph(changed, data_dir / "processed" / "tiny")
    with pytest.raises(ValueError, match="图指纹"):
        benchmark.run_benchmark(config, data_dir=data_dir, resume=root)
    for relative, expected in protected.items():
        assert (directory / relative).read_bytes() == expected, f"resume overwrote {relative}"


def test_interrupt_then_resume_preserves_completed_case_worker_and_task_evidence(
        tmp_path, monkeypatch, mock_experiment):
    data_dir, config, successful_call = mock_experiment
    calls = {"algorithm": 0}

    def interrupt_second_algorithm(kind, graph, params, seconds):
        if kind == "algorithm":
            calls["algorithm"] += 1
            if calls["algorithm"] == 2:
                raise KeyboardInterrupt
        return successful_call(kind, graph, params, seconds)

    monkeypatch.setattr(benchmark, "bounded_call", interrupt_second_algorithm)
    with pytest.raises(KeyboardInterrupt):
        benchmark.run_benchmark(config, data_dir=data_dir, output_dir=tmp_path / "results")
    manifest_file, = (tmp_path / "results").glob("*/manifest.json")
    root = manifest_file.parent
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    completed = [row for row in manifest["cases"] if row["status"] == "ok"]
    assert len(completed) == 1
    key = completed[0]["case_key"]
    worker_file = root / "tiny" / "worker_summary.csv"
    task_file = root / "tiny" / "tasks.jsonl"
    assert worker_file.exists(), "completed case must persist worker evidence before next case"
    assert task_file.exists(), "completed case must persist task evidence before next case"
    old_workers = [r for r in _read_csv(worker_file) if r["case_key"] == key]
    old_tasks = [r for r in _read_tasks(task_file) if r["case_key"] == key]
    assert old_workers and old_tasks
    resumed_calls = []

    def count_resumed(kind, graph, params, seconds):
        if kind == "algorithm":
            resumed_calls.append(params)
        return successful_call(kind, graph, params, seconds)

    monkeypatch.setattr(benchmark, "bounded_call", count_resumed)
    benchmark.run_benchmark(config, data_dir=data_dir, resume=root)
    assert len(resumed_calls) == 1, "the completed case must be reused rather than remeasured"
    new_workers = _read_csv(worker_file)
    new_tasks = _read_tasks(task_file)
    assert [r for r in new_workers if r["case_key"] == key] == old_workers
    assert [r for r in new_tasks if r["case_key"] == key] == old_tasks
    finished = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert len(finished["cases"]) == 2
    assert all(row["status"] == "ok" for row in finished["cases"])
    assert set(r["case_key"] for r in new_workers) == set(r["case_key"] for r in finished["cases"])
    assert set(r["case_key"] for r in new_tasks) == set(r["case_key"] for r in finished["cases"])


def _long_lived_grandchild():
    time.sleep(60)


def _spawn_grandchildren_case(conn, kind, graph, params):
    """Actual spawned grandchildren, without waiting for costly graph computation."""
    ctx = mp.get_context("spawn")
    children = [ctx.Process(target=_long_lived_grandchild) for _ in range(2)]
    for child in children:
        child.start()
    Path(params["marker"]).write_text(json.dumps({"supervisor": os.getpid(),
        "workers": [child.pid for child in children]}), encoding="utf-8")
    time.sleep(60)


def test_hard_deadline_reclaims_already_started_grandchildren(tmp_path, monkeypatch):
    marker = tmp_path / "descendant_pids.json"
    graph = graph_from_edges([(1, 2)])
    monkeypatch.setattr(benchmark, "_case_child", _spawn_grandchildren_case)
    before = {process.pid for process in mp.active_children()}
    started = time.perf_counter()
    known_pids = []
    try:
        status, result, error = benchmark.bounded_call("algorithm", graph,
            {"marker": str(marker)}, 3.0)
        assert marker.exists(), "budget must expire after grandchildren were actually spawned"
        evidence = json.loads(marker.read_text(encoding="utf-8"))
        known_pids = [evidence["supervisor"], *evidence["workers"]]
        assert len(set(known_pids)) == 3
        assert status == "timeout" and result is None
        assert time.perf_counter() - started < 12
        assert {process.pid for process in mp.active_children()} <= before
        remaining = [pid for pid in known_pids if psutil.pid_exists(pid)]
        assert not remaining, f"budget left supervisor/worker processes alive: {remaining}"
    finally:
        # A regression must not leave a 60-second background process in the test host.
        if marker.exists() and not known_pids:
            evidence = json.loads(marker.read_text(encoding="utf-8"))
            known_pids = [evidence["supervisor"], *evidence["workers"]]
        for pid in known_pids:
            try:
                process = psutil.Process(pid)
                process.kill()
                process.wait(timeout=3)
            except psutil.Error:
                pass


def test_resume_recovers_preflight_deferred_cases_without_stale_failed_keys(
        tmp_path, monkeypatch, mock_experiment):
    data_dir, config, successful_call = mock_experiment

    def failed_preflight(kind, graph, params, seconds):
        assert kind == "preflight"
        return "timeout", None, "simulated preflight deadline"

    monkeypatch.setattr(benchmark, "bounded_call", failed_preflight)
    root = benchmark.run_benchmark(config, data_dir=data_dir, output_dir=tmp_path / "results")
    initial = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert len(initial["cases"]) == 2
    assert {row["status"] for row in initial["cases"]} == {"not_run_preflight"}
    initial_keys = {row["case_key"] for row in initial["cases"]}
    monkeypatch.setattr(benchmark, "bounded_call", successful_call)
    benchmark.run_benchmark(config, data_dir=data_dir, resume=root)
    completed = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert len(completed["cases"]) == 2
    assert {row["case_key"] for row in completed["cases"]} == initial_keys
    assert {row["status"] for row in completed["cases"]} == {"ok"}
    assert completed["status"] == "complete"


def test_resume_failed_preflight_preserves_previously_measured_root_timings(
        tmp_path, monkeypatch, mock_experiment):
    data_dir, config, _ = mock_experiment
    root = benchmark.run_benchmark(config, data_dir=data_dir, output_dir=tmp_path / "results")
    before = _read_csv(root / "timings_raw.csv")
    assert len(before) == 2 and all(row["status"] == "ok" for row in before)

    def failed_preflight(kind, graph, params, seconds):
        assert kind == "preflight"
        return "timeout", None, "simulated retry deadline"

    monkeypatch.setattr(benchmark, "bounded_call", failed_preflight)
    benchmark.run_benchmark(config, data_dir=data_dir, resume=root)
    assert _read_csv(root / "timings_raw.csv") == before
