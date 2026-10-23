from pathlib import Path
import json
import multiprocessing as mp
import subprocess
import sys
import time

import numpy as np
import pytest

from parallel_bc.benchmark import bounded_call, summarize, run_benchmark
from parallel_bc.data import graph_from_edges


def test_summary_excludes_failures_and_warmups():
    base = dict(dataset_id="g", graph_sha256="h", mode="serial", p_used=1, batch_size=8,
                source_order_hash="s", task_logging=False, n=3, m=2, status="ok", warmup=False)
    rows = [dict(base, t_total=2, t_steady=1), dict(base, t_total=4, t_steady=3),
            dict(base, t_total=999, t_steady=999, warmup=True),
            dict(base, t_total=None, t_steady=None, status="timeout")]
    result = summarize(rows)[0]
    assert result["count"] == 2
    assert result["median_total"] == 3
    assert result["speedup_total"] == 1


def test_hard_budget_reaps_processes():
    graph = graph_from_edges(((i, i + 1) for i in range(300)), nodes=range(301))
    before = {p.pid for p in mp.active_children()}
    start = time.perf_counter()
    status, result, error = bounded_call("algorithm", graph, dict(mode="dynamic_batches", processes=2, batch_size=1, timeout_seconds=30), .05)
    assert status == "timeout"
    assert result is None
    assert time.perf_counter() - start < 10
    assert {p.pid for p in mp.active_children()} <= before


def test_budget_records_not_run(tmp_path):
    config = dict(seed=42, datasets=[dict(id="tiny", generator=dict(family="er", nodes=10))],
                  benchmark=dict(modes=["serial", "dynamic_batches"], processes=[2], batch_sizes=[2], repeats=1,
                                 max_case_seconds=1, max_total_seconds=.000001), robustness=dict(enabled=False))
    root = run_benchmark(config, output_dir=tmp_path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "partial"
    assert manifest["limitations"]
    assert (root / "report.md").exists()


def cli(*args):
    return subprocess.run([sys.executable, "-m", "parallel_bc", *map(str, args)], capture_output=True, text=True, encoding="utf-8", timeout=120)


def test_cli_help():
    result = cli("--help")
    assert result.returncode == 0, result.stderr
    assert "centrality" in result.stdout


def test_manual_import_cli(tmp_path):
    path = tmp_path / "含 空格.txt"
    path.write_text("8 3\n3 8\n9 9\n", encoding="utf-8")
    result = cli("data", "import", "--dataset", "manual", "--path", path, "--data-dir", tmp_path / "data")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["n"] == 3


def test_cli_offline_smoke(tmp_path):
    result = cli("demo", "--profile", "smoke", "--data-dir", tmp_path / "data", "--output-dir", tmp_path / "results")
    assert result.returncode == 0, result.stdout + result.stderr
    manifests = list((tmp_path / "results").glob("*/manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert len(manifest["cases"]) == 5
    assert all(r["status"] == "ok" for r in manifest["cases"])
    assert (manifests[0].parent / "report.md").exists()
    assert list((manifests[0].parent / "figures").rglob("*.png"))
