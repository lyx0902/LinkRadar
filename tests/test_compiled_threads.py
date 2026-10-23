"""Native source partitioning, real OS thread evidence, errors and supervision."""
import json
import multiprocessing as mp
import threading

import networkx as nx
import numpy as np
import psutil
import pytest

pytest.importorskip("numba")

from parallel_bc.brandes import finalize_scores
from parallel_bc.compiled_scaling import run_scaling
from parallel_bc.compiled_threads import threaded_contributions
from parallel_bc.csr import csr_from_edges, save_csr
from parallel_bc.large_benchmark import bounded_job, subset_contributions


@pytest.mark.parametrize("p", [1, 2, 4])
@pytest.mark.parametrize("kind", ["random", "disconnected", "diamond"])
def test_full_and_subset_against_reference(p, kind, tmp_path):
    g = (nx.gnp_random_graph(29, .2, seed=42) if kind == "random" else
         nx.disjoint_union(nx.path_graph(9), nx.empty_graph(4)) if kind == "disconnected" else
         nx.Graph([(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]))
    csr = csr_from_edges(g.edges(), nodes=np.arange(len(g)))
    arrays_before = [a.copy() for a in (csr.indptr, csr.indices)]
    sources = list(reversed(range(csr.n)))
    actual, meta = threaded_contributions(csr, sources, threads=p, batch_size=1,
                                           progress_path=tmp_path / "progress.jsonl")
    expected, _ = subset_contributions(csr, sources, backend="numba_csr")
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-12)
    _, normalized = finalize_scores(actual, csr.n)
    ref = nx.betweenness_centrality(g, normalized=True)
    np.testing.assert_allclose(normalized, [ref[i] for i in range(csr.n)], atol=1e-10)
    assert meta["p_used"] == meta["active_workers"] == p
    assert len({w["native_thread_id"] for w in meta["workers"]}) == p
    assert sum(w["processed_sources"] for w in meta["workers"]) == len(sources)
    assert all(w["processed_sources"] > 0 for w in meta["workers"])
    assert [r["source_index"] for r in meta["tasks"]] == sources
    assert meta["workspace_bytes"] == p * csr.n * 24
    assert meta["local_total_bytes"] == p * csr.n * 8
    assert meta["exact_all_sources"] and not meta["scaled"]
    assert meta["graph_shared_readonly"]
    assert len((tmp_path / "progress.jsonl").read_text().splitlines()) == csr.n
    for a, before in zip((csr.indptr, csr.indices), arrays_before):
        np.testing.assert_array_equal(a, before)
    partial, info = threaded_contributions(csr, [0, csr.n - 1], threads=p, batch_size=4)
    expected, _ = subset_contributions(csr, [0, csr.n - 1], backend="python_csr")
    np.testing.assert_allclose(partial, expected, atol=1e-10)
    assert info["p_used"] == 1 and info["result_kind"] == "source_subset_contribution"
    assert not info["exact_all_sources"]


def test_concurrent_bigint_fallback():
    layers = [[0]] + [[2 * i - 1, 2 * i] for i in range(1, 66)]
    edges = [(a, b) for left, right in zip(layers, layers[1:]) for a in left for b in right]
    csr = csr_from_edges(edges)
    sources = [0, layers[-1][0], layers[32][0], layers[33][1]]
    actual, info = threaded_contributions(csr, sources, threads=4, batch_size=1)
    expected, _ = subset_contributions(csr, sources, backend="python_csr")
    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-12)
    assert info["fallback_sources"] >= 2
    assert info["active_workers"] == 4


def test_worker_exception_and_timeout_join_all_threads(monkeypatch):
    csr = csr_from_edges([(i, i + 1) for i in range(15)])
    before = {t.ident for t in threading.enumerate()}
    def broken(*args):
        raise RuntimeError("deliberate worker failure")
    monkeypatch.setattr("parallel_bc.compiled.CompiledAccumulator.accumulate", broken)
    with pytest.raises(RuntimeError, match="deliberate"):
        threaded_contributions(csr, range(csr.n), threads=4, batch_size=1)
    assert {t.ident for t in threading.enumerate()} == before
    with pytest.raises(TimeoutError):
        threaded_contributions(csr, range(csr.n), threads=4, timeout_seconds=1e-12)
    assert {t.ident for t in threading.enumerate()} == before


@pytest.mark.parametrize("kwargs", [dict(sources=[]), dict(sources=[0, 0]), dict(sources=[99]),
    dict(sources=[.5]), dict(sources=[0], threads=0), dict(sources=[0], batch_size=0)])
def test_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        threaded_contributions(csr_from_edges([(0, 1)]), **kwargs)


def test_real_bounded_scaling_and_rotating_order(tmp_path):
    cache = tmp_path / "csr"
    save_csr(csr_from_edges([(i, i + 1) for i in range(8)]), cache)
    run = run_scaling(cache, tmp_path / "out", source_count=8, threads=(1, 2, 4),
        batch_size=2, repeats=2, max_case_seconds=60, max_total_seconds=180, min_available_bytes=0)
    data = json.loads((run / "manifest.json").read_text())
    assert data["status"] == "complete"
    assert [r["threads_requested"] for r in data["cases"]] == [1, 2, 4, 2, 4, 1]
    assert len(data["comparisons"]) == 6 and all(c["status"] == "ok" for c in data["comparisons"])
    assert all(c["validation"]["allclose"] for c in data["comparisons"])
    assert data["warmup"]["source_count"] == 1
    assert (run / "report.md").exists()
    before = {p.pid for p in mp.active_children()}
    row = bounded_job("thread_subset", {}, seconds=.001)
    assert row["status"] == "timeout"
    assert {p.pid for p in mp.active_children()} == before


def test_scaling_resource_rejection_preserves_status(tmp_path):
    cache = tmp_path / "csr"
    save_csr(csr_from_edges([(i, i + 1) for i in range(8)]), cache)
    run = run_scaling(cache, tmp_path / "out", source_count=8, threads=(1, 2),
        batch_size=2, repeats=1, min_available_bytes=2**100)
    data = json.loads((run / "manifest.json").read_text())
    assert data["status"] == "partial"
    assert data["warmup"]["status"] == "resource_not_run"
    assert not data["summary"]


def test_hard_timeout_reaps_native_thread_process_without_publishing_vector(tmp_path):
    graph = nx.fast_gnp_random_graph(15000, 6 / 15000, seed=42)
    csr = csr_from_edges(graph.edges(), nodes=np.arange(15000))
    cache, output = tmp_path / "csr", tmp_path / "out"
    save_csr(csr, cache)
    output.mkdir()
    before = {p.pid for p in mp.active_children()}
    row = bounded_job("thread_subset", dict(cache_directory=str(cache), output=str(output),
        sources=list(range(csr.n)), threads=4, batch_size=4, timeout_seconds=60,
        progress_path=str(output / "sources_completed.jsonl")), seconds=4)
    assert row["status"] == "timeout"
    assert not (output / "contributions.npy").exists()
    assert {p.pid for p in mp.active_children()} == before


def test_scaling_applies_same_affinity_in_child_and_keeps_parent_unchanged(tmp_path):
    if not hasattr(psutil.Process(), "cpu_affinity"):
        pytest.skip("CPU affinity unavailable")
    before = psutil.Process().cpu_affinity()
    if len(before) < 2:
        pytest.skip("Need two allowed CPUs")
    selected = sorted(before[:2])
    cache = tmp_path / "csr"
    save_csr(csr_from_edges([(i, i + 1) for i in range(8)]), cache)
    root = run_scaling(cache, tmp_path / "out", source_count=8, threads=(1, 2),
        batch_size=2, repeats=1, cpu_affinity=selected, max_case_seconds=60,
        max_total_seconds=120, min_available_bytes=0)
    data = json.loads((root / "manifest.json").read_text())
    assert data["status"] == "complete"
    assert all(c["cpu_affinity_used"] == selected for c in data["cases"])
    assert psutil.Process().cpu_affinity() == before
    with pytest.raises(ValueError, match="Affinity"):
        run_scaling(cache, tmp_path / "bad", source_count=8, threads=(1, 2), cpu_affinity=[selected[0]])
