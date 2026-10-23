"""Native correctness, overflow fallback and genuine single-worker comparison."""
import json
import subprocess
import sys

import networkx as nx
import numpy as np
import pytest

pytest.importorskip("numba")

from parallel_bc.brandes import finalize_scores, single_source_accumulate
from parallel_bc.compiled import CompiledAccumulator, warmup
from parallel_bc.compiled_benchmark import compare_vectors, run_comparison
from parallel_bc.csr import csr_from_edges, save_csr
from parallel_bc.large_benchmark import subset_contributions


@pytest.fixture(scope="module", autouse=True)
def native_warmup():
    info = warmup()
    assert info["native_nopython"] is True
    assert info["jit_options"]["parallel"] is False


@pytest.mark.parametrize("graph", [nx.empty_graph(1), nx.empty_graph(5), nx.path_graph(9),
    nx.star_graph(8), nx.cycle_graph(8), nx.complete_graph(8),
    nx.disjoint_union(nx.path_graph(4), nx.cycle_graph(5)), nx.karate_club_graph(),
    *[nx.gnp_random_graph(35, p, seed=seed) for p in (.08, .4) for seed in (0, 7, 42)]])
def test_all_nodes_against_original_and_networkx(graph):
    csr = csr_from_edges(graph.edges(), nodes=range(len(graph)))
    expected = np.zeros(csr.n)
    actual = np.zeros(csr.n)
    runner = CompiledAccumulator(csr)
    for source in reversed(range(csr.n)):
        before = expected.copy()
        single_source_accumulate(csr.adj, source, expected)
        metadata = runner.accumulate(source, actual)
        assert metadata["execution"] == "numba_native"
        # Workspace resets must not retain unreachable vertices from prior sources.
        assert metadata["reachable_nodes"] == len(nx.single_source_shortest_path_length(graph, source))
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-12)
        distances = nx.single_source_shortest_path_length(graph, source)
        identity = sum(d - 1 for v, d in distances.items() if v != source)
        assert (expected - before).sum() == pytest.approx(identity, abs=1e-9)
    raw, normalized = finalize_scores(actual, csr.n)
    for values, normalize in ((raw, False), (normalized, True)):
        reference = nx.betweenness_centrality(graph, normalized=normalize)
        np.testing.assert_allclose(values, [reference[i] for i in range(csr.n)], atol=1e-10, rtol=1e-12)


def test_bigint_overflow_retries_before_committing_and_workspace_recovers():
    layers = [[0]] + [[2 * i - 1, 2 * i] for i in range(1, 66)]
    edges = [(a, b) for left, right in zip(layers, layers[1:]) for a in left for b in right]
    csr = csr_from_edges(edges)
    runner = CompiledAccumulator(csr)
    actual = np.full(csr.n, 7.0)
    expected = actual.copy()
    single_source_accumulate(csr.adj, 0, expected)
    info = runner.accumulate(0, actual)
    assert info["execution"] == "python_bigint_fallback"
    assert info["fallback_reason"] == "sigma_int64_overflow"
    np.testing.assert_array_equal(actual, expected)
    # A middle source has smaller multiplicities; same workspace remains usable.
    source = layers[32][0]
    single_source_accumulate(csr.adj, source, expected)
    info = runner.accumulate(source, actual)
    assert info["execution"] == "numba_native"
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-12)


def test_subset_scope_and_optional_backend_metadata(tmp_path):
    csr = csr_from_edges([(10, 20), (20, 30), (10, 30), (30, 40)], nodes=[50])
    sources = [1, 4, 0]
    actual, meta = subset_contributions(csr, sources, backend="numba_csr")
    expected, _ = subset_contributions(csr, sources, backend="python_csr")
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    assert meta["p_used"] == 1 and meta["fallback_sources"] == 0
    assert meta["native_nopython"] and meta["workspace_bytes"] == csr.n * 24
    assert not meta["scaled"] and not meta["exact_all_sources"]
    assert meta["result_kind"] == "source_subset_contribution"
    assert meta["t_jit_warmup"] >= 0 and meta["t_kernel"] > 0


def test_nonfinite_native_source_never_commits(monkeypatch):
    csr = csr_from_edges([(0, 1), (1, 2)])
    runner = CompiledAccumulator(csr)
    total = np.ones(3)
    monkeypatch.setattr("parallel_bc.compiled._source_dependencies", lambda *args: (2, 3, 1))
    with pytest.raises(FloatingPointError, match="not committed"):
        runner.accumulate(0, total)
    np.testing.assert_array_equal(total, np.ones(3))


@pytest.mark.parametrize("source,total", [(3, np.zeros(3)), (-1, np.zeros(3)),
    (1, np.zeros(2)), (1, np.zeros(3, np.int64))])
def test_invalid_native_inputs(source, total):
    runner = CompiledAccumulator(csr_from_edges([(0, 1), (1, 2)]))
    with pytest.raises(ValueError):
        runner.accumulate(source, total)


def test_comparison_checks_all_values_not_only_top_nodes():
    assert compare_vectors(np.array([1., 0.]), np.array([1., 0.]))["allclose"]
    assert not compare_vectors(np.array([1., 0.]), np.array([1., .1]))["allclose"]
    assert not compare_vectors(np.array([1., 0.]), np.array([1., np.nan]))["allclose"]


def test_real_spawn_comparison_and_cache_reuse(tmp_path):
    cache = tmp_path / "csr"
    save_csr(csr_from_edges([(0, 1), (1, 2), (2, 3)], nodes=[4]), cache)
    root = run_comparison(cache, tmp_path / "out", source_count=2, repeats=2,
                          max_case_seconds=60, max_total_seconds=160)
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert [r["backend"] for r in manifest["cases"]] == ["python_csr", "numba_csr", "numba_csr", "python_csr"]
    assert all(p["validation"]["allclose"] for p in manifest["pairs"])
    assert all(r["source_order_hash"] == manifest["source_order_hash"] for r in manifest["cases"])
    assert len(list((root / "jit_cache").rglob("*.nbc"))) >= 2
    assert (root / "report.md").exists()


def test_base_cli_import_does_not_require_numba():
    code = "import sys; sys.modules['numba'] = None; from parallel_bc.cli import main; main(['--help'])"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr


def test_missing_optional_dependency_has_install_hint(tmp_path, monkeypatch):
    from importlib.metadata import PackageNotFoundError
    cache = tmp_path / "csr"
    save_csr(csr_from_edges([(0, 1)]), cache)
    def missing(name):
        raise PackageNotFoundError(name)
    monkeypatch.setattr("parallel_bc.compiled_benchmark.version", missing)
    with pytest.raises(RuntimeError, match="compiled"):
        run_comparison(cache, tmp_path / "out", source_count=1)
