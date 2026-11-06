"""Lossless representation and exact recovery on every original vertex."""
from dataclasses import replace
import json

import networkx as nx
import numpy as np
import pytest

pytest.importorskip("numba")

from parallel_bc.brandes import finalize_scores
from parallel_bc.csr import csr_from_edges, save_csr
from parallel_bc.large_benchmark import bounded_job, subset_contributions
from parallel_bc.quotient_benchmark import run_quotient_comparison, run_quotient_preparation
from parallel_bc.quotient_brandes import quotient_contributions
from parallel_bc.tree_quotient import factor_trees, load_quotient, reconstructed_digest, save_quotient, validate_quotient


def fixtures():
    attached = nx.cycle_graph(5)
    attached.add_edges_from([(0, 5), (5, 6), (5, 7), (1, 8), (1, 9), (9, 10)])
    return [nx.empty_graph(0), nx.empty_graph(1), nx.empty_graph(5), nx.path_graph(2),
        nx.path_graph(9), nx.star_graph(10), nx.balanced_tree(2, 4), nx.cycle_graph(7),
        nx.complete_graph(6), nx.barbell_graph(5, 2), attached,
        nx.disjoint_union_all([attached, nx.path_graph(5), nx.empty_graph(2)]),
        *[nx.gnp_random_graph(30, p, seed=seed) for p in (.03, .08, .25) for seed in (0, 7, 42)]]


@pytest.mark.parametrize("g", fixtures())
def test_lossless_full_source_against_networkx_and_subset_against_original(g):
    graph = csr_from_edges(g.edges(), nodes=np.arange(len(g)))
    before = graph.indices.copy()
    q = factor_trees(graph)
    assert reconstructed_digest(q) == graph.graph_sha256
    assert q.core.m + len(q.peel_order) == graph.m
    assert q.mass.sum() == graph.n
    total, meta = quotient_contributions(q, threads=4, batch_size=1)
    raw, normalized = finalize_scores(total, graph.n)
    for actual, norm in ((raw, False), (normalized, True)):
        ref = nx.betweenness_centrality(g, normalized=norm)
        np.testing.assert_allclose(actual, [ref[v] for v in range(graph.n)], atol=1e-9, rtol=1e-11)
    assert meta["exact_all_sources"] and meta["restored_original_nodes"] == graph.n
    if graph.n:
        sources = list(dict.fromkeys([graph.n - 1, 0, graph.n // 2]))
        actual, info = quotient_contributions(q, sources, threads=2, batch_size=1)
        expected, _ = subset_contributions(graph, sources, backend="python_csr")
        np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-11)
        assert info["source_count"] == len(sources) == info["represented_source_count"]
    np.testing.assert_array_equal(graph.indices, before)


def test_same_tree_sources_are_grouped_but_endpoints_and_internal_nodes_restored():
    g = nx.cycle_graph(4)
    g.add_edges_from([(0, 4), (4, 5), (4, 6), (6, 7), (0, 8)])
    graph = csr_from_edges(g.edges())
    q = factor_trees(graph)
    actual, info = quotient_contributions(q, [5, 7, 8], threads=4, batch_size=1)
    expected, _ = subset_contributions(graph, [5, 7, 8], backend="python_csr")
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    assert info["core_traversals"] == 1 and info["represented_source_count"] == 3
    assert actual[4] > 0 and actual[6] > 0  # represented non-leaves still have scores
    assert actual[5] == actual[7] == actual[8] == 0
    # A naive core-only computation omits endpoint mass and anchor contributions.
    naive, _ = subset_contributions(q.core, range(q.core.n), backend="python_csr")
    full, _ = quotient_contributions(q)
    assert not np.allclose(full[q.anchor_nodes], naive)


def test_all_tree_components_need_no_bfs():
    g = nx.disjoint_union(nx.path_graph(15), nx.star_graph(5))
    graph = csr_from_edges(g.edges())
    q = factor_trees(graph)
    total, info = quotient_contributions(q)
    assert q.core.n == 2 and q.core.m == 0
    assert info["core_traversals"] == 0 and info["p_used"] == 0
    raw, _ = finalize_scores(total, graph.n)
    ref = nx.betweenness_centrality(g, normalized=False)
    np.testing.assert_allclose(raw, list(ref.values()))


def test_roundtrip_keeps_original_ids_isolates_and_refuses_tampering(tmp_path):
    graph = csr_from_edges([(-7, 20), (20, 91), (91, -7), (20, 40), (40, 50)], nodes=[999])
    q = factor_trees(graph)
    path = save_quotient(q, tmp_path / "q")
    loaded = load_quotient(path)
    np.testing.assert_array_equal(loaded.original_ids, graph.ids)
    np.testing.assert_array_equal(quotient_contributions(loaded)[0], quotient_contributions(q)[0])
    with pytest.raises(FileExistsError):
        save_quotient(q, path)
    bad = q.subtree_size.copy(); bad[0] += 1
    with pytest.raises(ValueError, match="subtree"):
        validate_quotient(replace(q, subtree_size=bad))
    meta = json.loads((path / "metadata.json").read_text())
    meta["source_graph_sha256"] = "tampered"
    (path / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="fingerprint"):
        load_quotient(path)


def test_weighted_path_count_overflow_uses_bigint_without_losing_tree_targets():
    layers = [[0]] + [[2 * i - 1, 2 * i] for i in range(1, 66)]
    edges = [(a, b) for left, right in zip(layers, layers[1:]) for a in left for b in right]
    edges += [(layers[35][0], 131), (131, 132)]
    graph = csr_from_edges(edges)
    q = factor_trees(graph)
    actual, meta = quotient_contributions(q, [0, 132], threads=2, batch_size=1)
    expected, _ = subset_contributions(graph, [0, 132], backend="python_csr")
    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-11)
    assert meta["fallback_sources"] >= 1


@pytest.mark.parametrize("sources", [[], [-1], [99], [0, 0], [1.5]])
def test_bad_sources_never_change_result_semantics(sources):
    q = factor_trees(csr_from_edges([(0, 1), (1, 2)]))
    with pytest.raises(ValueError):
        quotient_contributions(q, sources)


def test_real_bounded_quotient_preparation_comparison_and_scope(tmp_path):
    g = nx.cycle_graph(5)
    g.add_edges_from([(0, 5), (5, 6), (5, 7), (1, 8)])
    cache = tmp_path / "csr"
    save_csr(csr_from_edges(g.edges()), cache)
    qpath = tmp_path / "q"
    row = run_quotient_preparation(cache, qpath, seconds=60)
    assert row["status"] == "ok"
    run = run_quotient_comparison(cache, qpath, tmp_path / "results", source_count=6,
        threads=2, batch_size=1, repeats=2, max_case_seconds=60, max_total_seconds=240)
    data = json.loads((run / "manifest.json").read_text())
    assert data["status"] == "complete"
    assert [r["variant"] for r in data["cases"]] == ["original", "quotient", "quotient", "original"]
    assert all(p["validation"]["allclose"] for p in data["pairs"])
    assert not data["exact_all_sources"] and not data["scaled"]
    for c in data["cases"]:
        assert len(np.load(c["contribution_path"])) == 9
    with pytest.raises(ValueError, match="overwrite"):
        run_quotient_preparation(cache, qpath)


def test_quotient_supervisor_hard_timeout_and_reserve(tmp_path):
    import multiprocessing as mp
    before = {p.pid for p in mp.active_children()}
    row = bounded_job("quotient_subset", {}, seconds=.001)
    assert row["status"] == "timeout"
    assert {p.pid for p in mp.active_children()} == before
    row = bounded_job("quotient_subset", {}, seconds=2, min_available_bytes=2**100)
    assert row["status"] == "resource_not_run"


def test_quotient_cli_end_to_end(tmp_path):
    import subprocess, sys
    graph = csr_from_edges([(0, 1), (1, 2), (2, 0), (0, 3), (3, 4)])
    save_csr(graph, tmp_path / "data/processed/tiny")
    common = ["--data-dir", str(tmp_path / "data"), "--reserve-gib", "0"]
    preparation = subprocess.run([sys.executable,"-m","parallel_bc","data","prepare-quotient",
        "--dataset","tiny",*common], capture_output=True,text=True,encoding="utf-8",timeout=60)
    assert preparation.returncode == 0, preparation.stdout + preparation.stderr
    result = subprocess.run([sys.executable,"-m","parallel_bc","quotient-compare","--dataset","tiny",
        "--sources","3","--threads","2","--batch-size","1","--repeats","1",
        "--output-dir",str(tmp_path/"out"),*common],capture_output=True,text=True,encoding="utf-8",timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
