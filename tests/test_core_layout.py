"""Permutation, source masses, restoration and matched protocol correctness."""
from dataclasses import replace
import json

import networkx as nx
import numpy as np
import psutil
import pytest

pytest.importorskip("numba")

from parallel_bc.brandes import finalize_scores
from parallel_bc.core_layout import build_layout, load_layout, save_layout, validate_layout, layout_contributions
from parallel_bc.csr import csr_from_edges
from parallel_bc.large_benchmark import bounded_job, subset_contributions
from parallel_bc.layout_benchmark import run_layout_tuning, cpu_configurations
from parallel_bc.quotient_brandes import quotient_contributions
from parallel_bc.tree_quotient import factor_trees, save_quotient


def attached():
    g = nx.cycle_graph(7)
    g.add_edges_from([(0, 7), (7, 8), (7, 9), (3, 10), (10, 11), (10, 12)])
    return g


@pytest.mark.parametrize("g", [nx.empty_graph(0), nx.empty_graph(5), nx.path_graph(12),
    nx.cycle_graph(9), attached(), nx.disjoint_union_all([attached(), nx.path_graph(5), nx.empty_graph(2)]),
    nx.gnp_random_graph(40, .12, seed=7), nx.gnp_random_graph(40, .2, seed=42)])
@pytest.mark.parametrize("kind", ["natural", "bfs"])
def test_full_original_node_scores_and_partial_groups(g, kind):
    csr = csr_from_edges(g.edges(), nodes=np.arange(len(g)))
    before = [a.copy() for a in (csr.ids, csr.indptr, csr.indices)]
    q = factor_trees(csr)
    layout = build_layout(q, kind, seed=3)
    total, core, meta = layout_contributions(q, layout, range(q.core.n), threads=2, batch_size=1)
    raw, normalized = finalize_scores(total, csr.n)
    ref = nx.betweenness_centrality(g, normalized=False)
    np.testing.assert_allclose(raw, list(ref.values()), atol=1e-9, rtol=1e-11)
    assert meta["exact_all_sources"] and meta["represented_source_count"] == len(g)
    if q.core.n:
        roots = list(dict.fromkeys([q.core.n-1, 0]))
        actual, _, info = layout_contributions(q, layout, roots, threads=2, batch_size=1)
        sources = np.flatnonzero(np.isin(q.anchor_index, roots))
        expected, _ = subset_contributions(csr, sources, backend="python_csr")
        np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-11)
        assert info["represented_source_count"] == len(sources)
        assert [r["source_index"] for r in info["tasks"]] == [r for r in roots if q.core.indptr[r+1]>q.core.indptr[r]]
    for old, now in zip(before, (csr.ids, csr.indptr, csr.indices)):
        np.testing.assert_array_equal(old, now)


def test_noncontiguous_original_ids_and_inverse_permutation_roundtrip(tmp_path):
    g = nx.relabel_nodes(attached(), lambda i: 100*i-37)
    csr = csr_from_edges(g.edges())
    q = factor_trees(csr)
    layout = build_layout(q, "bfs", seed=2)
    assert not np.array_equal(layout.new_to_old, np.arange(q.core.n))
    save_layout(layout, tmp_path / "layout")
    loaded = load_layout(q, tmp_path / "layout")
    actual, _, meta = layout_contributions(q, loaded, range(q.core.n), threads=2, batch_size=1)
    expected, _ = quotient_contributions(q)
    np.testing.assert_allclose(actual, expected, atol=1e-9)
    assert {r["original_id"] for r in meta["tasks"]} == set(q.core.ids)
    bad = layout.mass.copy(); bad[0] += 1
    with pytest.raises(ValueError, match="masses"):
        validate_layout(q, replace(layout, mass=bad))
    with (tmp_path / "layout/new_to_old.npy").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="fingerprint"):
        load_layout(q, tmp_path / "layout")
    with pytest.raises(FileExistsError):
        save_layout(layout, tmp_path / "layout")


def test_overflow_fallback_keeps_reordered_weighted_sources():
    layers = [[0]] + [[2*i-1,2*i] for i in range(1,66)]
    g = nx.Graph([(a,b) for left,right in zip(layers,layers[1:]) for a in left for b in right])
    g.add_edges_from([(70,131),(131,132)])
    q = factor_trees(csr_from_edges(g.edges()))
    layout = build_layout(q, "bfs", seed=7)
    actual, _, meta = layout_contributions(q, layout, [0,q.core.n-1], threads=2, batch_size=1)
    sources = np.flatnonzero(np.isin(q.anchor_index,[0,q.core.n-1]))
    expected, _ = quotient_contributions(q, sources)
    np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-10)
    assert meta["fallback_sources"] > 0


@pytest.mark.parametrize("roots", [[-1],[999],[0,0],[1.5]])
def test_invalid_groups_are_not_silently_rescaled(roots):
    q = factor_trees(csr_from_edges(attached().edges()))
    with pytest.raises((ValueError, TypeError)):
        layout_contributions(q, build_layout(q), roots)


def test_real_supervised_staged_protocol_and_protection(tmp_path):
    q = factor_trees(csr_from_edges(attached().edges()))
    cache = save_quotient(q, tmp_path / "q")
    protected = tmp_path / "paused"
    protected.mkdir()
    (protected / "manifest.json").write_text('{"status":"paused"}')
    cpus = sorted(psutil.Process().cpu_affinity())
    configs = {1:cpus[:1],2:cpus[:2]}
    output = run_layout_tuning(cache, tmp_path / "out", source_count=3, confirm_count=5,
        thread_counts=(1,2), repeats=1, batch_size=1, max_case_seconds=60,
        max_total_seconds=300, min_available_bytes=0, protect_run=protected, cpu_sets=configs)
    m = json.loads((output / "manifest.json").read_text())
    assert m["status"] == "complete", m.get("error")
    assert m["protected_paths_unchanged"] and m["code_unchanged"]
    assert len(m["layout_pairs"]) == len(m["thread_pairs"]) == len(m["confirmation_pairs"]) == 1
    assert all(r["validation"]["original_nodes"]["allclose"] for r in m["layout_pairs"]+m["thread_pairs"]+m["confirmation_pairs"])
    assert all(r["restored_original_nodes"] == q.n and not r["exact_all_sources"] for r in m["cases"])
    assert len({r["root_source_order_hash"] for r in m["cases"] if not r["label"].startswith("confirm")}) == 1
    assert not m["exact_all_sources"] and (output / "report.md").exists()
    assert bounded_job("layout_experiment", {}, seconds=.001)["status"] == "timeout"
    assert bounded_job("layout_experiment", {}, seconds=1, min_available_bytes=2**100)["status"] == "resource_not_run"


def test_current_topology_one_logical_per_physical_core():
    configs, audit = cpu_configurations((1,2))
    assert len(configs[1]) == 1 and len(configs[2]) == 2
    assert set(configs[1]) <= set(configs[2]) <= set(psutil.Process().cpu_affinity())
    if audit.get("cpu_sets"):
        rows = [r for r in audit["cpu_sets"] if r["group"] == 0 and r["logical_cpu"] in configs[2]]
        assert len({r["core_index"] for r in rows}) == 2
