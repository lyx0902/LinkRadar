"""Exact edge partition, cuts, weighted bridge sides and read-only supervision."""
import csv
import json
import subprocess
import sys

import networkx as nx
import numpy as np
import pytest

pytest.importorskip("numba")

from parallel_bc.core_diagnostic import diagnose_graph, run_core_diagnostic
from parallel_bc.csr import csr_from_edges
from parallel_bc.large_benchmark import bounded_job
from parallel_bc.tree_quotient import factor_trees, save_quotient


def fixtures():
    return [nx.empty_graph(0), nx.empty_graph(4), nx.path_graph(2), nx.path_graph(9),
        nx.star_graph(8), nx.cycle_graph(7), nx.complete_graph(6), nx.barbell_graph(5, 3),
        nx.windmill_graph(5, 3), nx.disjoint_union_all([nx.cycle_graph(4), nx.path_graph(5), nx.empty_graph(2)]),
        *[nx.gnp_random_graph(35, p, seed=s) for p in (.02, .06, .15, .35) for s in (0, 7, 42)]]


@pytest.mark.parametrize("graph", fixtures())
def test_every_bcc_edge_and_articulation_matches_networkx(graph):
    csr = csr_from_edges(graph.edges(), nodes=np.arange(len(graph)))
    original = [a.copy() for a in (csr.ids, csr.indptr, csr.indices)]
    mass = np.arange(1, csr.n + 1, dtype=np.int64)
    summary, arrays = diagnose_graph(csr, mass)
    blocks = arrays["block_edges"]
    ptr = arrays["block_offsets"]
    actual = set()
    for i in range(len(ptr) - 1):
        edges = blocks[ptr[i]:ptr[i+1]]
        actual.add(frozenset(tuple(map(int, edge)) for edge in edges))
        vertices = np.unique(edges)
        assert len(vertices) == arrays["block_vertex_counts"][i]
        assert mass[vertices].sum() == arrays["block_member_mass"][i]
    expected = {frozenset(tuple(sorted(edge)) for edge in component)
                for component in nx.biconnected_component_edges(graph)}
    assert actual == expected
    assert set(np.flatnonzero(arrays["articulation_mask"])) == set(nx.articulation_points(graph))
    bridges = {tuple(map(int, blocks[ptr[i]])) for i in range(len(ptr)-1) if ptr[i+1]-ptr[i] == 1}
    assert bridges == {tuple(sorted(edge)) for edge in nx.bridges(graph)}
    assert summary["connected_components"] == nx.number_connected_components(graph)
    assert summary["edge_count_sum"] == csr.m and summary["edge_partition_valid"]
    for u, v in bridges:
        child = v if arrays["dfs_parent"][v] == u else u
        copy = graph.copy()
        copy.remove_edge(u, v)
        side = nx.node_connected_component(copy, child)
        assert arrays["dfs_subtree_mass"][child] == mass[list(side)].sum()
    for old, current in zip(original, (csr.ids, csr.indptr, csr.indices)):
        np.testing.assert_array_equal(old, current)


def test_long_chain_needs_no_python_recursion():
    n = 20000
    csr = csr_from_edges(((i, i+1) for i in range(n-1)))
    summary, arrays = diagnose_graph(csr)
    assert summary["bridges"] == n-1
    assert summary["articulation_vertices"] == n-2
    assert summary["edge_blocks"] == n-1
    assert summary["work_proxy"]["cyclic_blocks_only"] == 0
    assert np.all(arrays["block_vertex_counts"] == 2)


def test_giant_block_proxy_and_shared_cut_vertices_are_explicit():
    g = nx.disjoint_union(nx.cycle_graph(6), nx.complete_graph(4))
    g.add_edge(0, 6)
    summary, arrays = diagnose_graph(csr_from_edges(g.edges()))
    assert summary["bridges"] == 1 and summary["articulation_vertices"] == 2
    assert summary["block_vertex_sum"] == len(g) + 2
    assert summary["top_blocks_by_vertices"][0]["nodes"] == 6
    assert summary["largest_block_by_edges"]["edges"] == 6
    expected = 6*(6+6) + 4*(4+6)
    assert summary["work_proxy"]["cyclic_blocks_only"] == expected


@pytest.mark.parametrize("mass", [[1.5, 2.], [1, 0], [1, -2], [1], np.array([2**63, 1], np.uint64)])
def test_invalid_endpoint_masses_rejected(mass):
    with pytest.raises(ValueError, match="mass"):
        diagnose_graph(csr_from_edges([(0, 1)]), mass)


def test_real_spawn_csv_and_protected_run_unchanged(tmp_path):
    g = nx.disjoint_union(nx.cycle_graph(4), nx.cycle_graph(5))
    g.add_edges_from([(0, 4), (1, 9), (9, 10), (4, 11)])
    cache = save_quotient(factor_trees(csr_from_edges(g.edges())), tmp_path / "q")
    protected = tmp_path / "checkpoint"
    protected.mkdir()
    (protected / "manifest.json").write_text('{"status":"paused"}')
    (protected / "run.lock").write_bytes(b"0")
    dest = run_core_diagnostic(cache, tmp_path / "out", protect_run=protected,
                               seconds=60, min_available_bytes=0)
    m = json.loads((dest / "manifest.json").read_text())
    d = json.loads((dest / "diagnostic.json").read_text())
    assert m["status"] == "complete" and m["protected_paths_unchanged"]
    assert json.loads((dest / "protected_before.json").read_text()) == json.loads((dest / "protected_after.json").read_text())
    assert d["bridges"] == 1 and d["articulation_vertices"] == 2
    with (dest / "bridges.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert int(row["cross_unordered_pairs"]) == 6 * 6
    assert d["result_kind"] == "core_structure_diagnostic" and not d["exact_all_sources"]
    assert not (dest / "bc_raw.npy").exists()
    assert (dest / "report.md").exists()
    with pytest.raises(ValueError, match="outside"):
        run_core_diagnostic(cache, cache / "bad")


def test_real_cli_and_resource_guards(tmp_path):
    save_quotient(factor_trees(csr_from_edges(nx.cycle_graph(5).edges())), tmp_path / "data/processed/tiny_treeq")
    proc = subprocess.run([sys.executable, "-m", "parallel_bc", "core-diagnose", "--dataset", "tiny",
        "--data-dir", str(tmp_path / "data"), "--output-dir", str(tmp_path / "out"), "--reserve-gib", "0"],
        capture_output=True, timeout=60)
    assert proc.returncode == 0, proc.stdout.decode(errors="replace") + proc.stderr.decode(errors="replace")
    assert bounded_job("core_structure", {}, seconds=.001)["status"] == "timeout"
    assert bounded_job("core_structure", {}, seconds=1, min_available_bytes=2**100)["status"] == "resource_not_run"
