"""CSR equivalence, provenance, strict formats and real bounded-spawn checks."""
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from parallel_bc.brandes import brandes_serial, finalize_scores, single_source_accumulate
from parallel_bc.csr import (CSRGraph, component_statistics, csr_from_edges, load_csr,
    prepare_csr, read_csr_edges, save_csr, validate_csr)
from parallel_bc.data import graph_from_edges
from parallel_bc.large_benchmark import bounded_job, run_pilot, subset_contributions


@pytest.mark.parametrize("edges,nodes", [
    ([], []), ([], [3]), ([(7, 7)], [99]),
    ([(9, 2), (2, 9), (7, 7)], [99]),
    ([(0, 1), (1, 2), (3, 4)], [0, 1, 2, 3, 4, 5]),
    ([(-17, 2), (2, 2), (2, 10000000)], [-99]),
])
def test_canonical_csr_matches_old_graph_including_isolates(edges, nodes):
    old = graph_from_edges(edges, nodes=nodes)
    csr = csr_from_edges(edges, nodes=nodes)
    assert tuple(csr.ids) == old.ids
    assert tuple(tuple(map(int, csr.adj[i])) for i in range(csr.n)) == old.adj
    assert csr.graph_sha256 == old.graph_sha256
    assert csr.metadata["raw_data_lines"] == old.metadata["raw_data_lines"]
    assert csr.metadata["self_loop_lines"] == old.metadata["self_loop_lines"]
    assert csr.metadata["duplicate_edge_lines"] == old.metadata["duplicate_edge_lines"]
    stats = component_statistics(csr)
    assert stats["connected_components"] == old.metadata["connected_components"]
    assert stats["largest_component_nodes"] == old.metadata["largest_component_nodes"]
    assert not csr.indices.flags.writeable


def test_konect_header_counts_isolates_and_explicit_opt_in(tmp_path):
    path = tmp_path / "konect.txt"
    path.write_text("% sym unweighted\n% 4 5 5\n1 2\n2 1\n3 3\n2 4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid edge row"):
        read_csr_edges(path)
    edges, nodes, parsed = read_csr_edges(path, file_format="konect")
    graph = csr_from_edges(edges, nodes=nodes)
    assert list(graph.ids) == [1, 2, 3, 4, 5]
    assert graph.m == 2
    assert graph.metadata["self_loop_lines"] == 1
    assert graph.metadata["duplicate_edge_lines"] == 1
    assert parsed["declared_nodes"] == 5


@pytest.mark.parametrize("body,error", [
    ("% asym unweighted\n% 1 2 2\n1 2\n", "unsupported"),
    ("% sym weighted\n% 1 2 2\n1 2\n", "unsupported"),
    ("% sym unweighted\n% 1 2 3\n1 2\n", "sizes"),
    ("% sym unweighted\n% 1 2 2\n0 2\n", "range"),
    ("% sym unweighted\n% 1 2 2\n1 2 1\n", "two integer"),
    ("% sym unweighted\n% 2 2 2\n1 2\n", "mismatch"),
    ("% sym unweighted\n% 1 2 2\n1 nope\n", "invalid edge"),
    ("% sym unweighted\n", "mismatch"),
    ("1 2\n", "missing"),
])
def test_konect_bad_rows_never_silently_skipped(tmp_path, body, error):
    path = tmp_path / "invalid.txt"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        read_csr_edges(path, file_format="konect")


def test_prepare_cache_keeps_provenance_and_no_overwrite(tmp_path):
    raw = tmp_path / "graph.txt"
    raw.write_text("% sym unweighted\n% 2 4 4\n1 2\n2 3\n", encoding="utf-8")
    provenance = tmp_path / "audit.json"
    provenance.write_text(json.dumps(dict(raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
                                         source_page="https://example.test/user-file")))
    cache = tmp_path / "csr"
    prepare_csr(raw, cache, dataset_id="test", file_format="konect", provenance_path=provenance)
    graph = load_csr(cache)
    assert (graph.n, graph.m) == (4, 2)
    assert graph.metadata["connected_components"] == 2
    assert graph.metadata["source_provenance"]["source_page"].endswith("user-file")
    assert isinstance(graph.indices, np.memmap)
    with pytest.raises(FileExistsError):
        save_csr(graph, cache)
    provenance.write_text('{"raw_sha256": "wrong"}')
    with pytest.raises(ValueError, match="Provenance"):
        prepare_csr(raw, tmp_path / "other", dataset_id="test", file_format="konect",
                    provenance_path=provenance)


def test_cache_hash_and_graph_metadata_tampering(tmp_path):
    cache = tmp_path / "csr"
    save_csr(csr_from_edges([(1, 2), (2, 3)]), cache)
    meta_file = cache / "metadata.json"
    meta = json.loads(meta_file.read_text())
    meta["graph_sha256"] = "wrong"
    meta_file.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="fingerprint"):
        load_csr(cache)
    with (cache / "indices.npy").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="SHA256"):
        load_csr(cache)


def test_invalid_csr_rejects_asymmetry_and_duplicate_neighbors():
    graph = csr_from_edges([(0, 1), (1, 2)], nodes=[0, 1, 2])
    bad = graph.indices.copy()
    bad[0] = 2
    with pytest.raises(ValueError, match="symmetric"):
        validate_csr(CSRGraph(graph.ids, graph.indptr, bad, {}))
    bad = graph.indices.copy()
    bad[1:3] = [0, 0]
    with pytest.raises(ValueError, match="duplicate"):
        validate_csr(CSRGraph(graph.ids, graph.indptr, bad, {}))


@pytest.mark.parametrize("seed", [0, 7, 42])
@pytest.mark.parametrize("backend", ["python_tuple", "python_csr"])
def test_subset_and_full_numeric_equivalence(seed, backend):
    nxgraph = nx.gnp_random_graph(27, .15, seed=seed)
    old = graph_from_edges(nxgraph.edges(), nodes=range(28))  # isolated node
    csr = csr_from_edges(list(nxgraph.edges()), nodes=np.arange(28))
    sources = [0, 3, 7, 27]
    expected = np.zeros(csr.n)
    for source in sources:
        single_source_accumulate(old.adj, source, expected)
    actual, meta = subset_contributions(csr, sources, backend=backend)
    np.testing.assert_array_equal(actual, expected)
    assert meta["exact_all_sources"] is False
    assert meta["scaled"] is False
    assert meta["result_kind"] == "source_subset_contribution"
    with pytest.raises(ValueError, match="permutation"):
        brandes_serial(old.adj, source_order=sources)
    total, meta = subset_contributions(csr, range(csr.n), backend=backend)
    raw, normalized = finalize_scores(total, csr.n)
    reference = brandes_serial(old.adj)
    np.testing.assert_allclose(raw, reference.raw)
    np.testing.assert_allclose(normalized, reference.normalized)
    # The old reference tests already exercise NetworkX; here also check all nodes with isolates.
    fullnx = nxgraph.copy()
    fullnx.add_node(27)
    expected_nx = nx.betweenness_centrality(fullnx, normalized=True)
    np.testing.assert_allclose(normalized, [expected_nx[i] for i in range(csr.n)], atol=1e-10)
    assert meta["exact_all_sources"] is True


@pytest.mark.parametrize("sources", [[], [0, 0], [-1], [3], [1.5]])
def test_subset_rejects_invalid_sources(sources):
    graph = csr_from_edges([(0, 1), (1, 2)])
    with pytest.raises(ValueError):
        subset_contributions(graph, sources)


def test_real_spawn_pilot_nested_sources_and_output_scope(tmp_path):
    cache = tmp_path / "csr"
    save_csr(csr_from_edges([(10, 20), (20, 30)], nodes=[10, 20, 30, 40]), cache)
    run = run_pilot(cache, tmp_path / "results", counts=(1, 2), max_case_seconds=20,
                    max_total_seconds=60, backend="python_tuple")
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    first, second = [json.loads((run / f"k{k}" / "sources.json").read_text()) for k in (1, 2)]
    assert first["indices"] == second["indices"][:1]
    assert first["original_ids"] == second["original_ids"][:1]
    assert all(row["completed_sources"] == row["k_requested"] for row in manifest["cases"])
    assert all(row["exact_all_sources"] is False for row in manifest["cases"])
    assert (run / "k2" / "contributions.npy").exists()
    assert not list(run.glob("*/centrality*.json"))
    # Different max-counts still select the same initial source for a given seed.
    other = run_pilot(cache, tmp_path / "results", counts=(1,), max_case_seconds=20,
                      max_total_seconds=60, backend="python_csr")
    np.testing.assert_array_equal(np.load(run / "k1/contributions.npy"),
                                  np.load(other / "k1/contributions.npy"))


def test_hard_budget_and_memory_reserve_do_not_leave_children(tmp_path):
    before = {p.pid for p in mp.active_children()}
    row = bounded_job("subset", {}, seconds=.001)
    assert row["status"] == "timeout"
    assert {p.pid for p in mp.active_children()} <= before
    row = bounded_job("subset", {}, seconds=1, min_available_bytes=2**100)
    assert row["status"] == "resource_not_run"
    assert {p.pid for p in mp.active_children()} <= before


def test_preparation_cli_small_konect(tmp_path):
    import subprocess
    import sys
    raw = tmp_path / "raw.txt"
    raw.write_text("% sym unweighted\n% 2 4 4\n1 2\n2 3\n", encoding="utf-8")
    proc = subprocess.run([sys.executable, "-m", "parallel_bc", "data", "prepare-csr",
        "--dataset", "tiny", "--path", str(raw), "--format", "konect",
        "--data-dir", str(tmp_path / "data"), "--reserve-gib", "0"],
        capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    graph = load_csr(tmp_path / "data/processed/tiny_csr")
    assert graph.metadata["isolated_nodes"] == 1


def test_pilot_skips_larger_cases_after_resource_limit(tmp_path, monkeypatch):
    cache = tmp_path / "csr"
    save_csr(csr_from_edges([(0, 1), (1, 2)], nodes=[0, 1, 2, 3]), cache)
    calls = []
    def blocked(*args, **kwargs):
        calls.append(args)
        return dict(status="resource_limit", error="test reserve")
    monkeypatch.setattr("parallel_bc.large_benchmark.bounded_job", blocked)
    run = run_pilot(cache, tmp_path / "out", counts=(1, 2, 3))
    data = json.loads((run / "manifest.json").read_text())
    assert len(calls) == 1
    assert [r["status"] for r in data["cases"]] == ["resource_limit", "resource_not_run", "resource_not_run"]
    assert data["status"] == "partial"
