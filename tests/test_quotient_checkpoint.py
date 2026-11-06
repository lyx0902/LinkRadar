"""Transaction boundaries, bounded scheduling and exact original BC recovery."""
import json
import shutil
import subprocess
import sys

import networkx as nx
import numpy as np
import pytest

pytest.importorskip("numba")

from parallel_bc.csr import csr_from_edges
from parallel_bc.large_benchmark import bounded_job
from parallel_bc.quotient_brandes import quotient_contributions
from parallel_bc.quotient_checkpoint import _run_lock, checkpoint_contributions, run_checkpoint
from parallel_bc.tree_quotient import factor_trees, load_quotient, save_quotient


def example():
    graph = nx.barbell_graph(5, 2)
    graph.add_edges_from([(0, 12), (12, 13), (12, 14), (3, 15), (15, 16)])
    return nx.disjoint_union_all([graph, nx.path_graph(6), nx.empty_graph(2)])


def cached(tmp_path, graph=None):
    graph = example() if graph is None else graph
    csr = csr_from_edges(graph.edges(), nodes=np.arange(len(graph)))
    path = save_quotient(factor_trees(csr), tmp_path / "q")
    return graph, path, load_quotient(path)


def manifest(root):
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("graph", [example(), nx.empty_graph(0), nx.empty_graph(5),
    nx.path_graph(15), nx.cycle_graph(9), nx.gnp_random_graph(35, .13, seed=7)])
def test_chunked_full_scores_and_pause_semantics(tmp_path, graph):
    graph, _, q = cached(tmp_path, graph)
    root = tmp_path / "run"
    options = dict(threads=2, batch_size=1, chunk_size=3, seed=7)
    row = checkpoint_contributions(q, root, max_chunks=1, **options)
    if row["run_status"] != "complete":
        assert not row["exact_all_sources"]
        assert row["result_kind"] == "incomplete_core_accumulator"
        assert not (root / "bc_raw.npy").exists()
        row = checkpoint_contributions(q, root, resume=True, **options)
    assert row["run_status"] == "complete" and row["exact_all_sources"]
    expected = nx.betweenness_centrality(graph, normalized=False)
    np.testing.assert_allclose(np.load(root / "bc_raw.npy"), list(expected.values()), atol=1e-9, rtol=1e-11)
    normalized = nx.betweenness_centrality(graph, normalized=True)
    np.testing.assert_allclose(np.load(root / "bc_normalized.npy"), list(normalized.values()), atol=1e-10)
    direct, _ = quotient_contributions(q, threads=2, batch_size=1)
    np.testing.assert_allclose(np.load(root / "contributions.npy"), direct, atol=1e-9)
    m = manifest(root)
    plan = np.load(root / "source_plan.npy")
    assert sorted(plan.tolist()) == np.flatnonzero(np.diff(q.core.indptr) > 0).tolist()
    assert m["completed_source_mass"] + m["analytic_source_mass"] == q.n
    assert "tasks" not in json.dumps(m)  # bounded summaries, no accumulated source rows
    hashes = [item["sha256"] for item in m["chunks"]]
    again = checkpoint_contributions(q, root, resume=True, **options)
    assert again["new_chunks"] == 0
    assert hashes == [item["sha256"] for item in manifest(root)["chunks"]]


@pytest.mark.parametrize("stage", ["after_compute", "after_score_file", "after_manifest"])
def test_fault_at_each_commit_boundary_exactly_once(tmp_path, stage):
    _, _, q = cached(tmp_path)
    options = dict(threads=2, batch_size=1, chunk_size=3, seed=7)
    root = tmp_path / "crashed"
    def fault(here, index):
        if here == stage and index == 1:
            raise RuntimeError("injected crash")
    with pytest.raises(RuntimeError, match="injected"):
        checkpoint_contributions(q, root, _fault=fault, **options)
    assert len(manifest(root)["chunks"]) == (2 if stage == "after_manifest" else 1)
    row = checkpoint_contributions(q, root, resume=True, **options)
    assert row["run_status"] == "complete"
    direct, _ = quotient_contributions(q)
    np.testing.assert_allclose(np.load(root / "contributions.npy"), direct, atol=1e-9)


@pytest.mark.parametrize("stage", ["after_score_file", "after_manifest"])
def test_real_process_death_releases_lock_and_resumes(tmp_path, stage):
    _, path, q = cached(tmp_path)
    root = tmp_path / "dead"
    script = """
import os, sys
from parallel_bc.tree_quotient import load_quotient
from parallel_bc.quotient_checkpoint import checkpoint_contributions
def fault(stage, index):
    if stage == sys.argv[3] and index == 1:
        os._exit(23)
checkpoint_contributions(load_quotient(sys.argv[1]), sys.argv[2], threads=2,
                         batch_size=1, chunk_size=3, seed=7, _fault=fault)
"""
    proc = subprocess.run([sys.executable, "-c", script, str(path), str(root), stage],
                          capture_output=True, timeout=60)
    assert proc.returncode == 23, proc.stderr.decode(errors="replace")
    before = len(manifest(root)["chunks"])
    assert before == (2 if stage == "after_manifest" else 1)
    row = run_checkpoint(path, root, seconds=60, min_available_bytes=0, resume=True,
                         threads=2, batch_size=1, chunk_size=3, seed=7)
    assert row["status"] == "ok" and row["run_status"] == "complete", row
    direct, _ = quotient_contributions(q)
    np.testing.assert_allclose(np.load(root / "contributions.npy"), direct, atol=1e-9)


@pytest.mark.parametrize("change", ["code", "config", "plan", "chunk", "coverage", "mass", "final"])
def test_resume_refuses_incompatible_or_corrupt_state(tmp_path, monkeypatch, change):
    _, _, q = cached(tmp_path)
    root = tmp_path / "run"
    options = dict(threads=2, chunk_size=3)
    checkpoint_contributions(q, root, max_chunks=None if change == "final" else 1, **options)
    if change == "code":
        monkeypatch.setattr("parallel_bc.quotient_checkpoint.checkpoint_code_digest", lambda: "changed")
    elif change == "config":
        options["chunk_size"] = 4
    elif change in {"plan", "chunk", "final"}:
        target = (root / "source_plan.npy" if change == "plan" else
                  root / "chunk_000000.npy" if change == "chunk" else root / "bc_raw.npy")
        with target.open("ab") as stream:
            stream.write(b"corruption")
    else:
        m = manifest(root)
        m["chunks"][0]["stop" if change == "coverage" else "source_mass"] += 1
        (root / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        checkpoint_contributions(q, root, resume=True, **options)


def test_lock_refusal_and_no_overwrite(tmp_path):
    _, _, q = cached(tmp_path)
    root = tmp_path / "run"
    checkpoint_contributions(q, root, max_chunks=0)
    with pytest.raises(FileExistsError):
        checkpoint_contributions(q, root)
    with _run_lock(root):
        with pytest.raises(RuntimeError, match="locked"):
            checkpoint_contributions(q, root, resume=True)
    with pytest.raises(ValueError, match="manifest"):
        checkpoint_contributions(q, tmp_path / "missing", resume=True)


def test_supervisor_timeout_and_reserve_are_preserved(tmp_path):
    assert bounded_job("quotient_checkpoint", {}, seconds=.001)["status"] == "timeout"
    assert bounded_job("quotient_checkpoint", {}, seconds=1,
                       min_available_bytes=2**100)["status"] == "resource_not_run"


def test_checkpoint_bigint_fallback_and_weighted_blocks(tmp_path):
    layers = [[0]] + [[2 * i - 1, 2 * i] for i in range(1, 66)]
    graph = nx.Graph([(a, b) for left, right in zip(layers, layers[1:])
                      for a in left for b in right])
    graph.add_edges_from([(layers[35][0], 131), (131, 132)])
    _, _, q = cached(tmp_path, graph)
    root = tmp_path / "run"
    checkpoint_contributions(q, root, threads=2, chunk_size=16)
    assert sum(c["fallback_sources"] for c in manifest(root)["chunks"]) > 0
    direct, _ = quotient_contributions(q)
    np.testing.assert_allclose(np.load(root / "contributions.npy"), direct, atol=1e-8, rtol=1e-10)


@pytest.mark.parametrize("stage", ["final_array", "final_manifest"])
def test_finalization_interruption_reuses_all_committed_sources(tmp_path, monkeypatch, stage):
    import parallel_bc.quotient_checkpoint as module
    _, _, q = cached(tmp_path)
    root = tmp_path / "run"
    writer = module._write_array if stage == "final_array" else module._write_json
    def fail(path, value):
        if ((stage == "final_array" and path.name == "bc_raw.npy")
                or (stage == "final_manifest" and path.name == "manifest.json"
                    and value.get("status") == "complete")):
            raise RuntimeError("finalization interrupted")
        writer(path, value)
    with monkeypatch.context() as patch:
        patch.setattr(module, "_write_array" if stage == "final_array" else "_write_json", fail)
        with pytest.raises(RuntimeError, match="finalization"):
            checkpoint_contributions(q, root, threads=2, chunk_size=3)
    before = manifest(root)
    assert not before["exact_all_sources"] and before["completed_core_groups"] == before["identity"]["core_traversals"]
    row = checkpoint_contributions(q, root, resume=True, threads=2, chunk_size=3)
    assert row["new_chunks"] == 0 and row["exact_all_sources"]
    assert before["chunks"] == manifest(root)["chunks"]
    direct, _ = quotient_contributions(q)
    np.testing.assert_allclose(np.load(root / "contributions.npy"), direct, atol=1e-9)


def test_cli_pause_resume_and_result_flags(tmp_path):
    _, path, _ = cached(tmp_path)
    root = tmp_path / "run"
    # CLI expects a data/processed/<id> path, so name the existing cache via it.
    data = tmp_path / "data" / "processed"
    data.mkdir(parents=True)
    shutil.copytree(path, data / "tiny_treeq")
    command = [sys.executable, "-m", "parallel_bc", "quotient-run", "--dataset", "tiny",
        "--data-dir", str(tmp_path / "data"), "--run-dir", str(root), "--threads", "2",
        "--chunk-size", "3", "--reserve-gib", "0", "--timeout", "60"]
    first = subprocess.run(command + ["--max-chunks", "1"], capture_output=True, timeout=60)
    assert first.returncode == 0, first.stderr.decode(errors="replace")
    assert not manifest(root)["exact_all_sources"]
    second = subprocess.run(command + ["--resume"], capture_output=True, timeout=60)
    assert second.returncode == 0, second.stderr.decode(errors="replace")
    assert manifest(root)["exact_all_sources"]
