"""Real spawn integration tests: scheduling, coverage, failure and reclamation."""
import multiprocessing as mp
import os

import networkx as nx
import numpy as np
import pytest

from parallel_bc.brandes import brandes_serial
from parallel_bc.parallel import ParallelWorkerError, available_cpu_count, brandes_parallel
from parallel_bc.validation import compare_results


def adjacency(graph):
    return tuple(tuple(sorted(graph.neighbors(i))) for i in range(len(graph)))


@pytest.mark.parametrize("mode", ["static_batches", "dynamic_batches"])
@pytest.mark.parametrize("processes,batch_size", [(1, 1), (2, 4), (3, 1), (3, 100), (3, 7)])
def test_spawn_results_coverage_and_timing(mode, processes, batch_size):
    adj = adjacency(nx.gnp_random_graph(17, 0.25, seed=42))
    order = list(reversed(range(len(adj))))
    serial = brandes_serial(adj, source_order=order, batch_size=batch_size, task_logging=True)
    result = brandes_parallel(adj, processes=processes, mode=mode, batch_size=batch_size,
                              source_order=order, task_logging=True, timeout_seconds=30)
    assert result.start_method == "spawn"
    assert compare_results(result, serial)["allclose"]
    assert result.source_order_hash == serial.source_order_hash
    assert result.p_used == min(processes, available_cpu_count(), len(adj))
    assert len({w["pid"] for w in result.workers}) == result.p_used
    assert all(w["pid"] != os.getpid() for w in result.workers)
    assert sum(w["processed_sources"] for w in result.workers) == len(adj)
    assert sum(w["processed_batches"] for w in result.workers) == len(serial.tasks)
    assert sorted(t["batch_id"] for t in result.tasks) == list(range(len(serial.tasks)))
    assert result.active_workers == sum(w["processed_sources"] > 0 for w in result.workers)
    for task in result.tasks:
        assert task["start_rel_seconds"] <= task["end_rel_seconds"]
        assert task["compute_seconds"] == pytest.approx(task["end_rel_seconds"] - task["start_rel_seconds"])
        assert task["source_order_hash"] == serial.tasks[task["batch_id"]]["source_order_hash"]
        if mode == "static_batches":
            assert task["worker_id"] == task["batch_id"] % result.p_used
    if batch_size > len(adj):
        assert result.active_workers == 1
    assert result.timings["t_steady"] == pytest.approx(result.timings["t_execute_reduce"] + result.timings["t_finalize"])
    phases = sum(result.timings[k] for k in ("t_startup", "t_execute_reduce", "t_finalize", "t_cleanup"))
    assert result.timings["t_total"] >= phases - 1e-8
    assert result.timings["t_total"] - phases < 1.0
    assert not mp.active_children()


def test_empty_graph_starts_no_process_and_limit_adjustment():
    empty = brandes_parallel((), processes=3)
    assert empty.p_requested == 3 and empty.p_used == 0 and empty.active_workers == 0
    assert empty.raw.size == 0 and empty.workers == []
    result = brandes_parallel(((1,), (0,)), processes=999, max_processes=1,
                              batch_size=1, timeout_seconds=30)
    assert result.p_requested == 999 and result.p_used == 1
    np.testing.assert_array_equal(result.raw, [0, 0])
    assert result.tasks == []
    assert not mp.active_children()


@pytest.mark.parametrize("mode", ["static_batches", "dynamic_batches"])
def test_child_error_is_reported_and_all_children_reclaimed(mode):
    # The malformed neighbor is reached inside the actual spawned worker.
    with pytest.raises(ParallelWorkerError, match="invalid adjacency index"):
        brandes_parallel(((99,), ()), processes=2, mode=mode, batch_size=1,
                          timeout_seconds=30)
    assert not mp.active_children()


def test_timeout_reclaims_workers():
    with pytest.raises(TimeoutError):
        brandes_parallel(adjacency(nx.path_graph(50)), processes=2, timeout_seconds=0.05)
    assert not mp.active_children()


def test_keyboard_interrupt_reclaims_workers(monkeypatch):
    import parallel_bc.parallel as module

    checks = [0]

    def interrupt(_deadline):
        checks[0] += 1
        if checks[0] >= 3:
            raise KeyboardInterrupt

    monkeypatch.setattr(module, "check_deadline", interrupt)
    with pytest.raises(KeyboardInterrupt):
        brandes_parallel(adjacency(nx.path_graph(5)), processes=2, timeout_seconds=30)
    assert not mp.active_children()


@pytest.mark.parametrize("kwargs", [
    {"processes": 0}, {"processes": 1.5}, {"max_processes": 0}, {"mode": "pool"},
    {"batch_size": 0}, {"source_order": [0, 0, 2]}, {"timeout_seconds": -1},
])
def test_parallel_validates_parameters_before_spawn(kwargs):
    with pytest.raises(ValueError):
        brandes_parallel(adjacency(nx.path_graph(3)), **kwargs)
    assert not mp.active_children()


def test_cpu_affinity_cap_is_applied_and_recorded(monkeypatch):
    import parallel_bc.parallel as module
    monkeypatch.setattr(module, "available_cpu_count", lambda: 1)
    result = module.brandes_parallel(adjacency(nx.path_graph(3)), processes=3,
                                     batch_size=1, timeout_seconds=30)
    assert result.p_requested == 3
    assert result.p_used == 1
    assert result.active_workers == 1
    assert not mp.active_children()
