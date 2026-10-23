"""Hand-computable fixtures and independent all-node NetworkX comparisons."""
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from parallel_bc.brandes import (brandes_serial, finalize_scores,
                                 single_source_accumulate)
from parallel_bc.validation import compare_results, validate_against_networkx


def adjacency(graph):
    return tuple(tuple(sorted(graph.neighbors(i))) for i in range(len(graph)))


def test_readable_five_node_fixture():
    # Retain the requested labels: S=0, A=1, B=2, T=3, U=4.
    labels = ("S", "A", "B", "T", "U")
    graph = nx.Graph([(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)])
    adj = adjacency(graph)
    contribution = np.zeros(5)
    single_source_accumulate(adj, 0, contribution)
    assert dict(zip(labels, contribution)) == {"S": 0, "A": 1, "B": 1, "T": 1, "U": 0}
    result = brandes_serial(adj)
    np.testing.assert_allclose(result.raw, [0.5, 1, 1, 3.5, 0])
    np.testing.assert_allclose(result.normalized, [1 / 12, 1 / 6, 1 / 6, 7 / 12, 0])


@pytest.mark.parametrize("graph, expected", [
    (nx.empty_graph(0), []), (nx.empty_graph(1), [0]),
    (nx.empty_graph(2), [0, 0]), (nx.path_graph(2), [0, 0]),
    (nx.path_graph(3), [0, 1, 0]), (nx.star_graph(4), [6, 0, 0, 0, 0]),
    (nx.complete_graph(5), [0] * 5), (nx.cycle_graph(4), [0.5] * 4),
])
def test_hand_computable_graphs(graph, expected):
    result = brandes_serial(adjacency(graph), batch_size=2, task_logging=True)
    np.testing.assert_allclose(result.raw, expected)
    clean = SimpleNamespace(ids=tuple(range(len(graph))), adj=adjacency(graph))
    assert validate_against_networkx(clean, result)["allclose"]
    assert sum(worker["processed_sources"] for worker in result.workers) == len(graph)


def test_disconnected_graph_uses_global_normalization_and_original_ids():
    graph = nx.Graph([(0, 1), (1, 2), (3, 4)])
    graph.add_node(5)
    result = brandes_serial(adjacency(graph))
    assert result.raw[1] == 1
    assert result.normalized[1] == pytest.approx(0.1)
    clean = SimpleNamespace(ids=(-11, 7, 21, 100, 200, 1000), adj=adjacency(graph))
    validation = validate_against_networkx(clean, result)
    assert validation["allclose"] and validation["finite"]
    assert validation["compared_nodes"] == 6
    assert validation["max_abs_error"] < 1e-10


@pytest.mark.parametrize("seed", [0, 7, 42])
@pytest.mark.parametrize("n,density", [(5, 0.1), (17, 0.2), (50, 0.05), (50, 0.5)])
def test_random_graphs_all_nodes(seed, n, density):
    adj = adjacency(nx.gnp_random_graph(n, density, seed=seed))
    clean = SimpleNamespace(ids=tuple(range(n)), adj=adj)
    result = brandes_serial(adj, source_order=list(reversed(range(n))))
    validation = validate_against_networkx(clean, result)
    assert validation["allclose"], validation


def test_single_source_sum_matches_runner_and_logs():
    adj = adjacency(nx.karate_club_graph())
    total = np.zeros(len(adj))
    for source in range(len(adj)):
        single_source_accumulate(adj, source, total)
    raw, normalized = finalize_scores(total, len(adj))
    result = brandes_serial(adj, batch_size=7, task_logging=True)
    np.testing.assert_array_equal(raw, result.raw)
    np.testing.assert_array_equal(normalized, result.normalized)
    assert len(result.tasks) == 5
    assert sum(task["source_count"] for task in result.tasks) == len(adj)
    for task in result.tasks:
        assert task["start_rel_seconds"] <= task["end_rel_seconds"]
        assert task["compute_seconds"] == pytest.approx(task["end_rel_seconds"] - task["start_rel_seconds"])
    assert result.timings["t_steady"] == pytest.approx(result.timings["t_execute_reduce"] + result.timings["t_finalize"])
    assert result.timings["t_total"] >= result.timings["t_steady"]


@pytest.mark.parametrize("order", [[0, 0, 2], [0, 1], [0, 1, 3], [0, 1, 1.5]])
def test_reject_partial_duplicate_or_invalid_source_order(order):
    with pytest.raises(ValueError, match="source_order"):
        brandes_serial(adjacency(nx.path_graph(3)), source_order=order)


@pytest.mark.parametrize("batch_size", [0, -1, 1.5])
def test_reject_invalid_batch_sizes(batch_size):
    with pytest.raises(ValueError):
        brandes_serial((), batch_size=batch_size)


def test_reject_nonfinite_scores_and_expired_serial_budget():
    with pytest.raises(FloatingPointError):
        finalize_scores(np.array([0.0, np.inf, 1.0]), 3)
    with pytest.raises(TimeoutError):
        brandes_serial(adjacency(nx.path_graph(10)), timeout_seconds=1e-12)


def test_compare_checks_all_nodes_including_low_rank_nodes():
    result = brandes_serial(adjacency(nx.path_graph(5)))
    other = brandes_serial(adjacency(nx.path_graph(5)))
    other.raw[4] = 0.1
    validation = compare_results(result, other, ids=[10, 20, 30, 40, 50])
    assert not validation["allclose"]
    assert validation["worst_node"] == 50
    assert validation["max_abs_error"] == pytest.approx(0.1)
