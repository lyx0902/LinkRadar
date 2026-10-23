"""Node-removal protocol tests independent of multiprocessing and the network."""
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from parallel_bc.data import graph_from_edges
from parallel_bc.robustness import ranking_rows, run_robustness


def _graph(edges=(), nodes=()):
    return graph_from_edges(edges, nodes=nodes, metadata={"dataset_id": "test_graph"})


def test_fixed_rankings_do_not_recompute_after_removals():
    graph = _graph([(0, 1), (1, 2), (2, 3), (3, 4)])
    result = run_robustness(graph, [0, 3, 4, 3, 0],
                            {"fractions": [0, .2, .4, .6, 1], "strategies": ["bc", "degree"]})
    bc = {row["k"]: row for row in result["raw"] if row["strategy"] == "bc"}
    # Original BC order is 2, 1, 3, 0, 4. At k=3 only isolated 0 and 4 remain.
    assert bc[3]["lcc_size"] == 1
    assert bc[3]["num_components"] == 2
    assert bc[3]["remaining_edges"] == 0
    assert bc[3]["lcc_original_fraction"] == .2
    assert bc[3]["lcc_remaining_fraction"] == .5
    assert bc[5]["lcc_size"] == bc[5]["num_components"] == 0
    assert bc[5]["lcc_remaining_fraction"] is None
    assert all(row["count"] == 1 for row in result["summary"])


def test_original_id_tie_breaking_and_ordinal_ranks():
    graph = _graph([(9, 2), (2, 7)], nodes=[-1])
    rows = ranking_rows(graph, [0, 0, 0, 0])
    assert [row["original_id"] for row in rows] == [-1, 2, 7, 9]
    assert [row["rank_bc"] for row in rows] == [1, 2, 3, 4]
    assert [row["rank_degree"] for row in rows] == [4, 1, 2, 3]
    assert all(row["bc_normalized"] == 0 for row in rows)
    result = run_robustness(graph, [0, 0, 0, 0], {"fractions": [.25], "strategies": ["bc"]})
    last = result["raw"][-1]
    assert last["lcc_size"] == 3  # Lowest original ID is the isolate.
    assert last["num_components"] == 1


def test_disconnected_baseline_preserves_original_denominator():
    graph = _graph([(1, 2), (3, 4)], nodes=[5])
    result = run_robustness(graph, np.zeros(5), {"fractions": [0, .01, .02, .19, .2, .21, 1], "strategies": ["bc"]})
    assert [row["k"] for row in result["raw"]] == [0, 1, 5]
    first = result["raw"][0]
    assert first["lcc_original_fraction"] == .4
    assert first["num_components"] == 3
    assert first["remaining_edges"] == 2
    assert result["raw"][1]["removed_fraction"] == .2


def test_random_curves_are_independent_reproducible_nested_prefixes():
    graph = _graph([(i, i+1) for i in range(11)])
    config = {"fractions": [0, .25, .5, .75, 1], "strategies": ["random"], "random_repeats": 4, "seed": 123}
    first = run_robustness(graph, np.zeros(graph.n), config)
    second = run_robustness(graph, np.zeros(graph.n), config)
    assert first["raw"] == second["raw"]
    assert first["summary"] == second["summary"]
    assert len({row["seed"] for row in first["raw"]}) == 4
    original = nx.Graph()
    original.add_nodes_from(range(graph.n))
    original.add_edges_from((u, v) for u, ns in enumerate(graph.adj) for v in ns if u < v)
    for row in first["raw"]:
        order = np.random.default_rng(row["seed"]).permutation(graph.n).tolist()
        remaining = original.subgraph(set(range(graph.n)) - set(order[:row["k"]]))
        sizes = [len(component) for component in nx.connected_components(remaining)]
        assert row["lcc_size"] == max(sizes, default=0)
        assert row["remaining_edges"] == remaining.number_of_edges()
        assert row["num_components"] == len(sizes)
    for row in first["summary"]:
        values = [r["lcc_original_fraction"] for r in first["raw"] if r["k"] == row["k"]]
        assert row["count"] == 4
        assert row["lcc_original_mean"] == pytest.approx(np.mean(values))
        assert row["lcc_original_p10"] == pytest.approx(np.quantile(values, .1))
        assert row["lcc_original_p90"] == pytest.approx(np.quantile(values, .9))
    assert first["raw"] != run_robustness(graph, np.zeros(graph.n), dict(config, seed=124))["raw"]


@pytest.mark.parametrize("nodes", [[], [10], [10, 20]])
def test_empty_and_tiny_graphs(nodes):
    graph = _graph(nodes=nodes)
    result = run_robustness(graph, np.zeros(graph.n), {"fractions": [0, .5, 1], "random_repeats": 2})
    for strategy in ("bc", "degree", "random"):
        rows = [row for row in result["raw"] if row["strategy"] == strategy]
        assert rows
        assert all(row["remaining_edges"] == 0 for row in rows)
        assert all(row["lcc_size"] == int(row["remaining_nodes"] > 0) for row in rows)
    assert all(row["bc_normalized"] == 0 for row in ranking_rows(graph, np.zeros(graph.n)))


@pytest.mark.parametrize("scores", [[0], [0, np.nan], [0, np.inf], [0, -1], [[0, 0]]])
def test_incomplete_nonfinite_or_negative_scores_fail(scores):
    graph = _graph([(1, 2)])
    with pytest.raises(ValueError):
        run_robustness(graph, scores)


def test_cache_provenance_rejects_wrong_graph_or_approximation():
    graph = _graph([(1, 2)])
    with pytest.raises(ValueError, match="graph_sha256"):
        run_robustness(graph, SimpleNamespace(raw=np.zeros(2), graph_sha256="other"))
    with pytest.raises(ValueError, match="全起点"):
        run_robustness(graph, {"raw": [0, 0], "graph_sha256": graph.graph_sha256, "exact_all_sources": False})
    scores = SimpleNamespace(raw=np.zeros(2), normalized=np.zeros(2), graph_sha256=graph.graph_sha256, exact_all_sources=True)
    assert run_robustness(graph, scores)["raw"]


@pytest.mark.parametrize("config", [{"fractions": [-.1]}, {"fractions": [1.1]}, {"fractions": [np.nan]},
                                    {"random_repeats": 0}, {"strategies": []}, {"strategies": ["adaptive"]}, {"seed": -1}])
def test_invalid_protocol_configuration_fails(config):
    graph = _graph([(1, 2)])
    with pytest.raises(ValueError):
        run_robustness(graph, np.zeros(graph.n), config)
