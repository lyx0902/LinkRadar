import networkx as nx
import numpy as np
import pytest

from parallel_bc.csr import csr_from_edges
from parallel_bc.quotient_brandes import WeightedAccumulator
from parallel_bc.quotient_dag_experiment import DagAccumulator

@pytest.mark.parametrize('graph', [nx.cycle_graph(8), nx.disjoint_union(nx.path_graph(6), nx.cycle_graph(5)),
                                   nx.gnp_random_graph(30, 0.14, seed=42)])
def test_candidate_matches_old_weighted_kernel_with_reused_workspace(graph):
    csr = csr_from_edges(graph.edges(), nodes=np.arange(len(graph)))
    mass = (np.arange(csr.n, dtype=np.int64) % 5) + 1
    counts = (np.arange(csr.n, dtype=np.int64) % 3) + 1
    original = WeightedAccumulator(csr, mass, counts)
    candidate = DagAccumulator(csr, mass, counts)
    old, new = np.zeros(csr.n), np.zeros(csr.n)
    for source in list(range(csr.n)) + [0, csr.n-1]:
        original.accumulate(source, old)
        candidate.accumulate(source, new)
        np.testing.assert_allclose(new, old, rtol=1e-10, atol=1e-8)

def test_overflow_fallback_recomputes_whole_source():
    layers = [[0]] + [[2*i-1, 2*i] for i in range(1, 66)]
    graph = nx.Graph([(a,b) for left,right in zip(layers, layers[1:]) for a in left for b in right])
    csr = csr_from_edges(graph.edges())
    mass = np.ones(csr.n, np.int64)
    old, new = np.zeros(csr.n), np.zeros(csr.n)
    original = WeightedAccumulator(csr, mass, mass)
    candidate = DagAccumulator(csr, mass, mass)
    assert original.accumulate(0, old)['execution'] == 'python_bigint_fallback'
    assert candidate.accumulate(0, new)['execution'] == 'python_bigint_fallback'
    candidate.accumulate(1, new)
    original.accumulate(1, old)
    np.testing.assert_allclose(new, old, rtol=1e-10, atol=1e-8)

def test_diamond_records_both_predecessors():
    graph = nx.Graph([(0,1), (0,2), (1,3), (2,3)])
    csr = csr_from_edges(graph.edges(), nodes=np.arange(4))
    mass = np.ones(4, dtype=np.int64)
    runner = DagAccumulator(csr, mass, mass)
    total = np.zeros(4)
    info = runner.accumulate(0, total)
    assert info['dag_edges'] == 4
    assert runner.workspace.sigma[3] == 2
    np.testing.assert_allclose(total, [0, 0.5, 0.5, 0])
