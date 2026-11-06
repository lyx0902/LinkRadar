import networkx as nx
import numpy as np

from parallel_bc.csr import csr_from_edges
from scripts.count_shortest_path_edges import counted_bfs, recount

def test_diamond_with_same_level_edge_and_isolate():
    graph = nx.Graph([(0,1), (0,2), (1,2), (1,3), (2,3)])
    graph.add_node(4)
    csr = csr_from_edges(graph.edges(), nodes=np.arange(5))
    dist = np.empty(5, np.int32)
    sigma = np.empty(5, np.int64)
    queue = np.empty(5, np.int32)
    status, visited, scanned, valid = counted_bfs(csr.indptr, csr.indices, 0, dist, sigma, queue)
    assert (status, visited, scanned, valid) == (0, 4, 10, 4)
    np.testing.assert_array_equal(dist, [0, 1, 1, 2, -1])
    np.testing.assert_array_equal(sigma, [1, 1, 1, 2, 0])
    assert recount(csr.indptr, csr.indices, dist) == (4, 2, 4)
