"""Experimental exact weighted Brandes storing BFS successor edges per worker."""
import numpy as np
import time
from numba import njit
from .compiled import CompiledWorkspace
from .quotient_brandes import _weighted_commit, _python_weighted_dependencies

MAX_INT64 = np.iinfo(np.int64).max

@njit(cache=True, nogil=True, fastmath=False)
def dag_dependencies(ptr, indices, mass, source, dist, sigma, delta, queue, offsets, successors):
    for v in range(len(dist)):
        dist[v], sigma[v], delta[v] = -1, 0, 0.0
    dist[source], sigma[source], queue[0] = 0, 1, source
    head, tail, maximum, edge_count = 0, 1, 1, 0
    offsets[0] = 0
    while head < tail:
        v = queue[head]
        next_distance = dist[v] + 1
        for edge in range(ptr[v], ptr[v+1]):
            w = indices[edge]
            if dist[w] == -1:
                dist[w] = next_distance
                queue[tail] = w
                tail += 1
            if dist[w] == next_distance:
                if sigma[v] > MAX_INT64 - sigma[w]:
                    return 1, tail, maximum, edge_count
                sigma[w] += sigma[v]
                maximum = max(maximum, sigma[w])
                successors[edge_count] = w
                edge_count += 1
        head += 1
        offsets[head] = edge_count
    for pos in range(tail-1, -1, -1):
        v = queue[pos]
        value = 0.0
        for edge in range(offsets[pos], offsets[pos+1]):
            w = successors[edge]
            value += (sigma[v] / sigma[w]) * (mass[w] + delta[w])
        if not np.isfinite(value):
            return 2, tail, maximum, edge_count
        delta[v] = value
    return 0, tail, maximum, edge_count

class DagAccumulator:
    def __init__(self, core, mass, source_counts):
        self.core, self.mass, self.source_counts = core, mass, source_counts
        self.workspace = CompiledWorkspace.allocate(core.n)
        # Safe upper bound: no directed adjacency item is recorded twice.
        self.successors = np.empty(len(core.indices), dtype=np.int32)
        self.offsets = np.empty(core.n + 1, dtype=np.int64)
        self.extra_bytes = self.successors.nbytes + self.offsets.nbytes

    def accumulate(self, source, total):
        w = self.workspace
        status, visited, maximum, edges = dag_dependencies(self.core.indptr, self.core.indices,
            self.mass, source, w.dist, w.sigma, w.delta, w.queue, self.offsets, self.successors)
        count = int(self.source_counts[source])
        if status == 1:
            delta = _python_weighted_dependencies(self.core, self.mass, source)
            total += count * delta
            return dict(execution='python_bigint_fallback', source_multiplicity=count,
                        fallback_reason='sigma_int64_overflow')
        if status:
            raise FloatingPointError('Nonfinite dependency; source not committed')
        _weighted_commit(total, w.delta, w.queue, visited, source, count)
        return dict(execution='numba_native', source_multiplicity=count,
                    reachable_core_nodes=int(visited), max_sigma=int(maximum), dag_edges=int(edges))

def dag_warmup():
    """Compile the readonly CSR signature before starting worker threads."""
    start = time.perf_counter()
    ptr = np.array([0, 1, 2], np.int64)
    indices = np.array([1, 0], np.int32)
    mass = np.ones(2, np.int64)
    for a in (ptr, indices, mass):
        a.flags.writeable = False
    work = CompiledWorkspace.allocate(2)
    offsets, successors = np.empty(3, np.int64), np.empty(2, np.int32)
    status, visited, _, _ = dag_dependencies(ptr, indices, mass, 0, work.dist,
        work.sigma, work.delta, work.queue, offsets, successors)
    if status:
        raise RuntimeError('DAG kernel warmup failed')
    _weighted_commit(np.zeros(2), work.delta, work.queue, visited, 0, 1)
    if not dag_dependencies.nopython_signatures or not _weighted_commit.nopython_signatures:
        raise RuntimeError('DAG native nopython compilation is required')
    return dict(t_jit_warmup=time.perf_counter() - start, native_nopython=True,
                jit_options=dict(cache=True, nogil=True, fastmath=False, parallel=False))
