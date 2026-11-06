"""Experimental weighted Brandes with a cached successor coefficient."""
import numpy as np
from numba import njit
from .compiled import CompiledWorkspace
from .quotient_brandes import _weighted_commit, _python_weighted_dependencies

_MAX = np.iinfo(np.int64).max

@njit(cache=True, nogil=True, fastmath=False)
def dependencies(ptr, indices, mass, source, dist, sigma, delta, queue, coeff):
    for v in range(len(dist)):
        dist[v], sigma[v], delta[v] = -1, 0, 0.0
    dist[source], sigma[source], queue[0] = 0, 1, source
    head, tail, maximum = 0, 1, 1
    while head < tail:
        v = queue[head]
        head += 1
        next_distance = dist[v] + 1
        for edge in range(ptr[v], ptr[v + 1]):
            w = indices[edge]
            if dist[w] == -1:
                dist[w] = next_distance
                queue[tail] = w
                tail += 1
            if dist[w] == next_distance:
                if sigma[v] > _MAX - sigma[w]:
                    return 1, tail, maximum
                sigma[w] += sigma[v]
                maximum = max(maximum, sigma[w])
    for pos in range(tail - 1, -1, -1):
        v = queue[pos]
        value = 0.0
        for edge in range(ptr[v], ptr[v + 1]):
            w = indices[edge]
            if dist[w] == dist[v] + 1:
                value += sigma[v] * coeff[w]
        if not np.isfinite(value):
            return 2, tail, maximum
        delta[v] = value
        coeff[v] = (mass[v] + value) / sigma[v]
        if not np.isfinite(coeff[v]):
            return 2, tail, maximum
    return 0, tail, maximum

class CoefficientAccumulator:
    def __init__(self, core, mass, source_counts):
        self.core, self.mass, self.source_counts = core, mass, source_counts
        self.workspace = CompiledWorkspace.allocate(core.n)
        self.coeff = np.empty(core.n, np.float64)

    def accumulate(self, source, total):
        w = self.workspace
        status, visited, maximum = dependencies(self.core.indptr, self.core.indices,
            self.mass, source, w.dist, w.sigma, w.delta, w.queue, self.coeff)
        count = int(self.source_counts[source])
        if status == 1:
            delta = _python_weighted_dependencies(self.core, self.mass, source)
            total += count * delta
            return dict(execution='python_bigint_fallback', source_multiplicity=count,
                        fallback_reason='sigma_int64_overflow')
        if status:
            raise FloatingPointError('Nonfinite coefficient; source not committed')
        _weighted_commit(total, w.delta, w.queue, visited, source, count)
        return dict(execution='numba_native', source_multiplicity=count,
                    reachable_core_nodes=int(visited), max_sigma=int(maximum))
