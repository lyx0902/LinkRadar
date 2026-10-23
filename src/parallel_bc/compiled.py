"""Optional single-threaded, overflow-checked CSR Brandes native kernel.

Only the numeric loops are JIT compiled. No source sampling, graph reduction,
fastmath, prange, or parallel execution is performed here.
"""
from __future__ import annotations

from dataclasses import dataclass
import operator
import time

import numpy as np

from .brandes import single_source_accumulate
from .csr import validate_csr

try:
    import numba
except ImportError as exc:
    raise RuntimeError('Compiled backend requires: pip install -e ".[compiled]"') from exc


_INT64_MAX = np.iinfo(np.int64).max


@numba.njit(cache=True, nogil=True, fastmath=False)
def _source_dependencies(indptr, indices, source, dist, sigma, delta, queue):
    """Return (status, visited, max_sigma); never mutate the caller's total.

    status 0: success; 1: path counts would overflow; 2: nonfinite dependency.
    On overflow the Python caller retries this source with arbitrary integers.
    """
    n = len(dist)
    for v in range(n):
        dist[v] = -1
        sigma[v] = 0
        delta[v] = 0.0
    dist[source] = 0
    sigma[source] = 1
    queue[0] = source
    head, tail, max_sigma = 0, 1, 1
    while head < tail:
        v = queue[head]
        head += 1
        next_distance = dist[v] + 1
        for edge in range(indptr[v], indptr[v + 1]):
            w = indices[edge]
            if dist[w] == -1:
                dist[w] = next_distance
                queue[tail] = w
                tail += 1
            if dist[w] == next_distance:
                if sigma[v] > _INT64_MAX - sigma[w]:
                    return 1, tail, max_sigma
                sigma[w] += sigma[v]
                max_sigma = max(max_sigma, sigma[w])
    # The shortest-path DAG is implicit in the CSR + distance labels.
    # Pull from successors instead of allocating Python predecessor lists.
    for pos in range(tail - 1, -1, -1):
        v = queue[pos]
        value = 0.0
        next_distance = dist[v] + 1
        for edge in range(indptr[v], indptr[v + 1]):
            w = indices[edge]
            if dist[w] == next_distance:
                value += (sigma[v] / sigma[w]) * (1.0 + delta[w])
        if not np.isfinite(value):
            return 2, tail, max_sigma
        delta[v] = value
    return 0, tail, max_sigma


@numba.njit(cache=True, nogil=True, fastmath=False)
def _commit_source(total, delta, queue, visited, source):
    for pos in range(visited):
        v = queue[pos]
        if v != source:
            total[v] += delta[v]


@dataclass
class CompiledWorkspace:
    dist: np.ndarray
    sigma: np.ndarray
    delta: np.ndarray
    queue: np.ndarray

    @classmethod
    def allocate(cls, n):
        return cls(np.empty(n, np.int32), np.empty(n, np.int64),
                   np.empty(n, np.float64), np.empty(n, np.int32))

    @property
    def nbytes(self):
        return sum(a.nbytes for a in (self.dist, self.sigma, self.delta, self.queue))


def warmup():
    """Compile/load the same readonly contiguous CSR signature on a tiny graph.

    Returns dispatch-first-call time, not pure compiler time: disk cache loading
    and execution of the two-node warmup are included. Never compile a big run.
    """
    start = time.perf_counter()
    indptr = np.array([0, 1, 2], dtype=np.int64)
    indices = np.array([1, 0], dtype=np.int32)
    indptr.flags.writeable = indices.flags.writeable = False
    workspace = CompiledWorkspace.allocate(2)
    total = np.zeros(2)
    status, visited, _ = _source_dependencies(indptr, indices, 0, workspace.dist,
        workspace.sigma, workspace.delta, workspace.queue)
    if status:
        raise RuntimeError("Compiled warmup failed")
    _commit_source(total, workspace.delta, workspace.queue, visited, 0)
    if not _source_dependencies.nopython_signatures or not _commit_source.nopython_signatures:
        raise RuntimeError("Native nopython compilation is required")
    return dict(t_jit_warmup=time.perf_counter() - start,
                numba_version=numba.__version__, native_nopython=True,
                jit_options=dict(cache=True, nogil=True, fastmath=False, parallel=False))


class CompiledAccumulator:
    """Prepared single-core runner for an already validated immutable CSR graph."""
    def __init__(self, graph):
        # Must be validated by load_csr/csr_from_edges before this trusted path.
        # Basic checks also avoid accidental type specializations after warmup.
        validate_csr(graph, full=False)
        n = graph.n
        if (graph.indptr.dtype != np.int64 or graph.indices.dtype != np.int32
                or graph.indptr.shape != (n + 1,) or graph.indices.ndim != 1
                or not graph.indptr.flags.c_contiguous or not graph.indices.flags.c_contiguous
                or n > np.iinfo(np.int32).max):
            raise ValueError("Compiled backend requires validated int64/int32 CSR")
        self.graph = graph
        self.indptr = graph.indptr.view()
        self.indices = graph.indices.view()
        self.indptr.flags.writeable = self.indices.flags.writeable = False
        self.workspace = CompiledWorkspace.allocate(n)

    def accumulate(self, source, total):
        source = operator.index(source)
        if (not 0 <= source < self.graph.n or total.shape != (self.graph.n,)
                or total.dtype != np.float64 or not total.flags.c_contiguous
                or not total.flags.writeable):
            raise ValueError("Invalid compiled source or float64 accumulator")
        work = self.workspace
        status, visited, max_sigma = _source_dependencies(self.indptr, self.indices, source,
            work.dist, work.sigma, work.delta, work.queue)
        if status == 1:
            # No part of this source has reached total: safe to recompute once.
            single_source_accumulate(self.graph.adj, source, total)
            return dict(execution="python_bigint_fallback", fallback_reason="sigma_int64_overflow",
                        max_sigma_before_fallback=int(max_sigma))
        if status != 0:
            raise FloatingPointError("Nonfinite compiled source dependency; source not committed")
        _commit_source(total, work.delta, work.queue, visited, source)
        return dict(execution="numba_native", reachable_nodes=int(visited),
                    max_sigma=int(max_sigma))
