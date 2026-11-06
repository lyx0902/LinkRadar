"""Exact original-graph source contributions via weighted quotient + recovery."""
from __future__ import annotations

import math
import operator
import time

import numpy as np
from numba import njit

from .brandes import source_order_digest
from .compiled import CompiledWorkspace
from .compiled_threads import threaded_contributions


_INT64_MAX = np.iinfo(np.int64).max


@njit(cache=True, nogil=True, fastmath=False)
def _weighted_dependencies(indptr, indices, mass, source, dist, sigma, delta, queue):
    for v in range(len(dist)):
        dist[v], sigma[v], delta[v] = -1, 0, 0.0
    dist[source], sigma[source], queue[0] = 0, 1, source
    head, tail, maximum = 0, 1, 1
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
                    return 1, tail, maximum
                sigma[w] += sigma[v]
                maximum = max(maximum, sigma[w])
    for pos in range(tail - 1, -1, -1):
        v = queue[pos]
        value = 0.0
        for edge in range(indptr[v], indptr[v + 1]):
            w = indices[edge]
            if dist[w] == dist[v] + 1:
                value += (sigma[v] / sigma[w]) * (mass[w] + delta[w])
        if not np.isfinite(value):
            return 2, tail, maximum
        delta[v] = value
    return 0, tail, maximum


@njit(cache=True, nogil=True, fastmath=False)
def _weighted_commit(total, delta, queue, visited, source, source_count):
    for pos in range(visited):
        v = queue[pos]
        if v != source:
            total[v] += source_count * delta[v]


def weighted_warmup():
    start = time.perf_counter()
    ptr = np.array([0, 1, 2], np.int64)
    indices = np.array([1, 0], np.int32)
    mass = np.ones(2, np.int64)
    for a in (ptr, indices, mass):
        a.flags.writeable = False
    work = CompiledWorkspace.allocate(2)
    status, visited, _ = _weighted_dependencies(ptr, indices, mass, 0,
        work.dist, work.sigma, work.delta, work.queue)
    if status:
        raise RuntimeError("Weighted kernel warmup failed")
    _weighted_commit(np.zeros(2), work.delta, work.queue, visited, 0, 1)
    if not _weighted_dependencies.nopython_signatures or not _weighted_commit.nopython_signatures:
        raise RuntimeError("Weighted native nopython compilation is required")
    return dict(t_jit_warmup=time.perf_counter() - start, native_nopython=True,
        jit_options=dict(cache=True, nogil=True, fastmath=False, parallel=False))


def _python_weighted_dependencies(core, mass, source):
    n = core.n
    dist, sigma = [-1] * n, [0] * n
    dist[source], sigma[source] = 0, 1
    order, cursor = [source], 0
    while cursor < len(order):
        v = order[cursor]
        cursor += 1
        for raw_w in core.adj[v]:
            w = int(raw_w)
            if dist[w] == -1:
                dist[w] = dist[v] + 1
                order.append(w)
            if dist[w] == dist[v] + 1:
                sigma[w] += sigma[v]
    delta = np.zeros(n)
    for v in reversed(order):
        for raw_w in core.adj[v]:
            w = int(raw_w)
            if dist[w] == dist[v] + 1:
                delta[v] += (sigma[v] / sigma[w]) * (int(mass[w]) + delta[w])
        if not math.isfinite(delta[v]):
            raise FloatingPointError("Nonfinite arbitrary-integer weighted dependency")
    delta[source] = 0.0
    return delta


class WeightedAccumulator:
    def __init__(self, core, mass, source_counts):
        self.core, self.mass, self.source_counts = core, mass, source_counts
        self.workspace = CompiledWorkspace.allocate(core.n)

    def accumulate(self, source, total):
        work = self.workspace
        status, visited, maximum = _weighted_dependencies(self.core.indptr, self.core.indices,
            self.mass, source, work.dist, work.sigma, work.delta, work.queue)
        count = int(self.source_counts[source])
        if status == 1:
            # Nothing committed until all arbitrary-precision dependencies succeed.
            delta = _python_weighted_dependencies(self.core, self.mass, source)
            total += count * delta
            return dict(execution="python_bigint_fallback", source_multiplicity=count,
                        fallback_reason="sigma_int64_overflow")
        if status:
            raise FloatingPointError("Nonfinite weighted dependency; source not committed")
        _weighted_commit(total, work.delta, work.queue, visited, source, count)
        return dict(execution="numba_native", source_multiplicity=count,
                    reachable_core_nodes=int(visited), max_sigma=int(maximum))


@njit(cache=True, nogil=True)
def _selected_subtrees(indicator, parent, peel, anchors, labels, component_count):
    selected = indicator.astype(np.int64)
    for v in peel:
        selected[parent[v]] += selected[v]
    root_counts = selected[anchors].copy()
    component_counts = np.zeros(component_count, np.int64)
    for r in range(len(anchors)):
        component_counts[labels[r]] += root_counts[r]
    return selected, root_counts, component_counts


@njit(cache=True, nogil=True, fastmath=False)
def _restore(parent, peel, anchors, mapping, sizes, labels, component_sizes,
             indicator, selected, root_counts, component_counts, core_scores):
    result = np.zeros(len(parent), np.float64)
    # Off-anchor v: sides after removing v are its child trees and parent side.
    # Start with parent-side term; child-side terms are added in one edge pass.
    for v in peel:
        r = mapping[v]
        comp = labels[r]
        n, c = component_sizes[comp], component_counts[comp]
        result[v] = (c - selected[v]) * (sizes[v] - 1)
    # Anchor correction for cross-block paths with an endpoint in its own tree.
    for r in range(len(anchors)):
        v = anchors[r]
        comp = labels[r]
        n, c, m, cr = component_sizes[comp], component_counts[comp], sizes[v], root_counts[r]
        result[v] = (core_scores[r] + (m - 1) * (c - cr)
                     + (cr - indicator[v]) * (n - m))
    for v in peel:
        p = parent[v]
        r = mapping[v]
        comp = labels[r]
        if parent[p] == -1:
            # Both endpoints in different pendant branches at their anchor.
            result[p] += selected[v] * (sizes[p] - 1 - sizes[v])
        else:
            # Child side can communicate with every other side, excluding p.
            result[p] += selected[v] * (component_sizes[comp] - 1 - sizes[v])
    return result


def _warmup_recovery():
    start = time.perf_counter()
    parent = np.array([1, -1], np.int32)
    peel, anchors, labels = np.array([0], np.int32), np.array([1], np.int32), np.array([0], np.int32)
    mapping, sizes, component_sizes = np.array([0, 0], np.int32), np.array([1, 2], np.int64), np.array([2], np.int64)
    for a in (parent, peel, anchors, labels, mapping, sizes, component_sizes):
        a.flags.writeable = False
    indicator = np.array([1, 0], np.int8)
    selected, counts, component_counts = _selected_subtrees(indicator, parent, peel, anchors, labels, 1)
    counts.flags.writeable = False
    _restore(parent, peel, anchors, mapping, sizes, labels, component_sizes, indicator,
             selected, counts, component_counts, np.zeros(1))
    return time.perf_counter() - start


def quotient_contributions(q, sources=None, *, threads=1, batch_size=4,
                           timeout_seconds=None, progress_path=None):
    """Exact A_S on all original vertices; None means explicitly all sources.

    Endpoint multiplicities are retained for both target blocks and source
    blocks. Tree-local paths are restored analytically; no sampling/rescaling.
    """
    start = time.perf_counter()
    try:
        selected_sources = tuple(range(q.n)) if sources is None else tuple(operator.index(s) for s in sources)
    except TypeError as exc:
        raise ValueError("sources must contain integer indices") from exc
    if (len(set(selected_sources)) != len(selected_sources)
            or (selected_sources and (min(selected_sources) < 0 or max(selected_sources) >= q.n))
            or (q.n and not selected_sources)):
        raise ValueError("Invalid original source subset")
    if timeout_seconds is not None and (not np.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("timeout_seconds must be positive and finite")
    if operator.index(threads) < 1 or operator.index(batch_size) < 1:
        raise ValueError("threads and batch_size must be positive")
    recovery_jit = _warmup_recovery()
    group_start = time.perf_counter()
    indicator = np.zeros(q.n, np.int8)
    indicator[list(selected_sources)] = 1
    selected, root_counts, component_counts = _selected_subtrees(indicator, q.parent, q.peel_order,
        q.anchor_nodes, q.component_index, len(q.component_sizes))
    mass = q.mass
    root_counts.flags.writeable = False
    source_groups = tuple(dict.fromkeys(int(q.anchor_index[s]) for s in selected_sources))
    # Pure-tree components need only analytic restoration, with no core BFS.
    roots = tuple(r for r in source_groups if q.core.indptr[r+1] > q.core.indptr[r])
    grouped = time.perf_counter()
    if roots:
        remaining = None if timeout_seconds is None else timeout_seconds - (grouped - start)
        if remaining is not None and remaining <= 0:
            raise TimeoutError("Quotient setup budget exceeded")
        core_total, meta = threaded_contributions(q.core, roots, threads=threads, batch_size=batch_size,
            timeout_seconds=remaining, progress_path=progress_path,
            _runner_factory=lambda core: WeightedAccumulator(core, mass, root_counts),
            _warmup=weighted_warmup)
    else:
        core_total = np.zeros(q.core.n)
        meta = dict(tasks=[], workers=[], p_used=0, active_workers=0, fallback_sources=0,
                    t_execute_reduce=0., t_execute_wall=0., t_reduce=0., t_jit_warmup=0.)
    before_restore = time.perf_counter()
    total = _restore(q.parent, q.peel_order, q.anchor_nodes, q.anchor_index, q.subtree_size,
        q.component_index, q.component_sizes, indicator, selected, root_counts, component_counts, core_total)
    if not np.isfinite(total).all() or np.any(total < 0):
        raise FloatingPointError("Invalid restored original-node contributions")
    end = time.perf_counter()
    if timeout_seconds is not None and end - start > timeout_seconds:
        raise TimeoutError("Quotient restoration budget exceeded")
    meta.update(backend="tree_quotient", n=q.n, m=q.metadata["original_edges"], graph_full=True,
        graph_sha256=q.source_graph_sha256, core_graph_sha256=q.core.graph_sha256,
        source_order_hash=source_order_digest(selected_sources), source_count=len(selected_sources),
        exact_all_sources=len(selected_sources) == q.n, scaled=False,
        result_kind="ordered_all_source_contribution" if len(selected_sources) == q.n else "source_subset_contribution",
        core_nodes=q.core.n, core_edges=q.core.m, core_traversals=len(roots),
        source_groups=len(source_groups),
        represented_source_count=int(root_counts.sum()), root_source_order_hash=source_order_digest(roots),
        source_task_kind="quotient_anchor_with_multiplicity", restored_original_nodes=q.n,
        t_recovery_jit_warmup=recovery_jit,
        t_group_sources=grouped - group_start, t_restore=end - before_restore,
        t_execute_restore=meta["t_execute_reduce"] + grouped - group_start + end - before_restore,
        t_quotient_compute_restore=end - grouped, t_compute_driver=end - start)
    return total, meta
