"""Exact all-source Brandes centrality for simple undirected unweighted graphs.

All runners use ``single_source_accumulate`` and apply undirected scaling once.
Shortest-path counts use Python integers; dependencies and scores use float64.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import operator
import os
import time
from typing import Iterable, Sequence

import numpy as np

Adjacency = Sequence[Sequence[int]]


@dataclass
class CentralityResult:
    raw: np.ndarray
    normalized: np.ndarray
    timings: dict[str, float]
    workers: list[dict] = field(default_factory=list)
    tasks: list[dict] = field(default_factory=list)
    source_order_hash: str = ""
    p_requested: int = 1
    p_used: int = 1
    active_workers: int = 1
    mode: str = "serial"
    batch_size: int = 32
    start_method: str = "none"
    task_logging: bool = False
    graph_sha256: str | None = None
    exact_all_sources: bool = True


def validate_source_order(n: int, source_order: Iterable[int] | None) -> tuple[int, ...]:
    """Require a permutation, never silently switch to sampled centrality."""
    try:
        order = tuple(range(n)) if source_order is None else tuple(operator.index(x) for x in source_order)
    except TypeError as exc:
        raise ValueError("source_order must contain integer indices") from exc
    if len(order) != n or set(order) != set(range(n)):
        raise ValueError("source_order must be an exact permutation of every node index")
    return order


def source_order_digest(order: Sequence[int]) -> str:
    return hashlib.sha256(",".join(map(str, order)).encode("ascii")).hexdigest()


def make_batches(order: Sequence[int], batch_size: int) -> list[tuple[int, tuple[int, ...]]]:
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    return [(i // batch_size, tuple(order[i:i + batch_size])) for i in range(0, len(order), batch_size)]


def timeout_deadline(start: float, timeout_seconds: float | None) -> float | None:
    if timeout_seconds is None:
        return None
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive, or None")
    return start + timeout_seconds


def check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.perf_counter() >= deadline:
        raise TimeoutError("Brandes computation exceeded timeout_seconds")


def single_source_accumulate(adj: Adjacency, source: int, accumulator: np.ndarray) -> None:
    """Add *unscaled ordered* dependencies for one source into accumulator.

    The source never contributes its own dependency (endpoints=False). Input
    adjacency must represent a clean simple undirected graph with indices 0..n-1.
    Python integer sigma prevents fixed-width shortest-path-count overflow.
    """
    n = len(adj)
    if not 0 <= source < n or len(accumulator) != n:
        raise ValueError("source or accumulator size does not match adjacency")
    pred: list[list[int]] = [[] for _ in range(n)]
    dist = [-1] * n
    sigma = [0] * n
    sigma[source] = 1
    dist[source] = 0
    # A list plus cursor is a FIFO and doubles as the reverse dependency stack.
    order = [source]
    cursor = 0
    while cursor < len(order):
        v = order[cursor]
        cursor += 1
        next_distance = dist[v] + 1
        for w in adj[v]:
            if w < 0 or w >= n:
                raise ValueError(f"invalid adjacency index {w} at node {v}")
            if dist[w] == -1:
                dist[w] = next_distance
                order.append(w)
            if dist[w] == next_distance:
                sigma[w] += sigma[v]
                pred[w].append(v)
    delta = [0.0] * n
    for w in reversed(order):
        for v in pred[w]:
            try:
                contribution = (sigma[v] / sigma[w]) * (1.0 + delta[w])
            except OverflowError as exc:
                raise FloatingPointError("shortest-path dependency conversion overflowed") from exc
            if not math.isfinite(contribution):
                raise FloatingPointError("non-finite shortest-path dependency")
            delta[v] += contribution
        if w != source:
            accumulator[w] += delta[w]


def finalize_scores(total: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Apply the sole undirected /2 and whole-graph normalization."""
    total = np.asarray(total, dtype=np.float64)
    if total.shape != (n,):
        raise ValueError("total must be a vector of length n")
    if not np.isfinite(total).all():
        raise FloatingPointError("non-finite ordered centrality total")
    if n <= 2:
        return np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.float64)
    raw = total / 2.0
    normalized = raw / ((n - 1) * (n - 2) / 2.0)
    if not np.isfinite(raw).all() or not np.isfinite(normalized).all():
        raise FloatingPointError("non-finite finalized centrality")
    return raw, normalized


def brandes_serial(
    adj: Adjacency, source_order: Iterable[int] | None = None,
    batch_size: int = 32, task_logging: bool = False,
    timeout_seconds: float | None = None,
) -> CentralityResult:
    """Exact serial baseline; cooperative deadline checks at source boundaries."""
    run_start = time.perf_counter()
    deadline = timeout_deadline(run_start, timeout_seconds)
    n = len(adj)
    order = validate_source_order(n, source_order)
    batches = make_batches(order, batch_size)
    total = np.zeros(n, dtype=np.float64)
    tasks: list[dict] = []
    pid = os.getpid()
    busy = 0.0
    first_start = None
    last_end = None
    execute_start = time.perf_counter()
    for batch_id, sources in batches:
        start = time.perf_counter()
        if first_start is None:
            first_start = start - run_start
        for source in sources:
            check_deadline(deadline)
            single_source_accumulate(adj, source, total)
        end = time.perf_counter()
        check_deadline(deadline)
        busy += end - start
        last_end = end - run_start
        if task_logging:
            tasks.append({"batch_id": batch_id, "worker_id": 0, "pid": pid,
                          "source_count": len(sources), "source_order_hash": source_order_digest(sources),
                          "start_rel_seconds": start - run_start, "end_rel_seconds": last_end,
                          "compute_seconds": end - start, "status": "ok"})
    execute_end = time.perf_counter()
    raw, normalized = finalize_scores(total, n)
    finalize_end = time.perf_counter()
    timings = {"t_startup": 0.0, "t_execute_reduce": execute_end - execute_start,
               "t_finalize": finalize_end - execute_end, "t_cleanup": 0.0,
               "t_total": finalize_end - run_start,
               "t_steady": finalize_end - execute_start}
    workers = [] if n == 0 else [{"worker_id": 0, "pid": pid, "processed_sources": n,
        "processed_batches": len(batches), "busy_seconds": busy,
        "first_task_start": first_start, "last_task_end": last_end}]
    return CentralityResult(raw=raw, normalized=normalized, timings=timings,
        workers=workers, tasks=tasks, source_order_hash=source_order_digest(order),
        p_used=int(n > 0), active_workers=int(n > 0), batch_size=batch_size,
        task_logging=task_logging)
