"""Independent physical CSR layouts for the frozen weighted quotient kernel.

Internal layout IDs are 0..n-1. Original graph IDs, anchor indices and tree
recovery arrays remain in the immutable quotient. Both permutation directions
and endpoint masses are saved/validated; output is unpermuted before recovery.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import operator
from pathlib import Path
import time

import numpy as np
from numba import njit

from .brandes import source_order_digest
from .csr import CSRGraph, graph_digest, load_csr, save_csr, validate_csr
from .data import _atomic_json, _sha256
from .large_benchmark import _save_array
from .quotient_brandes import (_restore, _selected_subtrees, _warmup_recovery,
                               WeightedAccumulator, weighted_warmup)
from .compiled_threads import threaded_contributions


@njit(cache=True, nogil=True)
def _bfs_order(ptr, indices, start):
    n = len(ptr) - 1
    order = np.empty(n, np.int32)
    seen = np.zeros(n, np.bool_)
    head = tail = 0
    # Seed component first, then every remaining component and isolate.
    for turn in range(n + 1):
        root = start if turn == 0 else turn - 1
        if root >= n or seen[root]:
            continue
        seen[root] = True
        order[tail] = root
        tail += 1
        while head < tail:
            v = order[head]
            head += 1
            for edge in range(ptr[v], ptr[v + 1]):
                w = indices[edge]
                if not seen[w]:
                    seen[w] = True
                    order[tail] = w
                    tail += 1
    return order


@njit(cache=True, nogil=True)
def _permute(ptr, indices, forward, inverse):
    n = len(forward)
    new_ptr = np.empty(n + 1, np.int64)
    new_ptr[0] = 0
    for v in range(n):
        old = forward[v]
        new_ptr[v + 1] = new_ptr[v] + ptr[old + 1] - ptr[old]
    new_indices = np.empty(len(indices), np.int32)
    for v in range(n):
        old = forward[v]
        cursor = new_ptr[v]
        for edge in range(ptr[old], ptr[old + 1]):
            new_indices[cursor] = inverse[indices[edge]]
            cursor += 1
        new_indices[new_ptr[v]:cursor].sort()
    return new_ptr, new_indices


@dataclass(frozen=True)
class CoreLayout:
    graph: CSRGraph
    new_to_old: np.ndarray
    old_to_new: np.ndarray
    mass: np.ndarray
    metadata: dict


def validate_layout(q, layout):
    n = q.core.n
    fwd, inv = layout.new_to_old, layout.old_to_new
    if (fwd.dtype != np.int32 or inv.dtype != np.int32 or fwd.shape != (n,) or inv.shape != (n,)
            or not np.array_equal(np.sort(fwd), np.arange(n))
            or not np.array_equal(inv[fwd], np.arange(n))):
        raise ValueError("Invalid layout permutation")
    if layout.graph.n != n or layout.graph.m != q.core.m:
        raise ValueError("Layout graph dimensions mismatch")
    validate_csr(layout.graph)
    if (layout.mass.dtype != np.int64 or layout.mass.shape != (n,)
            or not np.array_equal(layout.mass, q.mass[fwd])):
        raise ValueError("Layout endpoint masses mismatch")
    if not np.array_equal(np.diff(layout.graph.indptr), np.diff(q.core.indptr)[fwd]):
        raise ValueError("Layout degrees mismatch")
    # Compare all directed edges after restoring OLD indices, not just counts.
    rows = np.repeat(np.arange(n, dtype=np.int32), np.diff(layout.graph.indptr))
    actual = ((fwd[rows].astype(np.uint64) << np.uint64(32))
              | fwd[layout.graph.indices].astype(np.uint64))
    actual.sort()
    old_rows = np.repeat(np.arange(n, dtype=np.int32), np.diff(q.core.indptr))
    expected = (old_rows.astype(np.uint64) << np.uint64(32)) | q.core.indices.astype(np.uint64)
    if not np.array_equal(actual, expected):
        raise ValueError("Layout original edge identity mismatch")


def build_layout(q, kind="bfs", *, seed=1234):
    if kind not in {"natural", "bfs"} or operator.index(seed) < 0:
        raise ValueError("Invalid layout kind or seed")
    started = time.perf_counter()
    # Compile/load tiny readonly signatures; separate from actual layout work.
    ptr, idx = np.array([0, 1, 2], np.int64), np.array([1, 0], np.int32)
    for a in (ptr, idx):
        a.flags.writeable = False
    warm_order = _bfs_order(ptr, idx, 0)
    _permute(ptr, idx, warm_order, warm_order)
    warmed = time.perf_counter()
    n = q.core.n
    start = int(np.random.default_rng(seed).integers(n)) if n else 0
    fwd = (_bfs_order(q.core.indptr, q.core.indices, start) if kind == "bfs"
           else np.arange(n, dtype=np.int32))
    inv = np.empty(n, np.int32)
    inv[fwd] = np.arange(n, dtype=np.int32)
    if kind == "bfs":
        new_ptr, new_idx = _permute(q.core.indptr, q.core.indices, fwd, inv)
    else:
        new_ptr, new_idx = q.core.indptr, q.core.indices
    graph = CSRGraph(np.arange(n, dtype=np.int64), new_ptr, new_idx,
        dict(final_nodes=n, final_edges=q.core.m, internal_layout_ids=True))
    for a in (graph.ids, graph.indptr, graph.indices):
        a.flags.writeable = False
    graph.metadata["graph_sha256"] = graph_digest(graph)
    mass = q.mass[fwd]
    for a in (fwd, inv, mass):
        a.flags.writeable = False
    meta = dict(format="LinkRadar-core-layout-v1", kind=kind, seed=int(seed), bfs_start_old_index=start,
        source_graph_sha256=q.source_graph_sha256, source_core_graph_sha256=q.core.graph_sha256,
        physical_graph_sha256=graph.graph_sha256, permutation_hash=source_order_digest(fwd),
        t_layout_jit=warmed - started)
    result = CoreLayout(graph, fwd, inv, mass, meta)
    validate_layout(q, result)
    meta["t_build_validate"] = time.perf_counter() - warmed
    return result


def save_layout(layout, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    save_csr(layout.graph, path / "core")
    files = {}
    for name in ("new_to_old", "old_to_new", "mass"):
        target = path / (name + ".npy")
        _save_array(target, getattr(layout, name))
        files[target.name] = _sha256(target)
    _atomic_json(path / "metadata.json", dict(layout.metadata, files=files))


def load_layout(q, path):
    path = Path(path)
    meta = json.loads((path / "metadata.json").read_text())
    if (meta.get("format") != "LinkRadar-core-layout-v1"
            or meta.get("source_graph_sha256") != q.source_graph_sha256
            or meta.get("source_core_graph_sha256") != q.core.graph_sha256
            or set(meta.get("files", {})) != {"new_to_old.npy", "old_to_new.npy", "mass.npy"}):
        raise ValueError("Layout provenance mismatch")
    for name, sha in meta["files"].items():
        if _sha256(path / name) != sha:
            raise ValueError("Layout file fingerprint mismatch")
    graph = load_csr(path / "core")
    result = CoreLayout(graph, *(np.load(path / (name + ".npy"), mmap_mode="r", allow_pickle=False)
        for name in ("new_to_old", "old_to_new", "mass")), meta)
    if graph.graph_sha256 != meta.get("physical_graph_sha256"):
        raise ValueError("Layout physical graph mismatch")
    if meta.get("kind") not in {"natural", "bfs"} or source_order_digest(result.new_to_old) != meta.get("permutation_hash"):
        raise ValueError("Layout permutation metadata mismatch")
    validate_layout(q, result)
    return result


def layout_contributions(q, layout, roots, *, threads=4, batch_size=4, timeout_seconds=None):
    """Selected core groups represent ALL their original source vertices."""
    started = time.perf_counter()
    roots = tuple(operator.index(r) for r in roots)
    if operator.index(threads) < 1 or operator.index(batch_size) < 1:
        raise ValueError("Positive thread and batch counts required")
    if (len(set(roots)) != len(roots) or (roots and (min(roots) < 0 or max(roots) >= q.core.n))):
        raise ValueError("Invalid original core source groups")
    if not roots and q.n:
        raise ValueError("Nonempty source groups required")
    if timeout_seconds is not None and (not np.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("Positive finite timeout required")
    recovery_jit = _warmup_recovery()
    grouping = time.perf_counter()
    indicator = np.isin(q.anchor_index, roots).astype(np.int8)
    selected, root_counts, component_counts = _selected_subtrees(indicator, q.parent,
        q.peel_order, q.anchor_nodes, q.component_index, len(q.component_sizes))
    root_counts.flags.writeable = False
    counts = np.ascontiguousarray(root_counts[layout.new_to_old])
    counts.flags.writeable = False
    traversal_roots = [r for r in roots if q.core.indptr[r+1] > q.core.indptr[r]]
    new_roots = [int(layout.old_to_new[r]) for r in traversal_roots]
    grouped = time.perf_counter()
    if new_roots:
        remaining = None if timeout_seconds is None else timeout_seconds - (grouped - started)
        if remaining is not None and remaining <= 0:
            raise TimeoutError("Layout setup exceeded budget")
        new_total, meta = threaded_contributions(layout.graph, new_roots, threads=threads,
            batch_size=batch_size, timeout_seconds=remaining,
            _runner_factory=lambda graph: WeightedAccumulator(graph, layout.mass, counts),
            _warmup=weighted_warmup)
    else:
        new_total = np.zeros(q.core.n)
        meta = dict(tasks=[], workers=[], p_used=0, active_workers=0, fallback_sources=0,
                    t_execute_reduce=0., t_jit_warmup=0.)
    before_restore = time.perf_counter()
    core_total = new_total[layout.old_to_new]
    total = _restore(q.parent, q.peel_order, q.anchor_nodes, q.anchor_index, q.subtree_size,
        q.component_index, q.component_sizes, indicator, selected, root_counts, component_counts, core_total)
    if not np.isfinite(total).all() or np.any(total < 0):
        raise FloatingPointError("Invalid layout-restored original contributions")
    ended = time.perf_counter()
    if timeout_seconds is not None and ended - started > timeout_seconds:
        raise TimeoutError("Layout recovery exceeded budget")
    for row in meta["tasks"]:
        new = row["source_index"]
        old = int(layout.new_to_old[new])
        row.update(layout_source_index=new, source_index=old, original_id=int(q.core.ids[old]))
    if [r["source_index"] for r in meta["tasks"]] != traversal_roots:
        raise RuntimeError("Layout source mapping coverage mismatch")
    originals = np.flatnonzero(indicator)
    meta.update(backend="layout_weighted_quotient", layout=layout.metadata["kind"],
        graph_sha256=q.source_graph_sha256, n=q.n, m=q.metadata["original_edges"],
        source_core_graph_sha256=q.core.graph_sha256, physical_graph_sha256=layout.graph.graph_sha256,
        root_source_order_hash=source_order_digest(roots), source_order_hash=source_order_digest(originals),
        source_count=len(originals), represented_source_count=int(root_counts.sum()),
        core_traversals=len(traversal_roots), restored_original_nodes=q.n, graph_full=True, scaled=False,
        exact_all_sources=len(originals) == q.n, result_kind="ordered_all_source_contribution"
            if len(originals) == q.n else "source_subset_contribution",
        t_recovery_jit=recovery_jit, t_group=grouped - grouping, t_unpermute_restore=ended - before_restore,
        t_work=grouped - grouping + meta["t_execute_reduce"] + ended - before_restore,
        t_layout_compute_driver=ended - started)
    return total, core_total, meta


def layout_job(cache_directory, output, *, kind, layout_directory, seed=1234, roots=None,
               threads=4, batch_size=4, cpu_affinity=None, timeout_seconds=None):
    import psutil
    from .tree_quotient import load_quotient
    started = time.perf_counter()
    if cpu_affinity is not None:
        psutil.Process().cpu_affinity(cpu_affinity)
    q = load_quotient(cache_directory)
    if kind == "prepare":
        layout = build_layout(q, layout_directory[0], seed=seed)
        save_layout(layout, layout_directory[1])
        return dict(layout.metadata, t_prepare_total=time.perf_counter() - started)
    layout = load_layout(q, layout_directory)
    loaded = time.perf_counter()
    total, core_total, meta = layout_contributions(q, layout, roots, threads=threads,
        batch_size=batch_size, timeout_seconds=timeout_seconds)
    output = Path(output)
    with (output / "tasks.jsonl").open("w", encoding="utf-8") as stream:
        for row in meta.pop("tasks"):
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    _save_array(output / "contributions.npy", total)
    _save_array(output / "core_contributions_old_indices.npy", core_total)
    meta.update(t_load_validate=loaded - started, t_child_total=time.perf_counter() - started,
        cpu_affinity_used=psutil.Process().cpu_affinity(),
        contribution_path=str((output / "contributions.npy").resolve()),
        core_contribution_path=str((output / "core_contributions_old_indices.npy").resolve()))
    return meta
