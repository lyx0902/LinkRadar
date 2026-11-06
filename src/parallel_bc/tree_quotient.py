"""Lossless pendant-tree factorization: core CSR + every original tree edge.

Endpoint masses belong to vertices, never to shortest-path edge lengths.
Tree components retain one anchor so isolates/forests are also representable.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from numba import njit

from .csr import CSRGraph, graph_digest, load_csr, save_csr, validate_csr
from .data import _atomic_json, _sha256


FORMAT = "LinkRadar-tree-quotient-v1"
ARRAY_NAMES = ("original_ids", "parent", "peel_order", "anchor_index", "anchor_nodes",
               "subtree_size", "component_index", "component_sizes")


@njit(cache=True, nogil=True)
def _factor(indptr, indices):
    n = len(indptr) - 1
    degree = np.diff(indptr)
    parent = np.full(n, -1, np.int32)
    size = np.ones(n, np.int64)
    active = np.ones(n, np.bool_)
    queue = np.empty(n, np.int32)
    peel = np.empty(n, np.int32)
    head, tail, removed = 0, 0, 0
    for v in range(n):
        if degree[v] == 1:
            queue[tail] = v
            tail += 1
    while head < tail:
        v = queue[head]
        head += 1
        # A zero-degree last vertex of a tree stays as its component anchor.
        if not active[v] or degree[v] != 1:
            continue
        for edge in range(indptr[v], indptr[v + 1]):
            u = indices[edge]
            if active[u]:
                parent[v] = u
                size[u] += size[v]
                degree[u] -= 1
                active[v] = False
                peel[removed] = v
                removed += 1
                if degree[u] == 1:
                    queue[tail] = u
                    tail += 1
                break
    anchors = np.flatnonzero(active).astype(np.int32)
    mapping = np.full(n, -1, np.int32)
    for r in range(len(anchors)):
        mapping[anchors[r]] = r
    for pos in range(removed - 1, -1, -1):
        v = peel[pos]
        mapping[v] = mapping[parent[v]]
    core_ptr = np.empty(len(anchors) + 1, np.int64)
    core_ptr[0] = 0
    for r in range(len(anchors)):
        core_ptr[r + 1] = core_ptr[r] + degree[anchors[r]]
    core_indices = np.empty(core_ptr[-1], np.int32)
    for r in range(len(anchors)):
        v = anchors[r]
        cursor = core_ptr[r]
        for edge in range(indptr[v], indptr[v + 1]):
            u = indices[edge]
            if active[u]:
                core_indices[cursor] = mapping[u]
                cursor += 1
    return parent, peel[:removed].copy(), mapping, anchors, size, core_ptr, core_indices


@njit(cache=True, nogil=True)
def _components(indptr, indices, mass):
    n = len(mass)
    labels = np.full(n, -1, np.int32)
    queue = np.empty(n, np.int32)
    sizes = np.zeros(n, np.int64)
    count = 0
    for root in range(n):
        if labels[root] >= 0:
            continue
        head, tail = 0, 1
        queue[0] = root
        labels[root] = count
        while head < tail:
            v = queue[head]
            head += 1
            sizes[count] += mass[v]
            for edge in range(indptr[v], indptr[v + 1]):
                w = indices[edge]
                if labels[w] == -1:
                    labels[w] = count
                    queue[tail] = w
                    tail += 1
        count += 1
    return labels, sizes[:count].copy()


@njit(cache=True, nogil=True)
def _forest_check(parent, peel, anchors):
    mapping = np.full(len(parent), -1, np.int32)
    size = np.ones(len(parent), np.int64)
    for r in range(len(anchors)):
        mapping[anchors[r]] = r
    for pos in range(len(peel) - 1, -1, -1):
        v = peel[pos]
        if parent[v] < 0 or mapping[parent[v]] < 0:
            return False, mapping, size
        mapping[v] = mapping[parent[v]]
    for v in peel:
        size[parent[v]] += size[v]
    return True, mapping, size


@dataclass(frozen=True)
class TreeQuotient:
    core: CSRGraph
    original_ids: np.ndarray
    parent: np.ndarray
    peel_order: np.ndarray
    anchor_index: np.ndarray
    anchor_nodes: np.ndarray
    subtree_size: np.ndarray
    component_index: np.ndarray
    component_sizes: np.ndarray
    metadata: dict

    @property
    def n(self):
        return len(self.original_ids)

    @property
    def mass(self):
        values = np.ascontiguousarray(self.subtree_size[self.anchor_nodes])
        values.flags.writeable = False
        return values

    @property
    def source_graph_sha256(self):
        return self.metadata["source_graph_sha256"]


def reconstructed_keys(q):
    """Every original undirected edge, without building a Python graph object."""
    parent = q.parent[q.peel_order]
    a, b = np.minimum(q.peel_order, parent), np.maximum(q.peel_order, parent)
    tree = (a.astype(np.uint64) << np.uint64(32)) | b.astype(np.uint64)
    rows = np.repeat(np.arange(q.core.n, dtype=np.int32), np.diff(q.core.indptr))
    mask = rows < q.core.indices
    left, right = q.anchor_nodes[rows[mask]], q.anchor_nodes[q.core.indices[mask]]
    core = (left.astype(np.uint64) << np.uint64(32)) | right.astype(np.uint64)
    keys = np.concatenate((tree, core))
    keys.sort()
    return keys


def reconstructed_digest(q):
    digest = hashlib.sha256(b"LinkRadar graph v1\n")
    digest.update(f"n {q.n}\n".encode("ascii"))
    for i, original_id in enumerate(q.original_ids):
        digest.update(f"v {i} {int(original_id)}\n".encode("ascii"))
    for key in reconstructed_keys(q):
        key = int(key)
        digest.update(f"e {key >> 32} {key & 0xffffffff}\n".encode("ascii"))
    return digest.hexdigest()


def validate_quotient(q, *, verify_fingerprint=True):
    validate_csr(q.core, full=False)
    n, k = q.n, q.core.n
    expected = {"original_ids": (np.int64, n), "parent": (np.int32, n),
        "peel_order": (np.int32, n-k), "anchor_index": (np.int32, n),
        "anchor_nodes": (np.int32, k), "subtree_size": (np.int64, n),
        "component_index": (np.int32, k), "component_sizes": (np.int64, len(q.component_sizes))}
    for name, (dtype, length) in expected.items():
        values = getattr(q, name)
        if values.dtype != dtype or values.shape != (length,):
            raise ValueError(f"Invalid quotient array: {name}")
    if (np.any(q.parent < -1) or np.any(q.parent >= n)
            or np.any(q.original_ids[1:] <= q.original_ids[:-1])
            or not np.array_equal(q.anchor_nodes, np.flatnonzero(q.parent == -1))
            or not np.array_equal(np.sort(q.peel_order), np.flatnonzero(q.parent >= 0))):
        raise ValueError("Invalid quotient forest coverage or parents")
    ok, mapping, size = _forest_check(q.parent, q.peel_order, q.anchor_nodes)
    if (not ok or not np.array_equal(mapping, q.anchor_index)
            or not np.array_equal(size, q.subtree_size)
            or not np.array_equal(q.original_ids[q.anchor_nodes], q.core.ids)):
        raise ValueError("Invalid quotient forest membership or subtree mass")
    labels, sizes = _components(q.core.indptr, q.core.indices, q.mass)
    if not np.array_equal(labels, q.component_index) or not np.array_equal(sizes, q.component_sizes):
        raise ValueError("Invalid quotient component masses")
    if int(q.mass.sum()) != n or q.core.m + len(q.peel_order) != q.metadata["original_edges"]:
        raise ValueError("Quotient does not preserve node/edge counts")
    if verify_fingerprint and reconstructed_digest(q) != q.source_graph_sha256:
        raise ValueError("Reconstructed original graph fingerprint mismatch")


def factor_trees(graph):
    start = time.perf_counter()
    validate_csr(graph)
    parent, peel, mapping, anchors, size, ptr, indices = _factor(graph.indptr, graph.indices)
    core = CSRGraph(graph.ids[anchors].copy(), ptr, indices, {})
    validate_csr(core)
    core.metadata.update(graph_sha256=graph_digest(core), final_nodes=core.n, final_edges=core.m)
    labels, component_sizes = _components(ptr, indices, size[anchors])
    meta = dict(cache_format=FORMAT, source_graph_sha256=graph.graph_sha256,
        original_nodes=graph.n, original_edges=graph.m, core_nodes=core.n, core_edges=core.m,
        represented_tree_nodes=len(peel), connected_components=len(component_sizes),
        representation="core CSR + all original forest parent edges + endpoint vertex masses")
    q = TreeQuotient(core, graph.ids, parent, peel, mapping, anchors, size, labels, component_sizes, meta)
    validate_quotient(q)
    meta["t_factor_validate"] = time.perf_counter() - start
    for name in ARRAY_NAMES:
        getattr(q, name).flags.writeable = False
    for array in (core.ids, core.indptr, core.indices):
        array.flags.writeable = False
    return q


def save_quotient(q, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    save_csr(q.core, directory / "core")
    files = {}
    for name in ARRAY_NAMES:
        path = directory / (name + ".npy")
        with path.open("wb") as stream:
            np.save(stream, getattr(q, name), allow_pickle=False)
        files[path.name] = _sha256(path)
    _atomic_json(directory / "metadata.json", dict(q.metadata, array_files=files,
        core_graph_sha256=q.core.graph_sha256))
    return directory


def load_quotient(directory):
    directory = Path(directory)
    meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    names = {name + ".npy" for name in ARRAY_NAMES}
    if meta.get("cache_format") != FORMAT or set(meta.get("array_files", {})) != names:
        raise ValueError("Invalid quotient cache manifest")
    for name in names:
        if _sha256(directory / name) != meta["array_files"][name]:
            raise ValueError(f"Quotient cache SHA256 mismatch: {name}")
    core = load_csr(directory / "core")
    if core.graph_sha256 != meta.get("core_graph_sha256"):
        raise ValueError("Quotient core fingerprint mismatch")
    arrays = [np.load(directory / (name + ".npy"), mmap_mode="r", allow_pickle=False) for name in ARRAY_NAMES]
    q = TreeQuotient(core, *arrays, meta)
    validate_quotient(q)
    if q.n != meta.get("original_nodes") or core.n != meta.get("core_nodes") or core.m != meta.get("core_edges"):
        raise ValueError("Quotient metadata dimensions mismatch")
    return q
