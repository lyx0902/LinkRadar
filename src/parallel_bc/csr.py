"""Compact, deterministic CSR caches; legacy GraphData/NPZ stays unchanged."""
from __future__ import annotations

from array import array
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import operator
from pathlib import Path
import time

import numpy as np
import psutil

from .data import _atomic_json, _open_text, _sha256


CACHE_FORMAT = "LinkRadar-csr-v1"


def _check_budget(deadline, min_available_bytes):
    if deadline is not None and time.perf_counter() >= deadline:
        raise TimeoutError("CSR preparation exceeded time budget")
    if min_available_bytes and psutil.virtual_memory().available < min_available_bytes:
        raise MemoryError("Available system memory below reserved threshold")


@dataclass(frozen=True)
class CSRGraph:
    ids: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray
    metadata: dict

    @property
    def n(self):
        return len(self.ids)

    @property
    def m(self):
        return len(self.indices) // 2

    @property
    def graph_sha256(self):
        return self.metadata["graph_sha256"]

    @property
    def adj(self):
        return CSRAdjacency(self)


class CSRAdjacency:
    """Sequence adapter: the *existing* Python Brandes kernel sees CSR rows."""
    def __init__(self, graph):
        self.graph = graph

    def __len__(self):
        return self.graph.n

    def __getitem__(self, index):
        index = operator.index(index)
        if index < 0 or index >= self.graph.n:
            raise IndexError(index)
        return self.graph.indices[self.graph.indptr[index]:self.graph.indptr[index + 1]]


def _integers(values, name):
    values = np.asarray(values)
    if values.size == 0:
        return values.astype(np.int64)
    if values.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integers")
    if values.dtype.kind == "u" and np.any(values > np.iinfo(np.int64).max):
        raise ValueError(f"{name} exceeds int64 range")
    return values.astype(np.int64, copy=False)


def _directed_keys(graph):
    rows = np.repeat(np.arange(graph.n, dtype=np.uint64), np.diff(graph.indptr))
    return (rows << np.uint64(32)) | graph.indices.astype(np.uint64)


def validate_csr(graph, *, full=True):
    """Full checks cover sorted unique rows, self loops and undirected symmetry."""
    n = graph.n
    if (graph.ids.ndim != 1 or graph.ids.dtype != np.int64
            or graph.indptr.dtype != np.int64 or graph.indices.dtype != np.int32
            or graph.indptr.shape != (n + 1,) or graph.indices.ndim != 1):
        raise ValueError("Invalid CSR array types or shapes")
    if n > np.iinfo(np.int32).max:
        raise ValueError("CSR node count exceeds int32 index range")
    if (graph.indptr[0] != 0 or graph.indptr[-1] != len(graph.indices)
            or np.any(np.diff(graph.indptr) < 0) or len(graph.indices) % 2):
        raise ValueError("Invalid CSR offsets")
    if np.any(graph.ids[1:] <= graph.ids[:-1]):
        raise ValueError("Original IDs must be sorted and unique")
    if graph.indices.size and (graph.indices.min() < 0 or graph.indices.max() >= n):
        raise ValueError("Invalid adjacency index")
    if full:
        keys = _directed_keys(graph)
        if np.any(keys[1:] <= keys[:-1]):
            raise ValueError("Unsorted or duplicate CSR neighbor")
        left, right = keys >> np.uint64(32), keys & np.uint64(0xffffffff)
        if np.any(left == right):
            raise ValueError("CSR self loop")
        reverse = (right << np.uint64(32)) | left
        reverse.sort()
        if not np.array_equal(keys, reverse):
            raise ValueError("CSR is not symmetric")


def graph_digest(graph):
    """Same canonical encoding as GraphData, including original IDs and isolates."""
    digest = hashlib.sha256(b"LinkRadar graph v1\n")
    digest.update(f"n {graph.n}\n".encode("ascii"))
    for i, raw_id in enumerate(graph.ids):
        digest.update(f"v {i} {int(raw_id)}\n".encode("ascii"))
    keys = _directed_keys(graph)
    left, right = keys >> np.uint64(32), keys & np.uint64(0xffffffff)
    for u, v in zip(left[left < right], right[left < right]):
        digest.update(f"e {int(u)} {int(v)}\n".encode("ascii"))
    return digest.hexdigest()


def csr_from_edges(edges, nodes=None, metadata=None):
    """Integer arrays instead of tuple/set/dict copies; self-loop nodes survive."""
    edges = _integers(edges if isinstance(edges, np.ndarray) else list(edges), "edges")
    if edges.size == 0:
        edges = edges.reshape(0, 2)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("Expected two columns of integer edges")
    extra = _integers([] if nodes is None else nodes, "nodes")
    if extra.ndim != 1:
        raise ValueError("nodes must be one dimensional")
    discovered = np.unique(edges)
    ids = np.union1d(discovered, extra)
    if len(ids) > np.iinfo(np.int32).max:
        raise ValueError("Too many nodes for int32 indices")
    mapped = np.searchsorted(ids, edges)
    self_loops = int(np.count_nonzero(mapped[:, 0] == mapped[:, 1]))
    lower = np.minimum(mapped[:, 0], mapped[:, 1]).astype(np.uint64)
    upper = np.maximum(mapped[:, 0], mapped[:, 1]).astype(np.uint64)
    mask = lower != upper
    keys = np.unique((lower[mask] << np.uint64(32)) | upper[mask])
    del mapped, lower, upper, mask
    reverse = ((keys & np.uint64(0xffffffff)) << np.uint64(32)) | (keys >> np.uint64(32))
    directed = np.concatenate((keys, reverse))
    directed.sort()
    counts = np.bincount((directed >> np.uint64(32)).astype(np.int64), minlength=len(ids))
    indptr = np.empty(len(ids) + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    indices = (directed & np.uint64(0xffffffff)).astype(np.int32)
    meta = dict(metadata or {})
    meta.update(raw_data_lines=len(edges), self_loop_lines=self_loops,
                duplicate_edge_lines=len(edges) - self_loops - len(keys),
                discovered_nodes=len(discovered), explicit_nodes=len(ids) - len(discovered),
                final_nodes=len(ids), final_edges=len(keys),
                average_degree=2. * len(keys) / len(ids) if len(ids) else 0.,
                cleaning_version="simple-undirected-v1",
                node_mapping_rule="original integer ID ascending -> contiguous index 0..n-1")
    graph = CSRGraph(ids, indptr, indices, meta)
    validate_csr(graph)
    meta["graph_sha256"] = graph_digest(graph)
    for value in (ids, indptr, indices):
        value.flags.writeable = False
    return graph


def read_csr_edges(path, *, file_format="snap", timeout_seconds=300,
                   min_available_bytes=0):
    """Explicit KONECT support; strict integer parsing reports original line."""
    if file_format not in {"snap", "konect"}:
        raise ValueError("file_format must be snap or konect")
    if timeout_seconds <= 0 or min_available_bytes < 0:
        raise ValueError("Invalid resource budget")
    deadline = time.perf_counter() + timeout_seconds
    path = Path(path)
    packed = array("q")
    header = []
    declared_edges = declared_nodes = None
    declared_kind = False
    with _open_text(path) as stream:
        for lineno, line in enumerate(stream, 1):
            if lineno % 8192 == 0:
                _check_budget(deadline, min_available_bytes)
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if file_format == "konect" and line.startswith("%"):
                fields = line[1:].split()
                if fields == ["sym", "unweighted"] and not packed and not declared_kind:
                    declared_kind = True
                elif declared_kind and declared_edges is None and not packed and len(fields) == 3:
                    try:
                        count, n1, n2 = map(int, fields)
                    except ValueError as exc:
                        raise ValueError(f"{path}:{lineno}: invalid KONECT sizes") from exc
                    if count < 0 or n1 < 0 or n1 != n2 or n1 > np.iinfo(np.int32).max:
                        raise ValueError(f"{path}:{lineno}: invalid KONECT sizes")
                    declared_edges, declared_nodes = count, n1
                else:
                    raise ValueError(f"{path}:{lineno}: unsupported KONECT header")
                header.append(line)
                continue
            if file_format == "konect" and declared_nodes is None:
                raise ValueError(f"{path}:{lineno}: missing KONECT sym unweighted/sizes header")
            fields = line.split()
            try:
                if len(fields) != 2:
                    raise ValueError("expected two integer node IDs")
                u, v = map(int, fields)
                if file_format == "konect" and not (1 <= u <= declared_nodes and 1 <= v <= declared_nodes):
                    raise ValueError("node ID outside declared 1..n range")
                packed.extend((u, v))
            except (ValueError, OverflowError) as exc:
                raise ValueError(f"{path}:{lineno}: invalid edge row: {line!r} ({exc})") from exc
    if file_format == "konect" and (declared_nodes is None or len(packed) // 2 != declared_edges):
        raise ValueError("KONECT edge count/header mismatch")
    _check_budget(deadline, min_available_bytes)
    edges = np.frombuffer(packed, dtype=np.int64).reshape(-1, 2)
    nodes = np.arange(1, declared_nodes + 1, dtype=np.int64) if declared_nodes is not None else None
    return edges, nodes, dict(input_format=file_format, input_header=header,
                             declared_nodes=declared_nodes, declared_edge_rows=declared_edges)


def component_statistics(graph, *, deadline=None, min_available_bytes=0):
    """One O(n+m) traversal, no NetworkX graph or Python node set."""
    seen = bytearray(graph.n)
    queue = np.empty(graph.n, dtype=np.int32)
    components = largest = 0
    for start in range(graph.n):
        if seen[start]:
            continue
        components += 1
        seen[start] = 1
        queue[0] = start
        head, tail = 0, 1
        while head < tail:
            if head % 8192 == 0:
                _check_budget(deadline, min_available_bytes)
            u = int(queue[head])
            head += 1
            for value in graph.indices[graph.indptr[u]:graph.indptr[u + 1]]:
                v = int(value)
                if not seen[v]:
                    seen[v] = 1
                    queue[tail] = v
                    tail += 1
        largest = max(largest, tail)
    return dict(connected_components=components, largest_component_nodes=largest,
                isolated_nodes=int(np.count_nonzero(np.diff(graph.indptr) == 0)))


def save_csr(graph, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)  # Never overwrite an existing cache.
    files = {}
    for name, value in (("original_ids", graph.ids), ("indptr", graph.indptr), ("indices", graph.indices)):
        target = directory / f"{name}.npy"
        temporary = directory / f"{name}.npy.tmp"
        with temporary.open("wb") as stream:
            np.save(stream, value, allow_pickle=False)
        temporary.replace(target)
        files[target.name] = _sha256(target)
    meta = dict(graph.metadata, cache_format=CACHE_FORMAT, cache_files=files,
                adjacency_array_bytes=graph.indptr.nbytes + graph.indices.nbytes,
                mapping_array_bytes=graph.ids.nbytes)
    _atomic_json(directory / "metadata.json", meta)
    return directory


def load_csr(directory, *, mmap_mode="r"):
    directory = Path(directory)
    meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if meta.get("cache_format") != CACHE_FORMAT:
        raise ValueError("Unsupported CSR cache format")
    names = ("original_ids.npy", "indptr.npy", "indices.npy")
    if set(meta.get("cache_files", {})) != set(names):
        raise ValueError("Invalid CSR cache file manifest")
    for name in names:
        if _sha256(directory / name) != meta["cache_files"][name]:
            raise ValueError(f"CSR cache SHA256 mismatch: {name}")
    arrays = [np.load(directory / name, mmap_mode=mmap_mode, allow_pickle=False) for name in names]
    graph = CSRGraph(*arrays, meta)
    validate_csr(graph, full=True)
    if graph.n != meta.get("final_nodes") or graph.m != meta.get("final_edges"):
        raise ValueError("CSR metadata dimensions mismatch")
    # Recomputing the canonical fingerprint also detects metadata graph-ID tampering.
    if graph_digest(graph) != meta.get("graph_sha256"):
        raise ValueError("CSR graph fingerprint mismatch")
    return graph


def prepare_csr(path, directory, *, dataset_id, file_format="snap", provenance_path=None,
                timeout_seconds=300, min_available_bytes=0):
    start = time.perf_counter()
    deadline = start + timeout_seconds
    _check_budget(deadline, min_available_bytes)
    path = Path(path)
    raw_hash = _sha256(path)
    provenance = {}
    if provenance_path:
        provenance = json.loads(Path(provenance_path).read_text(encoding="utf-8"))
        if provenance.get("raw_sha256") != raw_hash:
            raise ValueError("Provenance raw_sha256 mismatch")
    meta = dict(dataset_id=dataset_id, source="manual_import", source_path=str(path.resolve()),
                raw_sha256=raw_hash, raw_bytes=path.stat().st_size, source_provenance=provenance,
                prepared_at_utc=datetime.now(timezone.utc).isoformat())
    edges, nodes, parsed = read_csr_edges(path, file_format=file_format,
        timeout_seconds=max(.001, deadline - time.perf_counter()), min_available_bytes=min_available_bytes)
    parsed_end = time.perf_counter()
    graph = csr_from_edges(edges, nodes=nodes, metadata=dict(meta, **parsed))
    del edges, nodes
    built_end = time.perf_counter()
    _check_budget(deadline, min_available_bytes)
    graph.metadata.update(component_statistics(graph, deadline=deadline,
                                              min_available_bytes=min_available_bytes))
    components_end = time.perf_counter()
    if _sha256(path) != raw_hash:
        raise ValueError("Source file changed during preparation")
    graph.metadata["preparation_timings"] = dict(
        t_parse=parsed_end - start, t_build_validate_fingerprint=built_end - parsed_end,
        t_components=components_end - built_end,
        t_before_save=time.perf_counter() - start)
    _check_budget(deadline, min_available_bytes)
    return save_csr(graph, directory)
