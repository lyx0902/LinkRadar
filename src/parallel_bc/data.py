"""Deterministic simple undirected graphs, provenance, and SNAP acquisition.

No pickle is read or written. Original IDs (including isolated nodes) are part
of every graph fingerprint; the cached adjacency is reconstructed from edges.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cached_property
import csv
import gzip
import hashlib
import http.client
import json
from pathlib import Path
import tempfile
import time
from typing import Iterable
import urllib.error
import urllib.request
import zlib

import numpy as np


CLEANING_VERSION = "simple-undirected-v1"
DATASETS = {
    "facebook": {
        "url": "https://snap.stanford.edu/data/facebook_combined.txt.gz",
        "page": "https://snap.stanford.edu/data/ego-Facebook.html",
        "description": "SNAP sampled ego-network union; not the full Facebook platform",
        "official_nodes": 4039,
        "official_edges": 88234,
        "citation": "J. McAuley and J. Leskovec. Learning to Discover Social Circles in Ego Networks. NIPS 2012.",
    },
    "ca_grqc": {
        "url": "https://snap.stanford.edu/data/ca-GrQc.txt.gz",
        "page": "https://snap.stanford.edu/data/ca-GrQc.html",
        "description": "SNAP General Relativity and Quantum Cosmology author collaboration network",
        "official_nodes": 5242,
        "official_edges": 14496,
        "citation": "J. Leskovec, J. Kleinberg and C. Faloutsos. Graphs over Time: Densification Laws, Shrinking Diameters and Possible Explanations. KDD 2005.",
    },
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class GraphData:
    ids: tuple[int, ...]
    adj: tuple[tuple[int, ...], ...]
    metadata: dict

    @property
    def n(self) -> int:
        return len(self.ids)

    @property
    def m(self) -> int:
        return sum(map(len, self.adj)) // 2

    @cached_property
    def graph_sha256(self) -> str:
        # ASCII line encoding is canonical, handles arbitrary integer IDs, and
        # explicitly includes every node even when it has degree zero.
        digest = hashlib.sha256(b"LinkRadar graph v1\n")
        digest.update(f"n {self.n}\n".encode("ascii"))
        for index, raw_id in enumerate(self.ids):
            digest.update(f"v {index} {raw_id}\n".encode("ascii"))
        for u, neighbors in enumerate(self.adj):
            for v in neighbors:
                if u < v:
                    digest.update(f"e {u} {v}\n".encode("ascii"))
        return digest.hexdigest()


def _integer(value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"Node ID must be an integer, got {value!r}")
    return int(value)


def _component_sizes(adj: tuple[tuple[int, ...], ...]) -> list[int]:
    seen = bytearray(len(adj))
    sizes = []
    for start in range(len(adj)):
        if seen[start]:
            continue
        seen[start] = 1
        stack = [start]
        size = 0
        while stack:
            u = stack.pop()
            size += 1
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = 1
                    stack.append(v)
        sizes.append(size)
    return sizes


def graph_from_edges(edges: Iterable[tuple[int, int]], nodes: Iterable[int] | None = None,
                     metadata: dict | None = None) -> GraphData:
    node_set = {_integer(node) for node in (() if nodes is None else nodes)}
    discovered = set()
    unique = set()
    rows = self_loops = duplicates = 0
    for pair in edges:
        if len(pair) != 2:
            raise ValueError(f"Expected two integer node IDs, got {pair!r}")
        u, v = map(_integer, pair)
        rows += 1
        discovered.update((u, v))
        node_set.update((u, v))
        if u == v:
            self_loops += 1
            continue
        edge = (min(u, v), max(u, v))
        if edge in unique:
            duplicates += 1
        unique.add(edge)
    ids = tuple(sorted(node_set))
    mapping = {raw_id: index for index, raw_id in enumerate(ids)}
    adjacency: list[list[int]] = [[] for _ in ids]
    for u, v in sorted(unique):
        i, j = mapping[u], mapping[v]
        adjacency[i].append(j)
        adjacency[j].append(i)
    adj = tuple(tuple(sorted(neighbors)) for neighbors in adjacency)
    sizes = _component_sizes(adj)
    meta = dict(metadata or {})
    meta.update({
        "dataset_id": meta.get("dataset_id", "manual"),
        "source": meta.get("source", "in_memory"),
        "source_url": meta.get("source_url"),
        "raw_sha256": meta.get("raw_sha256"),
        "raw_data_lines": rows,
        "self_loop_lines": self_loops,
        "duplicate_edge_lines": duplicates,
        "discovered_nodes": len(discovered),
        "explicit_nodes": len(node_set - discovered),
        "final_nodes": len(ids),
        "final_edges": len(unique),
        "connected_components": len(sizes),
        "largest_component_nodes": max(sizes, default=0),
        "average_degree": 2.0 * len(unique) / len(ids) if ids else 0.0,
        "cleaning_version": CLEANING_VERSION,
        "node_mapping_rule": "original integer ID ascending -> contiguous index 0..n-1",
    })
    graph = GraphData(ids, adj, meta)
    meta["graph_sha256"] = graph.graph_sha256
    return graph


def _open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8-sig") if path.suffix.lower() == ".gz" else path.open("r", encoding="utf-8-sig")


def load_graph(path: str | Path, nodes_path: str | Path | None = None,
               dataset_id: str | None = None, strict: bool = True) -> GraphData:
    """Read an edge list. Strict malformed rows fail with filename and line."""
    path = Path(path)
    edges = []
    nodes = []
    skipped = []
    total_data_lines = 0
    with _open_text(path) as stream:
        for lineno, line in enumerate(stream, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            total_data_lines += 1
            fields = line.split()
            try:
                if len(fields) != 2:
                    raise ValueError("expected exactly two integer node IDs")
                edges.append((int(fields[0]), int(fields[1])))
            except ValueError as exc:
                message = f"{path}:{lineno}: invalid edge row: {line!r} ({exc})"
                if strict:
                    raise ValueError(message) from exc
                skipped.append({"line": lineno, "reason": str(exc)})
    if nodes_path is not None:
        nodes_path = Path(nodes_path)
        with _open_text(nodes_path) as stream:
            for lineno, line in enumerate(stream, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    fields = line.split()
                    if len(fields) != 1:
                        raise ValueError("expected exactly one integer node ID")
                    nodes.append(int(fields[0]))
                except ValueError as exc:
                    # A node list establishes graph identity; never drop an
                    # invalid node silently, including edge-list lenient mode.
                    raise ValueError(f"{nodes_path}:{lineno}: invalid node row: {line!r}") from exc
    metadata = {
        "dataset_id": dataset_id or path.name.removesuffix(".gz").removesuffix(".txt"),
        "source": "manual_import",
        "source_path": str(path.resolve()),
        "raw_sha256": _sha256(path),
        "raw_bytes": path.stat().st_size,
        "raw_input_data_lines": total_data_lines,
        "skipped_invalid_lines": len(skipped),
        "invalid_lines": skipped,
        "strict": strict,
        "nodes_path": str(nodes_path.resolve()) if nodes_path else None,
        "nodes_sha256": _sha256(nodes_path) if nodes_path else None,
    }
    return graph_from_edges(edges, nodes=nodes, metadata=metadata)


def save_graph(graph: GraphData, directory: str | Path) -> Path:
    """Write graph.npz, metadata.json and an explicit original-ID mapping."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    edges = np.asarray([(u, v) for u, ns in enumerate(graph.adj) for v in ns if u < v], dtype=np.int64).reshape(-1, 2)
    with tempfile.NamedTemporaryFile("wb", dir=directory, prefix="graph.", suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, ids=np.asarray([str(i) for i in graph.ids], dtype=str),
                            edges=edges, n=np.asarray(graph.n, dtype=np.int64))
    try:
        temporary.replace(directory / "graph.npz")
    finally:
        temporary.unlink(missing_ok=True)
    meta = dict(graph.metadata)
    meta.update({"graph_sha256": graph.graph_sha256, "cache_format": "LinkRadar-npz-v1",
                 "cache_sha256": _sha256(directory / "graph.npz")})
    _atomic_json(directory / "metadata.json", meta)
    with (directory / "node_mapping.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("index", "original_id"))
        writer.writerows(enumerate(graph.ids))
    return directory


def load_cached_graph(directory: str | Path) -> GraphData:
    directory = Path(directory)
    with (directory / "metadata.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("cache_format") != "LinkRadar-npz-v1":
        raise ValueError("Unsupported graph cache format")
    if metadata.get("cache_sha256") != _sha256(directory / "graph.npz"):
        raise ValueError("Graph cache file SHA256 mismatch")
    with np.load(directory / "graph.npz", allow_pickle=False) as archive:
        ids = tuple(int(i) for i in archive["ids"])
        edges = archive["edges"]
        n = int(archive["n"])
    if n != len(ids) or ids != tuple(sorted(set(ids))):
        raise ValueError("Invalid cached node mapping")
    if edges.ndim != 2 or edges.shape[1] != 2 or edges.dtype.kind not in "iu":
        raise ValueError("Invalid cached edge array")
    adjacency: list[list[int]] = [[] for _ in ids]
    previous = (-1, -1)
    for u, v in edges.tolist():
        if not 0 <= u < v < n or (u, v) <= previous:
            raise ValueError("Invalid or unsorted cached edge")
        adjacency[u].append(v)
        adjacency[v].append(u)
        previous = (u, v)
    graph = GraphData(ids, tuple(tuple(sorted(ns)) for ns in adjacency), metadata)
    if graph.graph_sha256 != metadata.get("graph_sha256"):
        raise ValueError("Graph fingerprint mismatch")
    return graph


def _registered(dataset: str) -> tuple[str, dict]:
    key = dataset.lower().replace("-", "_")
    if key not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {', '.join(DATASETS)}")
    return key, DATASETS[key]


def _verify_gzip(path: Path) -> None:
    with gzip.open(path, "rb") as stream:
        for _ in iter(lambda: stream.read(1024 * 1024), b""):
            pass


def download_dataset(dataset: str, data_dir: str | Path, timeout: float = 30,
                     retries: int = 2) -> Path:
    """Download only a registered official URL; failures leave an audit manifest."""
    key, specification = _registered(dataset)
    if timeout <= 0 or retries < 0:
        raise ValueError("timeout must be positive and retries nonnegative")
    directory = Path(data_dir) / "raw" / key
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / specification["url"].rsplit("/", 1)[-1]
    manifest_path = directory / "manifest.json"
    if target.exists() and manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("status") == "downloaded" and previous.get("raw_sha256") == _sha256(target):
            _verify_gzip(target)
            return target
    attempts = []
    for attempt in range(retries + 1):
        temporary = None
        try:
            request = urllib.request.Request(specification["url"], headers={"User-Agent": "LinkRadar/0.1 academic reproducibility"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                # Redirects are recorded, never silently replaced with mirrors.
                actual_url = response.geturl()
                with tempfile.NamedTemporaryFile("wb", dir=directory, prefix="download.", suffix=".tmp", delete=False) as output:
                    temporary = Path(output.name)
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        output.write(block)
            _verify_gzip(temporary)
            raw_sha256 = _sha256(temporary)
            size = temporary.stat().st_size
            temporary.replace(target)
            _atomic_json(manifest_path, {"dataset_id": key, "status": "downloaded",
                "source_url": specification["url"], "actual_url": actual_url,
                "downloaded_at_utc": _utc_now(), "raw_sha256": raw_sha256,
                "raw_bytes": size, "attempts": attempts, "path": str(target.resolve()),
                "source_page": specification["page"], "citation": specification["citation"]})
            return target
        except (OSError, EOFError, urllib.error.URLError, http.client.HTTPException, zlib.error) as exc:
            attempts.append({"attempt": attempt + 1, "error": f"{type(exc).__name__}: {exc}", "at_utc": _utc_now()})
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2 ** attempt, 4))
    _atomic_json(manifest_path, {"dataset_id": key, "status": "failed", "source_url": specification["url"],
                               "checked_at_utc": _utc_now(), "attempts": attempts,
                               "manual_import_available": True})
    raise RuntimeError(f"Download failed for {key}: {attempts[-1]['error']}. Use the manual import command.")


def _dataset_metadata(key: str, specification: dict) -> dict:
    return {"dataset_id": key, "source_url": specification["url"], "source_page": specification["page"],
            "official_statistics": {"nodes": specification["official_nodes"], "edges": specification["official_edges"]},
            "description": specification["description"], "citation": specification["citation"],
            "data_use_note": "Follow official source usage terms; public download does not imply unrestricted redistribution."}


def prepare_dataset(dataset: str, data_dir: str | Path) -> GraphData:
    key, specification = _registered(dataset)
    path = download_dataset(key, data_dir)
    graph = load_graph(path, dataset_id=key)
    manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
    graph.metadata.update(_dataset_metadata(key, specification))
    graph.metadata.update({"source": "official_download", "download_manifest": manifest})
    save_graph(graph, Path(data_dir) / "processed" / key)
    return graph


def import_dataset(dataset: str, path: str | Path, data_dir: str | Path,
                   nodes_path: str | Path | None = None) -> GraphData:
    # User-defined dataset names must not escape the processed cache root.
    if not dataset or dataset in {".", ".."} or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in dataset):
        raise ValueError("Dataset ID must contain only letters, digits, '_' and '-'")
    key = dataset.lower().replace("-", "_")
    graph = load_graph(path, nodes_path=nodes_path, dataset_id=key)
    if key in DATASETS:
        graph.metadata.update(_dataset_metadata(key, DATASETS[key]))
    save_graph(graph, Path(data_dir) / "processed" / key)
    return graph
