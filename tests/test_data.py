import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from parallel_bc.data import (download_dataset, graph_from_edges, import_dataset,
                              load_cached_graph, load_graph, save_graph)
from parallel_bc.generators import generate_graph


def test_deterministic_cleaning_preserves_self_loop_nodes_and_isolates(tmp_path):
    source = tmp_path / "edges.txt"
    source.write_text("# comment\n\n9 2\n2 9\n2 9\n7 7\n", encoding="utf-8")
    nodes = tmp_path / "nodes.txt"
    nodes.write_text("# isolated node\n99\n", encoding="utf-8")
    graph = load_graph(source, nodes)
    assert graph.ids == (2, 7, 9, 99)
    assert graph.adj == ((2,), (), (0,), ())
    assert (graph.n, graph.m) == (4, 1)
    assert graph.metadata["raw_data_lines"] == 4
    assert graph.metadata["self_loop_lines"] == 1
    assert graph.metadata["duplicate_edge_lines"] == 2
    assert graph.metadata["connected_components"] == 3
    assert graph.metadata["largest_component_nodes"] == 2
    reordered = graph_from_edges([(7, 7), (2, 9)], nodes=[99])
    assert graph.graph_sha256 == reordered.graph_sha256
    assert graph.graph_sha256 != graph_from_edges([(2, 9), (7, 7)]).graph_sha256


def test_bad_lines_report_location_and_lenient_mode_counts(tmp_path):
    source = tmp_path / "bad.txt"
    source.write_text("# title\n1 2\nnope 3\n1 2 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad.txt:3:"):
        load_graph(source)
    graph = load_graph(source, strict=False)
    assert graph.metadata["skipped_invalid_lines"] == 2
    assert graph.metadata["raw_input_data_lines"] == 3
    assert graph.m == 1


def test_gzip_import_and_pickle_free_cache(tmp_path):
    source = tmp_path / "sample.txt.gz"
    with gzip.open(source, "wt", encoding="utf-8") as stream:
        stream.write("-3 2\n2 8\n8 8\n")
    graph = import_dataset("example", source, tmp_path / "data")
    cached = load_cached_graph(tmp_path / "data" / "processed" / "example")
    assert cached.ids == graph.ids
    assert cached.adj == graph.adj
    assert cached.graph_sha256 == graph.graph_sha256
    assert cached.metadata["source"] == "manual_import"
    with np.load(tmp_path / "data" / "processed" / "example" / "graph.npz", allow_pickle=False) as archive:
        assert all(archive[name].dtype.kind != "O" for name in archive.files)
    assert (tmp_path / "data" / "processed" / "example" / "node_mapping.csv").exists()


def test_cache_rejects_metadata_fingerprint_tampering(tmp_path):
    save_graph(graph_from_edges([(1, 2)], nodes=[3]), tmp_path)
    path = tmp_path / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["graph_sha256"] = "wrong"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_cached_graph(tmp_path)


def test_cache_rejects_file_tampering(tmp_path):
    save_graph(graph_from_edges([]), tmp_path)
    with (tmp_path / "graph.npz").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_cached_graph(tmp_path)


def test_cache_rejects_pickle_payload_even_with_updated_file_hash(tmp_path):
    save_graph(graph_from_edges([(1, 2)]), tmp_path)
    np.savez(tmp_path / "graph.npz", ids=np.array([1, 2], dtype=object),
             edges=np.array([[0, 1]], dtype=np.int64), n=np.array(2))
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["cache_sha256"] = hashlib.sha256((tmp_path / "graph.npz").read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        load_cached_graph(tmp_path)


def test_invalid_node_file_never_silently_discards_identity(tmp_path):
    source = tmp_path / "edges.txt"
    source.write_text("1 2\n", encoding="utf-8")
    nodes = tmp_path / "nodes.txt"
    nodes.write_text("3\nwrong\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"nodes.txt:2:"):
        load_graph(source, nodes, strict=False)


def test_unknown_download_does_not_access_network(tmp_path):
    with pytest.raises(ValueError, match="Unknown dataset"):
        download_dataset("unknown", tmp_path)
    with pytest.raises(ValueError, match="Dataset ID"):
        import_dataset("../escape", tmp_path / "nope", tmp_path)


def test_download_atomic_gzip_validation_and_cache(tmp_path, monkeypatch):
    payload = gzip.compress(b"1 2\n2 3\n")

    class Response(io.BytesIO):
        def geturl(self):
            return "https://snap.stanford.edu/data/facebook_combined.txt.gz"

    calls = []

    def fake_open(request, timeout):
        calls.append((request.full_url, timeout))
        return Response(payload)

    monkeypatch.setattr("parallel_bc.data.urllib.request.urlopen", fake_open)
    path = download_dataset("facebook", tmp_path, timeout=1, retries=0)
    assert path.read_bytes() == payload
    manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "downloaded"
    assert manifest["raw_bytes"] == len(payload)
    assert len(manifest["raw_sha256"]) == 64
    assert download_dataset("facebook", tmp_path, timeout=1, retries=0) == path
    assert len(calls) == 1
    assert not list(path.parent.glob("*.tmp"))


def test_download_failure_saves_audit_without_partial_target(tmp_path, monkeypatch):
    class Response(io.BytesIO):
        def geturl(self):
            return "https://snap.stanford.edu/data/ca-GrQc.txt.gz"

    monkeypatch.setattr("parallel_bc.data.urllib.request.urlopen", lambda *args, **kwargs: Response(b"<html>failure</html>"))
    monkeypatch.setattr("parallel_bc.data.time.sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="manual import"):
        download_dataset("ca_grqc", tmp_path, retries=1)
    directory = tmp_path / "raw" / "ca_grqc"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert len(manifest["attempts"]) == 2
    assert not (directory / "ca-GrQc.txt.gz").exists()
    assert not list(directory.glob("*.tmp"))


@pytest.mark.parametrize("family", ["er", "ba", "sbm"])
@pytest.mark.parametrize("n", [0, 1, 2, 31])
def test_generators_reproducible_and_keep_all_nodes(family, n):
    spec = {"family": family, "nodes": n, "seed": 42}
    first, second = generate_graph(spec), generate_graph(spec)
    assert first.ids == tuple(range(n))
    assert first.graph_sha256 == second.graph_sha256
    assert first.metadata["source"] == "synthetic"
    assert first.metadata["seed"] == 42


def test_er_sparse_target_and_isolates():
    graph = generate_graph({"family": "er", "nodes": 50, "avg_degree": 0, "seed": 7})
    assert (graph.n, graph.m) == (50, 0)
    assert graph.metadata["connected_components"] == 50


def test_large_integer_ids_round_trip(tmp_path):
    graph = graph_from_edges([(2**100, -2**100)], nodes=[0])
    save_graph(graph, tmp_path)
    assert load_cached_graph(tmp_path).ids == graph.ids
