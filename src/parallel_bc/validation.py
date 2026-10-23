"""Full-node correctness checks using NetworkX public APIs only."""
from __future__ import annotations

import math
import time

import networkx as nx
import numpy as np

from .brandes import CentralityResult


def _comparison(actual, expected, ids, atol, rtol):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape or actual.shape != (len(ids),):
        raise ValueError("score vectors and node IDs must have equal length")
    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    if len(ids) == 0:
        maximum, worst = 0.0, None
    elif finite:
        errors = np.abs(actual - expected)
        index = int(np.argmax(errors))
        maximum, worst = float(errors[index]), int(ids[index])
    else:
        maximum, worst = None, None
    return {"allclose": finite and bool(np.allclose(actual, expected, atol=atol, rtol=rtol)),
            "finite": finite, "max_abs_error": maximum, "worst_node": worst}


def compare_results(actual: CentralityResult, expected: CentralityResult,
                    atol=1e-10, rtol=1e-9, ids=None) -> dict:
    """Compare every raw and normalized score, preserving original-ID attribution."""
    if not all(math.isfinite(x) and x >= 0 for x in (atol, rtol)):
        raise ValueError("atol and rtol must be finite and nonnegative")
    ids = tuple(range(len(actual.raw))) if ids is None else tuple(ids)
    raw = _comparison(actual.raw, expected.raw, ids, atol, rtol)
    normalized = _comparison(actual.normalized, expected.normalized, ids, atol, rtol)
    candidates = [(item["max_abs_error"], item["worst_node"], label)
                  for label, item in (("raw", raw), ("normalized", normalized))
                  if item["max_abs_error"] is not None]
    maximum, worst, score_kind = max(candidates, key=lambda item: item[0]) if candidates else (None, None, None)
    return {"allclose": raw["allclose"] and normalized["allclose"],
            "finite": raw["finite"] and normalized["finite"],
            "max_abs_error": maximum, "worst_node": worst, "worst_score_kind": score_kind,
            "raw": raw, "normalized": normalized, "atol": float(atol), "rtol": float(rtol),
            "compared_nodes": len(ids)}


def validate_against_networkx(graph, scores: CentralityResult,
                             atol=1e-10, rtol=1e-9) -> dict:
    """Validate the same complete clean graph, including isolates and all components.

    t_reference covers the two public exact betweenness API calls. Conversion to
    NetworkX is timed separately and never enters measured Brandes timings.
    """
    convert_start = time.perf_counter()
    reference_graph = nx.Graph()
    reference_graph.add_nodes_from(graph.ids)
    reference_graph.add_edges_from((graph.ids[u], graph.ids[v])
        for u, neighbors in enumerate(graph.adj) for v in neighbors if u < v)
    convert_end = time.perf_counter()
    raw_map = nx.betweenness_centrality(reference_graph, k=None, normalized=False,
                                       weight=None, endpoints=False)
    norm_map = nx.betweenness_centrality(reference_graph, k=None, normalized=True,
                                        weight=None, endpoints=False)
    reference_end = time.perf_counter()
    expected = CentralityResult(
        raw=np.array([raw_map[node] for node in graph.ids], dtype=np.float64),
        normalized=np.array([norm_map[node] for node in graph.ids], dtype=np.float64), timings={})
    result = compare_results(scores, expected, atol=atol, rtol=rtol, ids=graph.ids)
    result.update({"t_reference": reference_end - convert_end,
                   "t_reference_graph": convert_end - convert_start,
                   "reference": "networkx.betweenness_centrality",
                   "reference_version": nx.__version__, "k": None,
                   "weight": None, "endpoints": False})
    return result
