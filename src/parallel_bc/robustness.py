"""Fixed-initial-ranking node removal; connectivity denominators stay explicit."""
from __future__ import annotations

from collections.abc import Mapping
import math
import time

import numpy as np


def _scores(graph, scores, metadata=None):
    """Validate complete score coverage and any supplied cache provenance."""
    supplied = dict(metadata or {})
    if isinstance(scores, Mapping):
        supplied.update({key: scores[key] for key in ("graph_sha256", "exact_all_sources") if key in scores})
        supplied.update(scores.get("metadata", {}))
        raw = scores.get("raw")
        normalized = scores.get("normalized")
    else:
        supplied.update(getattr(scores, "metadata", {}) or {})
        for key in ("graph_sha256", "exact_all_sources"):
            if hasattr(scores, key):
                supplied[key] = getattr(scores, key)
        raw = getattr(scores, "raw", scores)
        normalized = getattr(scores, "normalized", None)
    if supplied.get("graph_sha256") is not None and supplied["graph_sha256"] != graph.graph_sha256:
        raise ValueError("中心性缓存的 graph_sha256 与当前图不匹配")
    if "exact_all_sources" in supplied and supplied["exact_all_sources"] is not True:
        raise ValueError("节点失效实验要求全起点精确中心性")
    raw = np.asarray(raw, dtype=float)
    if raw.shape != (graph.n,) or not np.isfinite(raw).all() or np.any(raw < 0):
        raise ValueError("中心性必须包含每个节点的有限非负分数")
    if normalized is None:
        normalized = raw * (2. / ((graph.n - 1) * (graph.n - 2))) if graph.n > 2 else np.zeros(graph.n)
    normalized = np.asarray(normalized, dtype=float)
    if normalized.shape != (graph.n,) or not np.isfinite(normalized).all():
        raise ValueError("归一化中心性必须覆盖全部节点且为有限值")
    return raw, normalized


def ranking_rows(graph, scores):
    """Return all node scores and ordinal ranks, resolving ties by original ID."""
    raw, normalized = _scores(graph, scores)
    bc_order = sorted(range(graph.n), key=lambda i: (-raw[i], graph.ids[i]))
    degree_order = sorted(range(graph.n), key=lambda i: (-len(graph.adj[i]), graph.ids[i]))
    bc_rank = {index: rank for rank, index in enumerate(bc_order, 1)}
    degree_rank = {index: rank for rank, index in enumerate(degree_order, 1)}
    return [dict(index=i, original_id=int(graph.ids[i]), degree=len(graph.adj[i]),
                 bc_raw=float(raw[i]), bc_normalized=float(normalized[i]),
                 rank_bc=bc_rank[i], rank_degree=degree_rank[i]) for i in range(graph.n)]


def _components(adj, active):
    seen = set()
    largest = components = 0
    for start in range(len(adj)):
        if not active[start] or start in seen:
            continue
        components += 1
        seen.add(start)
        stack = [start]
        size = 0
        while stack:
            v = stack.pop()
            size += 1
            for w in adj[v]:
                if active[w] and w not in seen:
                    seen.add(w)
                    stack.append(w)
        largest = max(largest, size)
    return largest, components


def run_robustness(graph, scores, config=None):
    """Measure fixed-ranking removals and independent seeded random prefixes.

    Random p10/p90 are empirical quantiles across permutations, not confidence
    intervals. A zero-removal baseline is always included; duplicate floor(q*n)
    values share one observation per curve.
    """
    started = time.perf_counter()
    config = config or {}
    raw, _ = _scores(graph, scores, config.get("score_metadata"))
    fractions = [float(q) for q in config.get("fractions", [0, .01, .02, .05, .1, .15, .2])]
    if any(not math.isfinite(q) or not 0 <= q <= 1 for q in fractions):
        raise ValueError("移除比例必须是 [0, 1] 内的有限值")
    targets = {}
    for q in sorted([0.] + fractions):
        targets.setdefault(math.floor(q * graph.n), q)
    strategies = list(dict.fromkeys(config.get("strategies", ["bc", "degree", "random"])))
    if not strategies or set(strategies) - {"bc", "degree", "random"}:
        raise ValueError("strategies 仅支持 bc、degree、random 且不能为空")
    repeats = config.get("random_repeats", 10)
    if isinstance(repeats, bool) or int(repeats) != repeats or repeats < 1:
        raise ValueError("random_repeats 必须是正整数")
    seed = config.get("seed", 42)
    if isinstance(seed, bool) or int(seed) != seed or seed < 0:
        raise ValueError("seed 必须是非负整数")
    seed = int(seed)
    permutations = []
    if "bc" in strategies:
        permutations.append(("bc", 0, None, sorted(range(graph.n), key=lambda i: (-raw[i], graph.ids[i]))))
    if "degree" in strategies:
        permutations.append(("degree", 0, None, sorted(range(graph.n), key=lambda i: (-len(graph.adj[i]), graph.ids[i]))))
    if "random" in strategies:
        for repeat, child in enumerate(np.random.SeedSequence(seed).spawn(int(repeats))):
            child_seed = int(child.generate_state(1, dtype=np.uint64)[0])
            order = np.random.default_rng(child_seed).permutation(graph.n).tolist()
            permutations.append(("random", repeat, child_seed, order))
    ranking_elapsed = time.perf_counter() - started
    connectivity_started = time.perf_counter()
    dataset_id = graph.metadata.get("dataset_id", graph.metadata.get("name", "graph"))
    rows = []
    for strategy, repeat, child_seed, order in permutations:
        active = [True] * graph.n
        removed = 0
        remaining_edges = graph.m
        for k, requested in targets.items():
            while removed < k:
                v = order[removed]
                remaining_edges -= sum(active[w] for w in graph.adj[v])
                active[v] = False
                removed += 1
            largest, components = _components(graph.adj, active)
            remaining = graph.n - k
            rows.append(dict(dataset_id=dataset_id, graph_sha256=graph.graph_sha256, strategy=strategy,
                             repeat=repeat, seed=child_seed, n_original=graph.n, q_requested=requested,
                             k=k, removed_fraction=k / graph.n if graph.n else 0., remaining_nodes=remaining,
                             remaining_edges=remaining_edges, lcc_size=largest, num_components=components,
                             lcc_original_fraction=largest / graph.n if graph.n else 0.,
                             lcc_remaining_fraction=largest / remaining if remaining else None))
    groups = {}
    for row in rows:
        groups.setdefault((row["strategy"], row["k"]), []).append(row)
    summary = []
    for (strategy, k), group in groups.items():
        values = np.asarray([row["lcc_original_fraction"] for row in group])
        summary.append(dict(dataset_id=dataset_id, graph_sha256=graph.graph_sha256, strategy=strategy,
                            k=k, removed_fraction=group[0]["removed_fraction"], n_original=graph.n,
                            count=len(group), lcc_original_mean=float(np.mean(values)),
                            lcc_original_p10=float(np.quantile(values, .1)),
                            lcc_original_p90=float(np.quantile(values, .9)),
                            lcc_original_min=float(np.min(values)), lcc_original_max=float(np.max(values)),
                            lcc_size_mean=float(np.mean([r["lcc_size"] for r in group])),
                            remaining_nodes=graph.n-k,
                            remaining_edges_mean=float(np.mean([r["remaining_edges"] for r in group])),
                            num_components_mean=float(np.mean([r["num_components"] for r in group]))))
    return dict(raw=rows, summary=summary,
                timings=dict(t_ranking=ranking_elapsed, t_connectivity=time.perf_counter()-connectivity_started,
                             t_total=time.perf_counter()-started, protocol="fixed_initial_ranking",
                             random_band="empirical_p10_p90", graph_sha256=graph.graph_sha256,
                             exact_all_sources=True))
