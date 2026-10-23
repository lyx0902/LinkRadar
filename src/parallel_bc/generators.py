"""Seeded sparse graph generators; all nodes, including isolates, are retained."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Mapping

import networkx as nx

from .data import GraphData, graph_from_edges


def generate_graph(spec: Mapping | object) -> GraphData:
    if is_dataclass(spec):
        spec = asdict(spec)
    elif not isinstance(spec, Mapping):
        spec = vars(spec)
    spec = dict(spec)
    family = str(spec.get("family", "er")).lower()
    n = int(spec.get("nodes", spec.get("n", 200)))
    seed = int(spec.get("seed", 42))
    if n < 0:
        raise ValueError("nodes must be nonnegative")
    parameters = {"family": family, "nodes": n, "seed": seed}
    if family == "er":
        degree = float(spec.get("avg_degree", 6))
        if degree < 0:
            raise ValueError("avg_degree must be nonnegative")
        m = min(n * (n - 1) // 2, round(n * degree / 2))
        generated = nx.gnm_random_graph(n, m, seed=seed)
        parameters.update({"avg_degree": degree, "target_edges": m})
    elif family == "ba":
        attach = int(spec.get("attach", spec.get("m_attach", 3)))
        if attach < 1:
            raise ValueError("BA attach must be positive")
        effective = min(attach, max(0, n - 1))
        generated = nx.barabasi_albert_graph(n, effective, seed=seed) if n > 1 else nx.empty_graph(n)
        parameters.update({"attach": attach, "effective_attach": effective})
    elif family == "sbm":
        sizes = [n // 2, n - n // 2]
        internal = float(spec.get("internal_degree", 6))
        cross = float(spec.get("cross_degree", 0.5))
        if internal < 0 or cross < 0:
            raise ValueError("SBM expected degrees must be nonnegative")
        p00 = min(1.0, internal / max(1, sizes[0] - 1)) if sizes[0] > 1 else 0.0
        p11 = min(1.0, internal / max(1, sizes[1] - 1)) if sizes[1] > 1 else 0.0
        p01 = min(1.0, cross / max(1.0, n / 2)) if n else 0.0
        probabilities = [[p00, p01], [p01, p11]]
        generated = nx.stochastic_block_model(sizes, probabilities, seed=seed, selfloops=False)
        parameters.update({"sizes": sizes, "internal_degree": internal, "cross_degree": cross,
                           "probabilities": probabilities})
    else:
        raise ValueError(f"Unknown graph family {family!r}; use er, ba, or sbm")
    return graph_from_edges(generated.edges(), nodes=range(n), metadata={
        "dataset_id": spec.get("dataset_id", f"{family}_n{n}_seed{seed}"),
        "source": "synthetic", "generator": f"networkx {nx.__version__}",
        "generation_parameters": parameters, "seed": seed,
    })
