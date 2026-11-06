"""Real-graph full-source checkpoint acceptance; no YouTube full run.

Run from the repository root with the installed compiled extra. Outputs use a
new directory and preserve every invocation's resource/transaction evidence.
"""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from parallel_bc.brandes import finalize_scores
from parallel_bc.compiled_benchmark import compare_vectors
from parallel_bc.csr import csr_from_edges
from parallel_bc.data import _atomic_json, load_cached_graph
from parallel_bc.large_benchmark import subset_contributions
from parallel_bc.quotient_checkpoint import run_checkpoint
from parallel_bc.tree_quotient import factor_trees, save_quotient


def main():
    root = Path("results") / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ_") + "checkpoint_real_validation")
    root.mkdir(parents=True, exist_ok=False)
    validations = []
    for dataset in ("facebook", "ca_grqc"):
        graph = load_cached_graph(Path("data/processed") / dataset)
        csr = csr_from_edges(((graph.ids[u], graph.ids[v]) for u, adj in enumerate(graph.adj)
                              for v in adj if u < v), nodes=graph.ids)
        assert csr.graph_sha256 == graph.graph_sha256
        q = factor_trees(csr)
        cache = save_quotient(q, root / (dataset + "_q"))
        run = root / dataset
        options = dict(threads=4, batch_size=4, chunk_size=256, seed=45)
        first = run_checkpoint(cache, run, seconds=120, max_chunks=2,
                               min_available_bytes=2**30, **options)
        assert first["status"] == "ok" and first["run_status"] == "paused", first
        old_chunks = json.loads((run / "manifest.json").read_text())["chunks"]
        resumed = run_checkpoint(cache, run, seconds=180, resume=True,
                                 min_available_bytes=2**30, **options)
        assert resumed["status"] == "ok" and resumed["run_status"] == "complete", resumed
        meta = json.loads((run / "manifest.json").read_text())
        assert meta["chunks"][:2] == old_chunks
        reference, _ = subset_contributions(csr, range(csr.n), backend="numba_csr")
        raw, normalized = finalize_scores(reference, csr.n)
        checks = {name: compare_vectors(values, np.load(run / (name + ".npy")))
                  for name, values in (("contributions", reference), ("bc_raw", raw),
                                       ("bc_normalized", normalized))}
        assert all(c["allclose"] for c in checks.values()), checks
        validations.append(dict(dataset=dataset, n=csr.n, m=csr.m,
            graph_sha256=csr.graph_sha256, exact_all_sources=meta["exact_all_sources"],
            completed_core_groups=meta["completed_core_groups"],
            represented_sources=meta["completed_source_mass"] + meta["analytic_source_mass"],
            committed_chunks=len(meta["chunks"]), original_chunks_unchanged=True,
            validation=checks, first_session=first, resumed_session=resumed))
        _atomic_json(root / "validation.json", {"status": "complete" if len(validations) == 2 else "running",
                                               "cases": validations})
        print(dataset, checks, flush=True)
    print(root.resolve(), flush=True)


if __name__ == "__main__":
    main()
