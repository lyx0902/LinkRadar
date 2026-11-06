"""Independently check the first committed YouTube chunk on the original CSR.

Restoring just this chunk uses its represented ORIGINAL source subset. This is
an acceptance check, not a full-source ranking or the production finalization.
Run after the soak test to avoid concurrent benchmark workloads.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from parallel_bc.compiled_benchmark import compare_vectors
from parallel_bc.data import _atomic_json, _sha256
from parallel_bc.large_benchmark import bounded_job, _save_array
from parallel_bc.quotient_brandes import _restore, _selected_subtrees
from parallel_bc.tree_quotient import load_quotient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    m = json.loads((args.run_dir / "manifest.json").read_text())
    first = m["chunks"][0]
    assert _sha256(args.run_dir / first["file"]) == first["sha256"]
    roots = np.load(args.run_dir / "source_plan.npy")[first["start"]:first["stop"]]
    q = load_quotient("data/processed/youtube_csr_treeq")
    assert q.source_graph_sha256 == m["identity"]["graph_sha256"]
    sources = np.flatnonzero(np.isin(q.anchor_index, roots))
    assert len(sources) == first["source_mass"] == int(q.mass[roots].sum())
    indicator = np.zeros(q.n, np.int8)
    indicator[sources] = 1
    selected, counts, component_counts = _selected_subtrees(indicator, q.parent,
        q.peel_order, q.anchor_nodes, q.component_index, len(q.component_sizes))
    assert np.array_equal(np.flatnonzero(counts), np.sort(roots))
    counts.flags.writeable = False
    core_part = np.load(args.run_dir / first["file"])
    restored = _restore(q.parent, q.peel_order, q.anchor_nodes, q.anchor_index,
        q.subtree_size, q.component_index, q.component_sizes, indicator, selected,
        counts, component_counts, core_part)
    root = Path("results") / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ_") + "youtube_checkpoint_validation")
    (root / "original").mkdir(parents=True, exist_ok=False)
    _save_array(root / "source_order.npy", sources)
    _save_array(root / "restored_chunk_original_subset.npy", restored)
    row = bounded_job("thread_subset", dict(cache_directory=str(Path("data/processed/youtube_csr").resolve()),
        output=str((root / "original").resolve()), sources=sources.tolist(),
        threads=4, batch_size=4, timeout_seconds=240, cpu_affinity=[0, 2, 4, 6]),
        seconds=240, min_available_bytes=2**30)
    result = dict(status=row["status"], graph_sha256=q.source_graph_sha256,
        original_nodes=q.n, original_edges=q.metadata["original_edges"],
        original_source_count=len(sources), core_source_groups=len(roots),
        exact_all_sources=False, scaled=False, result_kind="source_subset_contribution",
        checkpoint_file_sha256=first["sha256"], native_case=row)
    if row["status"] == "ok":
        check = compare_vectors(np.load(row["contribution_path"]), restored)
        result.update(status="verified" if check["allclose"] else "mismatch", validation=check)
    _atomic_json(root / "validation.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "native_case"}, indent=2))
    print(json.dumps({k: row[k] for k in ("status", "error", "source_count", "t_execute_reduce",
        "t_supervised_total", "child_peak_rss_bytes", "min_available_bytes") if k in row}, indent=2))
    print(root.resolve())
    return 0 if result["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
