"""Read committed evidence and summarize a checkpoint run (no computation)."""
import argparse
import json
from pathlib import Path
import statistics

import numpy as np

from parallel_bc.brandes import source_order_digest
from parallel_bc.data import _atomic_json, _sha256
from parallel_bc.quotient_checkpoint import checkpoint_code_digest


def audit(root):
    m = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    identity = m["identity"]
    assert identity["code_sha256"] == checkpoint_code_digest()
    assert _sha256(root / "source_plan.npy") == m["plan_file_sha256"]
    plan = np.load(root / "source_plan.npy", allow_pickle=False)
    assert len(plan) == len(set(plan.tolist())) == identity["core_traversals"]
    assert source_order_digest(plan) == identity["plan_hash"]
    cursor = source_mass = 0
    chunks = m["chunks"]
    for index, row in enumerate(chunks):
        assert row["index"] == index and row["start"] == cursor
        end = min(cursor + identity["chunk_size"], len(plan))
        assert row["stop"] == end
        assert row["source_order_hash"] == source_order_digest(plan[cursor:end])
        assert _sha256(root / row["file"]) == row["sha256"]
        values = np.load(root / row["file"], allow_pickle=False)
        assert values.shape == (identity["core_nodes"],) and values.dtype == np.float64
        assert np.isfinite(values).all() and np.all(values >= 0)
        cursor = end
        source_mass += row["source_mass"]
    assert cursor == m["completed_core_groups"]
    assert source_mass == m["completed_source_mass"]
    baseline_checks = {}
    for name in ("pre_soak_commits.json", "pre_recovery_commits.json"):
        baseline = root / name
        if baseline.exists():
            before = json.loads(baseline.read_text(encoding="utf-8"))
            baseline_checks[name] = chunks[:len(before)] == before
            assert baseline_checks[name]
    sessions = [dict(json.loads(path.read_text(encoding="utf-8")), file=path.name)
                for path in sorted(root.glob("session_*.json"))]
    execute = sum(r["t_execute_reduce"] for r in chunks)
    durations = [r["t_execute_reduce"] for r in chunks]
    rss = [r["rss_after_bytes"] for r in chunks]
    med = statistics.median(durations) if durations else None
    summary = dict(status="verified", run_status=m["status"], exact_all_sources=m["exact_all_sources"],
        graph_sha256=identity["graph_sha256"], code_sha256=identity["code_sha256"],
        completed_core_groups=cursor, total_core_groups=len(plan), completion_fraction=cursor / len(plan) if len(plan) else 1.,
        completed_source_mass=source_mass, analytic_source_mass=m["analytic_source_mass"],
        committed_chunks=len(chunks), baseline_commits_unchanged=all(baseline_checks.values()) if baseline_checks else None,
        baseline_checks=baseline_checks,
        committed_work_seconds=execute, median_chunk_work_seconds=med,
        min_chunk_work_seconds=min(durations) if durations else None,
        max_chunk_work_seconds=max(durations) if durations else None,
        core_groups_per_second=cursor / execute if execute else None,
        projected_full_execution_hours=len(plan) * execute / cursor / 3600 if cursor else None,
        projected_remaining_execution_hours=(len(plan) - cursor) * execute / cursor / 3600 if cursor else None,
        total_file_commit_seconds=sum(r["commit_file_seconds"] for r in chunks),
        commit_io_fraction_of_committed_work=sum(r["commit_file_seconds"] for r in chunks) / execute if execute else None,
        rss_after_first_bytes=rss[0] if rss else None, rss_after_last_bytes=rss[-1] if rss else None,
        rss_after_min_bytes=min(rss) if rss else None, rss_after_max_bytes=max(rss) if rss else None,
        rss_after_first4_median=statistics.median(rss[:4]) if rss else None,
        rss_after_last4_median=statistics.median(rss[-4:]) if rss else None,
        fallback_core_sources=sum(r["fallback_sources"] for r in chunks),
        sessions=sessions,
        notes=["Incomplete core totals are not original-node centrality or source-subset contributions.",
               "Projection excludes future interruptions, loading, restore and machine load changes.",
               "Commit time measures score file write/fsync/hash, not manifest replacement.",
               "RSS snapshots are per chunk; session peaks are supervised separately."])
    if not m["exact_all_sources"] and cursor < len(plan):
        assert not any((root / (name + ".npy")).exists()
                       for name in ("contributions", "bc_raw", "bc_normalized"))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = audit(args.run_dir)
    _atomic_json(args.run_dir / "audit.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "sessions"}, indent=2))
