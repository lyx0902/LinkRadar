"""Isolated natural-vs-BFS layout A/B using the recorded-DAG kernel.

This script is experimental and does not touch the default quotient-run or
quotient-snapshot code path. It builds two layouts of the existing quotient
core, runs the same core source groups with DagAccumulator, and compares the
restored original-graph contribution vectors.

Usage (from the repository root):
    .\\.venv\\Scripts\\python.exe -B scripts/benchmark_layout_dag.py
"""
from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parallel_bc.brandes import source_order_digest
from parallel_bc.compiled_benchmark import compare_vectors
from parallel_bc.large_benchmark import bounded_job

CACHE = ROOT / "data" / "processed" / "as-skitter_csr_treeq"
THREADS = 8
BATCH_SIZE = 4
GROUPS = 1024
REPEATS = 2
SEED = 46
LAYOUT_SEED = 1234
CPU_AFFINITY = [0, 2, 4, 6, 8, 9, 10, 11]
CASE_SECONDS = 600
RESERVE_BYTES = 512 * 2**20


def _code_digest():
    digest = hashlib.sha256()
    for name in ("core_layout.py", "quotient_dag_experiment.py", "compiled_threads.py",
                 "compiled.py", "quotient_brandes.py", "tree_quotient.py", "csr.py",
                 "brandes.py", "large_benchmark.py"):
        digest.update(name.encode())
        digest.update((ROOT / "src" / "parallel_bc" / name).read_bytes())
    return digest.hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def _prepare(label, kind, layouts_dir, run_root):
    output = run_root / f"prepare_{label}"
    output.mkdir()
    params = dict(cache_directory=str(CACHE), output=str(output), kind="prepare",
                  layout_directory=(kind, str(layouts_dir / kind)), seed=LAYOUT_SEED,
                  cpu_affinity=CPU_AFFINITY, timeout_seconds=CASE_SECONDS)
    row = bounded_job("layout_experiment", params, seconds=CASE_SECONDS,
                      min_available_bytes=RESERVE_BYTES)
    row.update(kind=kind, label=label)
    _write_json(run_root / f"prepare_{label}.json", row)
    print(f"prepare {label}: {row['status']} wall={row.get('t_supervised_total', 0):.2f}s", flush=True)
    if row["status"] != "ok":
        raise RuntimeError(f"layout preparation failed: {label}: {row.get('error')}")
    return row


def _case(label, kind, repeat, roots, layouts_dir, run_root):
    output = run_root / f"{label}_r{repeat}"
    output.mkdir()
    params = dict(cache_directory=str(CACHE), output=str(output), kind="case",
                  layout_directory=str(layouts_dir / kind), roots=roots.tolist(),
                  threads=THREADS, batch_size=BATCH_SIZE, cpu_affinity=CPU_AFFINITY,
                  timeout_seconds=CASE_SECONDS, kernel="dag")
    row = bounded_job("layout_experiment", params, seconds=CASE_SECONDS,
                      min_available_bytes=RESERVE_BYTES)
    row.update(layout=kind, label=label, repeat=repeat,
               output_directory=str(output.resolve()))
    _write_json(output / "case.json", row)
    print(f"{label} repeat={repeat}: {row['status']} work={row.get('t_work', 0):.3f}s "
          f"exec={row.get('t_execute_reduce', 0):.3f}s wall={row.get('t_supervised_total', 0):.2f}s",
          flush=True)
    return row


def _compare(natural, bfs):
    if natural["status"] != "ok" or bfs["status"] != "ok":
        raise RuntimeError("Cannot compare unfinished layout cases")
    coverage = all(natural[key] == bfs[key] for key in (
        "graph_sha256", "root_source_order_hash", "source_order_hash", "source_count",
        "represented_source_count", "core_traversals", "restored_original_nodes"))
    a = np.load(natural["contribution_path"], mmap_mode="r", allow_pickle=False)
    b = np.load(bfs["contribution_path"], mmap_mode="r", allow_pickle=False)
    check = compare_vectors(a, b)
    core_a = np.load(natural["core_contribution_path"], mmap_mode="r", allow_pickle=False)
    core_b = np.load(bfs["core_contribution_path"], mmap_mode="r", allow_pickle=False)
    core_check = compare_vectors(core_a, core_b)
    if not coverage or not check["allclose"] or not core_check["allclose"]:
        raise RuntimeError(f"layout validation mismatch: coverage={coverage}, "
                           f"original={check}, core={core_check}")
    return dict(coverage_valid=coverage,
                original_nodes=check, old_core_indices=core_check,
                work_speedup=natural["t_work"] / bfs["t_work"],
                execute_speedup=natural["t_execute_reduce"] / bfs["t_execute_reduce"],
                wall_speedup=natural["t_supervised_total"] / bfs["t_supervised_total"],
                natural_work_seconds=natural["t_work"],
                bfs_work_seconds=bfs["t_work"],
                natural_execute_seconds=natural["t_execute_reduce"],
                bfs_execute_seconds=bfs["t_execute_reduce"],
                natural_wall_seconds=natural["t_supervised_total"],
                bfs_wall_seconds=bfs["t_supervised_total"])


def main():
    if not CACHE.is_dir():
        raise SystemExit(f"Missing quotient cache: {CACHE}")
    if not set(CPU_AFFINITY) <= set(psutil.Process().cpu_affinity()):
        raise SystemExit("Requested CPU affinity is not available on this machine")
    run_root = ROOT / "results" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + "as_skitter_layout_dag_ab")
    run_root.mkdir(parents=True, exist_ok=False)
    layouts_dir = run_root / "layouts"
    layouts_dir.mkdir()
    core_indptr = np.load(CACHE / "core" / "indptr.npy", mmap_mode="r", allow_pickle=False)
    eligible = np.flatnonzero(np.diff(core_indptr) > 0)
    roots = np.random.default_rng(SEED).permutation(eligible)[:GROUPS].astype(np.int32)
    np.save(run_root / "source_groups.npy", roots)
    meta = json.loads((CACHE / "metadata.json").read_text(encoding="utf-8"))
    summary = dict(status="running", started_utc=datetime.now(timezone.utc).isoformat(),
        cache_directory=str(CACHE.resolve()), graph_sha256=meta["source_graph_sha256"],
        core_graph_sha256=meta["core_graph_sha256"], original_nodes=meta["original_nodes"],
        original_edges=meta["original_edges"], core_nodes=meta["core_nodes"],
        core_edges=meta["core_edges"], groups=GROUPS, seed=SEED, threads=THREADS,
        batch_size=BATCH_SIZE, cpu_affinity=CPU_AFFINITY, layout_seed=LAYOUT_SEED,
        source_order_hash=source_order_digest(roots), code_sha256=_code_digest(),
        result_kind="source_subset_contribution", exact_all_sources=False, scaled=False,
        kernel="recorded_shortest_path_edges_v1", protocol="alternating order; same roots; isolated A/B",
        preparations=[], cases=[], pairs=[])
    _write_json(run_root / "summary.json", summary)
    try:
        summary["preparations"].append(_prepare("natural", "natural", layouts_dir, run_root))
        summary["preparations"].append(_prepare("bfs", "bfs", layouts_dir, run_root))
        for repeat in range(1, REPEATS + 1):
            order = ("natural", "bfs") if repeat % 2 == 1 else ("bfs", "natural")
            rows = {}
            for kind in order:
                row = _case(kind, kind, repeat, roots, layouts_dir, run_root)
                summary["cases"].append(row)
                _write_json(run_root / "summary.json", summary)
                rows[kind] = row
            pair = dict(repeat=repeat, validation=_compare(rows["natural"], rows["bfs"]))
            summary["pairs"].append(pair)
            _write_json(run_root / "summary.json", summary)
    except BaseException as exc:
        summary["status"] = "incomplete"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        _write_json(run_root / "summary.json", summary)
        raise
    successful = summary["pairs"]
    if successful:
        summary["summary"] = {
            "successful_pairs": len(successful),
            "median_work_speedup": statistics.median(p["validation"]["work_speedup"] for p in successful),
            "median_execute_speedup": statistics.median(p["validation"]["execute_speedup"] for p in successful),
            "median_wall_speedup": statistics.median(p["validation"]["wall_speedup"] for p in successful),
            "work_speedups": [p["validation"]["work_speedup"] for p in successful],
            "execute_speedups": [p["validation"]["execute_speedup"] for p in successful],
            "max_abs_error": max(p["validation"]["original_nodes"]["max_abs_error"] for p in successful),
        }
    summary["status"] = "complete" if successful else "incomplete"
    summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(run_root / "summary.json", summary)
    lines = ["# as-skitter layout A/B (DAG kernel)", "",
        f"Status: {summary['status']}; groups={GROUPS}; seed={SEED}; threads={THREADS}; B={BATCH_SIZE}.",
        f"Graph `{meta['source_graph_sha256']}`; core `{meta['core_graph_sha256']}`.",
        "The default quotient-run / snapshot path was not used or modified; this is an isolated layout experiment.", "",
        "| Repeat | Natural work s | BFS work s | Work speedup | Execute speedup | Wall speedup | Max abs error |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for pair in summary["pairs"]:
        v = pair["validation"]
        lines.append(f"| {pair['repeat']} | {v['natural_work_seconds']:.3f} | {v['bfs_work_seconds']:.3f} | "
                     f"{v['work_speedup']:.4f} | {v['execute_speedup']:.4f} | {v['wall_speedup']:.4f} | "
                     f"{v['original_nodes']['max_abs_error']:.3e} |")
    if "summary" in summary:
        lines += ["", "```json", json.dumps(summary["summary"], indent=2), "```"]
    lines += ["", "Work includes grouping, core execution/reduction and full original-node restoration;",
              "wall additionally includes process startup, quotient/layout loading and output writes.",
              "A result above 1 means BFS is faster than natural. This is a partial source-group experiment, not full BC."]
    (run_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"RESULT={run_root.resolve()}", flush=True)


if __name__ == "__main__":
    main()
