"""Bounded native-thread scaling with matched source sets and rotating order."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import operator
import os
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import psutil

from .brandes import source_order_digest
from .compiled_benchmark import compare_vectors
from .data import _atomic_json
from .large_benchmark import _save_array, bounded_job


def scaling_code_digest():
    digest = hashlib.sha256()
    for name in ("compiled.py", "compiled_threads.py", "compiled_scaling.py", "compiled_benchmark.py",
                 "large_benchmark.py", "csr.py", "brandes.py", "cli.py"):
        digest.update(name.encode())
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def run_scaling(cache_directory, output_root, *, source_count=64, threads=(1, 2, 4),
                batch_size=4, repeats=3, seed=42, max_case_seconds=120,
                max_total_seconds=900, min_available_bytes=2 * 2**30, cpu_affinity=None):
    source_count, batch_size, repeats = map(operator.index, (source_count, batch_size, repeats))
    threads = tuple(operator.index(p) for p in threads)
    if (source_count < 1 or batch_size < 1 or repeats < 1 or not threads or threads[0] != 1
            or threads != tuple(sorted(set(threads))) or not np.isfinite(max_case_seconds)
            or not np.isfinite(max_total_seconds) or max_case_seconds <= 0
            or max_total_seconds <= 0 or min_available_bytes < 0):
        raise ValueError("Invalid scaling counts, ordered threads (must start with 1), or budget")
    if threads[-1] > (source_count + batch_size - 1) // batch_size:
        raise ValueError("Need at least one batch per requested worker; increase sources or reduce batch_size")
    if cpu_affinity is not None:
        cpu_affinity = tuple(operator.index(cpu) for cpu in cpu_affinity)
        if (len(cpu_affinity) < threads[-1] or cpu_affinity != tuple(sorted(set(cpu_affinity)))
                or not hasattr(psutil.Process(), "cpu_affinity")
                or not set(cpu_affinity) <= set(psutil.Process().cpu_affinity())):
            raise ValueError("Affinity must be distinct allowed CPUs, with at least one CPU per requested thread")
    meta = json.loads((Path(cache_directory) / "metadata.json").read_text(encoding="utf-8"))
    if meta.get("cache_format") != "LinkRadar-csr-v1" or source_count > meta["final_nodes"]:
        raise ValueError("Invalid CSR metadata or source count exceeds n")
    try:
        versions = {name: version(name) for name in ("numpy", "numba", "llvmlite", "psutil")}
    except PackageNotFoundError as exc:
        raise RuntimeError('Native scaling requires: pip install -e ".[compiled]"') from exc
    sources = np.random.default_rng(seed).permutation(meta["final_nodes"])[:source_count].tolist()
    root = Path(output_root) / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ_") + "compiled_scaling")
    root.mkdir(parents=True, exist_ok=False)
    (root / "jit_cache").mkdir()
    _save_array(root / "source_order.npy", np.asarray(sources, dtype=np.int64))
    manifest = dict(status="running", supervisor_pid=os.getpid(), graph_sha256=meta["graph_sha256"],
        n=meta["final_nodes"], m=meta["final_edges"], graph_full=True, source_count=source_count,
        exact_all_sources=source_count == meta["final_nodes"], scaled=False,
        result_kind="ordered_all_source_contribution" if source_count == meta["final_nodes"] else "source_subset_contribution",
        source_order_hash=source_order_digest(sources), seed=seed, threads=threads,
        batch_size=batch_size, repeats=repeats, code_sha256=scaling_code_digest(),
        max_case_seconds=max_case_seconds, max_total_seconds=max_total_seconds,
        min_available_bytes_reserved=min_available_bytes,
        cpu_affinity_requested=cpu_affinity,
        environment=dict(python=platform.python_version(), os=platform.platform(),
            cpu_model=platform.processor(), logical_cpus=psutil.cpu_count(),
            cpu_affinity=psutil.Process().cpu_affinity() if hasattr(psutil.Process(), "cpu_affinity") else None,
            **versions),
        protocol="separate JIT warmup; rotating threads order; same full graph/sources/batches; bounded spawn per case",
        cases=[], comparisons=[])
    _atomic_json(root / "manifest.json", manifest)
    begin = time.perf_counter()
    resource_stop = False
    def case(label, selected, p):
        nonlocal resource_stop
        output = root / label
        output.mkdir()
        remaining = min(max_case_seconds, max_total_seconds - (time.perf_counter() - begin))
        if resource_stop:
            row = dict(status="resource_not_run", error="Earlier case reached memory reserve")
        elif remaining <= 0:
            row = dict(status="budget_not_run", error="Total scaling budget exhausted")
        else:
            row = bounded_job("thread_subset", dict(cache_directory=str(Path(cache_directory).resolve()),
                output=str(output.resolve()), sources=selected, threads=p, batch_size=batch_size,
                timeout_seconds=remaining, progress_path=str((output / "sources_completed.jsonl").resolve()),
                cpu_affinity=cpu_affinity,
                jit_cache_directory=str((root / "jit_cache").resolve())), seconds=remaining,
                min_available_bytes=min_available_bytes)
        resource_stop |= row["status"] in {"resource_limit", "resource_not_run"}
        row.update(label=label, threads_requested=p, output_directory=str(output.resolve()))
        _atomic_json(output / "case.json", row)
        print(f"{label}: {row['status']}; execute+reduce={row.get('t_execute_reduce', 0):.3f}s; "
              f"wall={row.get('t_supervised_total', 0):.3f}s", flush=True)
        return row
    try:
        manifest["warmup"] = case("warmup", sources[:1], 1)
        _atomic_json(root / "manifest.json", manifest)
        if manifest["warmup"]["status"] != "ok":
            resource_stop = True  # Do not repeat a missing-dependency/startup failure.
        for repeat in range(repeats):
            shift = repeat % len(threads)
            order = threads[shift:] + threads[:shift]
            current = {}
            for p in order:
                row = case(f"repeat{repeat + 1}_t{p}", sources, p)
                row["repeat"] = repeat + 1
                current[p] = row
                manifest["cases"].append(row)
                _atomic_json(root / "manifest.json", manifest)
            reference = current[1]
            for p in threads:
                actual = current[p]
                comparison = dict(repeat=repeat + 1, threads=p, status="incomplete")
                if reference["status"] == actual["status"] == "ok":
                    a = np.load(reference["contribution_path"], mmap_mode="r", allow_pickle=False)
                    b = np.load(actual["contribution_path"], mmap_mode="r", allow_pickle=False)
                    check = compare_vectors(a, b)
                    coverage = (actual["source_order_hash"] == manifest["source_order_hash"]
                                and actual["p_used"] == actual["active_workers"] == p
                                and len({w["native_thread_id"] for w in actual["workers"]}) == p
                                and (cpu_affinity is None or actual["cpu_affinity_used"] == list(cpu_affinity)))
                    comparison.update(status="ok" if check["allclose"] and coverage else "mismatch",
                        validation=check, coverage_valid=coverage,
                        speedup_execute=reference["t_execute_reduce"] / actual["t_execute_reduce"],
                        speedup_wall=reference["t_supervised_total"] / actual["t_supervised_total"],
                        efficiency=(reference["t_execute_reduce"] / actual["t_execute_reduce"]) / p,
                        t_execute_reduce=actual["t_execute_reduce"], t_wall=actual["t_supervised_total"],
                        average_worker_cpu_cores=actual["average_worker_cpu_cores"],
                        fallback_sources=actual["fallback_sources"],
                        child_peak_rss_bytes=actual["child_peak_rss_bytes"])
                    del a, b
                manifest["comparisons"].append(comparison)
                _atomic_json(Path(actual["output_directory"]) / "validation.json", comparison)
            _atomic_json(root / "manifest.json", manifest)
    except BaseException:
        manifest["status"] = "interrupted"
        _atomic_json(root / "manifest.json", manifest)
        raise
    good = [row for row in manifest["comparisons"] if row["status"] == "ok"]
    manifest["status"] = "complete" if len(good) == repeats * len(threads) else "partial"
    manifest["t_scaling_wall"] = time.perf_counter() - begin
    manifest["summary"] = []
    for p in threads:
        values = [row for row in good if row["threads"] == p]
        if values:
            manifest["summary"].append(dict(threads=p, successful_repeats=len(values), **{
                "median_" + key: statistics.median(row[key] for row in values)
                for key in ("t_execute_reduce", "t_wall", "speedup_execute", "speedup_wall", "efficiency",
                            "average_worker_cpu_cores", "child_peak_rss_bytes")}))
    _atomic_json(root / "manifest.json", manifest)
    lines = ["# Compiled native-thread scaling", "",
        f"Status: {manifest['status']}; n={manifest['n']}, m={manifest['m']}, sources={source_count}, seed={seed}, B={batch_size}.",
        "JIT warmup is excluded from summary; each measured case still includes cache loading in its wall time.",
        "Same native kernel, full graph, sources and batch size; one-thread baseline runs inline.", "",
        "| Threads | Repeats | Execute+reduce median s | Wall median s | Paired execute speedup | Efficiency | Worker CPU cores | Peak RSS MiB |",
        "|---|---|---|---|---|---|---|---|"]
    for row in manifest["summary"]:
        lines.append(f"| {row['threads']} | {row['successful_repeats']} | {row['median_t_execute_reduce']:.4f} | "
            f"{row['median_t_wall']:.4f} | {row['median_speedup_execute']:.3f} | {row['median_efficiency']:.3f} | "
            f"{row['median_average_worker_cpu_cores']:.3f} | {row['median_child_peak_rss_bytes']/2**20:.2f} |")
    lines += ["", "Execute includes thread launch, barrier, source work, progress logging and join; reduction is separately recorded.",
        "Worker CPU cores = summed worker CPU time / execution wall time; not physical-core placement.",
        "Partial source contributions are unscaled and are not full-source centrality.",
        "All failures remain in manifest. Speedup is within-repeat paired, then summarized; no universal scaling guarantee."]
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root
