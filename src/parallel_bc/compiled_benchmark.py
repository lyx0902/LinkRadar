"""Same-source, alternating-order single-core comparison with bounded workers."""
from __future__ import annotations

from datetime import datetime, timezone
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
from .data import _atomic_json
from .large_benchmark import _code_digest, _save_array, bounded_job


def compare_vectors(reference, actual, *, atol=1e-8, rtol=1e-10):
    if reference.shape != actual.shape:
        return dict(allclose=False, reason="shape mismatch")
    finite = bool(np.isfinite(reference).all() and np.isfinite(actual).all())
    error = np.abs(actual - reference)
    return dict(allclose=finite and bool(np.allclose(actual, reference, atol=atol, rtol=rtol)),
        finite=finite, compared_nodes=int(reference.size), atol=atol, rtol=rtol,
        max_abs_error=float(error.max()) if finite and error.size else None,
        worst_node_index=int(np.argmax(error)) if finite and error.size else None)


def run_comparison(cache_directory, output_root, *, source_count=8, repeats=3, seed=42,
                   max_case_seconds=240, max_total_seconds=1200, min_available_bytes=0):
    source_count, repeats = operator.index(source_count), operator.index(repeats)
    if (source_count < 1 or repeats < 1 or not np.isfinite(max_case_seconds)
            or not np.isfinite(max_total_seconds) or max_case_seconds <= 0
            or max_total_seconds <= 0 or min_available_bytes < 0):
        raise ValueError("Invalid comparison source count, repeats or budget")
    meta = json.loads((Path(cache_directory) / "metadata.json").read_text(encoding="utf-8"))
    if meta.get("cache_format") != "LinkRadar-csr-v1" or source_count > meta["final_nodes"]:
        raise ValueError("Invalid CSR metadata or source count exceeds n")
    try:
        dependencies = {name: version(name) for name in ("numpy", "numba", "llvmlite", "psutil")}
    except PackageNotFoundError as exc:
        raise RuntimeError('Compiled comparison requires: pip install -e ".[compiled]"') from exc
    sources = np.random.default_rng(seed).permutation(meta["final_nodes"])[:source_count].tolist()
    root = Path(output_root) / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ_") + "compiled_compare")
    root.mkdir(parents=True, exist_ok=False)
    # New empty cache makes the first native case a real compile observation.
    # Subsequent subprocesses reuse this run's cache, independently of user caches.
    jit_cache = root / "jit_cache"
    jit_cache.mkdir()
    _save_array(root / "source_order.npy", np.asarray(sources, dtype=np.int64))
    source_hash = source_order_digest(sources)
    import hashlib
    code_hash = hashlib.sha256((_code_digest() + Path(__file__).read_text(encoding="utf-8")).encode()).hexdigest()
    manifest = dict(status="running", graph_sha256=meta["graph_sha256"],
        n=meta["final_nodes"], m=meta["final_edges"], graph_full=True,
        source_count=source_count, source_order_hash=source_hash, seed=seed, repeats=repeats,
        exact_all_sources=source_count == meta["final_nodes"], scaled=False,
        result_kind="ordered_all_source_contribution" if source_count == meta["final_nodes"] else "source_subset_contribution",
        p_used=1, supervisor_pid=os.getpid(), code_sha256=code_hash,
        max_case_seconds=max_case_seconds, max_total_seconds=max_total_seconds,
        min_available_bytes_reserved=min_available_bytes,
        environment=dict(python=platform.python_version(), os=platform.platform(),
            cpu_model=platform.processor(), logical_cpus=psutil.cpu_count(),
            cpu_affinity=psutil.Process().cpu_affinity() if hasattr(psutil.Process(), "cpu_affinity") else None,
            **dependencies),
        protocol="alternating Python/native order; same fixed source order; independent bounded spawn per case",
        jit_cache_directory=str(jit_cache.resolve()), cases=[], pairs=[])
    _atomic_json(root / "manifest.json", manifest)
    start = time.perf_counter()
    resource_stop = False
    try:
        for repeat in range(repeats):
            backends = ("python_csr", "numba_csr") if repeat % 2 == 0 else ("numba_csr", "python_csr")
            pair_rows = {}
            for backend in backends:
                output = root / f"repeat{repeat + 1}_{backend}"
                output.mkdir()
                budget = min(max_case_seconds, max_total_seconds - (time.perf_counter() - start))
                if resource_stop:
                    row = dict(status="resource_not_run", error="Earlier case reached memory reserve")
                elif budget <= 0:
                    row = dict(status="budget_not_run", error="Total comparison budget exhausted")
                else:
                    row = bounded_job("subset", dict(cache_directory=str(Path(cache_directory).resolve()),
                        output=str(output.resolve()), sources=sources, backend=backend,
                        timeout_seconds=budget, progress_path=str((output / "sources_completed.jsonl").resolve()),
                        jit_cache_directory=str(jit_cache.resolve())), seconds=budget,
                        min_available_bytes=min_available_bytes)
                resource_stop |= row["status"] in {"resource_limit", "resource_not_run"}
                row.update(repeat=repeat + 1, backend=backend, source_order_hash=source_hash,
                    output_directory=str(output.resolve()))
                _atomic_json(output / "case.json", row)
                pair_rows[backend] = row
                manifest["cases"].append(row)
                _atomic_json(root / "manifest.json", manifest)
                print(f"repeat={repeat + 1} {backend}: {row['status']}; "
                      f"kernel={row.get('t_kernel', 0):.4f}s; wall={row.get('t_supervised_total', 0):.3f}s", flush=True)
            reference, native = pair_rows["python_csr"], pair_rows["numba_csr"]
            pair = dict(repeat=repeat + 1, status="incomplete")
            if reference["status"] == native["status"] == "ok":
                a = np.load(reference["contribution_path"], mmap_mode="r", allow_pickle=False)
                b = np.load(native["contribution_path"], mmap_mode="r", allow_pickle=False)
                validation = compare_vectors(a, b)
                pair.update(status="ok" if validation["allclose"] else "mismatch", validation=validation,
                    speedup_kernel=reference["t_kernel"] / native["t_kernel"],
                    speedup_wall=reference["t_supervised_total"] / native["t_supervised_total"],
                    python_kernel_seconds=reference["t_kernel"], native_kernel_seconds=native["t_kernel"],
                    native_seconds_per_source=native["t_kernel"] / source_count,
                    native_fallback_sources=native["fallback_sources"])
                del a, b
            manifest["pairs"].append(pair)
            _atomic_json(root / "manifest.json", manifest)
    except BaseException:
        manifest["status"] = "interrupted"
        _atomic_json(root / "manifest.json", manifest)
        raise
    successful = [pair for pair in manifest["pairs"] if pair["status"] == "ok"]
    manifest["status"] = "complete" if len(successful) == repeats else "partial"
    manifest["t_comparison_wall"] = time.perf_counter() - start
    if successful:
        manifest["summary"] = {"successful_pairs": len(successful), **{
            "median_" + key: statistics.median(pair[key] for pair in successful)
            for key in ("speedup_kernel", "speedup_wall", "python_kernel_seconds",
                        "native_kernel_seconds", "native_seconds_per_source")}}
    _atomic_json(root / "manifest.json", manifest)
    lines = ["# Python / Numba single-core comparison", "",
        f"Status: {manifest['status']}; n={manifest['n']}, m={manifest['m']}, sources={source_count}.",
        "Same full graph, same source indices and order. Contributions are unscaled; no approximate BC estimate.",
        "Native warmup includes compilation or disk cache load and is excluded from kernel timing.", "",
        "| Repeat | Status | Python kernel s | Native kernel s | Kernel speedup | Wall speedup | Max abs error |",
        "|---|---|---|---|---|---|---|"]
    for pair in manifest["pairs"]:
        lines.append(f"| {pair['repeat']} | {pair['status']} | {pair.get('python_kernel_seconds', '')} | "
            f"{pair.get('native_kernel_seconds', '')} | {pair.get('speedup_kernel', '')} | "
            f"{pair.get('speedup_wall', '')} | {pair.get('validation', {}).get('max_abs_error', '')} |")
    lines += ["", "Medians (successful pairs only):", "```json",
        json.dumps(manifest.get("summary", {}), indent=2), "```", "",
        "Wall timings include spawn, imports, graph verification, JIT/cache, allocation, calculation and output.",
        "First native case uses a new empty JIT cache; later cases reuse it. All cases are single-threaded.",
        "Speedup reflects JIT plus array workspaces and implicit-DAG traversal, not isolated compiler effects.",
        "Repeated fixed sources measure this workload only; full-source runtime is not established."]
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root
