"""Same-original-problem benchmark: native CSR versus lossless tree quotient."""
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


def quotient_code_digest():
    digest = hashlib.sha256()
    for name in ("tree_quotient.py", "quotient_brandes.py", "quotient_benchmark.py", "compiled.py",
                 "compiled_threads.py", "large_benchmark.py", "csr.py", "brandes.py", "cli.py"):
        digest.update(name.encode())
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def run_quotient_preparation(cache_directory, directory, *, seconds=300, min_available_bytes=0):
    if Path(directory).exists():
        raise ValueError("Quotient cache exists; refusing to overwrite")
    row = bounded_job("quotient_prepare", dict(cache_directory=str(Path(cache_directory).resolve()),
        output=str(Path(directory).resolve())), seconds=seconds, min_available_bytes=min_available_bytes)
    _atomic_json(Path(directory).with_suffix(".preparation.json"), row)
    return row


def run_quotient_comparison(cache_directory, quotient_directory, output_root, *, source_count=256,
                            threads=4, batch_size=4, repeats=3, seed=42, max_case_seconds=120,
                            max_total_seconds=900, min_available_bytes=0, cpu_affinity=None):
    source_count, threads, batch_size, repeats = map(operator.index, (source_count, threads, batch_size, repeats))
    if (min(source_count, threads, batch_size, repeats) < 1 or not np.isfinite(max_case_seconds)
            or not np.isfinite(max_total_seconds) or max_case_seconds <= 0
            or max_total_seconds <= 0 or min_available_bytes < 0):
        raise ValueError("Invalid quotient benchmark counts or budget")
    if cpu_affinity is not None:
        cpu_affinity = tuple(operator.index(v) for v in cpu_affinity)
        if (len(cpu_affinity) < threads or cpu_affinity != tuple(sorted(set(cpu_affinity)))
                or not set(cpu_affinity) <= set(psutil.Process().cpu_affinity())):
            raise ValueError("Invalid allowed CPU affinity")
    meta = json.loads((Path(cache_directory) / "metadata.json").read_text(encoding="utf-8"))
    qmeta = json.loads((Path(quotient_directory) / "metadata.json").read_text(encoding="utf-8"))
    if (source_count > meta["final_nodes"] or meta["graph_sha256"] != qmeta["source_graph_sha256"]
            or meta["final_nodes"] != qmeta["original_nodes"] or meta["final_edges"] != qmeta["original_edges"]):
        raise ValueError("Original graph and quotient provenance mismatch")
    sources = np.random.default_rng(seed).permutation(meta["final_nodes"])[:source_count].tolist()
    try:
        versions = {name: version(name) for name in ("numpy", "numba", "llvmlite", "psutil")}
    except PackageNotFoundError as exc:
        raise RuntimeError('Tree quotient requires: pip install -e ".[compiled]"') from exc
    root = Path(output_root) / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ_") + "quotient_compare")
    root.mkdir(parents=True, exist_ok=False)
    (root / "jit_cache").mkdir()
    _save_array(root / "source_order.npy", np.asarray(sources, dtype=np.int64))
    manifest = dict(status="running", supervisor_pid=os.getpid(), graph_sha256=meta["graph_sha256"],
        n=meta["final_nodes"], m=meta["final_edges"], core_nodes=qmeta["core_nodes"],
        core_edges=qmeta["core_edges"], source_count=source_count, seed=seed,
        source_order_hash=source_order_digest(sources), graph_full=True, scaled=False,
        exact_all_sources=source_count == meta["final_nodes"],
        result_kind="ordered_all_source_contribution" if source_count == meta["final_nodes"] else "source_subset_contribution",
        threads=threads, batch_size=batch_size, repeats=repeats, code_sha256=quotient_code_digest(),
        cpu_affinity_requested=cpu_affinity, min_available_bytes_reserved=min_available_bytes,
        max_case_seconds=max_case_seconds, max_total_seconds=max_total_seconds,
        environment=dict(python=platform.python_version(), os=platform.platform(), **versions),
        protocol="same original sources and output vertices; separate warmups; alternating variant order",
        warmups=[], cases=[], pairs=[])
    _atomic_json(root / "manifest.json", manifest)
    begin = time.perf_counter()
    stopped = False
    def run(label, backend, selected):
        nonlocal stopped
        output = root / label
        output.mkdir()
        budget = min(max_case_seconds, max_total_seconds - (time.perf_counter() - begin))
        if stopped:
            row = dict(status="resource_not_run", error="Earlier preparation or resource check failed")
        elif budget <= 0:
            row = dict(status="budget_not_run", error="Total budget exhausted")
        else:
            row = bounded_job("thread_subset" if backend == "original" else "quotient_subset",
                dict(cache_directory=str(Path(cache_directory if backend == "original" else quotient_directory).resolve()),
                    output=str(output.resolve()), sources=selected, threads=threads, batch_size=batch_size,
                    timeout_seconds=budget, cpu_affinity=cpu_affinity,
                    progress_path=str((output / "sources_completed.jsonl").resolve()),
                    jit_cache_directory=str((root / "jit_cache").resolve())),
                seconds=budget, min_available_bytes=min_available_bytes)
        stopped |= row["status"] in {"resource_limit", "resource_not_run"}
        row.update(variant=backend, output_directory=str(output.resolve()), label=label)
        if row["status"] == "ok":
            row["t_work"] = row["t_execute_reduce"] if backend == "original" else row["t_execute_restore"]
        _atomic_json(output / "case.json", row)
        print(f"{label}: {row['status']}; work={row.get('t_work', 0):.3f}s; "
              f"wall={row.get('t_supervised_total', 0):.3f}s", flush=True)
        return row
    try:
        for backend in ("original", "quotient"):
            row = run("warmup_" + backend, backend, sources[:1])
            manifest["warmups"].append(row)
            stopped |= row["status"] != "ok"
            _atomic_json(root / "manifest.json", manifest)
        for repeat in range(repeats):
            pair_rows = {}
            for backend in (("original", "quotient") if repeat % 2 == 0 else ("quotient", "original")):
                row = run(f"repeat{repeat+1}_{backend}", backend, sources)
                row["repeat"] = repeat + 1
                pair_rows[backend] = row
                manifest["cases"].append(row)
                _atomic_json(root / "manifest.json", manifest)
            a, b = pair_rows["original"], pair_rows["quotient"]
            pair = dict(repeat=repeat + 1, status="incomplete")
            if a["status"] == b["status"] == "ok":
                av = np.load(a["contribution_path"], mmap_mode="r", allow_pickle=False)
                bv = np.load(b["contribution_path"], mmap_mode="r", allow_pickle=False)
                check = compare_vectors(av, bv)
                coverage = (a["source_order_hash"] == b["source_order_hash"] == manifest["source_order_hash"]
                            and b["represented_source_count"] == source_count and b["restored_original_nodes"] == meta["final_nodes"])
                pair.update(status="ok" if check["allclose"] and coverage else "mismatch", validation=check,
                    coverage_valid=coverage, original_work_seconds=a["t_work"], quotient_work_seconds=b["t_work"],
                    speedup_work=a["t_work"] / b["t_work"], speedup_wall=a["t_supervised_total"] / b["t_supervised_total"],
                    original_wall_seconds=a["t_supervised_total"], quotient_wall_seconds=b["t_supervised_total"],
                    core_traversals=b["core_traversals"], fallback_sources=b["fallback_sources"],
                    core_execution_seconds=b["t_execute_reduce"], grouping_seconds=b["t_group_sources"],
                    restoration_seconds=b["t_restore"])
                del av, bv
            manifest["pairs"].append(pair)
            _atomic_json(root / "manifest.json", manifest)
    except BaseException:
        manifest["status"] = "interrupted"
        _atomic_json(root / "manifest.json", manifest)
        raise
    good = [p for p in manifest["pairs"] if p["status"] == "ok"]
    manifest["status"] = "complete" if len(good) == repeats else "partial"
    manifest["t_comparison_wall"] = time.perf_counter() - begin
    manifest["summary"] = {"successful_pairs": len(good)}
    if good:
        for key in ("original_work_seconds", "quotient_work_seconds", "speedup_work", "speedup_wall",
                    "original_wall_seconds", "quotient_wall_seconds", "core_execution_seconds",
                    "grouping_seconds", "restoration_seconds"):
            manifest["summary"]["median_" + key] = statistics.median(p[key] for p in good)
    _atomic_json(root / "manifest.json", manifest)
    lines = ["# Lossless tree-quotient comparison", "",
        f"Status: {manifest['status']}; original n={manifest['n']}, m={manifest['m']}; sources={source_count}.",
        f"Core n={manifest['core_nodes']}, m={manifest['core_edges']}; every original output vertex is restored.",
        "Work includes original execution+reduce, or quotient grouping+weighted execution+reduce+restoration.", "",
        "| Repeat | Status | Original work s | Quotient work s | Work speedup | Wall speedup | Max abs error |",
        "|---|---|---|---|---|---|---|"]
    for p in manifest["pairs"]:
        lines.append(f"| {p['repeat']} | {p['status']} | {p.get('original_work_seconds','')} | "
            f"{p.get('quotient_work_seconds','')} | {p.get('speedup_work','')} | {p.get('speedup_wall','')} | "
            f"{p.get('validation',{}).get('max_abs_error','')} |")
    lines += ["", "```json", json.dumps(manifest["summary"], indent=2), "```", "",
        "Wall includes validation of the lossless representation, including reconstructed original graph digest.",
        "One-time factorization/cache construction is separate; full-source time is not measured here.",
        "Endpoint vertex masses are not edge distances. No original nodes or pairs are discarded."]
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root
