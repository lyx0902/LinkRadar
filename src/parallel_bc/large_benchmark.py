"""Bounded single-worker CSR preparation and fixed-source pilot.

No sampled estimates or normalized centrality are produced. Optional compiled
sources detect int64 overflow and retry with the original arbitrary-int kernel.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import operator
import os
from pathlib import Path
import time
import traceback

import numpy as np
import psutil

from .brandes import single_source_accumulate, source_order_digest
from .csr import load_csr, prepare_csr
from .data import _atomic_json


def _save_array(path, values):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
    temporary.replace(path)


def subset_contributions(graph, sources, *, backend="python_csr", timeout_seconds=None,
                         progress_path=None):
    """Keep the full-source contract in brandes.py untouched."""
    if backend not in {"python_csr", "python_tuple", "numba_csr"}:
        raise ValueError("Unsupported pilot backend")
    try:
        sources = tuple(operator.index(value) for value in sources)
    except TypeError as exc:
        raise ValueError("sources must contain integer indices") from exc
    if (not sources or len(set(sources)) != len(sources)
            or min(sources) < 0 or max(sources) >= graph.n):
        raise ValueError("sources must be nonempty, unique and in range")
    if timeout_seconds is not None and (not np.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("timeout_seconds must be positive and finite")
    start = time.perf_counter()
    adjacency = graph.adj
    if backend == "python_tuple":
        adjacency = tuple(tuple(map(int, adjacency[i])) for i in range(graph.n))
    adapter_end = time.perf_counter()
    setup = dict(t_backend_import=0.0, t_jit_warmup=0.0, t_workspace=0.0,
                 workspace_bytes=0, native_nopython=False)
    compiled = None
    if backend == "numba_csr":
        import_start = time.perf_counter()
        from .compiled import CompiledAccumulator, warmup
        setup["t_backend_import"] = time.perf_counter() - import_start
        setup.update(warmup())
        workspace_start = time.perf_counter()
        compiled = CompiledAccumulator(graph)
        setup["t_workspace"] = time.perf_counter() - workspace_start
        setup["workspace_bytes"] = compiled.workspace.nbytes
    total = np.zeros(graph.n, dtype=np.float64)
    tasks = []
    stream = Path(progress_path).open("w", encoding="utf-8") if progress_path else None
    try:
        for index, source in enumerate(sources):
            if timeout_seconds is not None and time.perf_counter() - start >= timeout_seconds:
                raise TimeoutError("Source-subset budget exceeded")
            source_start = time.perf_counter()
            if compiled is None:
                single_source_accumulate(adjacency, source, total)
                execution = dict(execution="python_reference")
            else:
                execution = compiled.accumulate(source, total)
            now = time.perf_counter()
            row = dict(source_index=source, original_id=int(graph.ids[source]),
                       source_ordinal=index, compute_seconds=now - source_start,
                       completed_at_rel_seconds=now - start, **execution)
            tasks.append(row)
            if stream is not None:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
            if timeout_seconds is not None and now - start > timeout_seconds:
                raise TimeoutError("Source-subset budget exceeded")
    finally:
        if stream is not None:
            stream.close()
    if not np.isfinite(total).all():
        raise FloatingPointError("Nonfinite source contributions")
    end = time.perf_counter()
    return total, dict(backend=backend, graph_sha256=graph.graph_sha256,
        n=graph.n, m=graph.m, graph_full=True, source_count=len(sources),
        source_order_hash=source_order_digest(sources), exact_all_sources=len(sources) == graph.n,
        result_kind="ordered_all_source_contribution" if len(sources) == graph.n else "source_subset_contribution",
        scaled=False, p_used=1, tasks=tasks, **setup,
        fallback_sources=sum(t["execution"] == "python_bigint_fallback" for t in tasks),
        t_adapter=adapter_end - start, t_kernel=sum(t["compute_seconds"] for t in tasks),
        t_compute_driver=end - start, value_min=float(total.min()), value_max=float(total.max()))


def _job_child(conn, kind, params):
    try:
        start, cpu_start = time.perf_counter(), time.process_time()
        if kind == "prepare":
            output = prepare_csr(**params)
            value = dict(cache_directory=str(output.resolve()))
        elif kind in {"subset", "thread_subset"}:
            options = dict(params)
            affinity = options.pop("cpu_affinity", None)
            if affinity is not None:
                psutil.Process().cpu_affinity(affinity)
            jit_cache = options.pop("jit_cache_directory", None)
            if jit_cache is not None:
                # Set before importing the optional Numba module in this child.
                cache_path = str(Path(jit_cache).resolve())
                if os.name == "nt" and not cache_path.startswith("\\\\?\\"):
                    # Numba appends a module hash and atomic-write suffix. Temp
                    # directories can exceed MAX_PATH even with a short root.
                    cache_path = ("\\\\?\\UNC\\" + cache_path[2:] if cache_path.startswith("\\\\")
                                  else "\\\\?\\" + cache_path)
                os.environ["NUMBA_CACHE_DIR"] = cache_path
            cache = options.pop("cache_directory")
            output = Path(options.pop("output"))
            graph = load_csr(cache)
            loaded = time.perf_counter()
            source_file = output / "sources.json"
            selected = dict(indices=options["sources"],
                            original_ids=[int(graph.ids[i]) for i in options["sources"]],
                            source_order_hash=source_order_digest(options["sources"]))
            _atomic_json(source_file, selected)
            if kind == "thread_subset":
                from .compiled_threads import threaded_contributions
                total, value = threaded_contributions(graph, **options)
            else:
                total, value = subset_contributions(graph, **options)
            _save_array(output / "contributions.npy", total)
            value.update(t_load_validate=loaded - start,
                         cpu_affinity_used=psutil.Process().cpu_affinity() if hasattr(psutil.Process(), "cpu_affinity") else None,
                         contribution_path=str((output / "contributions.npy").resolve()))
        else:
            raise ValueError(kind)
        value.update(t_child_total=time.perf_counter() - start,
                     cpu_seconds=time.process_time() - cpu_start)
        conn.send(("ok", value))
    except BaseException as exc:
        conn.send(("timeout" if isinstance(exc, TimeoutError) else
                   "resource_limit" if isinstance(exc, MemoryError) else "failed",
                   {"error": traceback.format_exc()}))
    finally:
        conn.close()


def bounded_job(kind, params, *, seconds, min_available_bytes=0):
    """Parent enforces wall time and system reserve, including spawn and I/O."""
    if not np.isfinite(seconds) or seconds <= 0 or min_available_bytes < 0:
        raise ValueError("Invalid job resource budget")
    started = time.perf_counter()
    available_min = psutil.virtual_memory().available
    if available_min < min_available_bytes:
        return dict(status="resource_not_run", error="Available memory below reserve before spawn",
                    t_supervised_total=0., min_available_bytes=available_min)
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_job_child, args=(child, kind, params), name=f"LinkRadar-{kind}-pilot")
    peak_rss = 0
    peak_private = None
    result = dict(status="failed", error="No child result")
    try:
        proc.start()
        child.close()
        watched = psutil.Process(proc.pid)
        while True:
            elapsed = time.perf_counter() - started
            available_min = min(available_min, psutil.virtual_memory().available)
            try:
                mem = watched.memory_info()
                peak_rss = max(peak_rss, mem.rss, getattr(mem, "peak_wset", mem.rss))
                private = getattr(mem, "private", None)
                if private is not None:
                    peak_private = max(peak_private or 0, private)
            except psutil.Error:
                pass
            if elapsed >= seconds:
                result = dict(status="timeout", error=f"Hard wall budget {seconds}s exceeded")
                break
            if available_min < min_available_bytes:
                result = dict(status="resource_limit", error="System available memory fell below reserve")
                break
            if parent.poll(.05):
                try:
                    status, value = parent.recv()
                except EOFError:
                    result = dict(status="failed", error=f"Child pipe closed; exitcode={proc.exitcode}")
                    break
                proc.join(timeout=min(2., max(.001, seconds - elapsed)))
                if proc.is_alive():
                    result = dict(status="timeout", error="Child cleanup exceeded budget")
                else:
                    result = dict(status=status, **value)
                break
            if not proc.is_alive():
                result = dict(status="failed", error=f"Child exited without result: {proc.exitcode}")
                break
    finally:
        if proc.pid is not None and proc.is_alive():
            proc.terminate()
            proc.join(timeout=2.)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=2.)
        parent.close()
        child.close()
        if proc.pid is not None:
            proc.join(timeout=.1)
        proc.close()
    result.update(t_supervised_total=time.perf_counter() - started,
                  child_peak_rss_bytes=peak_rss, child_peak_private_bytes_sampled=peak_private,
                  min_available_bytes=available_min, memory_sample_interval_seconds=.05)
    return result


def run_preparation(path, directory, *, dataset_id, file_format="snap", provenance_path=None,
                    timeout_seconds=300, min_available_bytes=0):
    params = dict(path=str(path), directory=str(directory), dataset_id=dataset_id,
                  file_format=file_format, provenance_path=str(provenance_path) if provenance_path else None,
                  timeout_seconds=timeout_seconds, min_available_bytes=min_available_bytes)
    row = bounded_job("prepare", params, seconds=timeout_seconds,
                      min_available_bytes=min_available_bytes)
    row.update(kind="csr_preparation", parameters=params)
    _atomic_json(Path(directory).with_suffix(".preparation.json"), row)
    return row


def _code_digest():
    digest = hashlib.sha256()
    for name in ("csr.py", "large_benchmark.py", "brandes.py", "compiled.py"):
        digest.update(name.encode())
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def run_pilot(cache_directory, output_root, *, counts=(1, 4, 8), backend="python_tuple",
              seed=42, max_case_seconds=120, max_total_seconds=1800, min_available_bytes=0):
    if backend not in {"python_csr", "python_tuple", "numba_csr"}:
        raise ValueError("Unsupported pilot backend")
    if (not np.isfinite(max_case_seconds) or not np.isfinite(max_total_seconds)
            or max_case_seconds <= 0 or max_total_seconds <= 0 or min_available_bytes < 0):
        raise ValueError("Invalid pilot resource budget")
    counts = tuple(operator.index(count) for count in counts)
    if not counts or counts != tuple(sorted(set(counts))) or counts[0] < 1:
        raise ValueError("counts must be increasing unique positive integers")
    # Only metadata needed in parent. Child always checks file hashes, CSR and graph ID.
    meta = json.loads((Path(cache_directory) / "metadata.json").read_text(encoding="utf-8"))
    if meta.get("cache_format") != "LinkRadar-csr-v1" or counts[-1] > meta["final_nodes"]:
        raise ValueError("Invalid CSR metadata or source count exceeds n")
    order = np.random.default_rng(seed).permutation(meta["final_nodes"])[:counts[-1]]
    root = Path(output_root) / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ_") + "youtube_pilot")
    root.mkdir(parents=True, exist_ok=False)
    _save_array(root / "source_order.npy", order)
    import platform
    from importlib.metadata import version
    manifest = dict(status="running", cache_directory=str(Path(cache_directory).resolve()),
        graph_sha256=meta["graph_sha256"], n=meta["final_nodes"], m=meta["final_edges"],
        graph_full=True, supervisor_pid=os.getpid(), backend=backend, seed=seed, source_order_hash=source_order_digest(order),
        code_sha256=_code_digest(), counts=counts, max_case_seconds=max_case_seconds,
        max_total_seconds=max_total_seconds, min_available_bytes_reserved=min_available_bytes,
        environment=dict(python=platform.python_version(), os=platform.platform(),
                         numpy=version("numpy"), psutil=version("psutil"),
                         cpu_model=platform.processor(), logical_cpus=psutil.cpu_count(),
                         cpu_affinity=psutil.Process().cpu_affinity() if hasattr(psutil.Process(), "cpu_affinity") else None),
        cases=[], notes=["Pilot timings are single observations, not a speedup study.",
                         "Partial-source contributions are not full-graph exact centrality.",
                         "t_kernel excludes load/validation/tuple-adapter and spawn.",
                         "Progress logging enabled identically for all pilot cases."])
    _atomic_json(root / "manifest.json", manifest)
    begin = time.perf_counter()
    stop_for_resources = False
    try:
        for k in counts:
            output = root / f"k{k}"
            output.mkdir()
            sources = [int(x) for x in order[:k]]
            _atomic_json(output / "sources.json", dict(indices=sources,
                original_ids=None, note="original IDs populated from verified cache in child",
                source_order_hash=source_order_digest(sources)))
            seconds = min(max_case_seconds, max_total_seconds - (time.perf_counter() - begin))
            if stop_for_resources:
                row = dict(status="resource_not_run", error="Earlier case reached the system reserve; remaining cases skipped")
            elif seconds <= 0:
                row = dict(status="budget_not_run", error="Total pilot budget exhausted")
            else:
                row = bounded_job("subset", dict(cache_directory=str(Path(cache_directory).resolve()),
                    output=str(output.resolve()), sources=sources, backend=backend,
                    timeout_seconds=seconds, progress_path=str((output / "sources_completed.jsonl").resolve())),
                    seconds=seconds, min_available_bytes=min_available_bytes)
            if row["status"] in {"resource_limit", "resource_not_run"}:
                stop_for_resources = True
            progress = output / "sources_completed.jsonl"
            completed = [json.loads(line) for line in progress.read_text(encoding="utf-8").splitlines()] if progress.exists() else []
            source_file = json.loads((output / "sources.json").read_text(encoding="utf-8"))
            # Completed progress is evidence for IDs, but it does not stand in for an unfinished vector.
            source_file["completed_original_ids"] = [item["original_id"] for item in completed]
            _atomic_json(output / "sources.json", source_file)
            row.update(k_requested=k, completed_sources=len(completed),
                       source_order_hash=source_order_digest(sources), graph_full=True,
                       exact_all_sources=k == meta["final_nodes"],
                       result_kind="ordered_all_source_contribution" if k == meta["final_nodes"] else "source_subset_contribution",
                       scaled=False)
            _atomic_json(output / "case.json", row)
            manifest["cases"].append(row)
            _atomic_json(root / "manifest.json", manifest)
            print(f"k={k}: {row['status']}; completed={len(completed)}; "
                  f"wall={row.get('t_supervised_total', 0):.3f}s", flush=True)
    except BaseException:
        manifest["status"] = "interrupted"
        _atomic_json(root / "manifest.json", manifest)
        raise
    manifest["status"] = "complete" if all(row["status"] == "ok" for row in manifest["cases"]) else "partial"
    manifest["t_pilot_wall"] = time.perf_counter() - begin
    _atomic_json(root / "manifest.json", manifest)
    lines = ["# YouTube fixed-source pilot", "",
             f"Status: {manifest['status']}; full graph n={manifest['n']}, m={manifest['m']}.",
             f"Backend: {backend}. No full-source centrality or normalized estimates.", "",
             "| k | Status | Completed | Kernel s | JIT warmup/cache s | Load/validate s | Supervised wall s | Child peak RSS MiB |",
             "|---|---|---|---|---|---|---|---|"]
    for row in manifest["cases"]:
        lines.append(f"| {row['k_requested']} | {row['status']} | {row['completed_sources']} | "
                     f"{row.get('t_kernel', '')} | {row.get('t_jit_warmup', '')} | {row.get('t_load_validate', '')} | "
                     f"{row.get('t_supervised_total', '')} | "
                     f"{row.get('child_peak_rss_bytes', 0) / 2**20:.2f} |")
    lines += ["", "Each k is one observation. Peak RSS includes child imports, cache validation,",
              "adapter construction, kernel and output; it is not kernel-only memory.",
              "Unfinished cases must not be treated as complete contribution vectors."]
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root
