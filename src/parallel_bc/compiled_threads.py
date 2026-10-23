"""Source-parallel native Brandes with shared immutable CSR and private totals."""
from __future__ import annotations

import json
import operator
import os
from pathlib import Path
import queue
import threading
import time

import numpy as np

from .brandes import source_order_digest


def threaded_contributions(graph, sources, *, threads=1, batch_size=4,
                           timeout_seconds=None, progress_path=None):
    """One private accumulator/workspace per worker; reduce only after success.

    Caller must supply a verified CSRGraph. The supervisor must enforce a hard
    process deadline, because native threads only check deadlines between sources.
    threads=1 runs inline, providing a same-driver single-thread reference.
    """
    started = time.perf_counter()
    try:
        sources = tuple(operator.index(s) for s in sources)
        threads, batch_size = operator.index(threads), operator.index(batch_size)
    except TypeError as exc:
        raise ValueError("Integer sources, threads and batch size required") from exc
    if (not sources or len(set(sources)) != len(sources) or min(sources) < 0
            or max(sources) >= graph.n or threads < 1 or batch_size < 1):
        raise ValueError("Invalid sources, threads or batch size")
    if timeout_seconds is not None and (not np.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("timeout_seconds must be positive and finite")
    deadline = None if timeout_seconds is None else started + timeout_seconds
    def check_deadline():
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError("Compiled thread source budget exceeded")
    import_started = time.perf_counter()
    from .compiled import CompiledAccumulator, warmup
    import_seconds = time.perf_counter() - import_started
    jit = warmup()  # Compile/load before creating threads; avoid concurrent JIT.
    check_deadline()
    batches = [(i // batch_size, sources[i:i + batch_size]) for i in range(0, len(sources), batch_size)]
    p = min(threads, len(batches))
    workspace_started = time.perf_counter()
    runners = [CompiledAccumulator(graph) for _ in range(p)]
    totals = [np.zeros(graph.n, dtype=np.float64) for _ in range(p)]
    workspace_seconds = time.perf_counter() - workspace_started
    # Reserve each worker a first batch; remaining batches are dynamic. Even
    # tiny fixtures exercise every declared active worker, with no shared total.
    pending = queue.Queue()
    for batch in batches[p:]:
        pending.put(batch)
    ordinals = {s: i for i, s in enumerate(sources)}
    worker_tasks = [[] for _ in range(p)]
    worker_rows = [None] * p
    cancelled = threading.Event()
    lock = threading.Lock()
    errors = []
    barrier = threading.Barrier(p) if p > 1 else None
    stream = Path(progress_path).open("w", encoding="utf-8") if progress_path else None
    def worker(worker_id):
        own = worker_tasks[worker_id]
        cpu_started = None
        first = None
        native_id = threading.get_native_id()
        try:
            if barrier is not None:
                remaining = 30. if deadline is None else max(.001, min(30., deadline - time.perf_counter()))
                barrier.wait(timeout=remaining)
            cpu_started = time.thread_time()
            batch = batches[worker_id]
            while not cancelled.is_set():
                batch_id, selected = batch
                for source in selected:
                    if cancelled.is_set():
                        break
                    check_deadline()
                    source_started = time.perf_counter()
                    if first is None:
                        first = source_started
                    execution = runners[worker_id].accumulate(source, totals[worker_id])
                    source_ended = time.perf_counter()
                    row = dict(worker_id=worker_id, native_thread_id=native_id, batch_id=batch_id,
                        source_index=source, original_id=int(graph.ids[source]), source_ordinal=ordinals[source],
                        start_rel_seconds=source_started - started, end_rel_seconds=source_ended - started,
                        compute_seconds=source_ended - source_started, **execution)
                    own.append(row)
                    if stream is not None:
                        with lock:
                            stream.write(json.dumps(row, allow_nan=False) + "\n")
                            stream.flush()
                    check_deadline()
                try:
                    batch = pending.get_nowait()
                except queue.Empty:
                    break
        except BaseException as exc:
            if isinstance(exc, threading.BrokenBarrierError):
                exc = (TimeoutError("Compiled thread startup exceeded budget")
                       if deadline is not None and time.perf_counter() >= deadline
                       else RuntimeError("Compiled thread startup barrier failed"))
            with lock:
                errors.append(exc)
            cancelled.set()
            if barrier is not None:
                barrier.abort()
        finally:
            worker_rows[worker_id] = dict(worker_id=worker_id, native_thread_id=native_id,
                pid=os.getpid(), processed_sources=len(own),
                processed_batches=len({r["batch_id"] for r in own}),
                busy_seconds=sum(r["compute_seconds"] for r in own),
                cpu_seconds=0. if cpu_started is None else time.thread_time() - cpu_started,
                first_source_start=None if first is None else first - started,
                last_source_end=None if not own else own[-1]["end_rel_seconds"],
                source_order_hash=source_order_digest([r["source_index"] for r in own]))
    execute_started = time.perf_counter()
    launched = []
    try:
        check_deadline()
        if p == 1:
            worker(0)
        else:
            try:
                for i in range(p):
                    thread = threading.Thread(target=worker, args=(i,), name=f"LinkRadar-native-{i}")
                    thread.start()
                    launched.append(thread)
                for thread in launched:
                    thread.join()
            finally:
                cancelled.set()
                barrier.abort()
                for thread in launched:
                    thread.join()
        if errors:
            raise errors[0]
        check_deadline()
    finally:
        if stream is not None:
            stream.close()
    execute_ended = time.perf_counter()
    tasks = sorted((r for own in worker_tasks for r in own), key=lambda r: r["source_ordinal"])
    if [r["source_index"] for r in tasks] != list(sources):
        raise RuntimeError("Threaded execution did not cover exactly the requested sources")
    reduce_started = time.perf_counter()
    # Fixed worker reduction order. Dynamic batch ownership can change floating
    # grouping between runs, so correctness is tolerance-based, not bitwise.
    total = totals[0]
    for other in totals[1:]:
        total += other
    if not np.isfinite(total).all():
        raise FloatingPointError("Nonfinite threaded contribution vector")
    ended = time.perf_counter()
    check_deadline()
    cpu_seconds = sum(w["cpu_seconds"] for w in worker_rows)
    return total, dict(backend="numba_threads", n=graph.n, m=graph.m,
        graph_sha256=graph.graph_sha256, graph_full=True, source_count=len(sources),
        source_order_hash=source_order_digest(sources), exact_all_sources=len(sources) == graph.n,
        result_kind="ordered_all_source_contribution" if len(sources) == graph.n else "source_subset_contribution",
        scaled=False, threads_requested=threads, p_used=p,
        active_workers=sum(w["processed_sources"] > 0 for w in worker_rows),
        batch_size=batch_size, scheduler="reserved first batch then dynamic queue",
        graph_shared_readonly=True, tasks=tasks, workers=worker_rows,
        fallback_sources=sum(r["execution"] == "python_bigint_fallback" for r in tasks),
        t_backend_import=import_seconds, **jit, t_workspace=workspace_seconds,
        workspace_bytes=sum(r.workspace.nbytes for r in runners),
        local_total_bytes=sum(a.nbytes for a in totals),
        t_execute_wall=execute_ended - execute_started, t_reduce=ended - reduce_started,
        t_execute_reduce=ended - execute_started, t_worker_busy_sum=sum(w["busy_seconds"] for w in worker_rows),
        worker_cpu_seconds=cpu_seconds, average_worker_cpu_cores=cpu_seconds / (execute_ended - execute_started),
        t_compute_driver=ended - started)
