"""Spawn-based static round-robin and shared-queue Brandes schedulers."""
from __future__ import annotations

import multiprocessing as mp
import operator
import os
import queue
import time
import traceback
from typing import Iterable

import numpy as np

from .brandes import (
    Adjacency, CentralityResult, check_deadline, finalize_scores, make_batches,
    single_source_accumulate, source_order_digest, timeout_deadline, validate_source_order,
)


def available_cpu_count() -> int:
    """Honor CPU affinity / interpreter allocation where the platform exposes it."""
    counts = [os.cpu_count() or 1]
    process_count = getattr(os, "process_cpu_count", None)
    if process_count is not None:
        counts.append(process_count() or 1)
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        try:
            counts.append(len(affinity(0)))
        except OSError:
            pass
    # psutil supplies Windows processor-affinity information. It is optional to
    # importing this module; the core also works without that diagnostic package.
    try:
        import psutil
    except ImportError:
        pass
    else:
        try:
            counts.append(len(psutil.Process().cpu_affinity()))
        except (AttributeError, psutil.Error, OSError):
            pass
    return max(1, min(count for count in counts if count > 0))


class ParallelWorkerError(RuntimeError):
    """A child process failed; all children have been reclaimed before raising."""


def _worker(worker_id, adj, task_queue, event_queue, run_start, task_logging):
    """One local vector per process; only DONE transmits that vector."""
    pid = os.getpid()
    try:
        local = np.zeros(len(adj), dtype=np.float64)
        event_queue.put(("READY", worker_id, pid))
        processed_sources = 0
        batch_ids = []
        tasks = []
        busy = 0.0
        first_start = None
        last_end = None
        while True:
            task = task_queue.get()
            if task is None:
                break
            batch_id, sources = task
            start = time.perf_counter()
            if first_start is None:
                first_start = start - run_start
            for source in sources:
                single_source_accumulate(adj, source, local)
            end = time.perf_counter()
            busy += end - start
            last_end = end - run_start
            processed_sources += len(sources)
            batch_ids.append(batch_id)
            if task_logging:
                tasks.append({"batch_id": batch_id, "worker_id": worker_id, "pid": pid,
                    "source_count": len(sources), "source_order_hash": source_order_digest(sources),
                    "start_rel_seconds": start - run_start, "end_rel_seconds": last_end,
                    "compute_seconds": end - start, "status": "ok"})
        summary = {"worker_id": worker_id, "pid": pid, "processed_sources": processed_sources,
            "processed_batches": len(batch_ids), "batch_ids": batch_ids, "busy_seconds": busy,
            "first_task_start": first_start, "last_task_end": last_end}
        event_queue.put(("DONE", worker_id, pid, local, summary, tasks))
    except BaseException:
        # The parent consumes this channel while workers are running, before join.
        event_queue.put(("ERROR", worker_id, pid, traceback.format_exc()))


def _reclaim(processes, queues, failed):
    """Bounded cleanup, including partially started workers and queue feeders."""
    if failed:
        for process in processes:
            if process.pid is not None and process.is_alive():
                process.terminate()
    for process in processes:
        if process.pid is None:
            continue
        process.join(timeout=2.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=2.0)
    for channel in queues:
        if failed:
            channel.cancel_join_thread()
        channel.close()
        if not failed:
            channel.join_thread()
    for process in processes:
        if process.pid is not None:
            process.close()


def brandes_parallel(
    adj: Adjacency, processes: int = 2, mode: str = "dynamic_batches",
    batch_size: int = 32, source_order: Iterable[int] | None = None,
    task_logging: bool = False, timeout_seconds: float | None = 120,
    max_processes: int | None = None,
) -> CentralityResult:
    """Compute exact centrality using fresh spawned processes on every call.

    Static batches use dedicated queues, assigned by batch_id % p_used. Dynamic
    batches use one shared queue. The source batches and single-source kernel are
    identical. A parent deadline covers startup, dispatch, computation and reduce.
    """
    run_start = time.perf_counter()
    deadline = timeout_deadline(run_start, timeout_seconds)
    try:
        p_requested = operator.index(processes)
        limit = operator.index(max_processes) if max_processes is not None else (os.cpu_count() or 1)
    except TypeError as exc:
        raise ValueError("processes and max_processes must be positive integers") from exc
    if p_requested < 1 or limit < 1:
        raise ValueError("processes and max_processes must be positive integers")
    if mode not in ("static_batches", "dynamic_batches"):
        raise ValueError("mode must be static_batches or dynamic_batches")
    n = len(adj)
    order = validate_source_order(n, source_order)
    batches = make_batches(order, batch_size)
    order_hash = source_order_digest(order)
    p_used = min(p_requested, limit, available_cpu_count(), n)
    total = np.zeros(n, dtype=np.float64)
    if n == 0:
        execute_end = time.perf_counter()
        raw, normalized = finalize_scores(total, n)
        end = time.perf_counter()
        return CentralityResult(raw=raw, normalized=normalized,
            timings={"t_startup": 0.0, "t_execute_reduce": 0.0,
                     "t_finalize": end - execute_end, "t_cleanup": 0.0,
                     "t_total": end - run_start, "t_steady": end - execute_end},
            source_order_hash=order_hash, p_requested=p_requested, p_used=0,
            active_workers=0, mode=mode, batch_size=batch_size, start_method="spawn",
            task_logging=task_logging)
    ctx = mp.get_context("spawn")
    child_processes = []
    all_queues = []
    workers = []
    tasks = []
    ready = {}
    done = set()
    seen_batches = []
    failed = True
    startup_start = time.perf_counter()
    try:
        events = ctx.Queue()
        all_queues.append(events)
        input_queues = [ctx.Queue() for _ in range(p_used if mode == "static_batches" else 1)]
        all_queues.extend(input_queues)
        for worker_id in range(p_used):
            check_deadline(deadline)
            target_queue = input_queues[worker_id] if mode == "static_batches" else input_queues[0]
            process = ctx.Process(target=_worker,
                args=(worker_id, adj, target_queue, events, run_start, task_logging),
                name=f"LinkRadar-{mode}-{worker_id}")
            child_processes.append(process)
            process.start()

        exited_without_result = {}

        def next_event():
            while True:
                check_deadline(deadline)
                try:
                    event = events.get(timeout=0.05)
                except queue.Empty:
                    for worker_id, process in enumerate(child_processes):
                        if process.exitcode not in (None, 0):
                            raise ParallelWorkerError(
                                f"worker PID {process.pid} exited with code {process.exitcode}")
                        if process.exitcode == 0 and worker_id not in done:
                            # Queue feeders are flushed before a clean child exit;
                            # allow a short transport grace period before failing.
                            first_seen = exited_without_result.setdefault(worker_id, time.perf_counter())
                            if time.perf_counter() - first_seen > 0.5:
                                raise ParallelWorkerError(f"worker PID {process.pid} exited without DONE")
                    continue
                if event[0] == "ERROR":
                    raise ParallelWorkerError(f"worker {event[1]} PID {event[2]} failed:\n{event[3]}")
                return event

        while len(ready) < p_used:
            event = next_event()
            if event[0] != "READY":
                raise ParallelWorkerError("received a non-READY event before dispatch")
            _, worker_id, pid = event
            if worker_id in ready or pid in ready.values() or child_processes[worker_id].pid != pid:
                raise ParallelWorkerError("invalid or duplicate READY worker identity")
            ready[worker_id] = pid
        startup_end = time.perf_counter()
        execute_start = startup_end
        for batch_id, sources in batches:
            check_deadline(deadline)
            target = input_queues[batch_id % p_used] if mode == "static_batches" else input_queues[0]
            target.put((batch_id, sources))
        if mode == "static_batches":
            for channel in input_queues:
                channel.put(None)
        else:
            for _ in range(p_used):
                input_queues[0].put(None)
        while len(done) < p_used:
            event = next_event()
            if event[0] != "DONE":
                raise ParallelWorkerError("unexpected worker event after dispatch")
            _, worker_id, pid, local, summary, worker_tasks = event
            if worker_id in done or ready.get(worker_id) != pid:
                raise ParallelWorkerError("invalid or duplicate DONE worker identity")
            if np.asarray(local).shape != (n,) or not np.isfinite(local).all():
                raise ParallelWorkerError("worker returned an invalid local score vector")
            batch_ids = summary["batch_ids"]
            if any(batch_id < 0 or batch_id >= len(batches) for batch_id in batch_ids):
                raise ParallelWorkerError("worker returned an unknown batch")
            if mode == "static_batches" and any(batch_id % p_used != worker_id for batch_id in batch_ids):
                raise ParallelWorkerError("static batch executed by the wrong worker")
            expected_sources = sum(len(batches[batch_id][1]) for batch_id in batch_ids)
            if expected_sources != summary["processed_sources"]:
                raise ParallelWorkerError("worker source count does not match batches")
            total += local
            seen_batches.extend(batch_ids)
            workers.append(summary)
            tasks.extend(worker_tasks)
            done.add(worker_id)
        if sorted(seen_batches) != list(range(len(batches))):
            raise ParallelWorkerError("batch coverage has duplicates or missing sources")
        if sum(w["processed_sources"] for w in workers) != n:
            raise ParallelWorkerError("not every source was processed exactly once")
        execute_end = time.perf_counter()
        raw, normalized = finalize_scores(total, n)
        finalize_end = time.perf_counter()
        failed = False
    finally:
        cleanup_start = time.perf_counter()
        _reclaim(child_processes, all_queues, failed)
        cleanup_end = time.perf_counter()
    timings = {"t_startup": startup_end - startup_start,
               "t_execute_reduce": execute_end - execute_start,
               "t_finalize": finalize_end - execute_end,
               "t_cleanup": cleanup_end - cleanup_start,
               "t_total": cleanup_end - run_start,
               "t_steady": finalize_end - execute_start}
    workers.sort(key=lambda item: item["worker_id"])
    tasks.sort(key=lambda item: (item["start_rel_seconds"], item["worker_id"]))
    return CentralityResult(raw=raw, normalized=normalized, timings=timings,
        workers=workers, tasks=tasks, source_order_hash=order_hash,
        p_requested=p_requested, p_used=p_used,
        active_workers=sum(w["processed_sources"] > 0 for w in workers),
        mode=mode, batch_size=batch_size, start_method="spawn", task_logging=task_logging)

