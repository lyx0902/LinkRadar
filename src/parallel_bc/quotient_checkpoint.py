"""Bounded-memory, restartable all-source weighted-core execution.

Only a manifest commit makes a chunk visible. Orphan files are overwritten on
retry; committed files are immutable and revalidated before reuse. The OS lock
is released on process death. Atomic replacement + fsync protect process-crash
recovery; storage/controller guarantees still govern sudden power loss.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import operator
import os
from pathlib import Path
import platform
import time

import numpy as np
import psutil

from .brandes import finalize_scores, source_order_digest
from .compiled_threads import threaded_contributions
from .data import _sha256
from .quotient_brandes import WeightedAccumulator, _restore, _warmup_recovery, weighted_warmup


FORMAT = "LinkRadar-quotient-checkpoint-v1"


def checkpoint_code_digest():
    digest = hashlib.sha256()
    for name in ("quotient_checkpoint.py", "quotient_brandes.py", "compiled_threads.py",
                 "compiled.py", "tree_quotient.py", "csr.py", "brandes.py"):
        digest.update(name.encode())
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def _write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _write_array(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


@contextmanager
def _run_lock(root):
    """Nonblocking OS lock; a stale lock file does not prevent recovery."""
    with (root / "run.lock").open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Checkpoint run is locked by another process") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _full_restore(q, core_total):
    # Full-source selected subtree counts are already known from factorization.
    # No tuple/set/dict of all original sources is constructed.
    mass = q.mass
    mass.flags.writeable = False
    return _restore(q.parent, q.peel_order, q.anchor_nodes, q.anchor_index,
        q.subtree_size, q.component_index, q.component_sizes,
        np.ones(q.n, np.int8), q.subtree_size.copy(), mass,
        q.component_sizes.copy(), core_total)


def _chunk_name(index):
    return f"chunk_{index:06d}.npy"


def checkpoint_contributions(q, run_directory, *, resume=False, threads=4, batch_size=4,
                             chunk_size=1024, seed=45, max_chunks=None,
                             timeout_seconds=None, _fault=None):
    """Run/pause the full-source plan. Only complete runs publish original BC.

    max_chunks bounds NEW chunks this invocation, never changes the global
    source/target masses. Core-only partial totals are not original BC scores.
    The private fault hook is for transaction-boundary recovery tests.
    """
    started = time.perf_counter()
    threads, batch_size, chunk_size, seed = map(operator.index, (threads, batch_size, chunk_size, seed))
    if min(threads, batch_size, chunk_size) < 1 or seed < 0:
        raise ValueError("Invalid checkpoint configuration")
    if max_chunks is not None:
        max_chunks = operator.index(max_chunks)
        if max_chunks < 0:
            raise ValueError("max_chunks must be nonnegative")
    if timeout_seconds is not None and (not np.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("timeout_seconds must be positive and finite")
    root = Path(run_directory)
    if not resume:
        root.mkdir(parents=True, exist_ok=False)
    elif not (root / "manifest.json").is_file():
        raise ValueError("No committed checkpoint manifest to resume")
    with _run_lock(root):
        return _execute(q, root, resume, threads, batch_size, chunk_size, seed, max_chunks,
                        timeout_seconds, started, _fault)


def _execute(q, root, resume, threads, batch_size, chunk_size, seed, max_chunks,
             timeout_seconds, started, fault):
    def check_time():
        if timeout_seconds is not None and time.perf_counter() - started >= timeout_seconds:
            raise TimeoutError("Checkpoint session budget exceeded; committed chunks retained")

    mass = q.mass
    mass.flags.writeable = False
    roots = np.flatnonzero(np.diff(q.core.indptr) > 0).astype(np.int32)
    plan = np.random.default_rng(seed).permutation(roots)
    identity = dict(graph_sha256=q.source_graph_sha256, core_graph_sha256=q.core.graph_sha256,
        original_nodes=q.n, original_edges=q.metadata["original_edges"],
        core_nodes=q.core.n, core_edges=q.core.m,
        quotient_arrays_sha256={name: hashlib.sha256(np.ascontiguousarray(getattr(q, name)).tobytes()).hexdigest()
            for name in ("original_ids", "parent", "peel_order", "anchor_index", "anchor_nodes",
                         "subtree_size", "component_index", "component_sizes")},
        code_sha256=checkpoint_code_digest(),
        environment={name: version(name) for name in ("numpy", "numba", "llvmlite")},
        python=platform.python_version(), cpu_affinity=psutil.Process().cpu_affinity(),
        threads=threads, batch_size=batch_size, chunk_size=chunk_size, seed=seed,
        plan_hash=source_order_digest(plan), core_traversals=len(plan),
        source_mass_sum=int(mass.sum()), dtype="float64", fastmath=False,
        semantics="all original sources; source_count=target_mass=block_size; restore once")
    if resume:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format") != FORMAT or manifest.get("identity") != identity:
            raise ValueError("Checkpoint identity mismatch: graph, code, dependencies or configuration changed")
        if (manifest.get("status") not in {"paused", "complete"}
                or manifest.get("exact_all_sources") != (manifest["status"] == "complete")
                or manifest.get("graph_full") is not True or manifest.get("scaled") is not False
                or manifest.get("result_kind") != ("exact_centrality" if manifest["status"] == "complete"
                                                   else "incomplete_core_accumulator")):
            raise ValueError("Invalid checkpoint result semantics")
        if manifest["status"] == "complete":
            if (manifest.get("source_count") != q.n or manifest.get("restored_original_nodes") != q.n
                    or manifest.get("completed_core_groups") != len(plan)):
                raise ValueError("Invalid complete checkpoint coverage")
            for name in ("contributions", "bc_raw", "bc_normalized"):
                item = manifest["outputs"][name]
                if (item["file"] != name + ".npy" or item["nodes"] != q.n
                        or _sha256(root / item["file"]) != item["sha256"]):
                    raise ValueError("Checkpoint final output fingerprint mismatch")
        if _sha256(root / "source_plan.npy") != manifest["plan_file_sha256"]:
            raise ValueError("Checkpoint source plan fingerprint mismatch")
        stored_plan = np.load(root / "source_plan.npy", allow_pickle=False)
        if not np.array_equal(stored_plan, plan):
            raise ValueError("Checkpoint source plan coverage mismatch")
    else:
        _write_array(root / "source_plan.npy", plan)
        manifest = dict(format=FORMAT, identity=identity, status="paused", exact_all_sources=False,
            graph_full=True, scaled=False, result_kind="incomplete_core_accumulator",
            plan_file_sha256=_sha256(root / "source_plan.npy"), chunks=[],
            completed_core_groups=0, completed_source_mass=0, analytic_source_mass=int(mass.sum() - mass[plan].sum()),
            created_utc=datetime.now(timezone.utc).isoformat())
        _write_json(root / "manifest.json", manifest)
    check_time()
    core_total = np.zeros(q.core.n)
    cursor, source_mass = 0, 0
    for index, chunk in enumerate(manifest["chunks"]):
        end = min(cursor + chunk_size, len(plan))
        if (cursor >= len(plan) or chunk["index"] != index or chunk["start"] != cursor
                or chunk["stop"] != end or chunk["file"] != _chunk_name(index)
                or chunk["source_order_hash"] != source_order_digest(plan[cursor:end])
                or chunk["source_mass"] != int(mass[plan[cursor:end]].sum())):
            raise ValueError("Checkpoint chunk coverage mismatch")
        path = root / chunk["file"]
        if _sha256(path) != chunk["sha256"]:
            raise ValueError("Checkpoint contribution fingerprint mismatch")
        part = np.load(path, allow_pickle=False)
        if part.shape != core_total.shape or part.dtype != np.float64 or not np.isfinite(part).all() or np.any(part < 0):
            raise ValueError("Invalid checkpoint contribution vector")
        core_total += part
        source_mass += chunk["source_mass"]
        cursor = end
        check_time()
    if (manifest["completed_core_groups"] != cursor or manifest["completed_source_mass"] != source_mass
            or manifest["analytic_source_mass"] != int(mass.sum() - mass[plan].sum())):
        raise ValueError("Checkpoint counters mismatch")
    setup_ended = time.perf_counter()
    # Retained immutable files are a deliberate disk-for-recovery tradeoff.
    remaining_chunks = (len(plan) - cursor + chunk_size - 1) // chunk_size
    required_disk = remaining_chunks * (q.core.n * 8 + 256) + q.n * 24 + 64 * 2**20
    if psutil.disk_usage(str(root)).free < required_disk:
        raise OSError(f"Insufficient checkpoint disk space: need approximately {required_disk} bytes")
    jit = weighted_warmup() if len(plan) > cursor and max_chunks != 0 else {"t_jit_warmup": 0.}
    new_chunks, session_work = 0, 0.
    while cursor < len(plan) and (max_chunks is None or new_chunks < max_chunks):
        check_time()
        end = min(cursor + chunk_size, len(plan))
        selected = plan[cursor:end]
        rss_before = psutil.Process().memory_info().rss
        remaining = None if timeout_seconds is None else timeout_seconds - (time.perf_counter() - started)
        part, meta = threaded_contributions(q.core, selected, threads=threads, batch_size=batch_size,
            timeout_seconds=remaining, _runner_factory=lambda core: WeightedAccumulator(core, mass, mass),
            _warmup=lambda: dict(t_jit_warmup=0., native_nopython=True))
        if fault:
            fault("after_compute", len(manifest["chunks"]))
        check_time()
        index = len(manifest["chunks"])
        path = root / _chunk_name(index)
        commit_started = time.perf_counter()
        _write_array(path, part)
        row = dict(index=index, start=cursor, stop=end, file=path.name, sha256=_sha256(path),
            source_order_hash=source_order_digest(selected), source_mass=int(mass[selected].sum()),
            fallback_sources=meta["fallback_sources"], t_execute_reduce=meta["t_execute_reduce"],
            t_compute_driver=meta["t_compute_driver"], worker_cpu_seconds=meta["worker_cpu_seconds"],
            average_worker_cpu_cores=meta["average_worker_cpu_cores"],
            workers=meta["workers"], commit_file_seconds=time.perf_counter() - commit_started)
        row.update(rss_before_bytes=rss_before, rss_after_bytes=psutil.Process().memory_info().rss,
                   system_available_bytes=psutil.virtual_memory().available)
        if fault:
            fault("after_score_file", index)
        # The atomic manifest replacement is the ONLY commit point.
        manifest["chunks"].append(row)
        manifest.update(completed_core_groups=end, completed_source_mass=source_mass + row["source_mass"])
        _write_json(root / "manifest.json", manifest)
        if fault:
            fault("after_manifest", index)
        core_total += part
        source_mass += row["source_mass"]
        cursor, new_chunks = end, new_chunks + 1
        session_work += meta["t_execute_reduce"]
        print(f"checkpoint {index}: committed {cursor}/{len(plan)} core groups; "
              f"work={meta['t_execute_reduce']:.3f}s", flush=True)
        # meta/tasks are bounded by chunk_size and discarded each iteration.
        del meta, part
    check_time()
    if cursor == len(plan):
        if source_mass + manifest["analytic_source_mass"] != q.n:
            raise ValueError("All-source mass coverage mismatch")
        _warmup_recovery()
        restore_started = time.perf_counter()
        total = _full_restore(q, core_total)
        if not np.isfinite(total).all() or np.any(total < 0):
            raise FloatingPointError("Invalid restored full-source contributions")
        raw, normalized = finalize_scores(total, q.n)
        outputs = {}
        for name, values in (("contributions", total), ("bc_raw", raw), ("bc_normalized", normalized)):
            path = root / (name + ".npy")
            _write_array(path, values)
            outputs[name] = dict(file=path.name, sha256=_sha256(path), nodes=len(values))
        check_time()
        manifest.update(status="complete", exact_all_sources=True, source_count=q.n,
            result_kind="exact_centrality", restored_original_nodes=q.n, outputs=outputs,
            t_restore_finalize_save=time.perf_counter() - restore_started)
        _write_json(root / "manifest.json", manifest)
    return dict(run_directory=str(root.resolve()), run_status=manifest["status"],
        exact_all_sources=manifest["exact_all_sources"], result_kind=manifest["result_kind"],
        completed_core_groups=cursor, total_core_groups=len(plan), new_chunks=new_chunks,
        represented_completed_source_mass=source_mass, analytic_source_mass=manifest["analytic_source_mass"],
        source_count=q.n if manifest["exact_all_sources"] else None,
        code_sha256=identity["code_sha256"], graph_sha256=q.source_graph_sha256,
        t_resume_validate=setup_ended - started, t_jit_warmup=jit["t_jit_warmup"],
        t_session_work=session_work, t_checkpoint_session=time.perf_counter() - started)


def run_checkpoint(cache_directory, run_directory, *, seconds=1800, min_available_bytes=2**30,
                   cpu_affinity=None, **options):
    """One supervised process loads/validates the quotient only once per session."""
    from .large_benchmark import bounded_job
    if cpu_affinity is not None:
        cpu_affinity = list(map(operator.index, cpu_affinity))
        if (cpu_affinity != sorted(set(cpu_affinity)) or not cpu_affinity
                or len(cpu_affinity) < options.get("threads", 4)
                or not set(cpu_affinity) <= set(psutil.Process().cpu_affinity())):
            raise ValueError("Invalid CPU affinity")
    row = bounded_job("quotient_checkpoint", dict(cache_directory=str(Path(cache_directory).resolve()),
        run_directory=str(Path(run_directory).resolve()), timeout_seconds=seconds,
        cpu_affinity=cpu_affinity, **options), seconds=seconds, min_available_bytes=min_available_bytes)
    # Each invocation has independent supervision evidence, including hard kills.
    root = Path(run_directory)
    if root.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        _write_json(root / f"session_{stamp}.json", dict(row,
            reserve_bytes=min_available_bytes, wall_budget_seconds=seconds,
            requested_cpu_affinity=cpu_affinity))
    return row
