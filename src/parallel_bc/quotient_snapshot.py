"""Crash-recoverable alternating cumulative snapshots for exact DAG Brandes.

The manifest is the only commit point.  Never write the slot it references.
An uncommitted inactive slot is ignored and may be overwritten after restart.
"""
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import operator
from pathlib import Path
import platform
import time

import numpy as np
import psutil

from .brandes import finalize_scores, source_order_digest
from .compiled_threads import threaded_contributions
from .data import _sha256
from .quotient_brandes import _warmup_recovery
from .quotient_checkpoint_dag import _full_restore, _run_lock, _write_array, _write_json
from .quotient_dag_experiment import DagAccumulator, dag_warmup

FORMAT = 'LinkRadar-quotient-dag-snapshot-v1'
SLOTS = ('snapshot_A.npy', 'snapshot_B.npy')

def checkpoint_code_digest():
    digest = hashlib.sha256()
    for name in ('quotient_snapshot.py', 'quotient_checkpoint_dag.py', 'quotient_dag_experiment.py',
                 'quotient_brandes.py', 'compiled_threads.py', 'compiled.py', 'tree_quotient.py',
                 'csr.py', 'brandes.py'):
        digest.update(name.encode())
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()

def checkpoint_contributions(q, run_directory, *, resume=False, threads=4, batch_size=4,
                             chunk_size=1024, seed=45, max_chunks=None,
                             timeout_seconds=None, _fault=None):
    started = time.perf_counter()
    threads, batch_size, chunk_size, seed = map(operator.index, (threads, batch_size, chunk_size, seed))
    if min(threads, batch_size, chunk_size) < 1 or seed < 0:
        raise ValueError('Invalid checkpoint configuration')
    if max_chunks is not None:
        max_chunks = operator.index(max_chunks)
        if max_chunks < 0:
            raise ValueError('max_chunks must be nonnegative')
    if timeout_seconds is not None and (not np.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError('timeout_seconds must be positive and finite')
    root = Path(run_directory)
    if not resume:
        root.mkdir(parents=True, exist_ok=False)
    elif not (root / 'manifest.json').is_file():
        raise ValueError('No committed checkpoint manifest to resume')
    with _run_lock(root):
        return _execute(q, root, resume, threads, batch_size, chunk_size, seed,
                        max_chunks, timeout_seconds, started, _fault)

def _execute(q, root, resume, threads, batch_size, chunk_size, seed, max_chunks,
             timeout_seconds, started, fault):
    def check_time():
        if timeout_seconds is not None and time.perf_counter() - started >= timeout_seconds:
            raise TimeoutError('Snapshot session budget exceeded; committed snapshot retained')

    mass = q.mass
    mass.flags.writeable = False
    roots = np.flatnonzero(np.diff(q.core.indptr) > 0).astype(np.int32)
    plan = np.random.default_rng(seed).permutation(roots)
    analytic_mass = int(mass.sum() - mass[plan].sum())
    identity = dict(graph_sha256=q.source_graph_sha256, core_graph_sha256=q.core.graph_sha256,
        original_nodes=q.n, original_edges=q.metadata['original_edges'], core_nodes=q.core.n,
        core_edges=q.core.m,
        quotient_arrays_sha256={name: hashlib.sha256(np.ascontiguousarray(getattr(q, name)).tobytes()).hexdigest()
            for name in ('original_ids', 'parent', 'peel_order', 'anchor_index', 'anchor_nodes',
                         'subtree_size', 'component_index', 'component_sizes')},
        code_sha256=checkpoint_code_digest(),
        environment={name: version(name) for name in ('numpy', 'numba', 'llvmlite')},
        python=platform.python_version(), cpu_affinity=psutil.Process().cpu_affinity(),
        threads=threads, batch_size=batch_size, chunk_size=chunk_size, seed=seed,
        plan_hash=source_order_digest(plan), core_traversals=len(plan), source_mass_sum=int(mass.sum()),
        dtype='float64', fastmath=False, kernel='recorded_shortest_path_edges_v1',
        semantics='all original sources; source_count=target_mass=block_size; restore once')
    if resume:
        manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('format') != FORMAT or manifest.get('identity') != identity:
            raise ValueError('Checkpoint identity mismatch: format, graph, code, dependencies or configuration changed')
        if (manifest.get('status') not in {'paused', 'complete'}
                or manifest.get('exact_all_sources') != (manifest['status'] == 'complete')
                or manifest.get('graph_full') is not True or manifest.get('scaled') is not False
                or manifest.get('result_kind') != ('exact_centrality' if manifest['status'] == 'complete'
                                                   else 'incomplete_core_accumulator')):
            raise ValueError('Invalid checkpoint result semantics')
        if _sha256(root / 'source_plan.npy') != manifest['plan_file_sha256']:
            raise ValueError('Checkpoint source plan fingerprint mismatch')
        if not np.array_equal(np.load(root / 'source_plan.npy', allow_pickle=False), plan):
            raise ValueError('Checkpoint source plan coverage mismatch')
    else:
        _write_array(root / 'source_plan.npy', plan)
        manifest = dict(format=FORMAT, identity=identity, status='paused', exact_all_sources=False,
            graph_full=True, scaled=False, result_kind='incomplete_core_accumulator',
            plan_file_sha256=_sha256(root / 'source_plan.npy'), active_snapshot=None,
            completed_core_groups=0, completed_source_mass=0, analytic_source_mass=analytic_mass,
            generation=0, last_checkpoint=None, created_utc=datetime.now(timezone.utc).isoformat())
        _write_json(root / 'manifest.json', manifest)
    check_time()
    cursor = manifest['completed_core_groups']
    generation = manifest['generation']
    if (not isinstance(cursor, int) or isinstance(cursor, bool) or not 0 <= cursor <= len(plan)
            or (cursor != len(plan) and cursor % chunk_size != 0)
            or generation != (cursor + chunk_size - 1) // chunk_size
            or manifest['completed_source_mass'] != int(mass[plan[:cursor]].sum())
            or manifest['analytic_source_mass'] != analytic_mass):
        raise ValueError('Checkpoint counters or source coverage mismatch')
    snapshot = manifest['active_snapshot']
    if generation == 0:
        if snapshot is not None:
            raise ValueError('Unexpected initial snapshot')
        core_total = np.zeros(q.core.n, dtype=np.float64)
    else:
        expected_file = SLOTS[(generation - 1) % 2]
        if (not isinstance(snapshot, dict) or snapshot.get('file') != expected_file
                or snapshot.get('generation') != generation or snapshot.get('completed_core_groups') != cursor
                or snapshot.get('completed_source_mass') != manifest['completed_source_mass']):
            raise ValueError('Snapshot metadata or coverage mismatch')
        path = root / expected_file
        if _sha256(path) != snapshot.get('sha256'):
            raise ValueError('Snapshot fingerprint mismatch')
        core_total = np.load(path, allow_pickle=False)
        if (core_total.shape != (q.core.n,) or core_total.dtype != np.float64
                or not np.isfinite(core_total).all() or np.any(core_total < 0)):
            raise ValueError('Invalid cumulative snapshot vector')
    source_mass = manifest['completed_source_mass']
    if manifest['status'] == 'complete':
        if (cursor != len(plan) or source_mass + analytic_mass != q.n
                or manifest.get('source_count') != q.n or manifest.get('restored_original_nodes') != q.n):
            raise ValueError('Invalid complete checkpoint coverage')
        for name in ('contributions', 'bc_raw', 'bc_normalized'):
            item = manifest['outputs'][name]
            if item['file'] != name + '.npy' or item['nodes'] != q.n or _sha256(root / item['file']) != item['sha256']:
                raise ValueError('Checkpoint final output fingerprint mismatch')
        return dict(run_directory=str(root.resolve()), run_status='complete', exact_all_sources=True,
                    result_kind='exact_centrality', completed_core_groups=cursor, total_core_groups=len(plan),
                    new_chunks=0, represented_completed_source_mass=source_mass, analytic_source_mass=analytic_mass,
                    source_count=q.n, code_sha256=identity['code_sha256'], graph_sha256=q.source_graph_sha256,
                    storage_mode='snapshot', t_resume_validate=time.perf_counter()-started, t_jit_warmup=0.,
                    t_session_work=0., t_checkpoint_session=time.perf_counter()-started)
    check_time()
    setup_ended = time.perf_counter()
    # Conservative: A+B+temporary, three final arrays+one temporary, plan and 64 MiB margin.
    required_disk = 3 * (q.core.n * 8 + 256) + 4 * (q.n * 8 + 256) + plan.nbytes + 64 * 2**20
    if psutil.disk_usage(str(root)).free < required_disk:
        raise OSError(f'Insufficient snapshot disk space: need approximately {required_disk} bytes')
    jit = dag_warmup() if cursor < len(plan) and max_chunks != 0 else {'t_jit_warmup': 0.}
    new_chunks, session_work = 0, 0.
    while cursor < len(plan) and (max_chunks is None or new_chunks < max_chunks):
        check_time()
        end = min(cursor + chunk_size, len(plan))
        selected = plan[cursor:end]
        remaining = None if timeout_seconds is None else timeout_seconds - (time.perf_counter() - started)
        part, meta = threaded_contributions(q.core, selected, threads=threads, batch_size=batch_size,
            timeout_seconds=remaining, _runner_factory=lambda core: DagAccumulator(core, mass, mass),
            _warmup=lambda: dict(t_jit_warmup=0., native_nopython=True))
        if fault:
            fault('after_compute', generation)
        check_time()
        # In-place sum is not committed until the manifest is replaced. Any error
        # aborts this invocation; restart reloads the last committed snapshot.
        core_total += part
        if not np.isfinite(core_total).all() or np.any(core_total < 0):
            raise FloatingPointError('Invalid cumulative snapshot')
        filename = SLOTS[generation % 2]
        if snapshot is not None and filename == snapshot['file']:
            raise RuntimeError('Refusing to overwrite active snapshot')
        commit_started = time.perf_counter()
        path = root / filename
        _write_array(path, core_total)
        persisted = np.load(path, allow_pickle=False)
        if (persisted.shape != core_total.shape or persisted.dtype != np.float64
                or not np.array_equal(persisted, core_total)):
            raise OSError('Persisted snapshot differs from cumulative vector')
        del persisted
        new_mass = source_mass + int(mass[selected].sum())
        new_snapshot = dict(file=filename, sha256=_sha256(path), generation=generation+1,
                            completed_core_groups=end, completed_source_mass=new_mass)
        if fault:
            fault('after_snapshot', generation)
        check_time()
        last = dict(start=cursor, stop=end, source_mass=int(mass[selected].sum()),
            source_order_hash=source_order_digest(selected), fallback_sources=meta['fallback_sources'],
            t_execute_reduce=meta['t_execute_reduce'], worker_cpu_seconds=meta['worker_cpu_seconds'],
            average_worker_cpu_cores=meta['average_worker_cpu_cores'],
            snapshot_file_seconds=time.perf_counter()-commit_started,
            base_workspace_bytes=meta['workspace_bytes'],
            dag_workspace_bytes=meta['p_used'] * (q.core.indices.nbytes + (q.core.n + 1) * 8),
            local_total_bytes=meta['local_total_bytes'],
            rss_bytes=psutil.Process().memory_info().rss, system_available_bytes=psutil.virtual_memory().available)
        manifest.update(active_snapshot=new_snapshot, generation=generation+1,
                        completed_core_groups=end, completed_source_mass=new_mass, last_checkpoint=last)
        _write_json(root / 'manifest.json', manifest)
        if fault:
            fault('after_manifest', generation)
        snapshot, cursor, source_mass, generation = new_snapshot, end, new_mass, generation+1
        new_chunks += 1
        session_work += meta['t_execute_reduce']
        print(f'checkpoint {generation-1}: committed {cursor}/{len(plan)} core groups; '
              f'work={meta["t_execute_reduce"]:.3f}s', flush=True)
        del meta, part
    check_time()
    if cursor == len(plan):
        if source_mass + analytic_mass != q.n:
            raise ValueError('All-source mass coverage mismatch')
        _warmup_recovery()
        restore_started = time.perf_counter()
        total = _full_restore(q, core_total)
        if not np.isfinite(total).all() or np.any(total < 0):
            raise FloatingPointError('Invalid restored full-source contributions')
        raw, normalized = finalize_scores(total, q.n)
        outputs = {}
        for name, values in (('contributions', total), ('bc_raw', raw), ('bc_normalized', normalized)):
            path = root / (name + '.npy')
            _write_array(path, values)
            outputs[name] = dict(file=path.name, sha256=_sha256(path), nodes=len(values))
        check_time()
        manifest.update(status='complete', exact_all_sources=True, source_count=q.n,
            result_kind='exact_centrality', restored_original_nodes=q.n, outputs=outputs,
            t_restore_finalize_save=time.perf_counter()-restore_started)
        _write_json(root / 'manifest.json', manifest)
    return dict(run_directory=str(root.resolve()), run_status=manifest['status'],
        exact_all_sources=manifest['exact_all_sources'], result_kind=manifest['result_kind'],
        completed_core_groups=cursor, total_core_groups=len(plan), new_chunks=new_chunks,
        represented_completed_source_mass=source_mass, analytic_source_mass=analytic_mass,
        source_count=q.n if manifest['exact_all_sources'] else None, code_sha256=identity['code_sha256'],
        graph_sha256=q.source_graph_sha256, storage_mode='snapshot',
        t_resume_validate=setup_ended-started, t_jit_warmup=jit['t_jit_warmup'],
        t_session_work=session_work, t_checkpoint_session=time.perf_counter()-started)

def run_checkpoint(cache_directory, run_directory, *, seconds=1800, min_available_bytes=2**30,
                   cpu_affinity=None, **options):
    from .large_benchmark import bounded_job
    root = Path(run_directory)
    existed_before = root.exists()
    if cpu_affinity is not None:
        cpu_affinity = list(map(operator.index, cpu_affinity))
        if (cpu_affinity != sorted(set(cpu_affinity)) or not cpu_affinity
                or len(cpu_affinity) < options.get('threads', 4)
                or not set(cpu_affinity) <= set(psutil.Process().cpu_affinity())):
            raise ValueError('Invalid CPU affinity')
    row = bounded_job('quotient_snapshot', dict(cache_directory=str(Path(cache_directory).resolve()),
        run_directory=str(Path(run_directory).resolve()), timeout_seconds=seconds,
        cpu_affinity=cpu_affinity, **options), seconds=seconds, min_available_bytes=min_available_bytes)
    may_record = root.exists() and (not existed_before or options.get('resume', False))
    if may_record and (root / 'manifest.json').exists():
        try:
            may_record = json.loads((root / 'manifest.json').read_text(encoding='utf-8')).get('format') == FORMAT
        except (OSError, ValueError):
            may_record = False
    if may_record:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        _write_json(root / f'session_{stamp}.json', dict(row,
            reserve_bytes=min_available_bytes, wall_budget_seconds=seconds, requested_cpu_affinity=cpu_affinity))
    return row
