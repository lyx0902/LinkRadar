"""Bounded 64-source diagnostic; preserves production kernel and checkpoint."""
from pathlib import Path
import hashlib
import json
import statistics
import time
from datetime import datetime, timezone

import numpy as np
import psutil
from numba import njit

from parallel_bc.brandes import source_order_digest
from parallel_bc.compiled import CompiledWorkspace
from parallel_bc.compiled_threads import threaded_contributions
from parallel_bc.quotient_brandes import WeightedAccumulator, weighted_warmup, _weighted_commit
from parallel_bc.tree_quotient import load_quotient

LIMIT = np.iinfo(np.int64).max

@njit(nogil=True, fastmath=False)
def initialize(dist, sigma, delta, queue, source):
    for v in range(len(dist)):
        dist[v], sigma[v], delta[v] = -1, 0, 0.0
    dist[source], sigma[source], queue[0] = 0, 1, source

@njit(nogil=True, fastmath=False)
def bfs(ptr, indices, dist, sigma, queue):
    head, tail, maximum = 0, 1, 1
    while head < tail:
        v = queue[head]
        head += 1
        next_distance = dist[v] + 1
        for edge in range(ptr[v], ptr[v + 1]):
            w = indices[edge]
            if dist[w] == -1:
                dist[w] = next_distance
                queue[tail] = w
                tail += 1
            if dist[w] == next_distance:
                if sigma[v] > LIMIT - sigma[w]:
                    return 1, tail, maximum
                sigma[w] += sigma[v]
                maximum = max(maximum, sigma[w])
    return 0, tail, maximum

@njit(nogil=True, fastmath=False)
def reverse(ptr, indices, mass, dist, sigma, delta, queue, visited):
    for pos in range(visited - 1, -1, -1):
        v = queue[pos]
        value = 0.0
        for edge in range(ptr[v], ptr[v + 1]):
            w = indices[edge]
            if dist[w] == dist[v] + 1:
                value += (sigma[v] / sigma[w]) * (mass[w] + delta[w])
        if not np.isfinite(value):
            return 2
        delta[v] = value
    return 0

class Profiled:
    def __init__(self, core, mass):
        self.core, self.mass = core, mass
        self.workspace = CompiledWorkspace.allocate(core.n)

    def accumulate(self, source, total):
        w, g = self.workspace, self.core
        t0 = time.perf_counter()
        initialize(w.dist, w.sigma, w.delta, w.queue, source)
        t1 = time.perf_counter()
        status, visited, maximum = bfs(g.indptr, g.indices, w.dist, w.sigma, w.queue)
        t2 = time.perf_counter()
        if status:
            raise OverflowError('diagnostic stopped: sigma overflow; production fallback required')
        status = reverse(g.indptr, g.indices, self.mass, w.dist, w.sigma, w.delta, w.queue, visited)
        t3 = time.perf_counter()
        if status:
            raise FloatingPointError('nonfinite dependency')
        _weighted_commit(total, w.delta, w.queue, visited, source, int(self.mass[source]))
        t4 = time.perf_counter()
        return dict(execution='numba_native', reachable_core_nodes=int(visited), max_sigma=int(maximum),
                    init=t1-t0, bfs=t2-t1, reverse=t3-t2, commit=t4-t3)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    root = Path(__file__).resolve().parents[1]
    checkpoint = root / 'results/20261024T0903_youtube_checkpoint/manifest.json'
    before = sha(checkpoint)
    output = root / 'results' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '_stage_profile_64')
    output.mkdir()
    psutil.Process().cpu_affinity([0, 2, 4, 6])
    if psutil.virtual_memory().available < 2**30:
        raise RuntimeError('Less than 1 GiB available')
    q = load_quotient(root / 'data/processed/youtube_csr_treeq')
    mass = q.mass
    mass.flags.writeable = False
    roots = np.flatnonzero(np.diff(q.core.indptr) > 0).astype(np.int32)
    selected = np.random.default_rng(46).choice(roots, size=64, replace=False).tolist()
    weighted_warmup()
    probe = Profiled(q.core, mass)
    test_total = np.zeros(q.core.n)
    probe.accumulate(selected[0], test_total)
    baseline = WeightedAccumulator(q.core, mass, mass)
    reference = np.zeros(q.core.n)
    baseline.accumulate(selected[0], reference)
    if not np.allclose(test_total, reference, rtol=1e-10, atol=1e-8):
        raise AssertionError('warmup mismatch')
    cases = []
    for repeat in range(3):
        for p in ([1, 4] if repeat % 2 == 0 else [4, 1]):
            vectors = {}
            for kind in (['original', 'profile'] if repeat % 2 == 0 else ['profile', 'original']):
                if psutil.virtual_memory().available < 2**30:
                    raise RuntimeError('Less than 1 GiB available')
                factory = (lambda graph: WeightedAccumulator(graph, mass, mass)) if kind == 'original' else (lambda graph: Profiled(graph, mass))
                vector, meta = threaded_contributions(q.core, selected, threads=p, batch_size=4,
                    timeout_seconds=180, _runner_factory=factory, _warmup=lambda: {'t_jit_warmup': 0.0})
                vectors[kind] = vector
                tasks = meta.pop('tasks')
                stage = {name: sum(t[name] for t in tasks) for name in ('init', 'bfs', 'reverse', 'commit')} if kind == 'profile' else None
                cases.append(dict(repeat=repeat+1, threads=p, kind=kind, stages_sum=stage, **meta))
                print(f'repeat={repeat+1} threads={p} kind={kind} wall={meta["t_execute_reduce"]:.3f}', flush=True)
            difference = float(np.max(np.abs(vectors['original'] - vectors['profile'])))
            if not np.allclose(vectors['original'], vectors['profile'], rtol=1e-10, atol=1e-8):
                raise AssertionError(f'profile mismatch {difference}')
            cases[-1]['max_abs_difference'] = difference
    summary = {}
    for p in (1, 4):
        rows = [x for x in cases if x['threads'] == p]
        profiles = [x for x in rows if x['kind'] == 'profile']
        original = [x for x in rows if x['kind'] == 'original']
        stage_medians = {k: statistics.median(x['stages_sum'][k] for x in profiles) for k in ('init', 'bfs', 'reverse', 'commit')}
        summary[str(p)] = dict(stage_median_source_wall_sums=stage_medians,
            stage_fraction_of_sum={k: v / sum(stage_medians.values()) for k, v in stage_medians.items()},
            original_wall_median=statistics.median(x['t_execute_reduce'] for x in original),
            profile_wall_median=statistics.median(x['t_execute_reduce'] for x in profiles),
            original_cpu_cores_median=statistics.median(x['average_worker_cpu_cores'] for x in original),
            original_reduce_median=statistics.median(x['t_reduce'] for x in original),
            original_workspace_median=statistics.median(x['t_workspace'] for x in original))
    result = dict(status='complete', source_groups=64, source_seed=46, source_order_hash=source_order_digest(selected),
        original_source_mass=int(mass[selected].sum()), graph_sha256=q.source_graph_sha256,
        core_sha256=q.core.graph_sha256, cpu_affinity=[0, 2, 4, 6], batch_size=4, repeat_count=3,
        checkpoint_manifest_unchanged=sha(checkpoint)==before, summary=summary, cases=cases,
        note='Stage values are summed per-source wall times, overlapping at four threads; diagnostic calls add overhead. Original runs are uninstrumented.')
    (output / 'profile.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print('RESULT=' + str(output.resolve()), flush=True)

if __name__ == '__main__':
    main()
