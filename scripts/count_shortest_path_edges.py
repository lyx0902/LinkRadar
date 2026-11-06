"""Measure forward shortest-path DAG edges for the existing 64-source sample."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import psutil
from numba import njit
from parallel_bc.brandes import source_order_digest
from parallel_bc.compiled import CompiledWorkspace
from parallel_bc.quotient_brandes import _weighted_dependencies, weighted_warmup
from parallel_bc.tree_quotient import load_quotient

MAX = np.iinfo(np.int64).max

@njit(nogil=True, fastmath=False)
def counted_bfs(ptr, indices, source, dist, sigma, queue):
    for i in range(len(dist)):
        dist[i], sigma[i] = -1, 0
    dist[source], sigma[source], queue[0] = 0, 1, source
    head, tail, scanned, valid = 0, 1, 0, 0
    while head < tail:
        v = queue[head]
        head += 1
        nxt = dist[v] + 1
        for edge in range(ptr[v], ptr[v+1]):
            scanned += 1
            w = indices[edge]
            if dist[w] == -1:
                dist[w] = nxt
                queue[tail] = w
                tail += 1
            if dist[w] == nxt:
                valid += 1
                if sigma[v] > MAX - sigma[w]:
                    return 1, tail, scanned, valid
                sigma[w] += sigma[v]
    return 0, tail, scanned, valid

@njit(nogil=True, fastmath=False)
def recount(ptr, indices, dist):
    forward, same, backward = 0, 0, 0
    for v in range(len(dist)):
        if dist[v] < 0:
            continue
        for i in range(ptr[v], ptr[v+1]):
            difference = dist[indices[i]] - dist[v]
            if difference == 1:
                forward += 1
            elif difference == 0:
                same += 1
            elif difference == -1:
                backward += 1
            else:
                raise ValueError('Invalid distance difference')
    return forward, same, backward

def main():
    root = Path(__file__).resolve().parents[1]
    manifest = root / 'results/20261024T0903_youtube_checkpoint/manifest.json'
    old_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    prior = json.loads((root/'results/20261102T062858_343786Z_coeff_64/summary.json').read_text())
    out = root/'results'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'_dag_edge_count_64')
    out.mkdir()
    if psutil.virtual_memory().available < 2**30:
        raise RuntimeError('Less than 1 GiB available')
    psutil.Process().cpu_affinity([0, 2, 4, 6])
    q = load_quotient(root/'data/processed/youtube_csr_treeq')
    mass = q.mass
    mass.flags.writeable = False
    roots = np.flatnonzero(np.diff(q.core.indptr)>0).astype(np.int32)
    selected = np.random.default_rng(46).choice(roots, size=64, replace=False).tolist()
    digest = source_order_digest(selected)
    if digest != prior['source_order_hash'] or q.source_graph_sha256 != prior['graph_sha256']:
        raise ValueError('Graph or source sample mismatch')
    a, b = CompiledWorkspace.allocate(q.core.n), CompiledWorkspace.allocate(q.core.n)
    weighted_warmup()
    rows, started = [], time.perf_counter()
    for i, source in enumerate(selected):
        status, visited, scanned, valid = counted_bfs(q.core.indptr, q.core.indices, source, a.dist, a.sigma, a.queue)
        if status:
            raise OverflowError(f'source {source}: int64 overflow; no partial results accepted')
        old_status, old_visited, _ = _weighted_dependencies(q.core.indptr, q.core.indices, mass, source,
            b.dist, b.sigma, b.delta, b.queue)
        if old_status or visited != old_visited or not np.array_equal(a.dist,b.dist) or not np.array_equal(a.sigma,b.sigma):
            raise AssertionError(f'BFS mismatch at source {source}')
        forward, same, backward = recount(q.core.indptr, q.core.indices, a.dist)
        if forward != valid or forward+same+backward != scanned or forward != backward:
            raise AssertionError(f'edge classification mismatch at source {source}')
        rows.append(dict(source_index=source, original_id=int(q.core.ids[source]), visited=int(visited),
            scanned=int(scanned), valid=int(valid), same=int(same), backward=int(backward), ratio=valid/scanned))
        if (i+1)%16==0:
            print(f'{i+1}/64 verified', flush=True)
    ratio = np.array([x['ratio'] for x in rows])
    valid = np.array([x['valid'] for x in rows])
    result = dict(status='complete', n_groups=len(rows), source_order_hash=digest,
        graph_sha256=q.source_graph_sha256, core_graph_sha256=q.core.graph_sha256,
        represented_sources=int(mass[selected].sum()),
        median_fraction=float(np.median(ratio)), min_fraction=float(ratio.min()), max_fraction=float(ratio.max()),
        mean_fraction=float(ratio.mean()), total_fraction=sum(x['valid'] for x in rows)/sum(x['scanned'] for x in rows),
        median_scanned=float(np.median([x['scanned'] for x in rows])), median_valid=float(np.median(valid)),
        median_same=float(np.median([x['same'] for x in rows])), median_backward=float(np.median([x['backward'] for x in rows])),
        median_edge_indices_bytes=float(np.median(valid))*4,
        timing_note='Diagnostic time includes JIT/recount/production checks and is not a performance benchmark.',
        diagnostic_seconds=time.perf_counter()-started, checkpoint_manifest_unchanged=old_hash==hashlib.sha256(manifest.read_bytes()).hexdigest(),
        cases=rows)
    (out/'summary.json').write_text(json.dumps(result, indent=2),encoding='utf-8')
    print('RESULT='+str(out.resolve()),flush=True)

if __name__ == '__main__':
    main()
