"""Matched, alternating 64-group experiment for coefficient caching."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import psutil
from parallel_bc.brandes import source_order_digest
from parallel_bc.compiled_threads import threaded_contributions
from parallel_bc.quotient_brandes import WeightedAccumulator, weighted_warmup, _selected_subtrees, _restore
from parallel_bc.quotient_coeff_experiment import CoefficientAccumulator
from parallel_bc.tree_quotient import load_quotient

def main():
    root = Path(__file__).resolve().parents[1]
    manifest = root / 'results/20261024T0903_youtube_checkpoint/manifest.json'
    before = hashlib.sha256(manifest.read_bytes()).hexdigest()
    out = root / 'results' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '_coeff_64')
    out.mkdir()
    psutil.Process().cpu_affinity([0, 2, 4, 6])
    q = load_quotient(root / 'data/processed/youtube_csr_treeq')
    mass = q.mass
    mass.flags.writeable = False
    roots = np.flatnonzero(np.diff(q.core.indptr) > 0).astype(np.int32)
    selected = np.random.default_rng(46).choice(roots, size=64, replace=False).tolist()
    indicator = np.isin(q.anchor_index, selected).astype(np.int8)
    selected_subtrees, counts, component_counts = _selected_subtrees(indicator, q.parent,
        q.peel_order, q.anchor_nodes, q.component_index, len(q.component_sizes))
    weighted_warmup()
    for cls in (WeightedAccumulator, CoefficientAccumulator):
        cls(q.core, mass, mass).accumulate(selected[0], np.zeros(q.core.n))
    records, comparisons = [], []
    for repeat in range(3):
        vectors = {}
        for kind in (('baseline', 'candidate') if repeat % 2 == 0 else ('candidate', 'baseline')):
            if psutil.virtual_memory().available < 2**30:
                raise RuntimeError('Less than 1 GiB available')
            cls = WeightedAccumulator if kind == 'baseline' else CoefficientAccumulator
            before_run = time.perf_counter()
            vector, meta = threaded_contributions(q.core, selected, threads=4, batch_size=4,
                timeout_seconds=180, _runner_factory=lambda graph: cls(graph, mass, mass),
                _warmup=lambda: {'t_jit_warmup': 0.0})
            driver_wall = time.perf_counter() - before_run
            meta.pop('tasks')
            vectors[kind] = vector
            records.append(dict(repeat=repeat+1, kind=kind, driver_wall=driver_wall, **meta))
            print(f'{repeat+1} {kind} {meta["t_execute_reduce"]:.3f}s', flush=True)
        a, b = vectors['baseline'], vectors['candidate']
        core_diff = float(np.max(np.abs(a-b)))
        if not np.allclose(a, b, atol=1e-8, rtol=1e-10):
            raise AssertionError(f'core mismatch {core_diff}')
        restored = [_restore(q.parent, q.peel_order, q.anchor_nodes, q.anchor_index, q.subtree_size,
            q.component_index, q.component_sizes, indicator, selected_subtrees, counts, component_counts, x)
            for x in (a,b)]
        original_diff = float(np.max(np.abs(restored[0]-restored[1])))
        if not np.allclose(restored[0], restored[1], atol=1e-8, rtol=1e-10):
            raise AssertionError(f'original mismatch {original_diff}')
        times = {r['kind']: r['t_execute_reduce'] for r in records if r['repeat']==repeat+1}
        comparisons.append(dict(repeat=repeat+1, speedup=times['baseline']/times['candidate'],
            max_core_abs=core_diff, max_original_abs=original_diff))
    ratios = [r['speedup'] for r in comparisons]
    summary = dict(status='complete', source_groups=64, seed=46, represented_sources=int(mass[selected].sum()),
        source_order_hash=source_order_digest(selected), graph_sha256=q.source_graph_sha256,
        core_sha256=q.core.graph_sha256, threads=4, affinity=[0,2,4,6], batch_size=4,
        comparisons=comparisons, records=records, median_paired_speedup=statistics.median(ratios),
        passes_screen=all(x > 1 for x in ratios) and statistics.median(ratios) >= 1.05,
        checkpoint_manifest_unchanged=before==hashlib.sha256(manifest.read_bytes()).hexdigest(),
        extra_coeff_bytes_per_thread=q.core.n*8,
        note='64 groups, original graph target masses; no full run. Single small sample is preliminary.')
    (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('RESULT=' + str(out.resolve()), flush=True)

if __name__ == '__main__':
    main()
