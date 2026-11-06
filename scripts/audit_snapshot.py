"""Read-only audit of a cumulative-snapshot run and its quotient cache."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from parallel_bc.brandes import source_order_digest
from parallel_bc.data import _sha256
from parallel_bc.quotient_snapshot import FORMAT, SLOTS, checkpoint_code_digest
from parallel_bc.tree_quotient import load_quotient

def audit(root, cache):
    root, cache = Path(root), Path(cache)
    m = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if m.get('format') != FORMAT:
        raise ValueError('Not a cumulative snapshot checkpoint')
    i = m['identity']
    q = load_quotient(cache)
    if (q.source_graph_sha256 != i['graph_sha256'] or q.core.graph_sha256 != i['core_graph_sha256']
            or q.n != i['original_nodes'] or q.metadata['original_edges'] != i['original_edges']
            or q.core.n != i['core_nodes'] or q.core.m != i['core_edges']
            or int(q.mass.sum()) != i['source_mass_sum'] or i['core_traversals'] < 0):
        raise ValueError('Graph fingerprint mismatch')
    for name, expected in i['quotient_arrays_sha256'].items():
        actual = hashlib.sha256(np.ascontiguousarray(getattr(q, name)).tobytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Quotient array mismatch: {name}')
    if _sha256(root/'source_plan.npy') != m['plan_file_sha256']:
        raise ValueError('Source plan file hash mismatch')
    plan = np.load(root/'source_plan.npy', allow_pickle=False)
    roots = np.flatnonzero(np.diff(q.core.indptr) > 0).astype(np.int32)
    expected_plan = np.random.default_rng(i['seed']).permutation(roots)
    if (not np.array_equal(plan, expected_plan) or len(plan) != i['core_traversals']
            or source_order_digest(plan) != i['plan_hash']):
        raise ValueError('Source plan coverage mismatch')
    cursor, generation = m['completed_core_groups'], m['generation']
    size = i['chunk_size']
    if (not isinstance(cursor, int) or not 0 <= cursor <= len(plan)
            or (cursor != len(plan) and cursor % size != 0)
            or generation != (cursor + size - 1)//size
            or m['completed_source_mass'] != int(q.mass[plan[:cursor]].sum())
            or m['analytic_source_mass'] != int(q.mass.sum()-q.mass[plan].sum())):
        raise ValueError('Snapshot source coverage mismatch')
    active = m['active_snapshot']
    if generation:
        filename = SLOTS[(generation-1)%2]
        if (active['file'] != filename or active['generation'] != generation
                or active['completed_core_groups'] != cursor
                or active['completed_source_mass'] != m['completed_source_mass']
                or _sha256(root/filename) != active['sha256']):
            raise ValueError('Active snapshot metadata or fingerprint mismatch')
        a = np.load(root/filename, mmap_mode='r', allow_pickle=False)
        if a.shape != (q.core.n,) or a.dtype != np.float64 or not np.isfinite(a).all() or np.any(a < 0):
            raise ValueError('Invalid active snapshot vector')
        del a
    elif active is not None:
        raise ValueError('Unexpected initial snapshot')
    complete = m.get('status') == 'complete'
    if (m.get('status') not in {'paused','complete'} or m.get('exact_all_sources') != complete
            or m.get('graph_full') is not True or m.get('scaled') is not False
            or m.get('result_kind') != ('exact_centrality' if complete else 'incomplete_core_accumulator')):
        raise ValueError('Invalid result semantics')
    if complete:
        if (cursor != len(plan) or m['completed_source_mass']+m['analytic_source_mass'] != q.n
                or m.get('source_count') != q.n or m.get('restored_original_nodes') != q.n):
            raise ValueError('Invalid full-source coverage')
        for name in ('contributions','bc_raw','bc_normalized'):
            item = m['outputs'][name]
            if item['file'] != name+'.npy' or item['nodes'] != q.n or _sha256(root/item['file']) != item['sha256']:
                raise ValueError(f'Final output fingerprint mismatch: {name}')
    return dict(status='verified', run_status=m['status'], exact_all_sources=complete,
        completed_core_groups=cursor, total_core_groups=len(plan), completed_source_mass=m['completed_source_mass'],
        snapshot_generation=generation, active_snapshot=active,
        code_matches_current=checkpoint_code_digest()==i['code_sha256'],
        run_files_bytes=sum(p.stat().st_size for p in root.iterdir() if p.is_file()),
        note='Read-only audit; no per-chunk vectors retained. Code match alone does not validate runtime dependencies.')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_dir', type=Path)
    p.add_argument('--quotient', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(audit(args.run_dir, args.quotient), ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
