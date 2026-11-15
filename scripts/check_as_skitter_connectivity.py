"""Independently check selected full-graph removals by forward CSR BFS."""
import csv
import json
import time
from pathlib import Path

import numpy as np
from numba import njit

from parallel_bc.data import _sha256

ROOT = Path(__file__).resolve().parents[1]
CSR = ROOT / 'data/processed/as-skitter_csr'
RUN = ROOT / 'results/as_skitter_full_12t_2048'
EXPERIMENT = ROOT / 'results/as_skitter_robustness_20261115T153130'


@njit
def largest_remaining(indptr, indices, active):
    n = len(active)
    seen = np.zeros(n, np.uint8)
    queue = np.empty(n, np.int32)
    largest = 0
    for root in range(n):
        if not active[root] or seen[root]:
            continue
        head, tail = 0, 1
        queue[0] = root
        seen[root] = 1
        while head < tail:
            v = queue[head]
            head += 1
            for j in range(indptr[v], indptr[v + 1]):
                w = indices[j]
                if active[w] and not seen[w]:
                    seen[w] = 1
                    queue[tail] = w
                    tail += 1
        largest = max(largest, tail)
    return largest


def main():
    started = time.perf_counter()
    metadata = json.loads((CSR / 'metadata.json').read_text(encoding='utf-8'))
    full = json.loads((RUN / 'manifest.json').read_text(encoding='utf-8'))
    experiment = json.loads((EXPERIMENT / 'manifest.json').read_text(encoding='utf-8'))
    if experiment['status'] != 'complete' or experiment['graph_sha256'] != metadata['graph_sha256']:
        raise ValueError('Experiment or graph identity mismatch')
    if full['status'] != 'complete' or full['identity']['graph_sha256'] != metadata['graph_sha256']:
        raise ValueError('Full-run or graph identity mismatch')
    for name, digest in metadata['cache_files'].items():
        if _sha256(CSR / name) != digest:
            raise ValueError(f'CSR hash mismatch: {name}')
    if _sha256(RUN / 'bc_raw.npy') != full['outputs']['bc_raw']['sha256']:
        raise ValueError('Score hash mismatch')
    if _sha256(EXPERIMENT / 'robustness_raw.csv') != experiment['output_hashes']['robustness_raw.csv']:
        raise ValueError('Experiment result hash mismatch')
    load = lambda path: np.load(path, mmap_mode='r', allow_pickle=False)
    ids, indptr, indices = (load(CSR / name) for name in ('original_ids.npy', 'indptr.npy', 'indices.npy'))
    raw = load(RUN / 'bc_raw.npy')
    n = len(ids)
    orders = {'bc': np.lexsort((ids, -raw)).astype(np.int32),
              'degree': np.lexsort((ids, -np.diff(indptr))).astype(np.int32)}
    child = np.random.SeedSequence(42).spawn(20)[0]
    seed = int(child.generate_state(1, dtype=np.uint64)[0])
    orders['random'] = np.random.default_rng(seed).permutation(n).astype(np.int32)
    with (EXPERIMENT / 'robustness_raw.csv').open(encoding='utf-8', newline='') as f:
        raw_rows = list(csv.DictReader(f))
    result = []
    for strategy, order in orders.items():
        for k in (int(.01 * n), int(.20 * n)):
            active = np.ones(n, np.uint8)
            active[order[:k]] = 0
            actual = int(largest_remaining(indptr, indices, active))
            matches = [r for r in raw_rows if r['strategy'] == strategy and int(r['repeat']) == 0 and int(r['k']) == k]
            if len(matches) != 1:
                raise ValueError('Expected unique result')
            expected = int(matches[0]['lcc_size'])
            result.append(dict(strategy=strategy, repeat=0, k=k, union_find_lcc=expected,
                               independent_bfs_lcc=actual, passed=actual == expected))
            print(f'{strategy}, k={k}: {actual} / {expected}', flush=True)
    report = dict(status='passed' if all(r['passed'] for r in result) else 'failed',
                  graph_sha256=metadata['graph_sha256'], method='forward BFS on original CSR after removal',
                  comparison='reverse union-find', cases=result, elapsed_seconds=time.perf_counter()-started,
                  script_sha256=_sha256(Path(__file__)))
    path = EXPERIMENT / 'connectivity_validation.json'
    with path.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    if report['status'] != 'passed':
        raise AssertionError('Independent connectivity check failed')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
