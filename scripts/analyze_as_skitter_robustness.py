"""Run an isolated fixed-ranking robustness study and small-component check."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from audit_snapshot import audit
from export_youtube_web import connectivity
from parallel_bc.csr import csr_from_edges, load_csr
from parallel_bc.data import _sha256

RUN = ROOT / 'results/as_skitter_full_12t_2048'
CSR = ROOT / 'data/processed/as-skitter_csr'
QUOTIENT = ROOT / 'data/processed/as-skitter_csr_treeq'
FRACTIONS = [0, .01, .02, .05, .10, .15, .20]
REPEATS = 20
SEED = 42
MIN_AVAILABLE = 1024 ** 3


def check_memory():
    available = psutil.virtual_memory().available
    if available < MIN_AVAILABLE:
        raise MemoryError(f'System available memory below 1 GiB: {available} bytes')
    return available


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def sanity():
    """Compare reverse-union connectivity against explicit deletion on small graphs."""
    examples = [nx.path_graph(7), nx.empty_graph(6),
                nx.disjoint_union(nx.cycle_graph(5), nx.path_graph(4)),
                nx.gnp_random_graph(31, .12, seed=7)]
    for graph in examples:
        n = len(graph)
        csr = csr_from_edges(graph.edges(), nodes=np.arange(n))
        order = np.random.default_rng(9).permutation(n).astype(np.int32)
        targets = np.array(sorted(set([0, 1, n // 2, n])), dtype=np.int64)
        expected = []
        for k in targets:
            h = graph.copy()
            h.remove_nodes_from(order[:k])
            expected.append(max((len(c) for c in nx.connected_components(h)), default=0))
        actual = connectivity(csr.indptr, csr.indices, order, targets)
        if not np.array_equal(actual, expected):
            raise AssertionError('Reverse-union connectivity mismatch')


def verify_components(graph, raw, normalized):
    """Validate every non-largest connected component against NetworkX."""
    load = lambda name: np.load(QUOTIENT / f'{name}.npy', mmap_mode='r', allow_pickle=False)
    anchors = load('anchor_index')
    core_labels = load('component_index')
    sizes = load('component_sizes')
    labels = core_labels[anchors]
    counts = np.bincount(labels, minlength=len(sizes))
    if not np.array_equal(counts, sizes) or len(sizes) != graph.metadata['connected_components']:
        raise ValueError('Original component sizes differ from quotient and CSR metadata')
    giant = int(np.argmax(sizes))
    if int(sizes[giant]) != graph.metadata['largest_component_nodes']:
        raise ValueError('Largest component differs from CSR metadata')
    selected = np.flatnonzero(labels != giant)
    order = selected[np.argsort(labels[selected], kind='stable')]
    bounds = np.r_[0, np.flatnonzero(np.diff(labels[order])) + 1, len(order)]
    nodes_rows, component_rows = [], []
    global_factor = ((graph.n - 1) * (graph.n - 2) / 2) if graph.n > 2 else None
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        members = order[lo:hi]
        member_set = set(map(int, members))
        g = nx.Graph()
        g.add_nodes_from(member_set)
        for v in members:
            for w in graph.indices[graph.indptr[v]:graph.indptr[v+1]]:
                w = int(w)
                if w not in member_set:
                    raise ValueError('Selected component has an external edge')
                if v < w:
                    g.add_edge(int(v), w)
        if not nx.is_connected(g):
            raise ValueError('Selected component is not connected in original CSR')
        reference = nx.betweenness_centrality(g, normalized=False, weight=None, endpoints=False)
        is_tree = nx.is_tree(g)
        for v in members:
            expected = reference[int(v)]
            actual = float(raw[v])
            expected_normalized = expected / global_factor if global_factor else 0.
            abs_diff = abs(actual - expected)
            norm_diff = abs(float(normalized[v]) - expected_normalized)
            nodes_rows.append(dict(component=int(labels[v]), original_id=int(graph.ids[v]),
                                   raw_actual=actual, raw_networkx=expected, raw_abs_diff=abs_diff,
                                   normalized_actual=float(normalized[v]), normalized_networkx_global=expected_normalized,
                                   normalized_abs_diff=norm_diff,
                                   passed=bool(np.isclose(actual, expected, rtol=1e-7, atol=1e-8)
                                               and np.isclose(normalized[v], expected_normalized, rtol=1e-7, atol=1e-12))))
        component_rows.append(dict(component=int(labels[members[0]]), n=len(members), m=g.number_of_edges(),
                                   is_tree=is_tree, minimum_original_id=int(graph.ids[members].min())))
    if len(component_rows) != len(sizes) - 1 or len(nodes_rows) != graph.n - int(sizes[giant]):
        raise ValueError('Non-largest component coverage mismatch')
    return nodes_rows, component_rows, int(sizes[giant])


def main():
    started = time.perf_counter()
    available_start = check_memory()
    output = ROOT / 'results' / time.strftime('as_skitter_robustness_%Y%m%dT%H%M%S')
    output.mkdir(parents=True, exist_ok=False)
    state = dict(status='running', data='as-skitter', seed=SEED, random_repeats=REPEATS,
                 fractions=FRACTIONS, memory_precheck_bytes=MIN_AVAILABLE,
                 script_sha256=_sha256(Path(__file__)), available_start_bytes=available_start)
    def save_state():
        (output / 'manifest.json').write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding='utf-8')
    save_state()
    try:
        sanity()
        print('Small-graph connectivity sanity passed', flush=True)
        check = audit(RUN, QUOTIENT)
        if not check['exact_all_sources'] or not check['code_matches_current']:
            raise ValueError('Completed matching snapshot required')
        check_memory()
        graph = load_csr(CSR)
        run_manifest = json.loads((RUN / 'manifest.json').read_text(encoding='utf-8'))
        if graph.graph_sha256 != run_manifest['identity']['graph_sha256']:
            raise ValueError('Original CSR fingerprint mismatch')
        ids = np.load(QUOTIENT / 'original_ids.npy', mmap_mode='r', allow_pickle=False)
        if not np.array_equal(ids, graph.ids):
            raise ValueError('Original ID mapping mismatch')
        raw = np.load(RUN / 'bc_raw.npy', mmap_mode='r', allow_pickle=False)
        norm = np.load(RUN / 'bc_normalized.npy', mmap_mode='r', allow_pickle=False)
        if raw.shape != (graph.n,) or norm.shape != (graph.n,) or not np.isfinite(raw).all() or np.any(raw < 0):
            raise ValueError('Invalid final score arrays')
        nodes, components, largest = verify_components(graph, raw, norm)
        write_csv(output / 'small_components.csv', components)
        write_csv(output / 'small_component_nodes.csv', nodes)
        failed = sum(not x['passed'] for x in nodes)
        state['component_validation'] = dict(components=len(components), nodes=len(nodes),
            tree_components=sum(x['is_tree'] for x in components),
            cyclic_components=sum(not x['is_tree'] for x in components),
            max_component_nodes=max(x['n'] for x in components),
            nodes_with_nonzero_reference=sum(x['raw_networkx'] > 0 for x in nodes), failed_nodes=failed,
            max_raw_abs_diff=max(x['raw_abs_diff'] for x in nodes),
            max_normalized_abs_diff=max(x['normalized_abs_diff'] for x in nodes),
            atol_raw=1e-8, rtol=1e-7, atol_normalized=1e-12, original_n_for_normalization=graph.n,
            graph_closure='checked every neighbor of each selected component',
            scope='all non-largest components, not giant component')
        save_state()
        print(f'Checked {len(components)} small components / {len(nodes)} nodes; failures={failed}', flush=True)
        if failed:
            raise AssertionError('Small component centrality mismatch')
        check_memory()
        degrees = np.diff(graph.indptr)
        bc_order = np.lexsort((graph.ids, -raw)).astype(np.int32)
        degree_order = np.lexsort((graph.ids, -degrees)).astype(np.int32)
        targets = np.array([int(q * graph.n) for q in FRACTIONS], dtype=np.int64)
        rows = []
        children = np.random.SeedSequence(SEED).spawn(REPEATS)
        strategies = [('bc', 0, None, bc_order), ('degree', 0, None, degree_order)]
        for strategy, repeat, random_seed, order in strategies:
            check_memory()
            sizes = connectivity(graph.indptr, graph.indices, order, targets)
            rows.extend(dict(strategy=strategy, repeat=repeat, seed=random_seed, k=int(k),
                             removed_fraction=int(k)/graph.n, lcc_size=int(size),
                             lcc_original_fraction=int(size)/graph.n) for k, size in zip(targets, sizes))
            print(f'{strategy}: completed', flush=True)
        for repeat, child in enumerate(children):
            check_memory()
            random_seed = int(child.generate_state(1, dtype=np.uint64)[0])
            order = np.random.default_rng(random_seed).permutation(graph.n).astype(np.int32)
            sizes = connectivity(graph.indptr, graph.indices, order, targets)
            rows.extend(dict(strategy='random', repeat=repeat, seed=random_seed, k=int(k),
                             removed_fraction=int(k)/graph.n, lcc_size=int(size),
                             lcc_original_fraction=int(size)/graph.n) for k, size in zip(targets, sizes))
            write_csv(output / 'robustness_raw.csv', rows)
            print(f'random: {repeat + 1}/{REPEATS}', flush=True)
        if any(r['k'] == 0 and r['lcc_size'] != largest for r in rows):
            raise AssertionError('Baseline largest component mismatch')
        summary = []
        for strategy in ('bc', 'degree', 'random'):
            for k in targets:
                group = [r for r in rows if r['strategy'] == strategy and r['k'] == k]
                a = np.array([r['lcc_original_fraction'] for r in group])
                summary.append(dict(strategy=strategy, k=int(k), removed_fraction=int(k)/graph.n,
                    count=len(group), lcc_original_mean=float(a.mean()),
                    lcc_original_p10=float(np.quantile(a,.1)), lcc_original_p90=float(np.quantile(a,.9)),
                    lcc_original_min=float(a.min()), lcc_original_max=float(a.max())))
        write_csv(output / 'robustness_summary.csv', summary)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for strategy, label, color in [('bc','Betweenness','#197b6e'),('degree','Degree','#c57d44'),('random','Random mean (20)','#6b88a5')]:
            selected = [r for r in summary if r['strategy'] == strategy]
            x = np.array([r['removed_fraction'] for r in selected])
            ax.plot(x, [r['lcc_original_mean'] for r in selected], marker='o', label=label, color=color)
            if strategy == 'random':
                ax.fill_between(x, [r['lcc_original_p10'] for r in selected],
                                [r['lcc_original_p90'] for r in selected], color=color, alpha=.2, label='Random p10–p90')
        ax.set(xlabel='Removed nodes / original nodes', ylabel='Largest component / original nodes',
               title='as-skitter: fixed initial rankings', ylim=(0, 1.02))
        ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
        fig.savefig(output / 'robustness.png', dpi=150); plt.close(fig)
        state.update(status='complete', graph_sha256=graph.graph_sha256, original_nodes=graph.n,
            original_edges=graph.m, initial_lcc=largest, protocol='fixed_initial_ranking',
            tie_break='original_id_ascending', random_band='empirical_p10_p90',
            snapshot_audit=check, seconds_total=time.perf_counter()-started,
            output_hashes={p.name: _sha256(p) for p in output.iterdir() if p.name != 'manifest.json'})
        save_state()
        print(json.dumps(dict(status='complete', output=str(output), seconds=state['seconds_total'],
                              validation=state['component_validation'], last=[r for r in summary if r['k']==targets[-1]]),
                         ensure_ascii=False, indent=2), flush=True)
    except Exception as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}', seconds_total=time.perf_counter()-started)
        save_state()
        raise


if __name__ == '__main__':
    main()
