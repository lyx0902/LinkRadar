"""Export the completed as-skitter run and its audited robustness experiment."""
import csv
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
from export_youtube_web import sanity
from parallel_bc.core_diagnostic import _blocks
from parallel_bc.csr import load_csr
from parallel_bc.data import _sha256

RUN = ROOT / 'results/as_skitter_full_12t_2048'
QUOTIENT = ROOT / 'data/processed/as-skitter_csr_treeq'
CSR = ROOT / 'data/processed/as-skitter_csr'
EXPERIMENT = ROOT / 'results/as_skitter_robustness_20261115T153130'
DEST = ROOT / 'web/linkradar/dist/assets'


def main():
    if psutil.virtual_memory().available < 2 * 1024**3:
        raise MemoryError('Need at least 2 GiB available before export')
    start = time.perf_counter()
    sanity()
    check = audit(RUN, QUOTIENT)
    if not check['exact_all_sources'] or not check['code_matches_current']:
        raise ValueError('Completed matching full-source run required')
    manifest = json.loads((RUN / 'manifest.json').read_text(encoding='utf-8'))
    experiment = json.loads((EXPERIMENT / 'manifest.json').read_text(encoding='utf-8'))
    independent = json.loads((EXPERIMENT / 'connectivity_validation.json').read_text(encoding='utf-8'))
    if (experiment['status'] != 'complete' or independent['status'] != 'passed'
            or independent['graph_sha256'] != manifest['identity']['graph_sha256']
            or experiment['graph_sha256'] != manifest['identity']['graph_sha256']):
        raise ValueError('Mismatch in experiment validation or graph identity')
    summary_path = EXPERIMENT / 'robustness_summary.csv'
    if _sha256(summary_path) != experiment['output_hashes']['robustness_summary.csv']:
        raise ValueError('Robustness summary hash mismatch')
    graph = load_csr(CSR)
    if graph.graph_sha256 != manifest['identity']['graph_sha256']:
        raise ValueError('CSR graph mismatch')
    ids, ptr, indices = graph.ids, graph.indptr, graph.indices
    n = graph.n
    if not np.array_equal(ids, np.arange(n, dtype=np.int64)):
        raise ValueError('Binary format requires verified zero-based contiguous original IDs')
    raw = np.load(RUN / 'bc_raw.npy', mmap_mode='r', allow_pickle=False)
    norm = np.load(RUN / 'bc_normalized.npy', mmap_mode='r', allow_pickle=False)
    if raw.shape != (n,) or norm.shape != (n,) or not np.isfinite(raw).all() or np.any(raw < 0):
        raise ValueError('Invalid full graph scores')
    degree = np.diff(ptr).astype(np.uint32)
    bc_order = np.lexsort((ids, -raw)).astype(np.int32)
    degree_order = np.lexsort((ids, -degree.astype(np.int64))).astype(np.int32)
    bc_rank = np.empty(n, np.uint32); bc_rank[bc_order] = np.arange(1, n + 1, dtype=np.uint32)
    degree_rank = np.empty(n, np.uint32); degree_rank[degree_order] = np.arange(1, n + 1, dtype=np.uint32)
    print('Computing articulation points and original component sizes', flush=True)
    blocks = _blocks(ptr, indices, np.ones(n, np.int64))
    cuts, labels, component_sizes = blocks[4], blocks[7], blocks[8]
    if (len(component_sizes) != graph.metadata['connected_components']
            or int(component_sizes.max()) != graph.metadata['largest_component_nodes']
            or int(component_sizes.sum()) != n):
        raise ValueError('Unexpected component coverage')
    sizes_by_node = component_sizes[labels].astype(np.uint32)
    del blocks, labels, component_sizes
    with summary_path.open(encoding='utf-8', newline='') as f:
        rows = [dict(strategy=r['strategy'], k=int(r['k']), removed_fraction=float(r['removed_fraction']),
                     count=int(r['count']), lcc_original_mean=float(r['lcc_original_mean']),
                     lcc_original_p10=float(r['lcc_original_p10']), lcc_original_p90=float(r['lcc_original_p90']))
                for r in csv.DictReader(f)]
    if len(rows) != 21 or {r['strategy'] for r in rows} != {'bc', 'degree', 'random'}:
        raise ValueError('Unexpected robustness summary')
    def node(i):
        return dict(index=int(i), original_id=int(ids[i]), degree=int(degree[i]), bc_raw=float(raw[i]),
                    bc_normalized=float(norm[i]), rank_bc=int(bc_rank[i]), rank_degree=int(degree_rank[i]),
                    component_size=int(sizes_by_node[i]), is_articulation=bool(cuts[i]))
    top = [node(i) for i in bc_order[:10]]
    anchors = list(map(int, bc_order[:10])); selected = set(anchors)
    neighbors = [sorted(map(int, indices[ptr[v]:ptr[v+1]]), key=lambda x: (-raw[x], ids[x])) for v in anchors]
    for level in range(max(map(len, neighbors))):
        for nb in neighbors:
            if level < len(nb): selected.add(nb[level])
            if len(selected) >= 120: break
        if len(selected) >= 120: break
    edges = [(v, int(w)) for v in sorted(selected) for w in indices[ptr[v]:ptr[v+1]]
             if int(w) in selected and v < int(w)]
    small = nx.Graph(); small.add_nodes_from(sorted(selected)); small.add_edges_from(edges)
    pos = nx.spring_layout(small, seed=42, iterations=60)
    sub = [dict(**node(v), x=float(pos[v][0]), y=float(pos[v][1]), top10=v in anchors) for v in sorted(selected)]
    sessions = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(RUN.glob('session_*.json'))]
    identity = manifest['identity']
    payload = dict(title='as-skitter 互联网拓扑网络', n=n, m=graph.m,
        components=graph.metadata['connected_components'], lcc_size=graph.metadata['largest_component_nodes'],
        average_degree=2*graph.m/n, graph_sha256=graph.graph_sha256, raw_sha256=graph.metadata['raw_sha256'],
        top10=top, subgraph=dict(nodes=sub, edges=edges, note='从全图前十及其邻居轮流选取，布局仅供解释。'),
        robustness=rows, robustness_seed=experiment['seed'], robustness_random_repeats=experiment['random_repeats'],
        full_run=dict(threads=identity['threads'], batch_size=identity['batch_size'], chunk_size=identity['chunk_size'],
            seed=identity['seed'], cpu_affinity=identity['cpu_affinity'], core_nodes=identity['core_nodes'],
            core_edges=identity['core_edges'], core_groups=identity['core_traversals'], chunks=manifest['generation'],
            python=identity['python'], environment=identity['environment'], analytic_source_mass=manifest['analytic_source_mass'],
            sessions=[dict(status=s['status'], seconds=s['t_supervised_total']) for s in sessions],
            seconds_total=sum(s['t_supervised_total'] for s in sessions), component_validation=experiment['component_validation']))
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST/'as_skitter-data.js').write_text('window.LINKRADAR_AS_SKITTER = '+json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+';\n', encoding='utf-8')
    # 28 bytes per node: raw score, degree, two ranks, articulation, component size.
    record = np.empty(n, dtype=[('raw','<f8'), ('degree','<u4'), ('bc_rank','<u4'), ('degree_rank','<u4'), ('cut','<u4'), ('component_size','<u4')])
    record['raw']=raw; record['degree']=degree; record['bc_rank']=bc_rank; record['degree_rank']=degree_rank
    record['cut']=cuts; record['component_size']=sizes_by_node
    with (DEST/'as_skitter-nodes.bin').open('wb') as f:
        record.tofile(f); bc_order.astype('<u4').tofile(f); degree_order.astype('<u4').tofile(f)
    with (DEST/'as_skitter_top10.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(top[0])); writer.writeheader(); writer.writerows(top)
    with (DEST/'as_skitter_robustness.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sample = np.random.default_rng(42).choice(n, size=min(30000, n), replace=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(degree[sample], norm[sample], s=2, alpha=.18, color='#197b6e')
    ax.scatter(degree[bc_order[:10]], norm[bc_order[:10]], s=26, color='#c57d44')
    for v in bc_order[:10]: ax.annotate(str(ids[v]), (degree[v], norm[v]), fontsize=8)
    ax.set_xscale('symlog', linthresh=1); ax.set_yscale('symlog', linthresh=1e-6)
    ax.set_xlabel('Degree'); ax.set_ylabel('Normalized betweenness'); ax.set_title('as-skitter: 30,000-node sample + full-graph top 10')
    fig.tight_layout(); fig.savefig(DEST/'as_skitter_degree_bc.png', dpi=130); plt.close(fig)
    print(json.dumps(dict(status='ok', seconds=time.perf_counter()-start, nodes=n, components=graph.metadata['connected_components'],
        node_bytes=(DEST/'as_skitter-nodes.bin').stat().st_size, node_record_bytes=28,
        top_node=top[0], snapshot_code_matches=check['code_matches_current']), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
