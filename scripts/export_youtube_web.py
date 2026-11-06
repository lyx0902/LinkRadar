"""Generate YouTube full-node data and fixed-rank failure experiment for the web."""
import csv
import json
import sys
from pathlib import Path
import numpy as np
import networkx as nx
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from audit_snapshot import audit
from parallel_bc.core_diagnostic import _blocks
from parallel_bc.csr import load_csr

@njit
def connectivity(indptr, indices, order, targets):
    n = len(order)
    parent = np.full(n, -1, np.int32)
    size = np.ones(n, np.int32)
    out = np.zeros(len(targets), np.int64)
    added = 0
    largest = 0
    for t in range(len(targets)-1, -1, -1):
        want = n - targets[t]
        while added < want:
            v = order[n - 1 - added]
            parent[v] = v
            if largest == 0:
                largest = 1
            for k in range(indptr[v], indptr[v+1]):
                w = indices[k]
                if parent[w] < 0:
                    continue
                a, b = v, w
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                while parent[b] != b:
                    parent[b] = parent[parent[b]]
                    b = parent[b]
                if a != b:
                    if size[a] < size[b]:
                        a, b = b, a
                    parent[b] = a
                    size[a] += size[b]
                    largest = max(largest, size[a])
            added += 1
        out[t] = largest
    return out

def sanity():
    g = nx.gnp_random_graph(31, .12, seed=7)
    p = np.array([0] + list(np.cumsum([len(g[v]) for v in g])), dtype=np.int64)
    ix = np.array([u for v in g for u in g[v]], dtype=np.int32)
    order = np.random.default_rng(9).permutation(len(g)).astype(np.int32)
    targets = np.array([0, 1, 6, 15, 31], dtype=np.int64)
    expected = []
    for k in targets:
        h = g.copy(); h.remove_nodes_from(order[:k])
        expected.append(max((len(c) for c in nx.connected_components(h)), default=0))
    assert np.array_equal(connectivity(p, ix, order, targets), expected)
    cuts = _blocks(p, ix, np.ones(len(g), np.int64))[4]
    assert set(np.flatnonzero(cuts)) == set(nx.articulation_points(g))

def main():
    sanity()
    run = ROOT/'results/youtube_all_snapshot_8t_2048'
    cache = ROOT/'data/processed/youtube_csr_treeq'
    csr = ROOT/'data/processed/youtube_csr'
    dest = ROOT/'web/linkradar/dist/assets'
    check = audit(run, cache)
    if not check['exact_all_sources'] or not check['code_matches_current']:
        raise ValueError('Completed matching checkpoint required')
    manifest = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    sessions = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(run.glob('session_*.json'))]
    if len(sessions) != 2 or [s['status'] for s in sessions] != ['resource_limit', 'ok']:
        raise ValueError('Unexpected full-run session records')
    ident = manifest['identity']
    graph = load_csr(csr)
    if graph.graph_sha256 != manifest['identity']['graph_sha256']:
        raise ValueError('Original CSR fingerprint mismatch')
    load = lambda path: np.load(path, mmap_mode='r', allow_pickle=False)
    ids, ptr, indices = (load(csr/name) for name in ('original_ids.npy','indptr.npy','indices.npy'))
    raw, norm = (load(run/name) for name in ('bc_raw.npy','bc_normalized.npy'))
    n = len(ids)
    if not np.array_equal(ids, load(cache/'original_ids.npy')) or not np.array_equal(ids, np.arange(1,n+1)):
        raise ValueError('Unexpected node ID mapping')
    if not np.isfinite(raw).all() or np.any(raw < 0):
        raise ValueError('Invalid centrality')
    degree = np.diff(ptr).astype(np.uint32)
    bc_order = np.lexsort((ids, -raw)).astype(np.int32)
    degree_order = np.lexsort((ids, -degree.astype(np.int64))).astype(np.int32)
    bc_rank = np.empty(n, np.uint32); bc_rank[bc_order] = np.arange(1,n+1,dtype=np.uint32)
    degree_rank = np.empty(n, np.uint32); degree_rank[degree_order] = np.arange(1,n+1,dtype=np.uint32)
    blocks = _blocks(ptr, indices, np.ones(n,np.int64)); cuts = blocks[4]; labels = blocks[7]; comp = blocks[8]
    if len(comp) != 1 or int(comp[0]) != n:
        raise ValueError('Expected connected YouTube graph')
    del blocks
    fractions = [0,.01,.02,.05,.10,.15,.20]
    targets = np.array([int(x*n) for x in fractions], dtype=np.int64)
    curves = {'bc':[connectivity(ptr,indices,bc_order,targets)], 'degree':[connectivity(ptr,indices,degree_order,targets)], 'random':[]}
    for child in np.random.SeedSequence(42).spawn(20):
        seed = int(child.generate_state(1,dtype=np.uint64)[0])
        curves['random'].append(connectivity(ptr, indices, np.random.default_rng(seed).permutation(n).astype(np.int32), targets))
    rows=[]
    for strategy, values in curves.items():
        a=np.array(values,dtype=float)/n
        for j,k in enumerate(targets):
            rows.append(dict(strategy=strategy,k=int(k),removed_fraction=float(k/n),count=len(a),
                lcc_original_mean=float(a[:,j].mean()),lcc_original_p10=float(np.quantile(a[:,j],.1)),lcc_original_p90=float(np.quantile(a[:,j],.9))))
    def node(i):
        return dict(index=int(i),original_id=int(ids[i]),degree=int(degree[i]),bc_raw=float(raw[i]),
                    bc_normalized=float(norm[i]),rank_bc=int(bc_rank[i]),rank_degree=int(degree_rank[i]),
                    component_size=n,is_articulation=bool(cuts[i]))
    top=[node(i) for i in bc_order[:10]]
    anchors=list(map(int,bc_order[:10])); selected=set(anchors)
    neighbors=[sorted(map(int,indices[ptr[v]:ptr[v+1]]),key=lambda x:(-raw[x],ids[x])) for v in anchors]
    for level in range(max(map(len,neighbors))):
        for nb in neighbors:
            if level<len(nb):selected.add(nb[level])
            if len(selected)>=120:break
        if len(selected)>=120:break
    edges=[]
    for v in sorted(selected):
        edges.extend((v,int(w)) for w in indices[ptr[v]:ptr[v+1]] if int(w) in selected and v<int(w))
    graph=nx.Graph();graph.add_nodes_from(sorted(selected));graph.add_edges_from(edges)
    pos=nx.spring_layout(graph,seed=42,iterations=60)
    sub=[dict(**node(v),x=float(pos[v][0]),y=float(pos[v][1]),top10=v in anchors) for v in sorted(selected)]
    payload=dict(title='YouTube 社交网络',n=n,m=manifest['identity']['original_edges'],components=1,lcc_size=n,
      average_degree=2*manifest['identity']['original_edges']/n,graph_sha256=manifest['identity']['graph_sha256'],
      top10=top,subgraph=dict(nodes=sub,edges=edges,note='从全图前十及其邻居轮流选取，布局仅供解释。'),robustness=rows,
      robustness_seed=42,robustness_random_repeats=20,
      full_run=dict(threads=ident['threads'],batch_size=ident['batch_size'],chunk_size=ident['chunk_size'],
        seed=ident['seed'],cpu_affinity=ident['cpu_affinity'],core_nodes=ident['core_nodes'],
        core_edges=ident['core_edges'],core_groups=ident['core_traversals'],chunks=manifest['generation'],
        python=ident['python'],environment=ident['environment'],reserve_bytes=sessions[0]['reserve_bytes'],
        first_seconds=sessions[0]['t_supervised_total'],second_seconds=sessions[1]['t_supervised_total'],
        second_work_seconds=sessions[1]['t_session_work'],second_new_chunks=sessions[1]['new_chunks'],
        first_min_available=sessions[0]['min_available_bytes'],first_status=sessions[0]['status'],
        second_status=sessions[1]['status']))
    dest.mkdir(parents=True,exist_ok=True)
    (dest/'youtube-data.js').write_text('window.LINKRADAR_YOUTUBE = '+json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False)+';\n',encoding='utf-8')
    # Node records in original-ID order, followed by BC and degree index orders.
    record=np.empty(n,dtype=[('raw','<f8'),('degree','<u4'),('bc_rank','<u4'),('degree_rank','<u4'),('cut','<u4')])
    record['raw']=raw;record['degree']=degree;record['bc_rank']=bc_rank;record['degree_rank']=degree_rank;record['cut']=cuts
    with (dest/'youtube-nodes.bin').open('wb') as f:
        record.tofile(f);bc_order.astype('<u4').tofile(f);degree_order.astype('<u4').tofile(f)
    with (dest/'youtube_top10.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(top[0]));w.writeheader();w.writerows(top)
    with (dest/'youtube_robustness.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rng=np.random.default_rng(42);sample=rng.choice(n,size=min(30000,n),replace=False)
    fig,ax=plt.subplots(figsize=(10,5));ax.scatter(degree[sample],norm[sample],s=2,alpha=.18,color='#197b6e');ax.scatter(degree[bc_order[:10]],norm[bc_order[:10]],s=26,color='#c57d44')
    for v in bc_order[:10]:ax.annotate(str(ids[v]),(degree[v],norm[v]),fontsize=8)
    ax.set_xscale('symlog',linthresh=1);ax.set_yscale('symlog',linthresh=1e-6);ax.set_xlabel('Degree');ax.set_ylabel('Normalized betweenness');ax.set_title('YouTube: 30,000-node sample + full-graph top 10');fig.tight_layout();fig.savefig(dest/'youtube_degree_bc.png',dpi=130);plt.close(fig)
    print(json.dumps(dict(status='ok',nodes=n,robustness=rows[-1],subgraph_nodes=len(selected),bytes=(dest/'youtube-nodes.bin').stat().st_size)))

if __name__ == '__main__':main()
