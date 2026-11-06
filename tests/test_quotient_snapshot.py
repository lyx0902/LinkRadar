"""End-to-end snapshot correctness and transaction boundaries."""
import json
import os
from pathlib import Path
import subprocess
import sys

import networkx as nx
import numpy as np
import pytest

from parallel_bc.csr import csr_from_edges
from parallel_bc.quotient_brandes import quotient_contributions
from parallel_bc.quotient_checkpoint_dag import checkpoint_contributions as chunks
from parallel_bc.quotient_snapshot import FORMAT, checkpoint_contributions, run_checkpoint
from parallel_bc.tree_quotient import factor_trees, load_quotient, save_quotient

def graph():
    g = nx.barbell_graph(5, 2)
    g.add_edges_from([(0,12),(12,13),(12,14)])
    return nx.disjoint_union_all([g, nx.path_graph(5), nx.empty_graph(2)])

def cached(tmp_path, g=None):
    g = graph() if g is None else g
    csr = csr_from_edges(g.edges(), nodes=np.arange(len(g)))
    path = save_quotient(factor_trees(csr), tmp_path / 'q')
    return g, path, load_quotient(path)

def manifest(root):
    return json.loads((root / 'manifest.json').read_text(encoding='utf-8'))

OPTIONS = dict(threads=2, batch_size=1, chunk_size=3, seed=7)

@pytest.mark.parametrize('g', [graph(), nx.empty_graph(0), nx.empty_graph(5), nx.path_graph(15),
                               nx.cycle_graph(9), nx.gnp_random_graph(35, .13, seed=7)])
def test_full_result_and_restart(tmp_path, g):
    g, _, q = cached(tmp_path, g)
    root = tmp_path/'run'
    row = checkpoint_contributions(q, root, max_chunks=1, **OPTIONS)
    if not row['exact_all_sources']:
        assert not (root/'bc_raw.npy').exists()
        row = checkpoint_contributions(q, root, resume=True, **OPTIONS)
    assert row['exact_all_sources'] and row['run_status'] == 'complete'
    raw, normalized = np.load(root/'bc_raw.npy'), np.load(root/'bc_normalized.npy')
    np.testing.assert_allclose(raw, list(nx.betweenness_centrality(g, normalized=False).values()), atol=1e-9, rtol=1e-11)
    np.testing.assert_allclose(normalized, list(nx.betweenness_centrality(g, normalized=True).values()), atol=1e-10)
    direct, _ = quotient_contributions(q, threads=2, batch_size=1)
    np.testing.assert_allclose(np.load(root/'contributions.npy'), direct, atol=1e-9)
    m = manifest(root)
    assert m['format'] == FORMAT and m['completed_source_mass'] + m['analytic_source_mass'] == q.n
    assert len(list(root.glob('snapshot_*.npy'))) <= 2
    assert not list(root.glob('chunk_*.npy'))
    assert checkpoint_contributions(q, root, resume=True, **OPTIONS)['new_chunks'] == 0

@pytest.mark.parametrize('stage', ['after_compute','after_snapshot','after_manifest'])
def test_fault_boundaries_no_missing_or_double_counts(tmp_path, stage):
    _, _, q = cached(tmp_path)
    root = tmp_path/'run'
    def fault(here, generation):
        if here == stage and generation == 1:
            raise RuntimeError('injected fault')
    with pytest.raises(RuntimeError, match='injected'):
        checkpoint_contributions(q, root, _fault=fault, **OPTIONS)
    m = manifest(root)
    assert m['generation'] == (2 if stage == 'after_manifest' else 1)
    assert m['active_snapshot']['file'] == ('snapshot_B.npy' if stage == 'after_manifest' else 'snapshot_A.npy')
    row = checkpoint_contributions(q, root, resume=True, **OPTIONS)
    assert row['exact_all_sources']
    direct, _ = quotient_contributions(q)
    np.testing.assert_allclose(np.load(root/'contributions.npy'), direct, atol=1e-9)

@pytest.mark.parametrize('stage', ['after_snapshot','after_manifest'])
def test_real_process_death_and_supervised_recovery(tmp_path, stage):
    _, path, q = cached(tmp_path)
    root = tmp_path/'run'
    code = '''
import os,sys
from parallel_bc.tree_quotient import load_quotient
from parallel_bc.quotient_snapshot import checkpoint_contributions
def fault(stage, generation):
    if stage == sys.argv[3] and generation == 1:
        os._exit(23)
checkpoint_contributions(load_quotient(sys.argv[1]),sys.argv[2],threads=2,batch_size=1,chunk_size=3,seed=7,_fault=fault)
'''
    p = subprocess.run([sys.executable,'-c',code,str(path),str(root),stage],capture_output=True,timeout=60)
    assert p.returncode == 23, p.stderr.decode(errors='replace')
    assert manifest(root)['generation'] == (1 if stage == 'after_snapshot' else 2)
    result = run_checkpoint(path, root, resume=True, seconds=60, min_available_bytes=0, **OPTIONS)
    assert result['status'] == 'ok' and result['exact_all_sources'], result
    direct, _ = quotient_contributions(q)
    np.testing.assert_allclose(np.load(root/'contributions.npy'), direct, atol=1e-9)

@pytest.mark.parametrize('change', ['snapshot','plan','count','mass','config','format','digest'])
def test_corruption_or_identity_change_refused(tmp_path, monkeypatch, change):
    _, _, q = cached(tmp_path)
    root = tmp_path/'run'
    checkpoint_contributions(q, root, max_chunks=1, **OPTIONS)
    opts = dict(OPTIONS)
    if change in ('snapshot','plan'):
        path = root/('snapshot_A.npy' if change == 'snapshot' else 'source_plan.npy')
        with path.open('ab') as f: f.write(b'corrupt')
    elif change == 'config':
        opts['chunk_size'] = 4
    elif change == 'digest':
        monkeypatch.setattr('parallel_bc.quotient_snapshot.checkpoint_code_digest', lambda:'changed')
    else:
        m=manifest(root)
        if change == 'count': m['completed_core_groups'] += 1
        elif change == 'mass': m['completed_source_mass'] += 1
        else: m['format'] = 'wrong'
        (root/'manifest.json').write_text(json.dumps(m),encoding='utf8')
    with pytest.raises(ValueError, match='mismatch'):
        checkpoint_contributions(q, root, resume=True, **opts)

def test_cross_format_refusal(tmp_path):
    _, _, q = cached(tmp_path)
    old, new = tmp_path/'old', tmp_path/'new'
    chunks(q, old, max_chunks=1, **OPTIONS)
    checkpoint_contributions(q, new, max_chunks=1, **OPTIONS)
    with pytest.raises(ValueError, match='mismatch'):
        checkpoint_contributions(q, old, resume=True, **OPTIONS)
    with pytest.raises(ValueError, match='mismatch'):
        chunks(q, new, resume=True, **OPTIONS)

def test_partial_snapshot_write_keeps_committed_slot(tmp_path, monkeypatch):
    import parallel_bc.quotient_snapshot as mod
    _, _, q = cached(tmp_path)
    root = tmp_path/'run'
    checkpoint_contributions(q, root, max_chunks=2, **OPTIONS)
    before = manifest(root)
    assert before['active_snapshot']['file'] == 'snapshot_B.npy'
    write = mod._write_array
    def fail_during_write(path, value):
        if path.name == 'snapshot_A.npy':
            path.with_suffix('.npy.tmp').write_bytes(b'partial')
            raise OSError('injected partial write')
        return write(path, value)
    with monkeypatch.context() as m:
        m.setattr(mod, '_write_array', fail_during_write)
        with pytest.raises(OSError, match='partial write'):
            checkpoint_contributions(q, root, resume=True, **OPTIONS)
    assert manifest(root) == before
    row = checkpoint_contributions(q, root, resume=True, **OPTIONS)
    assert row['exact_all_sources']
    direct, _ = quotient_contributions(q)
    np.testing.assert_allclose(np.load(root/'contributions.npy'), direct, atol=1e-9)

def test_lock_no_overwrite_and_resource_guard(tmp_path):
    from parallel_bc.large_benchmark import bounded_job
    from parallel_bc.quotient_snapshot import _run_lock
    _, _, q = cached(tmp_path)
    root = tmp_path/'run'
    checkpoint_contributions(q, root, max_chunks=0, **OPTIONS)
    with pytest.raises(FileExistsError):
        checkpoint_contributions(q, root, **OPTIONS)
    with _run_lock(root):
        with pytest.raises(RuntimeError, match='locked'):
            checkpoint_contributions(q, root, resume=True, **OPTIONS)
    assert bounded_job('quotient_snapshot', {}, seconds=.001)['status'] == 'timeout'
    assert bounded_job('quotient_snapshot', {}, seconds=1, min_available_bytes=2**100)['status'] == 'resource_not_run'

def test_readonly_audit_and_corruption_detection(tmp_path):
    import hashlib
    from scripts.audit_snapshot import audit
    _, cache, q = cached(tmp_path)
    root = tmp_path/'run'
    checkpoint_contributions(q, root, max_chunks=1, **OPTIONS)
    def state():
        return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()}
    before = state()
    result = audit(root, cache)
    assert result['status'] == 'verified' and result['completed_core_groups'] == 3
    assert result['code_matches_current'] and state() == before
    with (root/'snapshot_A.npy').open('ab') as f: f.write(b'broken')
    with pytest.raises(ValueError, match='fingerprint mismatch'):
        audit(root, cache)

def test_snapshot_with_valid_hash_but_invalid_values_is_rejected(tmp_path):
    import hashlib
    _, _, q = cached(tmp_path)
    root=tmp_path/'run'
    checkpoint_contributions(q,root,max_chunks=1,**OPTIONS)
    m=manifest(root)
    p=root/m['active_snapshot']['file']
    arr=np.load(p,allow_pickle=False)
    arr[0]=np.nan
    with p.open('wb') as f: np.save(f,arr,allow_pickle=False)
    m['active_snapshot']['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    (root/'manifest.json').write_text(json.dumps(m),encoding='utf8')
    with pytest.raises(ValueError,match='Invalid cumulative snapshot'):
        checkpoint_contributions(q,root,resume=True,**OPTIONS)

def test_snapshot_cli_default_and_old_chunk_auto_resume(tmp_path):
    import shutil
    _, path, _ = cached(tmp_path)
    data = tmp_path/'data'/'processed'
    data.mkdir(parents=True)
    shutil.copytree(path, data/'tiny_treeq')
    def command(root):
        return [sys.executable,'-m','parallel_bc','quotient-run','--dataset','tiny',
            '--data-dir',str(tmp_path/'data'),'--run-dir',str(root),'--threads','2',
            '--batch-size','1','--chunk-size','3','--seed','7','--reserve-gib','0','--timeout','60']
    snapshot, chunk, legacy = tmp_path/'snapshot', tmp_path/'chunk', tmp_path/'legacy'
    for root, mode, resume_mode, fmt in (
            (snapshot, [], [], FORMAT),
            (chunk, ['--checkpoint-mode','chunks'], [], 'LinkRadar-quotient-dag-checkpoint-v1'),
            (legacy, ['--kernel','legacy'], ['--kernel','legacy'], 'LinkRadar-quotient-checkpoint-v1')):
        first = subprocess.run(command(root)+mode+['--max-chunks','1'],capture_output=True,timeout=60)
        assert first.returncode == 0, first.stderr.decode(errors='replace')
        assert manifest(root)['format'] == fmt
        second = subprocess.run(command(root)+resume_mode+['--resume'],capture_output=True,timeout=60)
        assert second.returncode == 0, second.stderr.decode(errors='replace')
        assert manifest(root)['exact_all_sources']
    np.testing.assert_allclose(np.load(snapshot/'bc_raw.npy'),np.load(chunk/'bc_raw.npy'),atol=1e-9)

@pytest.mark.parametrize('stage', ['final_array','final_manifest'])
def test_final_interruption_reuses_snapshot(tmp_path, monkeypatch, stage):
    import parallel_bc.quotient_snapshot as mod
    _, _, q = cached(tmp_path)
    root=tmp_path/'run'
    writer=mod._write_array if stage == 'final_array' else mod._write_json
    def fail(path, value):
        if (stage == 'final_array' and path.name == 'bc_raw.npy') or (stage == 'final_manifest' and path.name == 'manifest.json' and value.get('status') == 'complete'):
            raise RuntimeError('finalization interrupted')
        writer(path,value)
    with monkeypatch.context() as m:
        m.setattr(mod,'_write_array' if stage == 'final_array' else '_write_json',fail)
        with pytest.raises(RuntimeError, match='finalization'):
            checkpoint_contributions(q,root,**OPTIONS)
    before=manifest(root)
    assert before['completed_core_groups']==before['identity']['core_traversals'] and not before['exact_all_sources']
    result=checkpoint_contributions(q,root,resume=True,**OPTIONS)
    assert result['new_chunks']==0 and result['exact_all_sources']
    assert manifest(root)['active_snapshot']==before['active_snapshot']

def test_bigint_fallback(tmp_path):
    layers = [[0]] + [[2*i-1,2*i] for i in range(1,66)]
    g = nx.Graph([(a,b) for left,right in zip(layers,layers[1:]) for a in left for b in right])
    g.add_edges_from([(layers[35][0],131),(131,132)])
    _, _, q=cached(tmp_path,g)
    root=tmp_path/'run'
    row=checkpoint_contributions(q,root,threads=2,chunk_size=16)
    assert row['exact_all_sources']
    direct, meta=quotient_contributions(q)
    assert meta['fallback_sources'] > 0
    np.testing.assert_allclose(np.load(root/'contributions.npy'),direct,atol=1e-8,rtol=1e-10)
