"""Full Facebook adjacent-block baseline check; never changes the Brandes kernel."""
from __future__ import annotations
import argparse,csv,hashlib,json,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import psutil
from parallel_bc.benchmark import bounded_call,environment,source_fingerprint,write_csv,write_json
from parallel_bc.data import load_cached_graph


def build_schedule():
    modes=[('serial',1),('dynamic_batches',1),('dynamic_batches',4)]
    return [(b,pos,*modes[(b+pos)%3]) for b in range(3) for pos in range(3)]


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference-run',default='results/20261023T054258_364875Z');p.add_argument('--budget',type=float,default=1200);a=p.parse_args()
    graph=load_cached_graph('data/processed/facebook');ref=Path(a.reference_run)/'facebook'
    meta=json.loads((ref/'centrality_metadata.json').read_text())
    if meta['graph_sha256']!=graph.graph_sha256 or not meta['exact_all_sources']:raise ValueError('参考图或全起点标记不匹配')
    with (ref/'centrality.csv').open(encoding='utf-8',newline='') as f: nodes=sorted(csv.DictReader(f),key=lambda r:int(r['index']))
    if len(nodes)!=graph.n or any(int(r['original_id'])!=graph.ids[i] for i,r in enumerate(nodes)):raise ValueError('参考节点覆盖不匹配')
    expected=np.array([float(r['bc_raw']) for r in nodes]);expected_norm=np.array([float(r['bc_normalized']) for r in nodes])
    run=Path('results')/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'_facebook_paired');run.mkdir()
    proc=psutil.Process();previous=proc.cpu_affinity();affinity=previous[:4]
    if len(affinity)<4:raise ValueError('本协议需要至少4个可用逻辑CPU')
    manifest=dict(protocol='3x3 cyclic adjacent blocks; each condition occupies each position once',dataset_id='facebook',n=graph.n,m=graph.m,graph_sha256=graph.graph_sha256,reference_run=a.reference_run,source_sha256=source_fingerprint(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),logical_cpu_affinity=affinity,physical_core_mapping=None,batch_size=128,task_logging=False,budget_seconds=a.budget,started_utc=datetime.now(timezone.utc).isoformat(),status='running',cases=[])
    write_json(run/'environment.json',environment());write_json(run/'manifest.json',manifest)
    start=time.perf_counter();rows=[]
    try:
        proc.cpu_affinity(affinity)
        # Full-source warmup is explicitly excluded from formal statistics.
        plan=[(-1,0,'serial',1)]+build_schedule()
        for block,position,mode,p_used in plan:
            remaining=a.budget-(time.perf_counter()-start)
            params=dict(mode=mode,processes=p_used,max_processes=4,batch_size=128,task_logging=False,timeout_seconds=max(.01,min(300,remaining)*.95))
            cpu_load=psutil.cpu_percent(interval=.2);freq=psutil.cpu_freq()
            before=time.perf_counter()
            status,result,error=bounded_call('algorithm',graph,params,min(300,remaining))
            row=dict(block=block,position=position,warmup=block<0,mode=mode,p_requested=p_used,p_used=None,status=status,error=error,cpu_load_before=cpu_load,cpu_mhz_before=freq.current if freq else None,t_supervised_wall=time.perf_counter()-before)
            if result is not None:
                row.update(result.timings,p_used=result.p_used,active_workers=result.active_workers,max_abs_error=float(np.max(np.abs(result.raw-expected))))
                if not (np.allclose(result.raw,expected,atol=1e-8,rtol=1e-7) and np.allclose(result.normalized,expected_norm,atol=1e-8,rtol=1e-7)):
                    row.update(status='failed_validation',error='全节点分数不一致')
                write_json(run/f'workers_{block}_{position}.json',result.workers)
            rows.append(row);manifest['cases']=rows;write_csv(run/'timings_raw.csv',rows);write_json(run/'manifest.json',manifest)
            print(json.dumps(dict(run_dir=str(run),block=block,position=position,mode=mode,p=p_used,status=row['status'],t_total=row.get('t_total')),ensure_ascii=False),flush=True)
            if status!='ok':break
            time.sleep(2)
    finally:
        proc.cpu_affinity(previous)
    formal=[r for r in rows if not r['warmup'] and r['status']=='ok'];summary=[]
    for mode,p_used in [('serial',1),('dynamic_batches',1),('dynamic_batches',4)]:
        group=[r for r in formal if r['mode']==mode and r['p_used']==p_used]
        if group:
            vals=np.array([r['t_total'] for r in group]);summary.append(dict(mode=mode,p_used=p_used,count=len(group),median_total=float(np.median(vals)),min_total=float(vals.min()),max_total=float(vals.max()),q1_total=float(np.quantile(vals,.25)),q3_total=float(np.quantile(vals,.75))))
    baseline=next((r['median_total'] for r in summary if r['mode']=='serial'),None)
    for r in summary:r['observed_speedup_total']=baseline/r['median_total'] if baseline else None
    pairs=[]
    for b in range(3):
        group=[r for r in formal if r['block']==b];serial=next((r for r in group if r['mode']=='serial'),None)
        if serial:
            for r in group:
                if r['mode']!='serial':pairs.append(dict(block=b,p_used=r['p_used'],serial_total=serial['t_total'],parallel_total=r['t_total'],observed_speedup_total=serial['t_total']/r['t_total']))
    p1=[r['observed_speedup_total'] for r in pairs if r['p_used']==1]
    comparable=len(p1)==3 and all(.8<=x<=1.25 for x in p1)
    manifest.update(status='complete' if len(formal)==9 and all(r['status']=='ok' for r in rows) else 'partial',elapsed_seconds=time.perf_counter()-start,completed_utc=datetime.now(timezone.utc).isoformat(),baseline_comparability_screen_passed=comparable,screen_note='p=1 within-block observed ratios in [0.8,1.25]; heuristic diagnostic, not statistical significance',limitations=['Only three blocks; CPU affinity fixes 4 logical CPUs, not necessarily 4 physical cores; CPU frequency/background load not controlled.']+([] if comparable else ['Single-worker baseline comparability screen failed; do not claim a reliable multicore speedup.']))
    write_csv(run/'timings_summary.csv',summary);write_csv(run/'paired_ratios.csv',pairs);write_json(run/'manifest.json',manifest)
    print('PAIRED_RESULT_DIR='+str(run),flush=True)
    return 0 if manifest['status']=='complete' else 2

if __name__=='__main__':raise SystemExit(main())
