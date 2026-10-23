"""Verified node rankings, full-graph parameters, explanatory subgraphs and figures."""
from __future__ import annotations
import argparse,csv,json,shutil,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import networkx as nx
import numpy as np
from parallel_bc.data import load_cached_graph
from parallel_bc.benchmark import write_csv,write_json


def read_rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def read_paired_evidence(directory, datasets):
    """Reject cross-network or incomplete pairings before presenting a chart."""
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    dataset=manifest.get('dataset_id')
    if dataset not in datasets or manifest.get('graph_sha256')!=datasets[dataset]['graph_sha256']:
        raise ValueError('配对复测数据集或图指纹不匹配')
    if manifest.get('status')!='complete':raise ValueError('只展示已完整完成的配对复测')
    raw=read_rows(directory/'timings_raw.csv')
    if len(raw)!=10 or any(r['status']!='ok' for r in raw):raise ValueError('配对复测需要一次成功热身和九个成功正式案例')
    formal=[r for r in raw if int(r['block'])>=0]
    if len(formal)!=9 or len([r for r in raw if int(r['block'])<0])!=1:raise ValueError('热身与正式案例数量不匹配')
    conditions={('serial',1),('dynamic_batches',1),('dynamic_batches',4)}
    summary=read_rows(directory/'timings_summary.csv')
    if {(r['mode'],int(r['p_used'])) for r in summary}!=conditions or len(summary)!=3:raise ValueError('配对条件不完整')
    for block in range(3):
        group=[r for r in formal if int(r['block'])==block]
        if len(group)!=3 or {int(r['position']) for r in group}!={0,1,2} or {(r['mode'],int(r['p_used'])) for r in group}!=conditions:
            raise ValueError('每个区组必须恰好包含三个不同条件与位置')
    for condition in conditions:
        matching=[r for r in formal if (r['mode'],int(r['p_used']))==condition]
        if len(matching)!=3 or {int(r['position']) for r in matching}!={0,1,2} or {int(r['block']) for r in matching}!={0,1,2}:
            raise ValueError('配对条件的区组或位置不平衡')
        values=np.array([float(r['t_total']) for r in matching])
        row=next(r for r in summary if (r['mode'],int(r['p_used']))==condition)
        if int(row['count'])!=3 or not np.isfinite(values).all():raise ValueError('配对耗时必须有三次有限实测')
        for field,value in [('median_total',np.median(values)),('min_total',values.min()),('max_total',values.max())]:
            if not np.isclose(float(row[field]),float(value),rtol=1e-12,atol=1e-12):raise ValueError('配对汇总与原始耗时不一致')
    baseline=float(next(r['median_total'] for r in summary if r['mode']=='serial'))
    for row in summary:
        if not np.isclose(float(row['observed_speedup_total']),baseline/float(row['median_total']),rtol=1e-12,atol=1e-12):raise ValueError('加速比与同网络基线不一致')
    p1_ratios=[]
    for block in range(3):
        group=[r for r in formal if int(r['block'])==block]
        serial=float(next(r['t_total'] for r in group if r['mode']=='serial'))
        p1=float(next(r['t_total'] for r in group if r['mode']=='dynamic_batches' and int(r['p_used'])==1))
        p1_ratios.append(serial/p1)
    if manifest.get('baseline_comparability_screen_passed')!=all(.8<=ratio<=1.25 for ratio in p1_ratios):raise ValueError('基线筛查标记与原始区组不一致')
    return dataset,dict(path=directory.as_posix(),manifest=manifest,summary=summary,ratios=read_rows(directory/'paired_ratios.csv'),raw=raw)


def export(reference_run,paired_run=None):
    base=Path(reference_run);out=Path('docs/node_insights');assets=Path('web/linkradar/dist/assets');out.mkdir(exist_ok=True,parents=True);assets.mkdir(exist_ok=True,parents=True)
    available={f.name for f in font_manager.fontManager.ttflist}
    chinese=next((f for f in ['Microsoft YaHei','SimHei','Noto Sans CJK SC'] if f in available),None)
    if chinese:plt.rcParams['font.sans-serif']=[chinese]
    plt.rcParams['axes.unicode_minus']=False
    payload=dict(reference_run=base.as_posix(),definition='无权无向图、全起点精确Brandes、端点不计入；按raw介数降序，同分按原始整数ID升序。',datasets={},paired=None,paired_by_dataset={})
    reports=['# LinkRadar 关键节点与参数报告','','排名标准：全图精确介数中心性；Top10是这一指标的排序，不是适用于任何目标的统一重要性。度数衡量直接连接数，割点表示单独删除会增加连通分量数，三者含义不同。','','数据均为匿名编号，不能据此还原人物身份或评价学术贡献。','']
    for name,title in [('facebook','Facebook'),('ca_grqc','ca-GrQc')]:
        graph=load_cached_graph(Path('data/processed')/name);folder=base/name;meta=json.loads((folder/'centrality_metadata.json').read_text());validation=json.loads((folder/'validation.json').read_text())
        if meta['graph_sha256']!=graph.graph_sha256 or meta['exact_all_sources'] is not True or validation['allclose'] is not True:raise ValueError('图/全起点标记/正确性证据不匹配')
        raw=read_rows(folder/'centrality.csv');raw.sort(key=lambda r:int(r['index']))
        if len(raw)!=graph.n or any(int(r['index'])!=i or int(r['original_id'])!=graph.ids[i] for i,r in enumerate(raw)):raise ValueError('节点映射覆盖不匹配')
        G=nx.Graph();G.add_nodes_from(range(graph.n));G.add_edges_from((i,j) for i,nb in enumerate(graph.adj) for j in nb if i<j)
        cuts=set(nx.articulation_points(G));comp={v:len(c) for c in nx.connected_components(G) for v in c}
        rows=[]
        for r in raw:
            i=int(r['index']);degree=int(r['degree'])
            rows.append(dict(index=i,original_id=int(r['original_id']),degree=degree,degree_normalized=degree/(graph.n-1) if graph.n>1 else 0.,bc_raw=float(r['bc_raw']),bc_normalized=float(r['bc_normalized']),rank_bc=int(r['rank_bc']),rank_degree=int(r['rank_degree']),component_size=comp[i],is_articulation=i in cuts))
        top=sorted(rows,key=lambda r:r['rank_bc'])[:10];write_csv(out/f'{name}_top10.csv',top);write_csv(assets/f'{name}_all_nodes.csv',rows);write_csv(assets/f'{name}_top10.csv',top)
        byindex={r['index']:r for r in rows};anchors=[r['index'] for r in top];selected=set(anchors)
        # Round-robin neighbor selection prevents a single hub consuming the entire view.
        neighbors=[sorted(graph.adj[i],key=lambda j:(-byindex[j]['bc_raw'],graph.ids[j])) for i in anchors]
        for level in range(max(map(len,neighbors),default=0)):
            for nb in neighbors:
                if level<len(nb):selected.add(nb[level])
                if len(selected)>=120:break
            if len(selected)>=120:break
        H=G.subgraph(sorted(selected));pos=nx.spring_layout(H,seed=42,iterations=60)
        subnodes=[dict(**byindex[i],x=float(pos[i][0]),y=float(pos[i][1]),top10=i in anchors) for i in sorted(selected)]
        subedges=[[u,v] for u,v in H.edges()]
        robust=read_rows(folder/'robustness_summary.csv');robust=[{k:(int(v) if k in ['k','count'] else float(v) if k not in ['dataset_id','graph_sha256','strategy'] and v else v) for k,v in r.items()} for r in robust]
        timings=read_rows(folder/'timings_summary.csv')
        payload['datasets'][name]=dict(title=title,n=graph.n,m=graph.m,graph_sha256=graph.graph_sha256,raw_sha256=graph.metadata['raw_sha256'],components=graph.metadata['connected_components'],lcc_size=graph.metadata['largest_component_nodes'],average_degree=graph.metadata['average_degree'],self_loop_rows=graph.metadata['self_loop_lines'],duplicate_edge_rows=graph.metadata['duplicate_edge_lines'],validation=validation,top10=top,nodes=rows,subgraph=dict(nodes=subnodes,edges=subedges,limit=120,seed=42,note='仅为围绕全图Top10的确定性局部展示；分数在完整原图计算。布局距离不表示实际距离。'),robustness=robust,timings=timings)
        fig,ax=plt.subplots(figsize=(9,6));labels=[str(r['original_id']) for r in top];vals=[r['bc_normalized'] for r in top]
        ax.barh(labels[::-1],vals[::-1],color='#197f76');ax.set(xlabel='全图归一化介数' if chinese else 'Normalized full-graph betweenness',title=f'{title} · Top 10');ax.grid(axis='x',alpha=.2)
        for y,v in enumerate(vals[::-1]):ax.text(v,y,f' {v:.6f}',va='center',fontsize=9)
        ax.margins(x=.2);fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(out/f'{name}_top10.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig);shutil.copy2(out/f'{name}_top10.png',assets/f'{name}_top10.png')
        fig,ax=plt.subplots(figsize=(9,7));bc=np.array([r['bc_normalized'] for r in rows]);deg=np.array([r['degree'] for r in rows]);ax.scatter(deg,bc,s=9,alpha=.32,color='#539e94')
        for r in top:ax.scatter([r['degree']],[r['bc_normalized']],s=35,color='#c66b37');ax.annotate(str(r['original_id']),(r['degree'],r['bc_normalized']),xytext=(4,4),textcoords='offset points',fontsize=8)
        ax.set_xscale('symlog',linthresh=1);ax.set_yscale('symlog',linthresh=1e-6);ax.set(xlabel='Degree (symlog)',ylabel='Full-graph normalized BC (symlog)',title=f'{title}: degree and betweenness; labeled Top10');ax.grid(alpha=.15);fig.tight_layout();fig.savefig(out/f'{name}_degree_bc.png',dpi=180);plt.close(fig);shutil.copy2(out/f'{name}_degree_bc.png',assets/f'{name}_degree_bc.png')
        fig,ax=plt.subplots(figsize=(10,8));nx.draw_networkx_edges(H,pos,ax=ax,alpha=.12,width=.5);others=[i for i in H if i not in anchors];nx.draw_networkx_nodes(H,pos,nodelist=others,node_size=22,node_color='#96c7bc',ax=ax);nx.draw_networkx_nodes(H,pos,nodelist=anchors,node_size=[80+700*byindex[i]['bc_normalized']/max(vals) for i in anchors],node_color='#d47b44',ax=ax);nx.draw_networkx_labels(H,pos,labels={i:str(graph.ids[i]) for i in anchors},font_size=8,ax=ax);ax.set_title(f'{title}: explanatory induced subgraph ({H.number_of_nodes()} nodes / {H.number_of_edges()} edges)\nScores from FULL graph; layout distance has no metric meaning');ax.axis('off');fig.tight_layout();fig.savefig(out/f'{name}_subgraph.png',dpi=180);plt.close(fig);shutil.copy2(out/f'{name}_subgraph.png',assets/f'{name}_subgraph.png')
        reports += [f'## {title}','',f"节点={graph.n}，边={graph.m}，平均度={graph.metadata['average_degree']:.6f}，连通分量={graph.metadata['connected_components']}，最大分量={graph.metadata['largest_component_nodes']}。",f'图指纹：`{graph.graph_sha256}`。',f"原文件SHA256：`{graph.metadata['raw_sha256']}`；原始自环行数={graph.metadata['self_loop_lines']}，去重行数={graph.metadata['duplicate_edge_lines']}。",f"全节点NetworkX对照通过；raw最大绝对误差={validation['raw']['max_abs_error']:.6g}，normalized最大绝对误差={validation['normalized']['max_abs_error']:.6g}；atol={validation['atol']}，rtol={validation['rtol']}。",'', '| 介数排名 | 原始ID | 度数 | 度排名 | raw介数 | 归一化介数 | 分量大小 | 割点 |','|---|---|---|---|---|---|---|---|']
        for r in top:reports.append(f"| {r['rank_bc']} | {r['original_id']} | {r['degree']} | {r['rank_degree']} | {r['bc_raw']:.6f} | {r['bc_normalized']:.9f} | {r['component_size']} | {'是' if r['is_articulation'] else '否'} |")
        reports += ['',f'![{title} Top10]({name}_top10.png)','',f'![{title} degree and BC]({name}_degree_bc.png)','',f'![{title} explanatory subgraph]({name}_subgraph.png)','','局部图最多120个节点，选择全图Top10及其邻居作确定性展示；未在局部子图重算排名，也不将示意布局解释为地理或最短路距离。','']
    fig,axes=plt.subplots(1,2,figsize=(15,7));fig.patch.set_facecolor('#f5f7f2')
    for ax,key in zip(axes,['facebook','ca_grqc']):
        item=payload['datasets'][key];top=item['top10'];ax.set_facecolor('#f5f7f2')
        labels=[f"{r['rank_bc']:02}  ID {r['original_id']}  (deg={r['degree']})" for r in top][::-1];values=[r['bc_normalized'] for r in top][::-1]
        ax.barh(labels,values,color='#277c70',height=.65)
        for i,v in enumerate(values):ax.text(v,i,f'  {v:.6f}',va='center',fontsize=9)
        ax.set_title(f"{item['title']}  |  n={item['n']:,}, m={item['m']:,}",loc='left',pad=18,fontweight='bold');ax.set_xlabel('Full-graph normalized betweenness');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);ax.margins(x=.28);ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0,labelsize=10)
    fig.suptitle('LinkRadar · 两个真实网络的介数 Top10' if chinese else 'LinkRadar · Exact full-graph betweenness Top10',fontsize=18,x=.06,ha='left',y=.98)
    fig.text(.06,.02,'匿名原始ID；分数在完整图计算，非局部示意子图。度数、介数和割点性质是不同指标。' if chinese else 'Original anonymized IDs. Scores from FULL graphs. Degree, BC and articulation differ.',fontsize=10,color='#53695b')
    fig.tight_layout(rect=(0,.05,1,.94))
    for ext in ['png','pdf']:fig.savefig(out/f'top10_comparison.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    reports += ['## 首轮实验参数与证据','',f'原始结果目录：`{base.as_posix()}`。全图无权无向、endpoints=False，全部起点，raw与normalized均对照；原图全部分量与孤立点保留。', '首轮p=1/2/4，B=128，重复3次，seed=42；随机失效20次。具体原始参数以该目录config_resolved.yaml为准。首轮Facebook基线异常，网页明确保留警示。', '归一化分母=(n−1)(n−2)/2；原始介数为无序端点对的最短路径比例之和。排名按raw降序、同分按整数原始ID升序；度排名也以原始ID打破并列。', '本轮新增割点与分量大小在完整原图上求得，不使用局部示意图推断。', '']
    for file in ('robustness.png','total_time.png','speedup_total.png','speedup_steady.png'):shutil.copy2(base/'figures'/file,assets/file)
    paired_paths=[paired_run] if isinstance(paired_run,(str,Path)) else list(paired_run or [])
    for directory in paired_paths:
        dataset,paired=read_paired_evidence(directory,payload['datasets'])
        if dataset in payload['paired_by_dataset']:raise ValueError('同一网络不能混入两次配对运行')
        payload['paired_by_dataset'][dataset]=paired
        title=payload['datasets'][dataset]['title']
        reports += [f'## {title}相邻区组复测','',f"复测状态：{paired['manifest']['status']}；基线可比性启发式检查：{paired['manifest']['baseline_comparability_screen_passed']}。实际逻辑CPU亲和性={paired['manifest']['logical_cpu_affinity']}，不代表相同数量的物理核心。",f"原始运行：`{paired['path']}`。",'','| 模式 | 实际p | 次数 | total中位数 | min–max | 观察加速比 |','|---|---|---|---|---|---|']
        for r in paired['summary']:reports.append(f"| {r['mode']} | {r['p_used']} | {r['count']} | {float(r['median_total']):.6f} | {float(r['min_total']):.3f}–{float(r['max_total']):.3f} | {float(r['observed_speedup_total']):.3f} |")
        reports += ['', 'p=1比值落在[0.8,1.25]仅作可比性筛查，不是显著性检验；若未通过，则四进程比值也仅作描述，不声称可靠多核加速。不同网络与不同协议分别统计。','']
        vals=paired['summary'];fig,ax=plt.subplots(figsize=(8,5));y=[float(r['median_total']) for r in vals]
        ax.bar([f"{r['mode']}\np={r['p_used']}" for r in vals],y,color=['#9aa5ad','#40978a','#255b68'])
        ax.set(ylabel='Total wall time (seconds)',title=f'{title}: adjacent blocks, fixed 4 logical CPUs')
        for i,(v,condition) in enumerate(zip(y,vals)):
            observations=[float(r['t_total']) for r in paired['raw'] if int(r['block'])>=0 and r['status']=='ok' and r['mode']==condition['mode'] and int(r['p_used'])==int(condition['p_used'])]
            ax.scatter(np.linspace(i-.09,i+.09,len(observations)),observations,s=30,c='#b75f36',zorder=5,label='Individual repeats' if i==0 else None)
            ax.errorbar(i,v,yerr=[[v-float(condition['min_total'])],[float(condition['max_total'])-v]],fmt='none',ecolor='#405e4f',capsize=5,zorder=4)
            ax.text(i,max(observations)+max(y)*.03,f'median {v:.2f}s',ha='center',va='bottom',fontsize=9)
        ax.legend(fontsize=9);ax.margins(y=.16)
        ax.set_title(f'{title}: adjacent blocks, fixed 4 logical CPUs\nBars: median; dots: repeats; whiskers: min-max')
        fig.tight_layout();fig.savefig(out/f'{dataset}_paired.png',dpi=180);plt.close(fig);shutil.copy2(out/f'{dataset}_paired.png',assets/f'{dataset}_paired.png')
        reports += [f'![{title}配对复测]({dataset}_paired.png)','']
    # Preserve the old Facebook entry for consumers of the first export format.
    payload['paired']=payload['paired_by_dataset'].get('facebook')
    for dataset in payload['datasets']:
        if dataset not in payload['paired_by_dataset']:reports += [f"## {payload['datasets'][dataset]['title']}复测状态",'', '尚未导入该网络的完整配对结果，不用其他网络替代。','']
    reports += ['## 如何理解和展示','', 'Python负责全图算法、验证和CSV导出；Web页面读取已经完成的真实结果，可筛选、排序、搜索匿名ID并查看局部连接。无需浏览器重算Brandes，也无需外网或云服务。', '当前交付为静态HTML/JS页面，可用本地HTTP预览或GitHub Pages工作流发布；实际线上状态以部署记录为准。若需从网页提交计算任务，再加入后端API及任务管理。', '度数高不一定介数高；高介数也不保证是割点，割点按完整原图验证。排名与失效实验结论只针对当前拓扑。','']
    (out/'report.md').write_text('\n'.join(reports),encoding='utf-8');write_json(assets/'data.json',payload)
    data_text=json.dumps(payload,ensure_ascii=False,allow_nan=False).replace('</','<\\/')
    (assets/'data.js').write_text('window.LINKRADAR_DATA = '+data_text+';\n',encoding='utf-8')
    write_json(out/'manifest.json',dict(reference_run=base.as_posix(),paired_run=payload['paired']['path'] if payload['paired'] else None,paired_runs={k:v['path'] for k,v in payload['paired_by_dataset'].items()},ranking='exact full-graph raw betweenness desc; original ID asc for ties',datasets={k:dict(graph_sha256=v['graph_sha256'],top10_original_ids=[r['original_id'] for r in v['top10']]) for k,v in payload['datasets'].items()}))
    print(out/'report.md')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference-run',default='results/20261023T054258_364875Z');p.add_argument('--paired-run',action='append',help='可重复指定，每张真实图一个配对运行目录');a=p.parse_args();export(a.reference_run,a.paired_run)
