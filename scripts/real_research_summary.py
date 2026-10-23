"""Summarize one completed real-graph experiment from saved evidence."""
import csv,json,sys
from pathlib import Path


def read_rows(path):
    with Path(path).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))


def table(headers,rows):
    return ['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(v) for v in row)+' |' for row in rows]+['']


def main():
    import argparse
    parser=argparse.ArgumentParser(description="生成指定真实图运行的独立阶段摘要")
    parser.add_argument('run_dir',type=Path)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'results/summaries/real_results.md')
    args=parser.parse_args()
    root=args.run_dir
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    target=args.output.resolve();target.parent.mkdir(parents=True,exist_ok=True)
    text='# LinkRadar 指定真实图运行的自动摘要\n\n本页只对应传入运行目录，不代表后续复测或整个项目当前状态。请看[统一结果](RESULTS.md)与[原文件审计](real_data_audit.json)。\n\n' 
    lines=['## 实际运行结果','',f"本轮状态：**{manifest['status']}**；完整[自动报告](../{root.as_posix()}/report.md)、[manifest](../{root.as_posix()}/manifest.json)。",
           f"实际案例数：{len(manifest['cases'])}；成功：{sum(r['status']=='ok' for r in manifest['cases'])}。源码SHA256：`{manifest['source_sha256']}`。",
           f"实际实验墙钟（含导出）：{manifest.get('elapsed_seconds_including_export','未记录')}秒。",'']
    for dataset in manifest['datasets']:
        d=root/dataset['id']; v=json.loads((d/'validation.json').read_text(encoding='utf-8'))
        lines += [f"### {dataset['id']}",'',f"图规模：{dataset['n']}节点、{dataset['m']}边；指纹：`{dataset['graph_sha256']}`。",
                  f"NetworkX全节点对照：allclose={v.get('allclose')}；raw最大误差={v.get('raw',{}).get('max_abs_error')}，normalized最大误差={v.get('normalized',{}).get('max_abs_error')}。",'']
        summary=read_rows(d/'timings_summary.csv')
        summary.sort(key=lambda x:(x['mode'],int(x['p_used'])))
        lines+=table(['模式','实际p','B','重复','total中位数(s)','steady中位数(s)','S_total','S_steady'],[[r['mode'],r['p_used'],r['batch_size'],r['count'],*[f"{float(r[k]):.6f}" for k in ('median_total','median_steady','speedup_total','speedup_steady')]] for r in summary])
        parallel=[r for r in summary if r['mode']!='serial']
        if parallel:
            best=max(parallel,key=lambda r:float(r['speedup_total']))
            lines += [f"本轮已测配置中S_total最大为{float(best['speedup_total']):.3f}（{best['mode']}，p={best['p_used']}，B={best['batch_size']}）。仅描述所测配置，不声称全局最优。",'']
        p1=[r for r in parallel if int(r['p_used'])==1]
        if any(float(r['speedup_total'])>1.5 for r in p1):
            ratios=', '.join(f"{r['mode']} p=1: {float(r['speedup_total']):.3f}x" for r in p1)
            lines += [f"**性能诊断警示：{ratios}。单工作进程相对同核串行已出现异常大的观察比值，不能解释为多核收益；因此本图4进程加速比也只作受运行条件影响的描述性比值，不作为可靠多核加速结论。** 应先在稳定环境中紧邻配对复测serial/p=1/p=4。原始记录与全部重复值保留，不择优删除。",'']
        for p in (1,2,4):
            pair={r['mode']:r for r in parallel if int(r['p_used'])==p}
            if set(pair)=={'static_batches','dynamic_batches'}:
                ratio=float(pair['dynamic_batches']['median_total'])/float(pair['static_batches']['median_total'])
                lines += [f"固定p={p}时 dynamic total / static total 中位数比为{ratio:.4f}。尚未进行统计显著性检验，不将微小差异解释成调度优势。"]
        lines += ['']
        score_rows=read_rows(d/'centrality.csv')
        top=sorted(score_rows,key=lambda r:int(r['rank_bc']))[:5]
        lines+=table(['介数排名','匿名原始ID','度','raw','normalized'],[[r['rank_bc'],r['original_id'],r['degree'],f"{float(r['bc_raw']):.6f}",f"{float(r['bc_normalized']):.6f}"] for r in top])
        rpath=d/'robustness_summary.csv'
        if rpath.exists():
            robust=read_rows(rpath); k=max(int(r['k']) for r in robust)
            endpoints={r['strategy']:r for r in robust if int(r['k'])==k}
            lines += [f"固定原始排名移除k={k}个节点后的结果：",'']
            lines+=table(['策略','实际移除比例','LCC/原始n均值','p10','p90','曲线数'],[[r['strategy'],r['removed_fraction'],*[f"{float(r[x]):.6f}" for x in ('lcc_original_mean','lcc_original_p10','lcc_original_p90')],r['count']] for r in endpoints.values()])
            if {'bc','degree'}<=set(endpoints):
                delta=float(endpoints['bc']['lcc_original_mean'])-float(endpoints['degree']['lcc_original_mean'])
                lines += [f"此末点 L_bc − L_degree={delta:.6f}；负值表示此点介数移除后的最大分量更小，不能推出所有比例或所有图上介数必胜。",'']
        else: lines += ['失效实验未完成，见manifest限制。','']
    lines += ['## 结论边界与下一步','',
              '此次原始数据来自用户手工导入；缓存校验确认本次计算对象与终端记录一致，没有独立向官方核验文件发布哈希。ca-GrQc的12个自环与正反重复已从本地原文件核实。',
              '性能结果同时提供total与steady。三次重复只刻画本机本轮波动，未控制温度/后台负载，不作跨机器推广；随机失效分位带不是置信区间。',
              '未出现在本目录清单中的实验不能视为已执行。当前项目已做的其他运行与限制见统一结果，后续复测不能混入这张表。',
              '上一轮110项测试通过，本轮没有修改计算内核；实际真实图全节点参照和串并行一致性检查构成本轮新增验收。', '']
    lines += ['- '+x for x in manifest.get('limitations',[])]
    target.write_text(rebase_links(text+'\n'.join(lines)+'\n',target),encoding='utf-8')
    (target.parent/'real_result_index.json').write_text(json.dumps({'run_dir':str(root),'status':manifest['status'],'source_sha256':manifest['source_sha256']},indent=2),encoding='utf-8')
    print(target)



def rebase_links(text, output):
    """Keep old docs-relative links correct for the selected report location."""
    import os, re
    old_base = Path(__file__).resolve().parents[1] / "docs"
    def replace(match):
        url = match.group(2)
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", url) or url.startswith("#"):
            return match.group(0)
        file, mark, anchor = url.partition("#")
        destination = Path(file[3:]) if re.match(r"^\.\./[A-Za-z]:/", file) else old_base / file
        try:
            relative = Path(os.path.relpath(destination.resolve(), output.parent.resolve())).as_posix()
        except ValueError:
            relative = destination.resolve().as_posix()
        return match.group(1) + relative + (mark + anchor if mark else "") + match.group(3)
    return re.sub(r"(!?\[[^\]\n]*\]\()([^)\n]+)(\))", replace, text)

if __name__=='__main__':main()
