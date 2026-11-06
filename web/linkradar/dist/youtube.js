'use strict';
// The large node array is loaded only when YouTube is selected.
const youtube = window.LINKRADAR_YOUTUBE;
DATA.datasets.youtube = youtube;
let youtubeBuffer = null, youtubeView = null, youtubeBcOrder = null, youtubeDegreeOrder = null;
let youtubeQuery = '', youtubeMode = '', youtubeMatches = null;
const oldRenderTable = renderTable, oldSelectNode = selectNode, oldMethod = renderMethod;
const pairedPanel = $('paired-title').closest('article');
const timingPanel = $('performance-table').closest('article');
const youtubeRunPanel = $('youtube-run-panel');
const originalTrust = document.querySelector('.trust span').textContent;
const scatterCaption = $('scatter-image').nextElementSibling;
const originalScatterCaption = scatterCaption.textContent;

function youtubeNode(i) {
  const p = i * 24, raw = youtubeView.getFloat64(p, true);
  return {index:i, original_id:i+1, bc_raw:raw,
    bc_normalized:raw / ((youtube.n-1)*(youtube.n-2)/2),
    degree:youtubeView.getUint32(p+8,true), rank_bc:youtubeView.getUint32(p+12,true),
    rank_degree:youtubeView.getUint32(p+16,true), is_articulation:!!youtubeView.getUint32(p+20,true),
    component_size:youtube.n};
}
selectNode = function(index) {
  if (current !== 'youtube') return oldSelectNode(index);
  selected=youtubeNode(index); renderDetails(); renderBars(); renderTable(); paintSelection();
};
renderTable = function() {
  if (current !== 'youtube') return oldRenderTable();
  const query=$('search').value.trim(), mode=$('sort').value;
  const order=mode==='bc'?youtubeBcOrder:youtubeDegreeOrder;
  if (query !== youtubeQuery || mode !== youtubeMode) {
    youtubeQuery=query; youtubeMode=mode; youtubeMatches=null;
    if (query) {
      const found=[];
      for (let i=0;i<order.length;i++) if (String(order[i]+1).includes(query)) found.push(order[i]);
      youtubeMatches=found;
    }
  }
  const rows=youtubeMatches || order;
  const count=query?rows.length:($('range').value==='top'?10:rows.length);
  const pages=Math.max(1,Math.ceil(count/10)); page=Math.min(page,pages-1);
  $('node-table').replaceChildren();
  for(let j=page*10;j<Math.min(count,page*10+10);j++) {
    const n=youtubeNode(rows[j]), tr=document.createElement('tr');
    tr.className='node-row'+(selected?.index===n.index?' selected':'');tr.tabIndex=0;
    tr.setAttribute('aria-label',`选择节点 ${n.original_id}`);
    tr.innerHTML=`<td>${mode==='bc'?n.rank_bc:n.rank_degree}</td><td class="id">${n.original_id}</td><td>${n.degree}</td><td>${n.rank_degree}</td><td>${fmt(n.bc_raw,3)}</td><td>${n.bc_normalized.toFixed(9)}</td><td>${fmt(n.component_size)}</td><td>${n.is_articulation?'是':'否'}</td>`;
    tr.onclick=()=>selectNode(n.index);tr.onkeydown=e=>{if(e.key==='Enter')selectNode(n.index)};
    $('node-table').append(tr);
  }
  if(!count) $('node-table').innerHTML='<tr><td colspan="8">没有匹配的原始 ID。</td></tr>';
  $('row-count').textContent=`${fmt(count)} 个匹配节点 / 全图 ${fmt(youtube.n)}`;
  $('page-state').textContent=`${page+1} / ${pages}`;$('previous').disabled=page===0;$('next').disabled=page+1>=pages;
};
renderMethod = function() {
  pairedPanel.hidden=current==='youtube';timingPanel.hidden=current==='youtube';
  youtubeRunPanel.hidden=current!=='youtube';
  if(current!=='youtube') return oldMethod();
  const r=youtube.full_run, total=r.first_seconds+r.second_seconds;
  const seconds=v=>`${fmt(v,3)} 秒`, mib=v=>`${fmt(v/1048576,2)} MiB`;
  const firstChunks=r.chunks-r.second_new_chunks;
  youtubeRunPanel.innerHTML=`<div class="panel-head"><div><span class="kicker">FULL RUN / 2026-11-06</span><h3>YouTube 全量计算：配置与耗时</h3></div><span class="pill">全源已完成</span></div>
  <div class="insight">两次会话监督墙钟合计 <strong>${seconds(total)}</strong>（约 2 小时 30 分 37 秒），包含会话启动与加载，不包含中间暂停时间。第一次因系统可用内存低于预留线中止，第二次从检查点恢复并完成。</div>
  <div class="table-scroll"><table><thead><tr><th>配置项目</th><th>本次全量任务</th></tr></thead><tbody>
  <tr><td>运行机器</td><td>Windows 11 家庭版 · Intel Core i5-13500H · 12 物理核 / 16 逻辑处理器 · 内存约 15.53 GiB</td></tr>
  <tr><td>原图 / 核心图</td><td>${fmt(youtube.n)} 节点、${fmt(youtube.m)} 边 / ${fmt(r.core_nodes)} 节点、${fmt(r.core_edges)} 边</td></tr>
  <tr><td>内核与检查点</td><td>Numba 边记录内核 · 无损树商图 · 双累计快照</td></tr>
  <tr><td>线程与任务粒度</td><td>${r.threads} 线程 · 调度批次 ${r.batch_size} · 每块 ${fmt(r.chunk_size)} 核心源组 · 种子 ${r.seed}</td></tr>
  <tr><td>CPU 亲和性</td><td>${r.cpu_affinity.join('、')}</td></tr>
  <tr><td>软件环境</td><td>Python ${r.python} · NumPy ${r.environment.numpy} · Numba ${r.environment.numba} · llvmlite ${r.environment.llvmlite}</td></tr>
  <tr><td>完成范围</td><td>${fmt(r.core_groups)} 核心源组 · ${fmt(youtube.n)} 原始源点 · ${r.chunks} 次检查点提交</td></tr>
  </tbody></table></div>
  <div class="table-scroll"><table><thead><tr><th>会话</th><th>状态</th><th>监督墙钟</th><th>提交进度</th></tr></thead><tbody>
  <tr><td>第一次</td><td>可用内存低于预留线后中止</td><td>${seconds(r.first_seconds)}</td><td>推算已提交 ${firstChunks} 块 / ${fmt(firstChunks*r.chunk_size)} 组</td></tr>
  <tr><td>第二次</td><td>续算完成</td><td>${seconds(r.second_seconds)}</td><td>新增 ${r.second_new_chunks} 块，最终全源完成</td></tr>
  </tbody></table></div>
  <p class="caption">系统可用内存预留 ${mib(r.reserve_bytes)}，第一次最低采样值 ${mib(r.first_min_available)}；这不是计算进程自身的内存上限。第一段进度由最终块数与第二段新增块数推算。第二段执行与归并累计 ${seconds(r.second_work_seconds)}；第一段没有完整核心累计时间，因此不把两个口径相加。该记录为一次全量任务，不是多轮平均或线程加速比。</p>`;
  $('provenance').textContent=`YouTube 全源结果；图 SHA256：${youtube.graph_sha256}。源组覆盖、快照与输出哈希已核验；外围节点经树结构公式核对，核心节点未进行独立全图逐点对照。`;
};
async function loadYoutube() {
  if(youtubeBuffer)return;
  const response=await fetch('assets/youtube-nodes.bin');
  if(!response.ok)throw new Error(`HTTP ${response.status}`);
  const buffer=await response.arrayBuffer(), n=youtube.n;
  if(buffer.byteLength!==n*32)throw new Error('节点文件长度不符');
  youtubeBuffer=buffer;youtubeView=new DataView(buffer);
  youtubeBcOrder=new Uint32Array(buffer,n*24,n);youtubeDegreeOrder=new Uint32Array(buffer,n*28,n);
}
function updateDatasetNotes() {
  const isYoutube=current==='youtube';
  document.querySelector('.trust span').textContent=isYoutube?'✓ 全源覆盖与结果哈希核验':'✓ 全节点 NetworkX 对照通过';
  scatterCaption.textContent=isYoutube?'散点为随机抽取的 30,000 个节点，橙色为全图介数前十；不是全部节点。':' '+originalScatterCaption;
  $('downloadAll').textContent=isYoutube?'下载全节点 CSV ↗':'全部节点 CSV ↗';
}
$('dataset').onchange=async e=>{
  const value=e.target.value;
  if(value==='youtube'){
    e.target.disabled=true;$('row-count').textContent='正在加载 YouTube 全节点数据…';
    try{await loadYoutube()}catch(error){e.target.value=current;$('row-count').textContent='';alert('YouTube 节点数据加载失败，请通过 HTTP 服务访问页面：'+error.message);return}finally{e.target.disabled=false}
  }
  current=value;$('search').value='';youtubeQuery='';youtubeMode='';youtubeMatches=null;updateDatasetNotes();render();
  if(current==='youtube')$('downloadAll').href='#';
};
$('downloadAll').addEventListener('click',e=>{
  if(current!=='youtube')return;
  e.preventDefault();
  const lines=['rank_bc,original_id,degree,rank_degree,bc_raw,bc_normalized,component_size,is_articulation\n'];
  for(let i=0;i<youtube.n;i++){const x=youtubeNode(i);lines.push(`${x.rank_bc},${x.original_id},${x.degree},${x.rank_degree},${x.bc_raw},${x.bc_normalized},${x.component_size},${x.is_articulation}\n`)}
  const url=URL.createObjectURL(new Blob(['\ufeff',...lines],{type:'text/csv;charset=utf-8'}));
  const a=document.createElement('a');a.href=url;a.download='youtube_all_nodes.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),10000);
});
