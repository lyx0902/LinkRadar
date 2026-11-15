'use strict';
// Metadata is small; the complete 0-based node array is fetched only on selection.
const asSkitter = window.LINKRADAR_AS_SKITTER;
DATA.datasets.as_skitter = asSkitter;
let asSkitterView = null, asSkitterBcOrder = null, asSkitterDegreeOrder = null;
let asSkitterQuery = '', asSkitterMode = '', asSkitterMatches = null;
const beforeAsSkitterTable = renderTable, beforeAsSkitterSelect = selectNode;
const beforeAsSkitterMethod = renderMethod, beforeAsSkitterChange = $('dataset').onchange;

function asSkitterNode(i) {
  const p = i * 28, raw = asSkitterView.getFloat64(p, true);
  return {index:i, original_id:i, bc_raw:raw,
    bc_normalized:raw / ((asSkitter.n-1)*(asSkitter.n-2)/2),
    degree:asSkitterView.getUint32(p+8,true), rank_bc:asSkitterView.getUint32(p+12,true),
    rank_degree:asSkitterView.getUint32(p+16,true), is_articulation:!!asSkitterView.getUint32(p+20,true),
    component_size:asSkitterView.getUint32(p+24,true)};
}
selectNode = function(index) {
  if (current !== 'as_skitter') return beforeAsSkitterSelect(index);
  selected = asSkitterNode(index); renderDetails(); renderBars(); renderTable(); paintSelection();
};
renderTable = function() {
  if (current !== 'as_skitter') return beforeAsSkitterTable();
  const query = $('search').value.trim(), mode = $('sort').value;
  const order = mode === 'bc' ? asSkitterBcOrder : asSkitterDegreeOrder;
  if (query !== asSkitterQuery || mode !== asSkitterMode) {
    asSkitterQuery = query; asSkitterMode = mode; asSkitterMatches = null;
    if (query) {
      const found = [];
      for (let i=0; i<order.length; i++) if (String(order[i]).includes(query)) found.push(order[i]);
      asSkitterMatches = found;
    }
  }
  const rows = asSkitterMatches || order;
  const count = query ? rows.length : ($('range').value === 'top' ? 10 : rows.length);
  const pages = Math.max(1, Math.ceil(count/10)); page = Math.min(page, pages-1);
  $('node-table').replaceChildren();
  for (let j=page*10; j<Math.min(count,page*10+10); j++) {
    const n = asSkitterNode(rows[j]), tr = document.createElement('tr');
    tr.className = 'node-row' + (selected?.index === n.index ? ' selected' : ''); tr.tabIndex = 0;
    tr.setAttribute('aria-label', `选择节点 ${n.original_id}`);
    tr.innerHTML = `<td>${mode==='bc'?n.rank_bc:n.rank_degree}</td><td class="id">${n.original_id}</td><td>${n.degree}</td><td>${n.rank_degree}</td><td>${fmt(n.bc_raw,3)}</td><td>${n.bc_normalized.toFixed(9)}</td><td>${fmt(n.component_size)}</td><td>${n.is_articulation?'是':'否'}</td>`;
    tr.onclick = () => selectNode(n.index);
    tr.onkeydown = e => { if (e.key === 'Enter') { e.preventDefault(); selectNode(n.index); } };
    $('node-table').append(tr);
  }
  if (!count) $('node-table').innerHTML = '<tr><td colspan="8">没有匹配的原始 ID。</td></tr>';
  $('row-count').textContent = `${fmt(count)} 个匹配节点 / 全图 ${fmt(asSkitter.n)}`;
  $('page-state').textContent = `${page+1} / ${pages}`;
  $('previous').disabled = page===0; $('next').disabled = page+1>=pages;
};
renderMethod = function() {
  if (current !== 'as_skitter') return beforeAsSkitterMethod();
  pairedPanel.hidden = true; timingPanel.hidden = true; youtubeRunPanel.hidden = false;
  const r = asSkitter.full_run, seconds = v => `${fmt(v,3)} 秒`;
  const statusNames = {resource_limit:'可用内存低于预留线后中止', timeout:'达到会话时间预算', ok:'全量完成'};
  const sessions = r.sessions.map((s,i) => `<tr><td>${i+1}</td><td>${statusNames[s.status] || s.status}</td><td>${seconds(s.seconds)}</td></tr>`).join('');
  youtubeRunPanel.innerHTML = `<div class="panel-head"><div><span class="kicker">FULL RUN / 2026-11-14</span><h3>as-skitter 全量计算：配置与耗时</h3></div><span class="pill">全源已完成</span></div>
    <div class="insight">四份现存会话监督墙钟合计 <strong>${seconds(r.seconds_total)}</strong>（约 13 小时 36 分 31 秒）；不包含会话间停顿，人工中断可能没有完整会话记录。任务在资源中止和超时后恢复，最终完成全部源组。</div>
    <div class="table-scroll"><table><thead><tr><th>配置项目</th><th>本次全量任务</th></tr></thead><tbody>
    <tr><td>运行机器</td><td>Windows 11 · Intel Core i5-13500H · 12 物理核 / 16 逻辑处理器 · 内存约 15.53 GiB</td></tr>
    <tr><td>原图 / 核心图</td><td>${fmt(asSkitter.n)} 节点、${fmt(asSkitter.m)} 边 / ${fmt(r.core_nodes)} 节点、${fmt(r.core_edges)} 边</td></tr>
    <tr><td>内核与检查点</td><td>Numba 边记录内核 · 无损树商图 · 双累计快照</td></tr>
    <tr><td>线程与任务粒度</td><td>${r.threads} 线程 · 调度批次 ${r.batch_size} · 每块 ${fmt(r.chunk_size)} 核心源组 · 种子 ${r.seed}</td></tr>
    <tr><td>CPU 亲和性</td><td>${r.cpu_affinity.join('、')}</td></tr>
    <tr><td>软件环境</td><td>Python ${r.python} · NumPy ${r.environment.numpy} · Numba ${r.environment.numba} · llvmlite ${r.environment.llvmlite}</td></tr>
    <tr><td>完成范围</td><td>${fmt(r.core_groups)} 核心源组 · ${fmt(asSkitter.n)} 原始源点 · ${r.chunks} 次检查点提交；其中 ${fmt(r.analytic_source_mass)} 个源点由纯树分量解析恢复</td></tr>
    </tbody></table></div><div class="table-scroll"><table><thead><tr><th>现存会话</th><th>状态</th><th>监督墙钟</th></tr></thead><tbody>${sessions}</tbody></table></div>
    <p class="caption">会话合计不是纯内核时间，也不是多轮平均或线程加速比。原文件头提到的 22,622 个零度节点未由解析器从头部补入，本次结果以实际边表和缓存为准。</p>`;
  $('provenance').textContent = `as-skitter 全源结果；图 SHA256：${asSkitter.graph_sha256}。源覆盖、文件哈希及外围树核验通过；755 个非最大连通分量的 1,799 个节点与 NetworkX 全源对照通过，最大连通分量核心节点尚未独立全图逐点重算。`;
};
async function loadAsSkitter() {
  if (asSkitterView) return;
  const response = await fetch('assets/as_skitter-nodes.bin');
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const buffer = await response.arrayBuffer(), n = asSkitter.n;
  if (buffer.byteLength !== n*36) throw new Error('节点文件长度不符');
  asSkitterView = new DataView(buffer);
  asSkitterBcOrder = new Uint32Array(buffer,n*28,n);
  asSkitterDegreeOrder = new Uint32Array(buffer,n*32,n);
}
$('dataset').onchange = async e => {
  if (e.target.value !== 'as_skitter') return beforeAsSkitterChange(e);
  e.target.disabled = true; $('row-count').textContent = '正在加载 as-skitter 全节点数据…';
  try { await loadAsSkitter(); }
  catch (error) { e.target.value = current; $('row-count').textContent = ''; alert('as-skitter 节点数据加载失败，请通过 HTTP 服务访问页面：' + error.message); return; }
  finally { e.target.disabled = false; }
  current = 'as_skitter'; $('search').value = ''; asSkitterQuery = ''; asSkitterMode = ''; asSkitterMatches = null;
  document.querySelector('.trust span').textContent = '✓ 全源覆盖与小分量对照';
  scatterCaption.textContent = '散点为随机抽取的 30,000 个节点，橙色为全图介数前十；不是全部节点。';
  $('downloadAll').textContent = '下载全节点 CSV ↗'; render(); $('downloadAll').href = '#';
};
$('downloadAll').addEventListener('click', e => {
  if (current !== 'as_skitter') return;
  e.preventDefault();
  const chunks = ['\ufeffrank_bc,original_id,degree,rank_degree,bc_raw,bc_normalized,component_size,is_articulation\n'];
  let lines = [];
  for (let i=0; i<asSkitter.n; i++) {
    const n = asSkitterNode(i);
    lines.push(`${n.rank_bc},${n.original_id},${n.degree},${n.rank_degree},${n.bc_raw},${n.bc_normalized},${n.component_size},${n.is_articulation}\n`);
    if (lines.length===10000) { chunks.push(lines.join('')); lines=[]; }
  }
  if (lines.length) chunks.push(lines.join(''));
  const url = URL.createObjectURL(new Blob(chunks,{type:'text/csv;charset=utf-8'}));
  const a = document.createElement('a'); a.href = url; a.download = 'as_skitter_all_nodes.csv'; a.click();
  setTimeout(() => URL.revokeObjectURL(url),10000);
});
