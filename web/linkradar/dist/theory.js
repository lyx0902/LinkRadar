/* A five-node teaching example and glossary; no full-network recomputation. */
'use strict';
(() => {
  const byId = id => document.getElementById(id);
  const labels = ['S', 'A', 'B', 'T', 'U'];
  const edges = [[0, 1], [0, 2], [1, 3], [2, 3], [3, 4]];
  const adjacency = labels.map((_, v) => edges.flatMap(([a, b]) => a === v ? [b] : b === v ? [a] : []));
  const positions = [[65, 150], [215, 50], [215, 250], [385, 150], [535, 150]];
  const svgNS = 'http://www.w3.org/2000/svg';
  const makeSvg = (tag, attrs, text) => {
    const element = document.createElementNS(svgNS, tag);
    for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, value);
    if (text !== undefined) element.textContent = text;
    return element;
  };
  function shortestPaths(source, target) {
    const distance = labels.map(() => -1), predecessors = labels.map(() => []), queue = [source];
    distance[source] = 0;
    for (let cursor = 0; cursor < queue.length; cursor++) {
      const v = queue[cursor];
      for (const w of adjacency[v]) {
        if (distance[w] < 0) { distance[w] = distance[v] + 1; queue.push(w); }
        if (distance[w] === distance[v] + 1) predecessors[w].push(v);
      }
    }
    const paths = [];
    function visit(node, suffix) {
      if (node === source) { paths.push([source, ...suffix]); return; }
      for (const parent of predecessors[node]) visit(parent, [node, ...suffix]);
    }
    if (distance[target] >= 0) visit(target, []);
    return paths;
  }
  const key = (a, b) => [a, b].sort((x, y) => x - y).join('-');
  function renderToy() {
    const [source, target] = byId('toy-pair').value.split('-').map(Number);
    const observed = Number(byId('toy-node').value), paths = shortestPaths(source, target);
    const through = paths.filter(path => path.slice(1, -1).includes(observed)).length;
    const isEndpoint = observed === source || observed === target;
    const edgeSet = new Set(paths.flatMap(path => path.slice(1).map((v, i) => key(path[i], v))));
    const svg = makeSvg('svg', { viewBox: '0 0 600 300', role: 'group', 'aria-label': '五节点教学图，当前端点为' + labels[source] + '和' + labels[target] });
    for (const [a, b] of edges) {
      const [x1, y1] = positions[a], [x2, y2] = positions[b];
      svg.append(makeSvg('line', { x1, y1, x2, y2, class: 'toy-edge' + (edgeSet.has(key(a, b)) ? ' on-path' : '') }));
    }
    positions.forEach(([cx, cy], v) => {
      const circle = makeSvg('circle', { cx, cy, r: 24, tabindex: '0', role: 'button', 'aria-label': '在教学图观察节点' + labels[v], 'aria-pressed': String(v === observed), class: 'toy-circle' + ([source, target].includes(v) ? ' endpoint' : '') + (v === observed ? ' observed' : '') });
      const choose = () => { byId('toy-node').value = String(v); renderToy(); };
      circle.addEventListener('click', choose);
      circle.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); choose(); byId('toy-graph').querySelectorAll('circle')[v].focus(); } });
      svg.append(circle, makeSvg('text', { x: cx, y: cy + 6, 'text-anchor': 'middle', class: 'toy-label' }, labels[v]));
    });
    byId('toy-graph').replaceChildren(svg);
    byId('toy-share').textContent = through + ' / ' + paths.length + ' = ' + (through / paths.length).toLocaleString('zh-CN', { maximumFractionDigits: 3 });
    byId('toy-explanation').textContent = isEndpoint
      ? labels[observed] + '是当前端点，本项目不计端点贡献，所以按0展示。'
      : '从' + labels[source] + '到' + labels[target] + '有' + paths.length + '条最短路径，其中' + through + '条在内部经过' + labels[observed] + '。这是该端点对给' + labels[observed] + '的贡献。';
    byId('toy-paths').replaceChildren(...paths.map(path => { const line = document.createElement('span'); line.textContent = path.map(v => labels[v]).join(' → '); return line; }));
  }
  function renderToyTotals() {
    const scores = labels.map(() => 0);
    for (let source = 0; source < labels.length; source++) {
      for (let target = source + 1; target < labels.length; target++) {
        const paths = shortestPaths(source, target);
        for (const path of paths) for (const v of path.slice(1, -1)) scores[v] += 1 / paths.length;
      }
    }
    byId('toy-scores').replaceChildren(...scores.map((score, v) => {
      const row = document.createElement('tr');
      for (const value of [labels[v], adjacency[v].length, score, (score / 6).toFixed(6)]) { const cell = document.createElement('td'); cell.textContent = value; row.append(cell); }
      return row;
    }));
  }
  function buildBrandesTrace() {
    const source = 0, distance = labels.map(() => -1), sigma = labels.map(() => 0);
    const predecessors = labels.map(() => []), delta = labels.map(() => 0), order = [source];
    distance[source] = 0; sigma[source] = 1;
    const trace = [];
    const snapshot = (title, description, equation, complete = false) => trace.push({ title, description, equation, complete, distance: [...distance], sigma: [...sigma], predecessors: predecessors.map(p => [...p]), delta: [...delta] });
    snapshot('初始化：只有起点已知', '固定s=S。S到自己的路径数约定为1；其他节点暂未发现，距离显示为“未发现”。所有依赖δ从0开始。', 'σ[S] = 1；其余σ = 0；所有δ = 0');
    for (let cursor = 0; cursor < order.length; cursor++) {
      const v = order[cursor];
      for (const w of adjacency[v]) {
        if (distance[w] < 0) { distance[w] = distance[v] + 1; order.push(w); }
        if (distance[w] === distance[v] + 1) { sigma[w] += sigma[v]; predecessors[w].push(v); }
      }
    }
    snapshot('正向BFS：记录全部等长最短路', '按S；A、B；T；U四层展开。T有A和B两个前驱，因此有2条最短路；U沿用到T的2条路径。前驱不是只能保留一个。', 'σ[T] = σ[A] + σ[B] = 2；σ[U] = σ[T] = 2');
    for (const w of [...order].reverse()) {
      for (const v of predecessors[w]) delta[v] += sigma[v] / sigma[w] * (1 + delta[w]);
      if (w === 4) snapshot('处理最远的U：把一份贡献传给T', 'δS(U)=0，因为它没有更远目标；但以U为终点的这1份贡献要沿前驱T回传。所有S到U的最短路都经过T。', 'δS(T) += (2 / 2) × (1 + 0) = 1');
      if (w === 3) snapshot('处理T：将两部分贡献分给A和B', '“1”来自以T为终点的贡献；δS(T)=1来自更远的U。合计2，再按σ[A]/σ[T]=1/2分给A，B同理。所以A得到S↔T的1/2与S↔U的1/2。', 'δS(A) += (1 / 2) × (1 + 1) = 1；δS(B)同理');
      if (w === 1) snapshot('处理A、B：贡献继续回传到S', 'A和B各向前驱S回传2，S的内部依赖值累计为4。这个值用于递推，但S是当前起点，不能把它写入自己的中心性。', 'δS(S) = (1 / 1) × (1 + 1) + (1 / 1) × (1 + 1) = 4');
    }
    snapshot('本源贡献：排除起点，再换下一个源点', '以S为起点这一轮，A、B、T各写入1分，U与S为0。这只是本源贡献；接着还要换A、B、T、U为起点。全部起点相加后统一除以2，才得到上方全图raw表。', '写入向量[S, A, B, T, U] = [0, 1, 1, 1, 0]', true);
    return trace;
  }
  const brandesTrace = buildBrandesTrace();
  let brandesStep = 0;
  function renderBrandesTrace() {
    const state = brandesTrace[brandesStep];
    byId('brandes-step-title').textContent = state.title;
    byId('brandes-step-description').textContent = state.description;
    byId('brandes-step-equation').textContent = state.equation;
    byId('brandes-step-count').textContent = (brandesStep + 1) + ' / ' + brandesTrace.length + '步';
    byId('brandes-trace').replaceChildren(...labels.map((label, v) => {
      const row = document.createElement('tr'); if (v === 0) row.className = 'trace-source';
      const values = [label, state.distance[v] < 0 ? '未发现' : state.distance[v], state.sigma[v], state.predecessors[v].map(i => labels[i]).join('、') || '—', state.delta[v], state.complete ? (v === 0 ? 0 : state.delta[v]) : '完成后显示'];
      values.forEach((value, index) => { const cell = document.createElement('td'); cell.textContent = value; if (index === 5 && state.complete) cell.className = 'trace-final'; row.append(cell); });
      return row;
    }));
    document.querySelectorAll('[data-brandes-step]').forEach(button => { const active = Number(button.dataset.brandesStep) === brandesStep; button.classList.toggle('active', active); if (active) button.setAttribute('aria-current', 'step'); else button.removeAttribute('aria-current'); });
    byId('brandes-previous').disabled = brandesStep === 0;
    byId('brandes-next').disabled = brandesStep === brandesTrace.length - 1;
  }
  function renderContext() {
    const item = window.LINKRADAR_DATA.datasets[byId('dataset').value], first = item.top10[0];
    const denominator = (item.n - 1) * (item.n - 2) / 2;
    const number = value => Number(value).toLocaleString('zh-CN', { maximumFractionDigits: 6 });
    byId('theory-current-score').textContent = '当前' + item.title + '：n=' + number(item.n) + '，m=' + number(item.m) + '，平均度2m/n=' + number(2 * item.m / item.n) + '。介数第一的原始ID为' + first.original_id + '；raw=' + number(first.bc_raw) + '，归一化分母=' + number(denominator) + '，normalized=' + Number(first.bc_normalized).toFixed(9) + '。切换上方网络即可对照另一张图。';
    byId('theory-lcc-current').textContent = '当前' + item.title + '原图有' + number(item.components) + '个连通分量，最大分量' + number(item.lcc_size) + '个节点。因此零删除起点LCC/n=' + item.lcc_size + '/' + item.n + '=' + (item.lcc_size / item.n).toFixed(6) + '。' + (item.components > 1 ? '它原本就不完全连通，起点小于1不是删除造成的损失。' : '它原本完全连通，所以起点为1。');
  }
  function filterGlossary() {
    const query = byId('glossary-search').value.trim().toLocaleLowerCase();
    const terms = Array.from(document.querySelectorAll('.term-card')); let visible = 0;
    for (const term of terms) { const match = (term.dataset.term + ' ' + term.textContent).toLocaleLowerCase().includes(query); term.hidden = !match; if (match) visible++; }
    byId('glossary-count').textContent = visible + ' / ' + terms.length + '个术语';
    byId('glossary-empty').hidden = visible !== 0;
  }
  byId('toy-pair').addEventListener('change', renderToy);
  byId('toy-node').addEventListener('change', renderToy);
  byId('dataset').addEventListener('change', renderContext);
  byId('glossary-search').addEventListener('input', filterGlossary);
  document.querySelectorAll('[data-brandes-step]').forEach(button => button.addEventListener('click', () => { brandesStep = Number(button.dataset.brandesStep); renderBrandesTrace(); }));
  byId('brandes-previous').addEventListener('click', () => { brandesStep = Math.max(0, brandesStep - 1); renderBrandesTrace(); });
  byId('brandes-next').addEventListener('click', () => { brandesStep = Math.min(brandesTrace.length - 1, brandesStep + 1); renderBrandesTrace(); });
  renderToy(); renderToyTotals(); renderContext(); filterGlossary(); renderBrandesTrace();
})();
