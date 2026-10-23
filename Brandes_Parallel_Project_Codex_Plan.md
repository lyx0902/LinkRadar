# 复杂网络关键节点识别的并行计算与性能分析

## 面向 Codex 的完整实施方案

版本：1.0  
编写日期：2026-10-23  
课程：高性能计算（Python 版）：从天气预报、深度学习到计算广告  
项目类型：个人项目；单机多核 CPU；Python；无图像处理要求。

项目名称：LinkRadar

> 给实施者的指令：请按本文实现一个可以运行、验证和复现实验的完整仓库。不要只生成示意代码或空目录。优先完成 P0，再实现 P1；遇到下载或算力限制时保留功能、记录实际阻碍，并完成不受阻碍的部分。不要虚构实验结果，不要把尚未运行的实验写成已经完成。无需为常规实现选择再次询问用户；不需要部署、网站、GPU 或云服务。

## 1. 项目简述与目标

输入一张网络的节点连接关系，计算每个节点的精确介数中心性，找出承担较强中转作用的节点。实现相同 Brandes 核心算法的串行版本与多进程版本，在结果一致的前提下研究并行加速、任务粒度和进程开销。

增加一个轻量应用验证：按照介数、度数和随机顺序移除相同数量的节点，比较网络连通性变化，判断这些排名在所用网络中是否识别出了结构上重要的位置。

项目主线：

1. 读取、清洗并固定一张无权无向简单图。
2. 自己实现 Brandes 单源贡献计算。
3. 按起点划分任务，执行串行、静态分批、多进程动态分批计算。
4. 与 NetworkX 核对数值结果。
5. 在固定计算问题上测量性能。
6. 用节点移除实验解释排名的结构意义。
7. 导出可复现的原始记录、图表与中文报告。

### 1.1 需要回答的研究问题

- RQ1：使用多个进程计算精确介数中心性，何时比串行更快？
- RQ2：进程数和任务批次大小怎样影响性能？
- RQ3：静态分批与动态领取任务，在相同批次和核心算法下有什么差别？
- RQ4：移除高介数节点对连通性的影响，与按度数移除、随机移除有何不同？

不得预设动态调度一定更快，也不得预设介数排名一定优于度数排名。

### 1.2 现实意义与结论边界

介数衡量最短路径中转作用，可用于分析社交群体或合作群体之间的连接位置，或为抽象网络的关键节点排查提供候选。

本项目不根据匿名节点编号推断真实人物身份，不把科研合作网络称为“科学家贡献度排名”。高介数不等于学术贡献高、影响力强、设备故障概率高，也不等于删除后一定断网。节点移除实验是拓扑失效模型，不是现实行为或通信流量预测。

## 2. 项目范围与优先级

| 优先级 | 内容 | 验收要求 |
|---|---|---|
| P0 | 自写串行 Brandes | 全图精确计算；原始值与归一化值 |
| P0 | 静态分批、多进程动态分批 | 相同起点集合、相同核心算法；实际使用 spawn |
| P0 | 数据管线 | Facebook、ca-GrQc 的下载入口、手工导入、清洗、缓存、元数据 |
| P0 | 正确性 | 手算图、随机小图、非连通图与 NetworkX 比较 |
| P0 | 性能实验 | 进程数、图规模、分批大小；原始记录与统计图 |
| P0 | 节点失效实验 | 固定排名的介数、度数和多次随机移除 |
| P0 | 可复现交付 | CLI、配置、测试、快速演示、中文 README 和实际运行报告 |
| P1 | 工作进程时间线 | 真实 PID、批次起止时间、忙碌时间 |
| P1 | 粗粒度静态分配 | 每个进程一组连续起点，作为补充比较 |
| P1 | 节点编号顺序实验 | 固定编号顺序与固定种子打乱顺序 |
| P2 | CSR/Numba、共享内存 | 只在 P0 完成后独立加入，并重新建立同后端串行基线 |
| P2 | 有权图、近似采样、GPU | 不属于本轮交付要求 |

图结构绘制仅用于解释结果。默认用小图或局部子图，避免对几千节点强行绘制无意义的大图。

## 3. 技术路线与环境

### 3.1 默认技术栈

- Python：优先 3.11–3.13；在实际可用环境中确认兼容性。
- 核心图表示：连续整数索引的邻接表，邻居顺序固定。
- 核心算法：普通 Python，自写 BFS 与反向贡献累加。
- 并行：标准库 multiprocessing，明确使用 get_context("spawn")。
- NumPy：float64 局部贡献和总分数组、缓存。
- NetworkX：数据生成、正确性参照、连通性验证；不代替自写核心算法。
- pandas、Matplotlib：实验汇总及 PNG/PDF 图表。
- PyYAML：配置。
- psutil：可选，用于硬件信息；不可用时使用标准库并填 null。
- pytest：正确性、数据清洗、多进程与 CLI 冒烟测试。

使用 pyproject.toml 描述依赖。完成安装后保存实际依赖版本，不凭空声称测试过某个环境。不要强制用户升级到最新 Python。

### 3.2 资源约束

普通笔记本或台式机即可。默认并行进程数取 min(4, 可用逻辑 CPU 数)，真实执行时再限制为不超过起点数。记录实际进程数；物理核数、逻辑核数、CPU 配额能检测多少记录多少，未知写 null。

Windows 下必须可运行：工作函数定义在可导入模块顶层，CLI 使用主入口保护；不得依赖 fork 继承的全局图对象。

## 4. 算法基本原理

### 4.1 介数中心性定义

在无向图中，对每个不包含 v 的无序节点对 {s,t}：

\[
C_B(v)=\sum_{\{s,t\}\subseteq V\setminus\{v\}}
\frac{\sigma_{st}(v)}{\sigma_{st}}.
\]

- sigma_st：s 到 t 的最短路径数量。
- sigma_st(v)：其中以 v 为内部节点的最短路径数量。
- s、t 本身不计入中转贡献，即 endpoints=False。
- 不可达节点对贡献为 0。
- 所有边视为长度 1，weight=None。

对于 n>2，标准无向图归一化：

\[
C_B^{norm}(v)=\frac{C_B(v)}{(n-1)(n-2)/2}.
\]

n≤2 时全体中心性为 0。非连通图仍使用全图 n 作为归一化分母，不按各分量分别归一化。

### 4.2 Brandes 的核心思想

固定一个起点 s，用一次 BFS 统计最短路径数量和前驱关系，再按距离从远到近累计依赖值。重复所有起点即可得到完整介数。

不枚举所有完整最短路径；以路径数量和最短路径前驱关系复用计算。

单源阶段维护：

| 变量 | 含义 |
|---|---|
| dist[w] | s 到 w 的最短距离，未访问为 -1 |
| sigma[w] | s 到 w 的最短路径数；初始 sigma[s]=1 |
| pred[w] | w 的所有最短路径前驱 |
| order | BFS 出队顺序，反向遍历时从远到近 |
| delta[w] | 固定起点 s 时，w 累积的中转贡献 |

正向统计：扫描边 v—w 时，如果 w 首次出现，设置距离并入队；如果 dist[w] == dist[v]+1，则将 sigma[v] 加到 sigma[w]，并将 v 加入 pred[w]。后一个判断不能只在首次访问时执行，否则会漏掉等长最短路径。

反向累计：从 order 的尾部弹出 w，对每个 v in pred[w]：

\[
\delta_s(v)\mathrel{+}=
\frac{\sigma[v]}{\sigma[w]}(1+\delta_s(w)).
\]

其中 1 代表以 w 为终点的贡献，delta_s(w) 是从更远终点累积的贡献。

若 w != s，将 delta[w] 加到该进程的总贡献数组。不要将源节点自己的 delta 加入中心性。

### 4.3 核心伪代码

~~~text
single_source_accumulate(adj, s, accumulator):
    pred = 每个节点的空前驱列表
    dist = 全 -1
    sigma = 全 0；sigma[s] = 1
    dist[s] = 0
    queue = [s]
    order = []

    while queue 非空:
        v = 出队
        order.append(v)
        for w in adj[v]:
            if dist[w] == -1:
                dist[w] = dist[v] + 1
                w 入队
            if dist[w] == dist[v] + 1:
                sigma[w] += sigma[v]
                pred[w].append(v)

    delta = 全 0.0
    for w in reverse(order):
        for v in pred[w]:
            delta[v] += sigma[v] / sigma[w] * (1 + delta[w])
        if w != s:
            accumulator[w] += delta[w]
~~~

sigma 建议使用 Python 整数保存路径数，delta 与最终分数使用 float64。算法为全起点的精确枚举计算，但浮点贡献仍有舍入误差；不声称任意大图下任意精度数值完全精确。极端路径数量导致无法转换或非有限结果时必须明确报错。

### 4.4 全局累加与归一化：只做一次

所有起点累加得到 ordered_total；无向图中每对端点被两个方向计算：

~~~text
raw = ordered_total / 2
normalized = raw / ((n-1)*(n-2)/2)    # n > 2
~~~

等价地 normalized = ordered_total / ((n-1)*(n-2))。

每个批次、每个工作进程均只返回未缩放的有序贡献。主进程完成全局归并后统一缩放，禁止重复除以 2。

### 4.5 复杂度与并行位置

单源在邻接表上的时间为 O(n+m)，常见连通无权图的完整计算为 O(nm)。单进程额外工作空间为 O(n+m)。

并行外层起点循环；单源内部保留正向 BFS、反向累计的依赖顺序。不将一张图拆成几个互不相连的诱导子图分别计算。

算法参考：[R1]；公式与输出语义参照 [R2]；参考实现阅读 [R3]。本文伪代码为实施说明，代码应自行实现，参考或改写第三方代码时保留其许可证和归属信息。

## 5. 可手算样例：必须作为验收夹具

五节点无向图，边为：

~~~text
S A
S B
A T
B T
T U
~~~

从 S 出发，sigma[S,A,B,T,U] = [1,1,1,2,2]；单源贡献中 A=1、B=1、T=1、U=0，不计 S。

完整原始介数：

| 节点 | raw | normalized |
|---|---:|---:|
| S | 0.5 | 1/12 |
| A | 1 | 1/6 |
| B | 1 | 1/6 |
| T | 3.5 | 7/12 |
| U | 0 | 0 |

实现可使用整数编号，但夹具或标签表必须保留 S/A/B/T/U 的可读映射。

## 6. 真实数据集

### 6.1 Facebook 社交网络：必须纳入

- 数据集：Social circles: Facebook，SNAP。
- 官方页面：https://snap.stanford.edu/data/ego-Facebook.html
- 可用的官方说明页：https://snap.stanford.edu/data/egonets-Facebook.html
- 合并边列表：https://snap.stanford.edu/data/facebook_combined.txt.gz
- 官方公布规模：4,039 个节点，88,234 条边。
- 含义：匿名用户及其好友连接；是所采集 ego networks 的合并网络，不代表完整 Facebook 平台。
- 项目只需合并边列表，不必下载 profile、circles 等其他文件。
- 实验问题：哪些节点在所采集网络的最短路径中承担更多中转作用？
- 数据论文：J. McAuley and J. Leskovec, Learning to Discover Social Circles in Ego Networks, NIPS 2012。使用时按官方页面保留引用。

### 6.2 ca-GrQc 科研合作网络：必须纳入

- 数据集：General Relativity and Quantum Cosmology collaboration network，SNAP。
- 官方页面：https://snap.stanford.edu/data/ca-GrQc.html
- 下载：https://snap.stanford.edu/data/ca-GrQc.txt.gz
- 官方公布规模：5,242 个节点，14,496 条边。
- 含义：作者为节点，共同发表过相关类别论文的作者之间连边。
- 本项目称“科研合作网络”或“作者合作网络”，不能称“科学家贡献网络”并据此评价学术贡献。
- 分析问题：哪些节点连接了不同合作群体？其结构性中转作用是否与节点失效后的连通性变化有关？

### 6.3 原始统计与清洗统计必须区分

上述数字是官方页面的原始统计，不是清洗后图的强制断言。原文件可能包含正反重复记录、自环等；清洗后边数可能不同，特别是 ca-GrQc。

不得为了凑到官方数字而添加边、保留不一致的重复计数或删除节点。对外报告同时列“官方统计”和“本项目清洗后统计”，解释差异。

本方案编写时查阅了官方说明和下载入口，没有实际下载并审计完整原始文件。因此 SHA256、实际文件大小、自环数量、清洗后边数必须由实施时计算，不得填入猜测值。

## 7. 下载、导入与清洗规范

### 7.1 下载功能

- 只对注册的数据集提供自动下载；HTTP 超时、有限重试、退避、临时文件和原子重命名。
- 下载后检查 gzip 可完整读取，避免将 HTML 错误页当数据文件。
- 计算原始压缩文件 SHA256，保存实际 URL、UTC 下载时间、字节数。
- 下载失败时明确显示原因，保留手工导入入口，不影响合成数据的离线演示。
- 不搜索或静默使用不明镜像。不声称公开下载等同于任意再分发许可；保留数据来源、引用与官方使用说明。
- 默认不将原始数据与大结果提交 Git，只提供下载脚本和 manifest。

### 7.2 清洗规则

1. 自动读取 txt 或 txt.gz；忽略空行和以 # 开头的注释。
2. 对这两个 SNAP 文件，数据行应为两个整数节点 ID；异常行默认报出行号并失败，不静默跳过。可提供显式宽松模式并报告跳过数。
3. 在移除自环之前收集出现过的全部节点，避免只存在于自环中的节点被误删。
4. 将无向边规范为 (min(u,v), max(u,v))，去重，移除自环。
5. 默认保留全图、所有连通分量与已知孤立点，不自动只取最大连通分量。
6. 按原始整数 ID 排序映射到 0..n-1，邻接表内邻居排序。
7. 如需导入完全没有边的孤立节点，支持可选节点列表；单靠边列表无法恢复从未出现的节点。
8. 输出原始 ID ↔ 连续索引的映射，排名结果保留原始 ID。

### 7.3 缓存与元数据

保存清洗后的唯一无向边、节点映射和 n。可用 NPZ + JSON，避免依赖不透明的 Python 对象 pickle 缓存。

metadata 至少包含：

- dataset_id、原始/来源 URL、原文件 SHA256。
- 原始数据行数、自环行数、规范化后重复边行数。
- 发现节点数、最终节点数、最终唯一边数。
- 连通分量数、最大分量节点数、平均度。
- 清洗规则版本、节点映射规则、生成参数和随机种子（合成图适用）。
- graph_sha256：由 n、排序后的唯一边和确定的节点映射规范编码计算。

图指纹必须包括孤立节点信息；只对边字节做哈希不足以区分“同样边但不同孤立点”的图。

## 8. 合成数据与规模控制

真实图用于实际结构验证；合成图用于可控的规模实验。

P0 生成器：

| 图类型 | 默认参数建议 | 用途 |
|---|---|---|
| ER 随机图 | 使用 gnm_random_graph；目标平均度约 6 | 规模与一般稀疏结构 |
| BA 优先连接图 | m_attach=3 | 度分布不均的网络 |
| SBM 社区图 | 两个规模近似相等社区；期望内部度约 6、跨社区度约 0.5 | 群体连接与失效实验 |

ER 按目标边数随 n 线性增长，不使用固定边概率导致密度随规模无意增长。SBM 根据社区规模计算概率并限制到合法区间；记录实际参数与生成后结构，不假定一定连通。

规模阶梯建议：200、500、1000、2000、4000；不是强制全部跑完。默认固定种子 42；正式研究可用 42、43、44 三张独立图，以区分结构差异与重复计时波动。

所有算法版本必须读取同一份缓存图。不能每个版本各随机生成一张图。

## 9. 代码结构与接口

建议仓库名：LinkRadar

~~~text
LinkRadar/
  pyproject.toml
  README.md
  configs/
    smoke.yaml
    quick.yaml
    full.yaml
    datasets.yaml
  src/parallel_bc/
    __init__.py
    __main__.py
    cli.py
    data.py
    generators.py
    brandes.py
    parallel.py
    validation.py
    benchmark.py
    robustness.py
    plotting.py
    reporting.py
  tests/
    fixtures/
    test_brandes.py
    test_data.py
    test_parallel.py
    test_robustness.py
    test_cli.py
  docs/
    README.md
    PROJECT_GUIDE.md
    PARALLEL_COMPUTING.md
    RESULTS.md
    DEMO_GUIDE.md
  data/raw/          # Git 忽略
  data/processed/    # Git 忽略
  results/          # 大文件 Git 忽略
~~~

模块数量可合并以保持简单，功能契约必须保留。

主要接口建议：

~~~python
load_graph(path, nodes_path=None) -> GraphData
generate_graph(spec) -> GraphData
single_source_accumulate(adj, source, accumulator) -> None
brandes_serial(adj, source_order=None) -> CentralityResult
brandes_parallel(adj, processes, mode, batch_size, source_order=None) -> CentralityResult
finalize_scores(ordered_total, n) -> tuple[raw, normalized]
validate_against_networkx(graph, scores) -> ValidationResult
run_benchmark(config) -> ExperimentManifest
run_robustness(graph, scores, config) -> RobustnessResult
~~~

CentralityResult 至少包含 raw、normalized、计时阶段、真实进程统计、图指纹与起点集合标识。返回结构推荐 dataclass；失败用异常或明确错误状态，不能返回全零伪装成功。

## 10. 并行实现：共享核心、局部累加、一次归并

### 10.1 共同约束

- serial、static_batches、dynamic_batches 调用同一个 single_source_accumulate。
- 图的邻接表在进程初始化时传入一次，后续任务仅携带 batch_id 和起点编号。
- 每个进程维护自己的 float64 长度 n 累加数组；不使用共享可写中心性数组或 Manager 高频更新。
- 工作进程处理完自己的全部任务后，只回传一次长度 n 的局部总分。
- 批次日志只回传小型统计，禁止每个起点回传一个长度 n 的贡献数组。
- 并行模式都不提前缩放分数；主进程逐节点求和后统一 finalize。
- 图对象在工作进程中只读。spawn 下每个进程有一份图副本，不能宣称零复制共享内存；报告其内存成本。

### 10.2 串行基线

一个进程按固定 source_order 遍历全部起点，将贡献直接累加到一个数组。source_order 默认连续索引升序；可固定种子打乱。

串行执行也应按相同批次边界记录日志，便于比较，但不创建工作进程。核心算法、数据类型、输入、邻居顺序均与并行一致。

### 10.3 静态分批 static_batches

先建立固定大小 B 的批次列表，按 batch_id % p 将各批次固定分配给某个工作进程。每个进程只执行预先分配给自己的批次。

不应把 Pool.map 默认调度称为严格静态分配。可使用独立任务队列或直接把预分配批次交给指定 Process。

### 10.4 动态分批 dynamic_batches

使用与 static_batches 完全相同的批次列表，放入公共任务队列。工作进程处理完一个批次后继续领取，直到读取结束标记。

静态与动态比较必须固定：graph_sha256、source_order、B、起点总集合、p、核心算法和日志开关。仅调度方式不同。

同一次完整计算中，每个节点恰好作为起点一次，不重复、不遗漏。

### 10.5 建议的运行器与进程生命周期

P0 可以用明确的 Process + Queue 实现两种调度，共用同一个工作循环：

1. 主进程创建队列、错误通道与进程。
2. 每个进程完成图初始化和局部数组分配，发送 READY(worker_id, pid)。
3. 收到所有独立 PID 的 READY 后，主进程开始计算阶段计时并发布任务。
4. 静态模式按指定进程投递，动态模式投递公共队列。
5. 进程按批次处理，记录真实时间，不按每条边发送日志。
6. 结束时每个进程发送 DONE，包括局部总分、完成批次数、起点数和统计。
7. 主进程持续消费结果和错误，归并局部向量；随后 join 和关闭队列。

避免先 join 所有进程再读取大结果队列，这可能造成队列缓冲阻塞。处理异常、超时、KeyboardInterrupt 时终止并回收所有子进程，清理队列，标记运行失败，禁止无限等待。

可选择更简单但语义等价的实现；必须保持真实静态分配、相同批次比较、每进程局部总分和真实 PID 日志。

### 10.6 批次、边界和公平性

- 默认 B=max(1, ceil(n/(8*p)))；静动态对照必须复用同一个 B。
- 正式扫描 B∈{1, 8, 32, 128, 512}，过滤无意义值后去重；也可按图规模自适应。
- n>0 时 p_used=min(p_requested, 可用CPU上限, n)；显式记录调整。
- n=0 返回空结果，不启动子进程，p_used=0。
- 即使 B 很大导致部分进程无任务，也必须报告实际情况，不能将请求的进程数误写成有效参与计算的进程数。
- 浮点累加顺序可能不同，允许容差内差异，不要求 bitwise 相等。
- 不允许串行纯 Python、并行换成 Numba 后将全部收益归为多核加速。

## 11. 正确性评测：先通过再跑性能

### 11.1 手算图

- 空图、1 个节点、2 节点有边/无边：全零。
- 3 节点路径：中间 raw=1、normalized=1，其余 0。
- 5 节点星形图：中心 raw=6、normalized=1，其余 0。
- 完全图：全部 0。
- 四节点菱形环：每节点 raw=0.5、normalized=1/6，检验等长路径。
- 第 5 节的五节点样例。
- 包含至少两个分量和孤立节点的图：检查全图归一化。

### 11.2 NetworkX 参照

使用同一张清洗图：

~~~python
nx.betweenness_centrality(G, k=None, normalized=False, weight=None, endpoints=False)
nx.betweenness_centrality(G, k=None, normalized=True, weight=None, endpoints=False)
~~~

自动比较全部节点；不得只比较 Top-10。

- 小图默认 atol=1e-10、rtol=1e-9。
- 较大图可用 atol=1e-8、rtol=1e-7，但必须保存实际容限和最大误差；不得无说明放宽到掩盖错误。
- 报告 max_abs_error、误差最大节点、allclose、有限值检查。
- 排名并列以原始节点 ID 升序打破；数值正确性依然按分数判断，不能因极近浮点分数引起次序变化误判算法错误。

随机测试覆盖多种种子和 n=5..50 的稀疏/较密图。测试 p=1、2、3 与 B=1、B>n、不整除批次等情况。

### 11.3 数据与并行专项测试

- 正反重复边、自环、注释、空行、非法行、只有自环的节点、显式孤立点。
- 单源贡献加总后与全图结果一致。
- 起点覆盖：各批次组成严格分割；实际处理计数与提交计划一致。
- 多进程冒烟必须以 spawn 实际运行，不用 mock 代替。
- 模拟工作进程错误时，主进程应及时失败且无遗留工作进程。
- 测试手工导入路径，不依赖外网；自动测试不强制下载真实数据。

## 12. 性能评测口径

### 12.1 分阶段计时

统一使用 time.perf_counter_ns 或 perf_counter，导出秒。图下载、读取、清洗、缓存、哈希、正确性参照、图片导出均不计入中心性执行时间，分别记录。

| 字段 | 精确定义 |
|---|---|
| t_startup | 开始创建工作进程到所有工作进程 READY；串行为 0 |
| t_execute_reduce | 全部 READY 后开始派发任务，到全部局部向量收到并归并完；包含队列等待和通信 |
| t_finalize | 生成 raw、normalized 的时间 |
| t_cleanup | 收尾 join、关闭队列等时间 |
| t_total | 本次新建执行器的完整算法调用时间，包含上述阶段与必要调度开销 |
| t_steady | t_execute_reduce + t_finalize；明确其不包含启动和清理 |

串行 t_execute_reduce 是实际串行核心累计耗时，不伪造通信阶段。要求阶段边界清楚、不重叠累计，t_total 与各阶段之和只有可解释的测量残差。

若计算期间主进程同步进行向量归并，就将其保留在 t_execute_reduce，不能通过减去 CPU 时间估算所谓“纯计算墙钟时间”。

### 12.2 性能指标

\[
S_p^{total}=\frac{\mathrm{median}(T_1^{total})}{\mathrm{median}(T_p^{total})},
\qquad
E_p^{total}=\frac{S_p^{total}}{p}.
\]

同样计算 S_p_steady，分开作图。核心结论同时提供 total 和 steady，不能只展示排除启动后的更好数字。

T1 是自写串行模式；额外提供 parallel(p=1) 诊断进程与通信开销。NetworkX 是外部参照，不替代同核心串行基线。

各进程指标：

- worker_busy = 所有本进程批次计算墙钟时长之和。
- 工作时间不均衡比 = max(worker_busy)/mean(worker_busy)，均值为 0 时为 null。
- 诊断忙碌比例 = sum(worker_busy)/(p_used * t_execute_reduce)；注明其不是操作系统测得的 CPU 利用率。
- 起点数、批次数、可选实际扫描邻接项数。无权连通图不同起点理论工作量通常近似，不把抖动都解释为算法负载不均。

### 12.3 重复、顺序和实验分组

- smoke：1 次；quick：3 次；full：默认 5 次正式计时。
- 有热身时标记 warmup=true，不混入正式统计；total 计时每轮仍重建进程。
- 不重复使用上一次算好的中心性代替计时。稳态图结构缓存可复用。
- 各方法与进程数的运行顺序按固定种子交错或打乱，减少温度、后台任务的顺序偏差。
- 汇总中位数、Q1/Q3、最小值、最大值；保留每次原始值。
- 相同图/参数内汇总重复，不把不同图种子的运行混成单一计时分布。
- 数据预处理、绘图、NetworkX 校验均在被测算法计时外执行。

## 13. 实验矩阵与运行预算

### 13.1 三套配置

| 配置 | 默认范围 | 目标 |
|---|---|---|
| smoke | n≈50–100；p=1/2；一个 B；合成图；随机失效3次 | 离线端到端演示，通常在约1分钟内，依机器而定 |
| quick | 两种合成图 n=200/500；可用 p∈{1,2,4}；一个 B；3次计时 | 日常开发验证，不自动下载或跑完整真实图 |
| full | 两个真实网络、规模阶梯、B扫描、5次计时 | 正式报告；显式启动，不在安装后自动运行 |

正式实验分开：

- 强扩展：固定图、固定 B，对比 serial 与 p∈{1,2,4,8}，不超过环境上限。
- 规模实验：固定图族与目标平均度，改变 n；固定 p。
- 调度/粒度：固定图和 p，按 B 对比 static_batches 与 dynamic_batches。
- 结构实验：选两三种图族；不要交叉穷举所有配置。

### 13.2 预算与中断

full 先对小图做 pilot，给出任务数量和粗略预算。CLI 支持 max_case_seconds、max_total_seconds 和 max_processes；预算耗尽标记未执行/超时项，保存已完成结果。

超时必须真正停止并回收该次计算的子进程，不能只对 Future 调用 timeout 后让后台继续跑。串行长任务可在独立受控案例子进程中执行；其外围启动不计入算法内部计时，但必须在总实验墙钟预算中计算。

若完整真实图太慢：

1. 保留全图配置和下载入口；
2. 可显式选择最大分量或确定性子图，并用独立 dataset_variant 名称、采样方法、种子和新图指纹；
3. 子图结果只代表子图，不冒充原网络；
4. 不静默改成采样起点的近似介数。

断点续跑按完整 case_key 判断：图指纹、代码版本、模式、p、B、source_order、日志开关和重复编号等。失败记录不能当成已完成；配置或代码变更不能错误复用旧结果。

## 14. 节点失效实验

### 14.1 目标与默认协议

检验原始网络中的节点排名与移除后的连通性变化之间的关系。使用固定初始排名，避免每删一个节点重算介数导致范围失控。

三种策略：

1. bc：按原始图的介数降序，原始 ID 升序打破并列。
2. degree：按原始图的度数降序，原始 ID 升序打破并列。
3. random：固定种子生成节点排列；每个重复使用独立可复现种子。

每条曲线从原始图开始，沿同一排列逐步移除节点，不能在不同移除比例重新随机抽取不相关的集合。

默认移除比例为 [0, .01, .02, .05, .10, .15, .20]，可选扩展到 1。k=floor(q*n_original)，去重后报告实际 k 和 k/n_original。full 的 random 默认20次；quick可10次、smoke可3次。

### 14.2 指标与非连通图处理

主指标：

\[
L(q)=\frac{|LCC(G\setminus R_q)|}{n_{\mathrm{original}}}.
\]

固定原始节点数作分母；原始图非连通时 L(0) 可以小于1，必须保留真实基线。

同时记录：

- remaining_nodes、remaining_edges。
- 最大连通分量节点数 lcc_size。
- 连通分量数 num_components，含剩余孤立点。
- 可选 lcc_size / remaining_nodes，明确是另一个指标。

空图时 lcc_size=0、num_components=0；以 remaining_nodes 为分母的可选指标为空值，不除零。组件数不保证随移除单调增加；连通分量的绝对最大节点数不能因纯删除而增加。

不在 P0 默认计算每个步骤的全对最短路径或全局效率，避免应用验证又引入过重任务。

### 14.3 展示与解释

介数、度数为固定排名的单条曲线；随机策略展示均值和10%–90%经验分位带，带区不能称为置信区间。

可以比较指定比例下 L(q) 的差值。允许结论为“该网络上没有明显优势”；不能预填介数策略一定造成最大破坏。

若缓存中心性用于失效实验，必须检查 graph_sha256、全起点精确计算标识和分数覆盖，不使用另一张图的排名。

## 15. CLI 契约与可复制命令

实现以下命令或语义等价入口；README 中的所有命令必须实际测试。

~~~bash
python -m pip install -e ".[dev]"
python -m parallel_bc --help

python -m parallel_bc data download --dataset facebook
python -m parallel_bc data download --dataset ca_grqc
python -m parallel_bc data prepare --dataset facebook
python -m parallel_bc data prepare --dataset ca_grqc

python -m parallel_bc data import --dataset facebook --path data/raw/facebook_combined.txt.gz
python -m parallel_bc data generate --family ba --nodes 500 --attach 3 --seed 42 --name ba_500

python -m parallel_bc centrality --dataset ba_500 --mode serial
python -m parallel_bc centrality --dataset ba_500 --mode static_batches --processes 4 --batch-size 32
python -m parallel_bc centrality --dataset ba_500 --mode dynamic_batches --processes 4 --batch-size 32

python -m parallel_bc validate --dataset ba_500 --against networkx
python -m parallel_bc benchmark --config configs/quick.yaml
python -m parallel_bc robustness --dataset ba_500 --random-repeats 10 --seed 42
python -m parallel_bc report --run-dir results/RUN_ID

python -m parallel_bc demo --profile smoke
python -m pytest
~~~

命令中的4进程按可用资源调整并明确记录。demo 必须生成自己的数据、运行各算法、检查正确性、执行小规模失效实验并生成图表和报告；不依赖网络或历史缓存。

centrality 命令导出的分数、元数据应能直接被 robustness 使用。找不到匹配分数时可以显式计算串行中心性，并将此步骤单独计时记录。

支持传入 data-dir、output-dir；路径处理使用 pathlib，支持 Windows 含空格路径。默认CLI正文中文，参数名英文。

## 16. 配置字段

YAML 至少支持下列语义；允许合理调整结构，但配置示例必须能直接运行：

~~~yaml
seed: 42
start_method: spawn
datasets:
  - id: ba_500
    generator:
      family: ba
      nodes: 500
      attach: 3
      seed: 42
benchmark:
  modes: [serial, static_batches, dynamic_batches]
  processes: [1, 2, 4]
  batch_sizes: [32]
  repeats: 3
  warmups: 0
  source_order: ascending
  task_logging: false
  max_case_seconds: 120
  max_total_seconds: 900
validation:
  enabled: true
  max_reference_nodes: 1000
  atol: 1.0e-10
  rtol: 1.0e-9
robustness:
  enabled: true
  fractions: [0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
  strategies: [bc, degree, random]
  random_repeats: 10
output:
  root: results
~~~

串行不因 processes 和 batch_sizes 的组合被无意义重复执行，建立明确的基线配对关系。参考计算超过 max_reference_nodes 时标记 skipped_reference；依然检查串并行一致。实际完成 full 时至少对预算允许的真实数据执行参照；若未执行要明确说明。

## 17. 结果文件与记录字段

每次实验生成独立 RUN_ID 目录，不覆盖已有结果：

~~~text
results/RUN_ID/
  config_resolved.yaml
  environment.json
  manifest.json
  graph_metadata.json
  timings_raw.csv
  timings_summary.csv
  validation.json
  centrality.csv
  worker_summary.csv
  tasks.jsonl                 # 诊断运行启用
  robustness_raw.csv
  robustness_summary.csv
  figures/
  report.md
~~~

多图实验可以按 dataset_id 建子目录；顶层 manifest 索引全部文件。

### 17.1 计时原始记录

至少保存 run_id、case_key、dataset_id、graph_sha256、n、m、mode、p_requested、p_used、active_workers、batch_size、source_order_hash、repeat、warmup、task_logging、start_method、t_startup、t_execute_reduce、t_finalize、t_cleanup、t_total、t_steady、status、error。

运行失败/超时的时间不作为正常值进入中位数。未测数据写 null 并带状态，不能写0冒充测量。

### 17.2 节点分数

centrality.csv：index、original_id、degree、bc_raw、bc_normalized、rank_bc、rank_degree。不要丢失原始编号映射。原始贡献无需按每个起点存盘。

### 17.3 进程与任务

worker_summary：worker_id、pid、processed_sources、processed_batches、busy_seconds、first_task_start、last_task_end。

tasks.jsonl（诊断时）：batch_id、worker_id、pid、source_count、start_rel_seconds、end_rel_seconds、compute_seconds、status。可选 source_start/source_end 或起点列表哈希，不把非连续起点误写成连续区间。

时间线使用同机可比较的 monotonic/perf_counter 时间基准并转换为相对本次 run 的时间；验证 start≤end、时长一致，不按批次编号伪造执行顺序。

详细日志会影响性能；正式主计时默认关闭详细任务日志，另跑标记明确的诊断案例生成时间线。静动态对照的日志设置必须一致。

### 17.4 环境信息

记录 Python、操作系统、CPU型号、逻辑/物理核心数、可检测的CPU限制、内存、包版本、启动方式、Git commit或源码指纹。不可获取的字段为 null。性能报告标注“环境可能受限”，不得虚构机器配置。

## 18. 必须生成的图表

1. 正确性：各方法最大绝对误差表。
2. 时间：固定图下 serial、static、dynamic 的总耗时对比，显示重复波动。
3. 加速比：进程数—S_total，理想线 y=p 只作参考。
4. 稳态加速：进程数—S_steady，注明排除启动和清理。
5. 规模：n—耗时，标注图族与实际边数。
6. 批次：B—耗时，静动态同图比较。
7. 失效：移除比例—LCC原始占比，含随机经验分位带。
8. P1 时间线或工作进程忙碌时长分布：必须来自真实记录。

预算未覆盖的图表显示“未执行相应实验”，不得补造曲线。PNG建议≥150dpi；可同时PDF。字体检测失败使用英文轴标签，避免中文乱码；报告正文中文。图例不要将 p_requested 当 p_used。

## 19. 实际报告与解释要求

自动生成中文 report.md，至少包括：

- 项目问题、输入输出、计算指标和应用边界。
- 本次实际图规模、来源、清洗统计。
- 同核心串并行设计及实现差异。
- 全部已执行正确性检查及最大误差。
- 主性能结果：total 与 steady 同时列出。
- 启动/通信/归并开销、任务粒度的观察。
- 节点移除实验：真实曲线与比较，不强行宣称介数更优。
- 资源限制、未完成实验、下一步建议。
- 运行命令、配置、版本、数据指纹和来源链接。

分析语句必须基于真实数据。如果没有显著加速，应解释可能原因，区分“有测量支持”和“待验证推测”。不声称课程实现优于所有成熟库。

## 20. 实施顺序与阶段验收

### 阶段 A：算法与数据骨架

交付自写串行内核、五节点夹具、小图测试、数据清洗和生成器。所有手算图与 NetworkX 对齐后进入下一阶段。

### 阶段 B：多进程

实现静态与动态同批次调度、局部向量归并、spawn、错误回收。以小图实际多进程比较全部节点，检查起点覆盖和真实 PID。

### 阶段 C：实验管线

实现计时边界、原始CSV、重复统计、预算控制。执行 smoke，生成一套真实结果。

### 阶段 D：应用验证与报告

实现固定排名失效实验、随机重复、曲线与中文报告。端到端 smoke 能从空结果目录独立完成。

### 阶段 E：真实数据与正式性能

尝试下载并清洗两个真实图；运行预算允许的正确性与性能实验。完整 full 必须由显式命令触发；若环境不够，保留可执行配置并说明未跑范围。

### 阶段 F：P1

如果基础验收全部通过，增加诊断时间线、粗粒度静态或起点顺序实验。不要为了界面或额外算法拖延 P0。

有效工时预估：算法与清洗6–9小时、多进程5–8小时、实验/失效/报告8–12小时，总计约19–29小时；初学理解和平台调试另有弹性。不是运行时间保证。

## 21. 验收清单

- [ ] 安装后 CLI help 可运行，README 命令与实际一致。
- [ ] 核心 Brandes 自写，不是仅包装 NetworkX。
- [ ] 五节点夹具与其他手算图的 raw、normalized 正确。
- [ ] 随机、非连通、孤立节点情况下与 NetworkX 全节点对齐。
- [ ] spawn 下 static_batches、dynamic_batches 实际完成多进程计算。
- [ ] 静动态使用同批次、同起点，准确记录 p_used。
- [ ] 每进程局部累加，不逐起点回传完整贡献向量。
- [ ] 两个真实数据集有来源、下载/导入入口、清洗和元数据。
- [ ] 真实数据下载失败有明确状态，合成演示仍能离线执行。
- [ ] 计时包含 total 与 steady，原始重复值与失败状态可追踪。
- [ ] 参数扫描有预算和超时退出，不遗留子进程。
- [ ] 失效实验用固定初始排名，随机多次重复，分母明确。
- [ ] 空图、结果并列、重复边、自环、孤立节点均处理。
- [ ] smoke 从数据生成到报告可一条命令完成。
- [ ] 至少交付一套实际执行的 smoke 结果。
- [ ] 中文报告列明已运行与未运行项，所有数字有原始记录。
- [ ] 测试通过；真实数据与长性能实验不作为普通单元测试前提。

完成 P0 的定义是代码与小规模端到端结果完整。正式真实图实验是否全部跑完必须另列，不能因工具环境受限就声称真实图已验证。

## 22. 常见错误：实施时逐项避免

1. 只求一条最短路径，漏掉其他等长路径。
2. 把起点或终点计入中转分数。
3. 在每个批次归一化，再在主进程重复归一化。
4. 无向图漏除以2，或先除以2后又套用有序贡献分母。
5. 将图拆开计算，破坏跨块最短路径。
6. 每个源节点都发送全图或回传长度n向量。
7. 使用线程跑纯 Python 核心后默认会获得多核加速。
8. Windows 依赖 fork 全局状态、嵌套工作函数或导入时启动进程。
9. 把请求进程数、伪造的任务顺序当实际测量。
10. 以不同图、不同采样数量或不同算法后端制造“并行加速”。
11. 只展示最好一次，丢弃正常但较慢的运行。
12. 将稳态加速误标为包含启动的总加速。
13. 静默移除小分量或孤立点，使图和归一化发生变化。
14. 为贴合官方边数而不透明修改清洗结果。
15. 删除节点后重算排名，却把实验称为固定排名移除。
16. 把随机波动带称为置信区间，或预设介数必胜。
17. 将“作者合作中转位置”描述成“科学家贡献度”。
18. 未执行真实图或full实验，却在报告填写预想数值。

## 23. 参考源

以下是算法、接口和数据来源；自定义工程方案、实验参数与验收要求属于本项目设计，不是来源对性能的承诺。

### R1. 原始算法论文

Ulrik Brandes. A Faster Algorithm for Betweenness Centrality. Journal of Mathematical Sociology, 25(2):163–177, 2001.

- DOI：https://doi.org/10.1080/0022250X.2001.9990249
- 论文公开副本：https://snap.stanford.edu/class/cs224w-readings/brandes01centrality.pdf
- 出版页面：https://www.tandfonline.com/doi/abs/10.1080/0022250x.2001.9990249
- 用途：最短路径计数、依赖递推、时间与空间复杂度。

### R2. NetworkX 介数定义与公共 API

https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.betweenness_centrality.html

用途：精确计算参照、端点处理、归一化、有向/无向语义。不要使用版本可能变化的私有API作为本项目核心依赖。

### R3. NetworkX 实现源码

https://networkx.org/documentation/stable/_modules/networkx/algorithms/centrality/betweenness.html

用途：理解 BFS 中的路径计数与反向累加。若复制或改写源代码，遵循其许可证并保留归属；推荐自己实现后用公共 API 核验。

### R4. NetworkX 官方并行示例

https://networkx.org/documentation/stable/auto_examples/algorithms/plot_parallel_betweenness.html

用途：按起点分组、局部贡献归并的参考。不要直接沿用示例中的计时结果；它不代表本项目性能。

### R5. nx-parallel 仓库

https://github.com/networkx/nx-parallel

用途：学习并行图算法的模块组织、分块思想；不要求安装或直接调用其后端作为自己的全部实现。

### R6. Facebook 数据

- https://snap.stanford.edu/data/ego-Facebook.html
- https://snap.stanford.edu/data/egonets-Facebook.html
- https://snap.stanford.edu/data/facebook_combined.txt.gz
- 数据论文：https://proceedings.neurips.cc/paper/2012/file/7a614fd06c325499f1680b9896beedeb-Paper.pdf

### R7. ca-GrQc 数据

- https://snap.stanford.edu/data/ca-GrQc.html
- https://snap.stanford.edu/data/ca-GrQc.txt.gz
- 总目录：https://snap.stanford.edu/data/

### R8. Python 多进程官方文档

https://docs.python.org/3/library/multiprocessing.html

用途：spawn、主入口保护、进程通信、资源回收。跨版本默认启动方式可能不同，本项目明确选择 spawn。

## 24. 可直接交给 Codex 的执行提示

> 请在当前工作区按本方案创建完整的 LinkRadar Python 项目。先检查已有文件与 AGENTS.md，保留用户已有工作。若是空目录则建立 src 布局、pyproject.toml、CLI、配置和测试。实现无权无向图的全起点精确 Brandes；serial、static_batches、dynamic_batches 共用同一单源内核，使用 spawn 多进程和每进程局部累加后归并。纳入 Facebook 与 ca-GrQc 下载及手工导入、确定性清洗和元数据，支持合成数据离线演示。实现正确性对照、分阶段性能计时、预算控制、固定排名节点失效实验、真实结果图表和中文报告。完成 P0 并执行 pytest 与 smoke，记录真实命令、结果及限制；有余力再做 P1。不要只写规划、伪代码或 TODO；不要杜撰性能。若真实数据下载不可用，明确说明，完成合成图全流程并保留真实数据可运行入口。交付时列出已完成项、测试结果、实际实验、运行命令、未完成项和原因。

