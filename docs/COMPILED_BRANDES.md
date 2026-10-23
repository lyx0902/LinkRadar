# 编译版单核 Brandes：实现原理、运行与验证

本轮实现 `numba_csr` 可选后端，处理完整的无权无向 CSR 图。这里的“编译”指把数值循环转换为 CPU 机器码；不是 GPU 计算，也没有开启多线程、结构压缩或近似采样。旧 Python 任意精度内核和旧全源计算/失效实验接口保留。

本说明描述单核后端；随后新增的源级线程并行和同后端1/2/4线程比较见[多核实现说明](COMPILED_PARALLEL.md)。

## 从数据到执行的完整路线

`原始图 → 已校验的只读 CSR → Numba 预热 → 分配复用工作数组 → 逐源 BFS → 反向依赖累积 → 汇总贡献 → 数值核对与计时报告`。

实现位置：

- `src/parallel_bc/compiled.py`：编译循环、工作数组、溢出检测与 Python 回退。
- `src/parallel_bc/large_benchmark.py`：后端接入、逐源日志、预算监督与结果原子发布。
- `src/parallel_bc/compiled_benchmark.py`：同源、交替顺序、多次配对比较。
- `tests/test_compiled.py`：逐源/全源、NetworkX、溢出和真实 spawn 验证。

## 为什么原来的 Python 实现慢

对每个源点，原实现遍历全图，创建前驱 `list[list[int]]`，保存距离、任意精度路径数和依赖值。逐边操作会反复执行 Python 字节码、对象访问、整数对象分配和动态类型处理。CSR 只改变了图的存储方式，不能自动消除这些执行开销。

新内核对数值函数使用：

```python
@numba.njit(cache=True, nogil=True, fastmath=False)
```

Numba 根据输入的数组类型生成专用的本机代码，通过 LLVM 编译执行。`njit` 要求 nopython 模式，无法编译时直接报错；不会悄悄降级为带 Python 对象的执行方式。`nogil=True` 允许该函数执行时释放 GIL，但本轮仍逐源、单线程运行；没有使用 `parallel=True` 或 `prange`。`fastmath=False` 不允许宽松浮点重排，但结果仍受 float64 舍入影响，不能承诺与 Python 逐位一致。

本轮加速还包含数组化和前驱表消除，因此对照结果代表整个新后端的效果，不能把全部差异归因于 JIT 本身。

## 数学含义没有改变

对一个源点 $s$，BFS 得到距离 $d_s(v)$ 和最短路径条数 $\sigma_s(v)$。初始条件为 $\sigma_s(s)=1$，其余为 0。

若边 $(v,w)$ 满足 $d_s(w)=d_s(v)+1$，则它位于从源点向外的最短路径有向无环图中：

$$\sigma_s(w)\mathrel{+}=\sigma_s(v).$$

从 BFS 队列的末尾向前处理，可以保证后继节点的依赖值先算好：

$$\delta_s(v)=\sum_{w\in N(v),\ d_s(w)=d_s(v)+1}
\frac{\sigma_s(v)}{\sigma_s(w)}\bigl(1+\delta_s(w)\bigr).$$

比例表示“到达后继的最短路径中，有多少经过当前节点”；1 表示后继自身作为终点的贡献；$\delta_s(w)$ 表示更远终点经过后继的贡献。

原版把前驱存下来，再向前驱推送贡献。编译版重新扫描 CSR 邻居，用距离差判断最短路径后继，再从后继拉取贡献。对每条无向边反向遍历至多做常数次检查，数学上求的是同一个和。这样无需为百万节点创建百万个前驱列表。

完整 BFS 队列同时充当反向栈。所有源点均计算后，才按原 `finalize_scores` 进行一次无向图除以 2，以及全图规模归一化。仅算源集合 $S$ 时，输出仍为：

$$A_S(v)=\sum_{s\in S,\ s\ne v}\delta_s(v).$$

它是未缩放的源子集贡献，不是全图精确介数，也不是自动换算的近似介数估计。

### 一个能手算的例子

边为 `S-A, S-B, A-T, B-T, T-U`，从S出发。BFS得到：

| 节点 | S | A | B | T | U |
|---|---:|---:|---:|---:|---:|
| 距离 | 0 | 1 | 1 | 2 | 3 |
| 最短路径数 | 1 | 1 | 1 | 2 | 2 |

反向计算时，U没有后继，所以依赖为0；T得到 `(2/2)*(1+0)=1`；A和B分别得到 `(1/2)*(1+1)=1`。S自身不计入该源贡献。最终这一源给出的向量是 `S=0, A=1, B=1, T=1, U=0`。

原版通过保存的前驱把T的贡献分给A和B；编译版处理A时查看后继T，再处理B时查看后继T。两者用同一个分配比例，变化在数据结构和执行方式。

## 工作数组与整数安全

| 数组 | 类型 | 用途 | 每节点字节 |
|---|---|---|---:|
| dist | int32 | 距离；-1 表示不可达 | 4 |
| sigma | int64 | 最短路径条数 | 8 |
| delta | float64 | 单源依赖 | 8 |
| queue | int32 | BFS 队列/反向顺序 | 4 |

工作区共 `24n` 字节；另有 `8n` 字节的贡献累加向量，以及 CSR、原始 ID 和运行库。工作区每次运行分配一次，每个源点清空后复用；分配时间和逐源计算时间分别记录。工作区大小不等于整个子进程峰值内存。

Python 整数可以任意精度增长，Numba 的 int64 不可以。因此在每次路径数相加前检查：

$$\sigma_s(v)>\operatorname{INT64\_MAX}-\sigma_s(w).$$

若条件成立，停止该源的编译计算，并用原 `single_source_accumulate` 从头重算该源；记录 `python_bigint_fallback`。编译阶段先把依赖存入临时数组，成功后才提交到总向量，因此回退不会重复累加半个源的结果。不会用浮点路径数或溢出后的整数继续计算。

CSR 缓存读取仍进行文件 SHA、结构和图指纹验证。编译入口另外检查偏移、索引范围和类型，避免向默认不做边界检查的本机循环传入错误索引。

## 安装与执行

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,compiled]"
.\.venv\Scripts\python.exe -m parallel_bc large-pilot --dataset youtube_csr --backend numba_csr --counts 1 4 8 --seed 42 --timeout 240 --total-timeout 1000 --reserve-gib 2
```

配对比较：

```powershell
.\.venv\Scripts\python.exe -m parallel_bc compiled-compare --dataset youtube_csr --sources 8 --repeats 3 --seed 42 --timeout 240 --total-timeout 1000 --reserve-gib 2
```

每个案例独立 spawn 一个子进程；这用于硬预算监督，不是多核计算。Python 与编译后端使用完全相同的完整图、源点及顺序。奇数轮先 Python，偶数轮先编译版，以减小固定执行顺序的影响。

每次比较新建独立 JIT 缓存：第一轮编译版使用空缓存，后续轮从该缓存加载。Windows 使用扩展长度缓存路径，避免 Numba 原子写入的临时文件名超过 MAX_PATH。

## 怎样看时间

- `t_load_validate`：读取并验证 CSR；不算进内核。
- `t_backend_import`：导入可选编译后端和 Numba。
- `t_jit_warmup`：同类型的两节点小图首次调用，包括编译或缓存加载、分发及小图执行；不声称是纯编译时间。
- `t_workspace`：检查编译入口及分配工作区。
- `t_kernel`：所有所选源点的计算调用之和，包含清空数组、BFS、依赖回传、提交和必要的溢出回退。
- `t_supervised_total`：父进程看到的整例墙钟，包含启动、加载、校验、预热、计算、日志、结果保存和回收。

因此需要同时报告“内核加速比”和“整例墙钟加速比”。小规模工作量中，缓存校验、Numba 导入和编译可能占很大比例，不能用内核加速比替代用户实际等待时间。

结果位于 `results/<RUN_ID>_compiled_compare/`。`manifest.json` 保存环境、代码/图/源指纹、每轮指标与逐轮验证；案例子目录保存 `case.json`、源点日志和贡献向量。比较全节点贡献向量，默认 `atol=1e-8, rtol=1e-10`，只有正确且完成的配对进入统计。

## 验证边界与讲解方式

测试覆盖手算图、孤立点、多个连通分量、环、星形、完全图和随机图。逐源对照原 Python，完整遍历所有源后对照 NetworkX 的 raw 和归一化值。分层图构造超过 int64 的最短路径条数，检查回退发生且只累加一次。

YouTube 固定源点的配对验证覆盖每个节点的贡献值；它验证这些源点，不代表完成了百万个源点的全源验证。重复相同源点的中位数只描述该源集合的性能，外推全源耗时仍有源点差异、长时间负载和回退概率的不确定性。

可以这样向老师解释：**“保留 Brandes 的数学定义，把逐边操作转换为机器码，并用连续数组和隐式最短路径 DAG 降低对象开销。先验证相同输入得到相同贡献，再将首次编译与稳定计算分开计时；本轮不改变 $O(n+m)$ 的单源复杂度和 $O(n(n+m))$ 的全源复杂度。”**

## 官方参考

- [Numba 入门：nopython、JIT 与测量方法](https://numba.readthedocs.io/en/stable/user/5minguide.html)
- [Numba 与 Python 语义差异：固定宽度整数与边界检查](https://numba.readthedocs.io/en/stable/reference/pysemantics.html)
- [Numba 版本兼容表](https://numba.readthedocs.io/en/stable/user/installing.html#version-support-information)

本轮实测与解释见 [编译版首轮实测报告](../refine-logs/COMPILED_FIRST_LOOP_RESULT.md)。
