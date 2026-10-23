# 编译版 Brandes 的多核线程实现

本说明对应 `numba_threads`：同一张完整CSR图上，把源点分配给多个CPU线程；每个源点仍用已验证的单核编译内核。当前提供1/2/4线程的同源扩展实验。原Python串行与spawn多进程接口继续保留。

## 为什么这次用线程

原Python数值循环受GIL约束，原项目使用spawn多进程。现在最耗时的函数已经通过Numba以 `nopython + nogil=True` 编译：进入本机循环时释放GIL，允许不同线程同时执行数值代码。

这使同一进程中的线程能直接读取同一份CSR，而不用向多个spawn计算进程传递图，也不用另外管理共享内存的生命周期。线程各自拥有可写工作区，避免数据竞争。外层仍用一个spawn子进程隔离每个实验案例，让监督进程可以在硬超时或系统内存不足时结束整个计算子进程及其线程。

实际结构：`监督进程 → 一个计算子进程 → p个本机计算线程 → 按worker编号归并`。p=1在计算子进程主线程内联运行，避免把多线程启动开销强加给单核基线。

这是从原计划V3/spawn方案调整出的线程路线；依据是已验证的nogil内核和现有内存条件。没有实现或测量“编译版spawn vs线程”的独立消融，不能据此声称线程普遍快于进程。

## 工作量怎么分，为什么结果相同

固定源点有序集合为S，分给各worker的源点集合为 $S_0,\ldots,S_{p-1}$，满足互不重复、合起来恰好覆盖S。第j个worker只累加自己的向量：

$$A_j(v)=\sum_{s\in S_j,\ s\ne v}\delta_s(v).$$

全部worker成功后，按编号相加：

$$A_S(v)=\sum_{j=0}^{p-1} A_j(v).$$

数学上改变了求和分组，没有减少源点，也没有切掉图节点。float64的加法次序会变化，因此逐节点容差比较，不要求逐位一致。动态调度在不同运行中可能改变源点归属，日志保存实际分配。

调度方法：先把有序源点分成长度B的批次，给每个worker预留一批；后续批次通过线程安全队列领取。预留首批保证每个计入active_workers的worker实际处理过源点。有效worker数为 `min(请求线程数, 批次数)`。扩展实验要求足够多的批次，不能把只处理一个批次的案例说成4线程扩展。

线程启动后在屏障汇合，再计算源点。所有数值函数在创建线程前完成预热，防止多个线程并发触发JIT。共享对象只有只读图和任务队列；工作区、依赖数组及局部贡献向量不共享写入。

## 内存与异常边界

每worker：`24n`字节工作区 + `8n`字节局部向量。YouTube每worker约34.63MiB，4个约138.54MiB，另有共享CSR、ID、运行库和图校验开销。归并复用worker0的局部向量，不额外复制所有worker总和。

源计数溢出仍回退原Python任意精度内核；回退发生在对应worker的私有向量上。Python回退受GIL约束，发生频繁时会影响扩展效率，必须报告回退源数。

worker异常或协作超时会设置取消标志；其他worker在源边界停止。正常异常路径会join本次线程，成功前不发布贡献向量。外层硬墙钟监督负责处理无法在单源循环中立即退出的情况，并保留已完成源日志。未完成向量不能用于排名或失效实验。

## 运行

```powershell
.\.venv\Scripts\python.exe -m parallel_bc compiled-scale --dataset youtube_csr --sources 64 --threads 1 2 4 --batch-size 4 --repeats 3 --seed 42 --timeout 120 --total-timeout 900 --reserve-gib 2
```

`--reserve-gib`是实验期间必须保留的系统可用内存。资源不足时记录拒绝启动/资源中止，不将该案例混入性能统计。本轮机器可用内存较低，实际运行如调整预留值，会在实测报告中列明。

`--cpu-affinity`可限定计算子进程的逻辑CPU集合，所有线程配置使用同一集合，监督进程和用户其他应用不变。不要在其他机器照抄CPU编号：本机经Windows `GetSystemCpuSetInformation`核对，逻辑CPU0/2/4/6分别对应四个不同CoreIndex，EfficiencyClass均为1；8—15的类别为0。本轮主实验使用 `--cpu-affinity 0 2 4 6`，让1/2/4线程都在同一组高性能核心范围内运行，每个物理核只选一个逻辑线程。API字段含义见[微软官方说明](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-system_cpu_set_information)。

比较入口：`src/parallel_bc/compiled_scaling.py`。计算实现：`src/parallel_bc/compiled_threads.py`。线程编号、原生OS线程ID、PID、源点归属、每源开始/结束时间、worker CPU时间均保存到JSON/JSONL中。

输出 `results/<RUN_ID>_compiled_scaling/`：

- `source_order.npy`：源点有序集合；各案例的sources.json保存原始ID与哈希。
- `warmup/`：单独的一源编译预热，不计入性能统计。
- `repeat*_t*/`：案例状态、逐源日志、完整贡献向量、逐worker证据与validation.json。
- `manifest.json`：参数、代码/图/源指纹、环境、每轮原始值、正确性和汇总。
- `report.md`：按线程数汇总的执行、墙钟、加速比、效率、CPU利用证据与内存。

每轮线程配置轮换顺序。例如三轮为1→2→4、2→4→1、4→1→2。同一轮以1线程为参照，对全部节点贡献向量检查。只将数值、源覆盖和worker证据均通过的案例纳入统计。

## 计时与HPC指标

- `t_execute_wall`：线程创建、同步、逐源计算、任务领取、日志和join的墙钟。p=1对应内联执行。
- `t_reduce`：归并局部向量与最终有限性检查。
- `t_execute_reduce`：完整执行/归并阶段的墙钟，用于同后端并行加速比较。
- `t_worker_busy_sum`：各源耗时求和；并行时会重叠，不能把它当总等待时间。
- `t_supervised_total`：包含spawn、导入、CSR校验、缓存加载、数组准备、执行、归并、保存和回收的整例墙钟。

加速比与效率：

$$S_p=T_1/T_p,\qquad E_p=S_p/p.$$

每轮先计算配对比值，再对重复取中位数。执行加速与整例墙钟加速分别报告，不能与上一轮不同源数的Python/编译比值直接相乘。

`average_worker_cpu_cores = sum(worker_cpu_seconds) / t_execute_wall` 表示执行阶段平均消耗的CPU时间并行度。它结合不同原生线程ID和重叠时间段，提供实际并行证据；不证明固定占用了多少个同规格物理核心。本机混合P/E核及用户其他负载均可能影响结果。

线程增多后，加速可能低于线程数：所有线程争用内存带宽和缓存，单源执行可能变慢；末尾批次可能分配不均；启动、同步、日志和归并也需要时间。Numba先消除了大量解释开销后，这些成本更容易成为限制。

## 验证与结果边界

测试覆盖随机、非连通和手算图的全源NetworkX对照、相同源子集对照、实际线程ID与源点覆盖、输入图不变、并发溢出回退、异常join、硬超时子进程回收，以及真实spawn扩展实验。

YouTube基准输出仍是完整图上固定源集合的未缩放贡献。64源不等于全源介数，不能导入现有要求全源精确分数的失效接口。全源耗时只能在同后端吞吐基础上粗估，需要另行设计检查点，不能直接启动多日任务。

实测见[多核首轮结果](../refine-logs/COMPILED_PARALLEL_RESULT.md)。

参考：[Numba nogil官方说明](https://numba.readthedocs.io/en/stable/user/jit.html#nogil)、[Python线程共享内存与GIL说明](https://docs.python.org/3/library/threading.html)。
