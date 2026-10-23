# YouTube CSR 导入与固定起点试算

本轮范围：完整YouTube图的CSR准备、Python参考内核的1/4/8起点短试算。
首轮使用Python参考内核；后续已接入可选Numba单核后端，见[编译版说明](COMPILED_BRANDES.md)。旧NPZ缓存、全起点精确接口与失效实验不变。

当前首轮已运行：准备29.52秒、邻接数组31.45MiB；1/4/8起点均有成功结果。初次k8触及3GiB预留，重试使用2GiB预留和240秒上限。完整实测、测试与限制见[首轮报告](../refine-logs/YOUTUBE_FIRST_LOOP_RESULT.md)。

后续编译版同8源三轮独立复测：内核中位数100.46秒→1.73秒，逐配对内核加速比中位数58.16倍；整例墙钟中位数105.35秒→8.02秒。全部节点贡献在容差内一致，详见[编译版实测](../refine-logs/COMPILED_FIRST_LOOP_RESULT.md)。这是组合实现优化，尚未完成YouTube全源计算。

再后续同编译内核的64源、1/2/4线程三轮实验已完成，执行归并中位数13.94/6.56/3.25秒；种子43独立集合也确认同方向结果。见[多核报告](../refine-logs/COMPILED_PARALLEL_RESULT.md)与[实现说明](COMPILED_PARALLEL.md)。当前4线程全源执行粗估约16小时，未启动全源任务。

## 输入与来源

本地文件：data/raw/youtube/youtube.txt，使用KONECT的% sym unweighted头部。
用户提供：https://github.com/dange-academic/real_network_datasets/blob/main/PRE2020/youtube.zip 。
该仓库README将YouTube条目指向KONECT。它不是本项目从SNAP成功下载的文件。
原文件审计：data/raw/youtube/local_audit.json；参考规模1134890节点、2987624唯一无向边。

## 生成独立CSR缓存

在仓库根目录运行：

~~~powershell
.\.venv\Scripts\python.exe -m parallel_bc data prepare-csr --dataset youtube --path data/raw/youtube/youtube.txt --format konect --provenance data/raw/youtube/local_audit.json --timeout 300 --reserve-gib 3
~~~

输出data/processed/youtube_csr/，已有目录时拒绝覆盖。
- original_ids.npy：int64，原始ID完整保留。
- indptr.npy：int64行偏移。
- indices.npy：int32双向邻接，行内有序无重复。
- metadata.json：图/文件哈希、来源、清洗统计、连通分量、数组字节数及准备计时。
- 同级youtube_csr.preparation.json：硬预算执行状态、墙钟与子进程峰值内存。

格式是显式选择的：
- snap：默认只识别#注释；两列整数，坏行报错。
- konect：要求% sym unweighted、% 边行数 n n，检查1..n ID与行数，保留头部声明的孤立点。
不在旧load_graph中无条件跳过任意%行，避免改变原数据契约。

数据路径使用整数数组去重、构建邻接，不创建NetworkX大图或边tuple/set/dict的多份副本。
图哈希沿用旧GraphData规范编码，包含节点ID与孤立点；新缓存用独立格式LinkRadar-csr-v1。
读取时验证文件SHA、CSR对称性/有序性/索引和规范图哈希；通过只读内存映射打开数组。

## 固定起点Python短试算

~~~powershell
.\.venv\Scripts\python.exe -m parallel_bc large-pilot --dataset youtube_csr --backend python_tuple --counts 1 4 8 --seed 42 --timeout 120 --total-timeout 1800 --reserve-gib 3
~~~

- 每个案例为独立spawn子进程，父进程监控硬墙钟和系统可用内存。
- 默认给系统预留3GiB，资源不足则记录resource_not_run或resource_limit。
- 超时回收仅本次子进程，退出后保留源完成日志和失败记录，不把未完成向量当成功。
- 某案例触及资源线后，后续更大案例记为resource_not_run，不继续重复启动。
- 固定种子生成排列的嵌套前缀；相同图/种子在不同最大k下仍选到同样前缀。
- python_tuple：从CSR构造一次Python邻接tuple，调用原single_source_accumulate。
- python_csr：原single_source_accumulate直接通过CSR邻接适配器运行。可用--counts 1与上面的k=1做同源对照。
- 两个后端均保留原Python任意精度路径计数；没有静默改成int64路径数。
- numba_csr：单核本机数组内核，int64路径数溢出前回退原Python；JIT预热与计算分开记录。用`--backend numba_csr`启用。

输出results/<RUN_ID>_youtube_pilot/：
- manifest.json与report.md：逐案例状态、计时、环境与代码/图/源顺序标识。
- source_order.npy：主源集合。
- k*/sources.json：索引、原始ID、顺序哈希。
- k*/sources_completed.jsonl：每个已完成源的耗时；未完成case中也可读取。
- k*/contributions.npy：只有完成整个案例后才以原子重命名发布。
- k*/case.json：状态、范围、内存、时间、完成源数。

## 输出语义与计时边界

结果是完整图上选定S的未缩放有序贡献：
A_S(v) = sum_{s in S} delta_s(v)。

k<n时exact_all_sources=False，result_kind=source_subset_contribution、scaled=False。
不会输出全图精确介数、归一化估计或伪造的全节点Top10，也不能导入旧精确失效接口。
k=n时这里仍保存有序贡献，尚需使用原finalize_scores统一/2和按全图n归一化。

t_kernel仅原单源核心之和；t_adapter、t_load_validate、t_child_total、t_supervised_total分别列出。
硬预算包含子进程启动、缓存校验、适配、计算和I/O。
child_peak_rss_bytes覆盖子进程整个生命周期，不能当作kernel-only内存或父子唯一物理内存总量。
Windows还采样private bytes；系统最小可用内存和采样间隔单列。

这些是短pilot的单次观察；两个后端可能按顺序运行，不用于宣称统计显著加速。
复测协议与后续优化见[计算计划](../refine-logs/YOUTUBE_COMPUTE_PLAN.md)。

## 验证

~~~powershell
.\.venv\Scripts\python.exe -m pytest -q
~~~

新增测试覆盖：KONECT严格解析、孤立点/自环/重复、原始ID与旧图哈希等价、
非连通随机图的全源/子集数值对照、NetworkX参照、缓存篡改、真实spawn、
硬超时回收和内存不足拒绝启动。
