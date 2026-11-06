# 当前计算链路与检查点

当前 YouTube 全源入口 `quotient-run`：原始边列表 → CSR 缓存 → 外围树无损分解及端点数量 → 按核心锚点分组 → Numba 边记录内核与源级线程 → 分组检查点 → 全部源组完成后恢复原图并归一化。详细算法见 [Algorithm.md](Algorithm.md)，CSR 与线程见 [CSR_and_Numba.md](CSR_and_Numba.md)，树商图和边记录见 [Graph_Optimization.md](Graph_Optimization.md)。

2026-11-06 的新建 8 线程双快照任务已完成全源：每块 2,048 核心源组，累计 230 块；第一次因系统可用内存低于 1 GiB 预留线中止，第二次续算完成，两次监督墙钟合计 9,037.049 秒，不含中间停顿。全源覆盖和文件哈希通过核验；外围节点结构公式全部一致，核心节点未做另一套算法的全图逐点对照。数据和验证边界见[全量汇总报告](汇总报告.md)。

默认 CLI 使用 `dag` 内核，新建任务采用双累计快照检查点。旧 8.06% 检查点必须使用 `--kernel legacy`，不得将不同内核的分组直接混合。恢复时默认根据已有清单格式选择快照或旧分组模式；旧的 Python API `parallel_bc.quotient_checkpoint` 仍使用 legacy，逐块 dag API 是 `parallel_bc.quotient_checkpoint_dag`，双快照 API 是 `parallel_bc.quotient_snapshot`。

<a id="checkpoint"></a>

## 全源分组计算、检查点与恢复

此执行层复用[无损树商图](Graph_Optimization.md#tree-quotient)的带权内核，面向数小时的全源计算。2026-11-06 起，新建的 `quotient-run` 默认使用最短路径边记录内核 `--kernel dag`；旧版可用 `--kernel legacy`。历史检查点证据见[检查点实测](RESULTS.md#checkpoint-evidence)，新内核见[边记录优化](Graph_Optimization.md#dag-recording)。

新旧内核使用不同的检查点格式与代码指纹。**旧任务必须用 `--kernel legacy` 恢复；不能直接将既有分组与新内核混合续算。** 旧代码及检查点指纹保留，旧的 8.06% 进度不受新任务影响。新任务使用单独目录；新旧内核都只在完整覆盖时发布全量结果。

### 双累计快照（新任务默认）

每完成一个分块，将本块核心贡献加到累计向量，交替写入 `snapshot_A.npy` / `snapshot_B.npy`。写入当前清单未引用的槽位、刷新、重新读取核对并计算哈希后，最后原子替换 `manifest.json`，更新有效槽位和源组进度。清单替换前中断时恢复上一个快照，当前块重算；清单替换后恢复新快照，不重复累加。错误退出后必须从已提交快照重新加载，不使用内存中未提交的累计值。恢复校验图、代码、依赖、CPU/线程和分块配置、源计划、进度、源数量和快照哈希；发现损坏则报错，不自动回退。

格式为 `LinkRadar-quotient-dag-snapshot-v1`；累计快照仍是核心贡献，不是原图介数。清单只保留最新一次分块摘要，不长期保存逐块向量或逐源记录，因此无法再独立审计历史每一块。两个核心向量总计约 7.18 MiB，写入期间另有临时文件；完整结果数组、源计划、清单、缓存和其他文件另计。磁盘预检以两个快照、临时文件、最终结果和安全余量估算，不随全量分块数线性增长。每块仍需写一次快照；小分块会增加累计写入和 fsync 次数。

`--checkpoint-mode auto`（默认）对新 `dag` 任务选快照，对新 `legacy` 任务选旧分组；续算时读取清单格式自动选择，内核和所有身份仍须匹配。可显式指定 `--checkpoint-mode snapshot` 或 `--checkpoint-mode chunks`；快照只支持 `dag`，不能直接恢复旧的分组检查点。旧检查点没有被迁移或覆盖。

2026-11-06 在独立真实 YouTube 目录完成两次各32组的暂停/恢复，两个快照各3,761,440字节，累计64组与旧逐块版同源核心向量通过 `atol=1e-8, rtol=1e-10` 比较，最大绝对差9.31e-10；运行目录文件合计9,409,520字节，包含源计划和会话记录，不含外部图缓存。64组仍是部分贡献，不是全源介数；该验证也不是长期吞吐或断电持久性验证。

验收：双快照专项27项和全项目364项测试通过，覆盖NetworkX小图全源对照、普通异常和真实进程退出、快照部分写入、清单提交前后恢复、哈希/源计划/计数损坏、最终输出中断、旧格式自动识别及大整数回退。原有64组4/8线程试跑和legacy 8.06%任务的清单与已提交分组文件哈希核验未变。测试入口：`python -m pytest tests/test_quotient_snapshot.py -q`。

### 数学范围

完整计划包含每个有核心边的锚点恰好一次，顺序由固定种子生成。每个锚点代表的原始节点数 `M[r]` 同时作为源数量和目标数量；分组只拆分执行任务，不改变整图的数量权重。纯树分量无需核心遍历，最终解析恢复处理其全部源。

各组只产生核心贡献 `K_chunk`，完成后按组序相加。全部组完成后，以全源条件 `C_r=M_r`、`C=N`、`a_v=1` 恢复所有原始节点；无向 `/2` 和归一化按原始节点数执行一次。源组尚未覆盖完整时，不产生原图介数文件，也不把核心累加量称为原始源子集分数。

设核心源组计划被划分为互不重叠的集合 `R_j`，则每个检查点计算

```text
K_j(v) = sum_{r in R_j, r != v} M_r * D_r(v)
K(v)   = sum_j K_j(v)
A_V    = E(K) + T_V
```

`D_r` 是保留全部目标数量权重的核心依赖；`E` 把核心向量放回原始锚点位置，外围位置补零；`T_V` 是全源条件下的树路径及锚点端点修正。由源集合的不交并和求和可加性，分组不改变 `K`。`T_V` 只加入一次；若每个检查点都加入完整的 `T_V`，就会重复计数。数值上仍允许不同求和分组引起的浮点舍入差异。

YouTube 的计划有470,164个核心源组，完整恢复1,134,890个原始节点。若每1024个源组提交一次，共460个检查点；本次已完成任务使用每块2048组，共230次提交。调度微批次 `--batch-size 4` 与持久化分组 `--chunk-size` 是两个参数。

### 旧逐块文件模式的提交和恢复

每个检查点先写独立贡献文件，刷新并 `fsync` 后原子替换，再计算文件哈希。最后原子替换 `manifest.json`，这一步是提交点。

- 清单中已经提交的文件保持不变，恢复时逐个验证哈希、向量形状/类型、源组顺序、范围和源数量，再重建累加向量。
- 贡献文件存在但清单尚未引用时，该文件是未提交文件；重试会重新计算并替换这一组。
- 清单提交后进程退出时，恢复会读取这组贡献，避免重复计算或重复累加。
- 进程持有操作系统文件锁；并发写入会失败，进程死亡后锁由系统释放。锁文件留存本身不表示仍被占用。

恢复必须匹配原图/核心图指纹、全部树映射数组、源组计划、代码指纹、Python和数值依赖版本、线程数、调度/检查点大小、种子及CPU亲和性。发现不一致会拒绝续算。哈希用于完整性审计，不提供对恶意篡改的认证。

以上验证针对进程中断和强制终止。突然断电后的文件持久性还取决于操作系统、文件系统和存储设备，不承诺跨所有存储配置的断电事务保证。

### 内存、磁盘和预算

商图只在每次会话启动时完整加载/验证一次。每个检查点内部复用现有线程调度，只保留当前组的逐源任务记录；完成后丢弃，清单保存精简的分组/worker摘要。每个worker有独占工作区和累加向量。边记录内核另预分配每线程 `4*邻接项数+8*(核心节点数+1)` 字节，YouTube 4线程约 85.24 MiB；清单分别记录 `base_workspace_bytes`、`dag_workspace_bytes` 和 `local_total_bytes`，这是数组容量而非实际峰值 RSS。全源调用不创建百万原始源的tuple/set/dict。

旧逐块模式保留不可变分组文件以便逐块核验：YouTube每组核心向量约3.59 MiB，完整460组约1.61 GiB，另外保存原始节点分数和清单。旧模式启动时检查余下整个计划所需的磁盘空间，恢复需要重读已提交文件。双快照模式的空间预检和恢复按上文处理。

父进程延续原来的硬墙钟与系统可用内存监督。预算到期可能强制终止计算进程，最后一个未提交组需要重算；已提交组仍可恢复。会话结果写入独立 `session_*.json`，包括失败、超时和内存中止。旧逐块模式保留每组摘要；双快照模式的 `last_checkpoint` 只保留最新一块的执行时间、工作线程CPU时间、快照写入/读回/哈希时间、RSS快照和系统可用内存。RSS快照不是连续采样峰值；会话峰值另由监督进程记录。

每完成并提交一块，终端输出 `checkpoint ... committed 已完成/总数 ...`；清单的 `completed_core_groups` 同步更新。计算一块期间不会逐源输出，清单的 `status=paused` 也不能单独用来判断进程是否还在运行；全部恢复并提交最终结果后才为 `complete`。

### 使用方法

已有 `data/processed/youtube_csr_treeq` 且已安装 `.[compiled]` 时：

```powershell
# 新建计划，只完成一个检查点后正常暂停。已有run-dir拒绝覆盖。
.\.venv\Scripts\python.exe -m parallel_bc quotient-run --dataset youtube_csr --run-dir results/youtube_checkpoint --threads 4 --batch-size 4 --chunk-size 1024 --seed 45 --cpu-affinity 0 2 4 6 --reserve-gib 1 --timeout 180 --max-chunks 1

# 从已提交组继续，本次最多再完成两个检查点。
.\.venv\Scripts\python.exe -m parallel_bc quotient-run --dataset youtube_csr --run-dir results/youtube_checkpoint --resume --threads 4 --batch-size 4 --chunk-size 1024 --seed 45 --cpu-affinity 0 2 4 6 --reserve-gib 1 --timeout 300 --max-chunks 2

# 恢复原有 legacy 检查点，必须使用其原有全部参数。
.\.venv\Scripts\python.exe -m parallel_bc quotient-run --dataset youtube_csr --run-dir results/20261024T0903_youtube_checkpoint --resume --kernel legacy --threads 4 --batch-size 4 --chunk-size 1024 --seed 45 --cpu-affinity 0 2 4 6 --reserve-gib 1 --timeout 300 --max-chunks 2
```

CPU编号来自本机审计，其他机器应选择自身可用CPU。`--max-chunks` 是本次新增检查点上限，不是缩小原始图/全源计划。省略时持续推进，直到完成或本次预算耗尽；`--timeout` 是每次会话硬上限，不是总计划上限。

正常暂停时命令成功，但清单保持 `exact_all_sources=false`。只有校验全部源组和源数量后，才发布 `contributions.npy`、`bc_raw.npy`、`bc_normalized.npy`，清单标记 `exact_all_sources=true`。输出节点顺序与商图缓存 `original_ids.npy` 一致。

如果全部核心组已完成，但进程恰好在最终文件写入期间退出，目录里可能有尚未提交的最终文件；它们必须以清单的完成标记和输出哈希为准。恢复会重建核心总量、重新完成恢复与文件提交，不需要重算已提交的源组。

小图和进程故障验证：`python -m pytest tests/test_quotient_checkpoint.py -q`。两张旧真实图的全源暂停/恢复回归：`python scripts/verify_checkpoint.py`。这些验证不会启动YouTube全源计算。

`python scripts/audit_checkpoint.py <run-dir>` 和 `python scripts/verify_youtube_checkpoint.py <run-dir>` 针对旧逐块检查点，不适用于双快照格式。双快照可使用只读核验（不启动计算、不写入报告）：

```powershell
.\.venv\Scripts\python.exe -B scripts/audit_snapshot.py results/youtube_checkpoint --quotient data/processed/youtube_csr_treeq
```

该工具核对当前快照、源组计划及商图映射；`code_matches_current` 表明本机代码指纹是否与任务创建时相同，不代表已核对运行依赖和CPU亲和性。完整任务的最终文件以清单完成标记和输出哈希为准。
