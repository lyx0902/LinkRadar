# as-skitter 加速优化调研

- 日期：2026-11-10
- 目标：在现有精确全源 Brandes/树商图/边记录/双快照链路上，寻找还能加快 as-skitter 的优化
- 前置报告：[as-skitter 导入与全量估算](as-skitter_导入与全量估算.md)
- 当前基线（同一条 seed 46 前 2,048 核心源组）：
  - 4 线程：134.705 s，外推全源 26.75 h
  - 6 线程：151.430 s，外推全源 30.07 h（受系统负载影响，见 §1）
  - 8 线程：109.132 s，外推全源 21.67 h
  - 8 线程独立 seed 47：142.858 s，外推全源 28.37 h
  - 12 线程：104.045 s，外推全源 20.66 h
- 数值代码指纹：`011c87cce5a43c915e2d6cf7439df3271aac23493c53d64f74d39947e2a5bad5`

> 本文的优化结论只针对 as-skitter、当前代码和这台机器。所有“预期收益”都区分了**实测**、**结构上界**和**文献报告值**，不能直接相加或当成保证。

## 0. 结论摘要

**没有低成本的“大加速”藏在当前算法里。** 当前 8 线程已经接近这台机器的实际扩展上限；12 线程只比 8 线程快 **4.9%**，却多消耗 **33%** 私有内存和 **35%** CPU 时间。BCC/BADIOS 和 degree-2 路径收缩这两个结构优化在 as-skitter 上收益很小，均不值得为一次全源任务实现。

按“收益/成本”排序：

| 优先级 | 方向 | 预期收益 | 成本 | 建议 |
|---|---|---:|---|---|
| A | 保持 8 线程；内存不足时 4 线程 | 0%（13.0→8 线程已有 1.23×；12 线程仅再 4.9%） | 低 | **直接采用** |
| B | BFS 图重排（DAG 内核 A/B 已做） | 1,024 组×2：核心中位 **1.20×**、墙钟中位 **1.07×**；但两轮 1.057/1.348、wall 0.919/1.228，不稳定 | 中 | **暂不采用，需 3–5 轮复测** |
| C | 内核微优化（int32 offsets、复用 runner、batch 8/16、任务日志开销） | 约 1–5% | 中低 | 可作为附带优化 |
| D | MS-BFS/批量 Brandes（多源同时 BFS） | 文献 2–7×（无权 BC） | 高 | **精确算法里最有潜力** |
| E | direction-optimizing / pull-based BFS | 文献最高约 2×（稠密/偏斜图） | 高 | 研究型备选 |
| F | GPU 精确 BC | 文献 5–38× | 很高（需 GPU + 重写） | 当前机器不适用 |
| G | 近似/采样 BC | 10–100× | 中 | 只在允许“非精确”时考虑 |
| — | BCC/BADIOS 分解 | 结构上界 **1.8%** | 高 | **不做** |
| — | degree-2 路径收缩 | 结构上界约 **4.2%** | 高 | **不做** |
| — | 12 线程（4P+8E） | 实测 **4.9%**，多 33% 内存 | 低 | **不推荐** |
| — | 检查点/快照调参 | <0.1% | 低 | 不做 |

**一句话建议：**

- **只跑一次精确全源**：用 8 线程，关闭大内存应用，按 **24–30 h** 规划；不要再投入 BCC/degree-2/12 线程。
- **做研究/论文**：原型 MS-BFS 批量 Brandes（目标 2–7×）；其次做图重排或 pull-based BFS。
- **允许近似**：采样源点可以把 25 h 降到分钟级，但语义从“精确全源”变成“估计”；这需要单独定义误差和验证。

## 1. 当前瓶颈：线程与内存

### 1.1 同源 2,048 组的线程曲线

四个配置都使用 **seed 46 的前 2,048 核心源组**（同一源计划；2,048 组代表 2,330 原始源），同一核心图、batch size 4、chunk 2,048：

| 线程 | 亲和性 | 核心执行 | 每组 | CPU 并行度 | 私有峰值 | 运行中最低可用 | 外推全源 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4 | 0,2,4,6 | 134.7054 s | 0.065774 s | 3.70 | 1,355 MiB | 5,546 MiB | 26.75 h |
| 6 | 0,2,4,6,8,9 | 151.4300 s | 0.073940 s | 4.92 | 1,413 MiB | 3,113 MiB | 30.07 h |
| 8 | 0,2,4,6,8,9,10,11 | 109.1324 s | 0.053287 s | 6.95 | 1,692 MiB | 850 MiB | 21.67 h |
| 12 | 0,2,4,6,8,9,10,11,12,13,14,15 | 104.0448 s | 0.050803 s | 9.83 | 2,247 MiB | 1,733 MiB | 20.66 h |

### 1.2 怎么读

1. **6 线程没有优势**：151.430 s，反而慢于 4 线程的 134.705 s；CPU 并行度只有 4.92/6，该轮加载/校验 18.88 s（其他轮约 10 s），说明该次运行受系统负载或 E 核调度影响，但至少没有显示 6 线程的收益。**6 线程不作为推荐配置。**
2. **4→8 线程有效**：134.705/109.132 = **1.23×**。8 线程使用 4 个 P 核 + 4 个 E 核，虽然 E 核较慢，但仍带来实际收益。
3. **8→12 线程几乎无效**：109.132/104.045 = **1.049×**，只快 4.9%。原因是多用了 4 个 E 核后，CPU 时间从约 759 s 涨到约 1,023 s（+35%），私有内存从 1.69 GiB 涨到 2.25 GiB（+33%），大部分额外算力消耗在内存带宽/缓存竞争和 E 核的低单核吞吐上。**12 线程不值得。**
4. **8 线程的内存已经偏紧**：2,048/8 时系统可用内存最低到 **850 MiB**；12 线程最低 1,733 MiB，但私有峰值 2.25 GiB。对这台 15.5 GiB 内存、测量时仅约 2.17 GiB 可用的机器，8 线程是“速度/内存”的折中点；内存不够时 4 线程（1.36 GiB 私有峰值）更稳。
5. **源组成本波动很大**：同规模 2,048 组的 seed 46 与 seed 47 相差 **31%**（0.0533 对 0.0698 s/组）。线程调优只能改变常数，不能消除这个长尾；全源平均要用多个样本估计。

### 1.3 结论

- **保留 8 线程**；内存不足或系统有其他大应用时用 4 线程。
- 不建议 6 线程（本次实测慢于 4 线程，受系统负载/调度影响但没有收益），也不建议 12/16 线程（12 线程仅 4.9% 提升，超线程会进一步争用缓存/带宽）。
- 系统层面可尝试：Windows 高性能电源计划、关闭后台大内存应用、保证散热；这些可能带来 5–10% 的间接改善，但不可控、也不改变算法复杂度。

## 2. 结构优化一：BCC/BADIOS 分解——实测不值得

### 2.1 实测

对 as-skitter 商图核心运行 `core-diagnose`（只读，不修改检查点）：

| 指标 | as-skitter 核心 | YouTube 核心（项目历史） |
|---|---:|---:|
| 核心节点 / 边 | 1,464,757 / 10,863,640 | 470,164 / 2,322,898 |
| 割点 | **5,441** | 8,387 |
| 桥 | **483** | 2,364 |
| 边块（含单桥二节点块） | **7,063** | 8,935 |
| 含环块 | **6,580** | 6,571 |
| 最大块节点 | **1,443,769（98.57%）** | 452,060（96.15%） |
| 最大块边 | **10,830,987（99.70%）** | 2,295,072（98.80%） |
| 结构工作指标比（含环块/全局） | **0.9819** | 0.9457 |
| 结构工作指标下降 | **1.8%** | 5.4% |

诊断本身：DFS 内核 **0.63 s**，加载/校验商图 **16.64 s**，受监督总墙钟 **19.88 s**，峰值 RSS **998.7 MiB**。

### 2.2 判断

- 最大双连通块保留了 **98.57% 的核心节点和 99.70% 的核心边**；即使把所有小块都完美并行/解析处理，结构工作量也只下降 **1.8%**。
- BADIOS 的收益来自“打碎成很多可独立计算的块”；as-skitter 的核心几乎是**一个巨型块**，打碎后主要计算仍在同一块里。BADIOS 的割点克隆、side-vertex 处理和块间权重修正还会增加开销。
- 文献 BADIOS 在更分散的图上报告 2.8–3.8×，但这不适用于 as-skitter。**不做 BCC 数值实现。**

## 3. 结构优化二：degree-2 路径收缩——上界只有约 4%

核心度分布（`core.indptr` 统计）：

| 核心度 | 节点数 | 占核心 |
|---|---:|---:|
| 0（纯树锚点，不遍历） | 737 | 0.05% |
| 2 | 259,370 | **17.71%** |
| 3 | 186,127 | 12.71% |
| ≥4 | 1,018,523 | **69.54%** |
| ≥32 | 104,537 | 7.14% |
| ≥1024 | 912 | 0.06% |

如果把所有 degree-2 节点收缩成带权路径：

- 节点最多减少 259,370（-17.71%）；
- 边最多减少约 259,370（-2.39%）；
- 按 `n+m` 的工作代理，最多减少约 **4.2%**。

而且在无权重图上做路径收缩后，Brandes 需要处理“路径长度权重 + 最短路径条数 + 端点质量”的联合恢复；这不是现有“树商图解析恢复”公式能直接覆盖的。为 4% 的结构上界引入新的精确恢复逻辑，**性价比很低，不建议**。

## 4. 内核微优化——合计约 1–5%

当前 8 线程每组 0.0533 s，BFS 要从每个源扫描核心图约 21,727,280 个有向邻接项；1,464,020 个源合计约 **3.18×10¹³** 次邻接扫描。可做但不改变数量级的优化：

1. **把 DAG offsets 从 int64 改成 int32**：核心有向邻接项 21,727,280 < 2³¹，int32 足够；每线程 offsets 从 11.7 MiB 降到 5.9 MiB，8 线程省约 47 MiB，可能改善缓存/带宽。预计 **1–3%**；需要改代码、重跑小图和全节点容差测试。
2. **跨 chunk 复用 runner/workspace**：当前每个 2,048 组 chunk 重新分配 8 份 DAG 数组和工作区；715 个 chunk 的分配/触页开销约每次 0.1–0.5 s，总计约 1–6 分钟（<0.1 h）。可在 checkpoint 循环外复用 runner；收益很小。
3. **减少每源 Python 记录开销**：`threaded_contributions` 为每个源构造 dict、记录时间戳、追加到 `worker_tasks`，全源 1.46M 条；预计 10–20 s（<0.01%），且会影响诊断信息。**不建议为性能删掉。**
4. **batch size 8/16**：当前 batch 4，2,048 组 = 512 个批次；12/8 线程下队列开销很小。大 batch 可能减少调度开销，但会增加尾部不均衡。预计 **0–2%**；可做一个 2,048 组 A/B 测试，但不值得为此改生产默认值。
5. **电源/散热/内存**：Windows 高性能计划、关闭后台应用、保证散热，可能 5–10% 的间接改善；这是环境优化，不是算法优化。

**结论：** 微优化合计约 1–5%，可把 25 h 变成约 24 h，不会带来质变。若目标是“尽快跑完一次”，先把系统内存和电源状态弄好，比改内核更划算。

## 5. 图重排：最值得先做的中等成本实验

### 5.1 文献与外推

- **FrontOrder**（PPoPP'25 poster）报告 BFS 在 Ligra 上平均 **2.65×**、GPOP 上 **1.73×**，L2 缺失减少 25.6%。
- **HisOrder**（2025）报告 BFS 平均 **1.20×** 于已有重排方法。
- 项目自己在 YouTube 上做过 BFS 重排 + 6 线程实验：三轮配对工作比约 **1.04 / 1.55 / 0.78**，中位 **1.044**，没有通过“每轮 >1 且中位 ≥1.05”的门槛，因此没有替换原编号。该实验用的是旧的 `WeightedAccumulator`，不是当前默认 DAG 内核。

### 5.2 对 as-skitter 的判断

- as-skitter 有大量高 degree 枢纽（104,537 个度≥32，912 个度≥1024），BFS 的 frontier 局部性可能比 YouTube 更差；图重排有机会。
- 本次已经在**实验分支**上实现 `layout_contributions_dag`（DAG 内核 + 布局），并用 `scripts/benchmark_layout_dag.py` 做了 1,024 组、8 线程、2 轮交替 A/B；默认 `quotient-run`/snapshot 路径未改。
- 实测结果（见 5.3）：核心/工作加速比两轮为 **1.057 / 1.348**，中位 **1.202**；但端到端墙钟为 **0.919 / 1.228**，中位 **1.073**，两轮差异很大。**没有观察到稳定、可重复的 >5% 端到端加速。**
- 如果继续做，需要 3–5 轮、2,048 组、安静机器上的复测；目标设为稳定 **>5% work 且 wall >1**，否则不采用。当前建议仍是 **natural / 8 线程**。

### 5.3 本次 DAG-layout A/B 实测（2026-11-10）

代码位置（实验分支 `experiment/as-skitter-dag-layout`）：

- `src/parallel_bc/core_layout.py`：新增 `layout_contributions_dag`，`layout_job(..., kernel="dag")`；默认 `kernel="weighted"`，`layout-tune` 行为不变。
- `tests/test_core_layout.py`：新增 DAG-layout 对照测试。
- `scripts/benchmark_layout_dag.py`：独立 A/B 脚本，不写检查点，不进入 `quotient-run`。
- 默认主链路 `quotient-run` / `quotient_snapshot` / `quotient_checkpoint_dag` 未修改。

**协议：**

- 核心源组：seed 46 的前 **1,024** 组（代表约 1,160 个原始源）
- 布局：`natural`（原编号）与 `bfs`（BFS 重排，seed 1234）
- 线程：8；batch 4；DAG 内核；CPU 亲和性 `[0,2,4,6,8,9,10,11]`
- 两轮交替顺序：round 1 natural→bfs，round 2 bfs→natural
- 每次独立 `bounded_job` 子进程；`reserve=512 MiB`；每例 600 s 上限
- 产出一致性：natural 与 bfs 的全原图贡献向量逐节点对照，`atol=1e-8, rtol=1e-10`

**结果：**

| 轮次 | natural work / exec / wall | bfs work / exec / wall | work 加速比 | exec 加速比 | wall 加速比 | 最大绝对误差 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 50.521 / 50.476 / 68.02 s | 47.808 / 47.742 / 74.02 s | **1.0567** | 1.0573 | **0.9190** | 8.94e-8 |
| 2 | 70.359 / 69.890 / 98.57 s | 52.195 / 52.130 / 80.29 s | **1.3480** | 1.3407 | **1.2278** | 8.20e-8 |
| 中位 | — | — | **1.2024** | **1.1990** | **1.0734** | 8.94e-8 |

- `work` = 源组选择 + 核心执行/归并 + 全原图恢复；`wall` = 另含子进程启动、商图/布局加载、结果写盘。
- 两轮都通过逐节点容差检查，`fallback_sources=0`；结果都是部分源组，不是全源 BC。
- **结果不稳定**：natural 第 2 轮从 50.52 s 涨到 70.36 s（系统负载/内存波动，加载时间也从 16.84 s 涨到 27.05 s），而 bfs 第 2 轮为 52.20 s；这使第 2 轮 1.348 的加速比明显受运行条件影响。第 1 轮较干净：核心约 **1.057×**，但 wall 为 **0.919**（bfs 的布局加载更慢）。
- 中位 **1.202×** 不能当作可靠加速比；可信的区间更接近 **核心 5–20%、端到端不确定（0.92–1.23）**。这与旧 YouTube 实验的“每轮波动大、独立确认不稳”一致。
- 全源时布局只需加载一次（而不是每个 1,024 组子进程加载一次），所以 wall 的布局加载开销会摊薄；但核心加速比本身的不确定性仍然存在。

**结论：**

- 这次 A/B **没有证明** BFS 布局能给 as-skitter 带来稳定 >5% 的端到端加速；不建议基于这 2 轮结果替换 natural。
- 若继续，建议在安静机器上跑 3–5 轮、每轮 2,048 组，并同时报告 work 与 wall；如果 work 中位 >1.05 且 wall 中位 >1，再考虑接入生产检查点。
- 当前默认主链路保持 natural/8 线程；实验代码只在 `experiment/as-skitter-dag-layout` 分支，尚未合并。

## 6. MS-BFS / 批量 Brandes：精确算法里最有潜力

### 6.1 文献

- **Then et al., “The More the Merrier: Efficient Multi-Source Graph Traversal” (VLDB 2014)**：MS-BFS 同时执行多个源点的 BFS，用位并行状态共享顶点探索；在强连通小世界上报告 **>10×** 的 BFS/接近度加速。
- **Manuel Then 博士论文 (TUM, 2019)**：把 MS-BFS 扩展到 Brandes 风格的 **batched betweenness centrality**；反向 MS-BFS + 隐式前驱，多个源点在同一顶点/距离上合并处理；报告无权 BC **2–7×**、加权 BC 最高约 **3×**，相对 Brandes 约 **5×**；内存约 **12 MB/execution/百万节点**（无权）。

### 6.2 与当前实现的差距

- 当前 `threaded_contributions` 是“每个线程独立做一个源点的 BFS”，源点之间不共享任何遍历状态；MS-BFS 会把**多个源点合并到同一个顶点扫描**，提高缓存/向量化利用率。
- 但当前生产链路是**带权树商图**：每个核心锚点有源数量权重 `M_r`，目标端点也有质量；MS-BFS 的位并行状态如何与这些权重、路径数 `sigma`、溢出回退、精确恢复公式结合，需要重新推导和验证。
- 可行路径：
  1. 先在**未压缩原图**上验证 MS-BFS 精确 BC（小图对照 NetworkX；中图对照现有内核）；
  2. 再把它接到核心图上，保持端点质量；
  3. 最后处理树商图恢复和检查点。
- 预计收益：**2–7×**（文献，不同图/实现）；成本：**高**（新内核、验证、检查点身份变化）。如果目标是研究/论文，这是最值得投入的精确方向。

## 7. Direction-optimizing / pull-based BFS：次优研究备选

- **Beamer et al., SC 2012** 的 direction-optimizing BFS 在 frontier 变大时从 top-down 切到 bottom-up，稠密图收益明显；但 Brandes 还需要 `sigma` 和前驱/后继，不能只换 BFS。
- 文献中的可行做法是 **pull-based（bottom-up）sigma/delta 聚合**：CMU 的工作报告在 Hyperlink2012 上最高约 **2×**；Saule 的综述也提醒 direction-optimization “不总是能直接搬到 centrality”。
- 对 as-skitter：核心平均度 14.8、hub 多，bottom-up 可能减少高 frontier 层的边扫描；但反向依赖阶段仍要按距离层处理，改造量大。
- 预计收益：**1.5–2×**；成本：高。优先级低于 MS-BFS，但可作为第二条研究路线。

## 8. GPU / 近似：收益大但改变硬件或语义

### 8.1 GPU 精确 BC

- **Galliot**（INFOCOM 2023）报告路径合并 GPU BC 相对基线最高 **11.3×/5.7×/38.8×**；Bernaschi 等的 multi-GPU 工作、Pichetti 等的 MACBETH 多 GPU 分区也报告大幅加速。
- 但需要 CUDA/GPU 硬件、重写内核、处理原子依赖累加和内存上限；当前机器是笔记本，**不适用**。如果未来有 GPU，这是最大的单点加速方向。

### 8.2 近似/采样 BC

- 只采样 k 个源点（例如 1% = 14,640 个核心组），时间可降到分钟级；但输出是估计，不是精确全源 BC，必须重新定义误差、置信区间和验证协议。
- 如果项目允许“近似排名/演示”，这是 **10–100×** 的捷径；如果坚持 `exact_all_sources=true`，不能用。

## 9. 优先级路线图

### Tier 0：不做（已证伪）

| 方案 | 证据 |
|---|---|
| BCC/BADIOS 分解 | 最大块 98.57%/99.70%，工作指标只降 1.8% |
| degree-2 路径收缩 | 度 2 占 17.7%，工作代理上界约 4.2%，精确恢复复杂 |
| 12 线程 | 8→12 仅 4.9%，私有内存 +33%、CPU +35% |
| 检查点/快照调参 | 快照约 0.03 s/chunk，加载约 10–20 s/session，占 25 h 的 <0.1% |

### Tier 1：低成本，1–5%

- 保持 8 线程；内存不足用 4 线程；
- 关闭大内存应用、高性能电源计划、保证散热；
- batch 8/16 A/B；
- int32 offsets、跨 chunk 复用 runner；
- 预期把 25 h 压到约 24 h，不会质变。

### Tier 2：中等成本，0–20%（本次布局 A/B 不稳定）

- **BFS 图重排 + 当前 DAG 内核 A/B**：已完成 1,024 组×2 的初测（§5.3），核心中位 1.20×、墙钟中位 1.07×，但两轮 1.057/1.348、wall 0.919/1.228，**不稳定，暂不采用**；若继续，需 3–5 轮、2,048 组复测；
- 先做阶段剖析（BFS / 反向 / 初始化 / commit；有效边比例），再决定是否值得改内核；
- 这是“在现有精确算法内”最值得先验证、但目前证据不足的方向。

### Tier 3：高成本，2–7×

- **MS-BFS / batched Brandes**：精确算法里最有潜力；需新内核和完整验证；
- **direction-optimizing / pull-based BFS**：备选；
- 预计 2–7× / 1.5–2×，但实现周期以周/月计。

### Tier 4：改变硬件或语义

- GPU 精确 BC：5–38×，需 GPU；
- 近似/采样：10–100×，但不再是精确全源。

## 10. 建议

1. **如果只是把 as-skitter 全源跑完一次**：用 8 线程，关闭大内存应用，按 **24–30 h** 规划；不要做 BCC/degree-2/12 线程。当前 25 h 的中心估计已经是这台机器上“精确 + 现有代码”的合理水平。
2. **如果目标是优化研究/论文**：图重排 A/B 初测不稳定（§5.3），若继续先做 3–5 轮、2,048 组复测；同时评估 **MS-BFS 批量 Brandes**（最有潜力的精确加速）；direction-optimizing 作为第二备选。
3. **如果允许近似**：直接采样源点，时间可降 1–2 个数量级，但必须单独定义误差和结果语义；不要把它混进 `exact_all_sources=true` 的结论。
4. **如果未来有 GPU**：优先做 GPU 精确 BC；这是最大的单点加速，但当前硬件不成立。

## 11. 复现与原始记录

### 11.1 线程实验

| 配置 | 运行目录 |
|---|---|
| 4 线程 | `results/as_skitter_pilot_2048_t4_20261110/` |
| 6 线程 | `results/as_skitter_pilot_2048_t6_seed46_20261110/` |
| 8 线程 | `results/as_skitter_pilot_2048_t8_20261110/` |
| 12 线程 | `results/as_skitter_pilot_2048_t12_seed46_20261110/` |

命令模板：

```powershell
.\.venv\Scripts\python.exe -m parallel_bc quotient-run `
  --dataset as-skitter_csr `
  --run-dir results/as_skitter_pilot_2048_t8_20261110 `
  --threads 8 --batch-size 4 --chunk-size 2048 --seed 46 `
  --max-chunks 1 --timeout 900 --reserve-gib 1 `
  --cpu-affinity 0 2 4 6 8 9 10 11
```

### 11.2 BCC 诊断

```powershell
.\.venv\Scripts\python.exe -m parallel_bc core-diagnose `
  --dataset as-skitter_csr --timeout 300 --reserve-gib 1
```

结果目录：`results/20261110T043108_257913Z_core_diagnostic/`。

### 11.3 关键数值

| 项目 | 值 |
|---|---|
| 核心节点 / 边 | 1,464,757 / 10,863,640 |
| 核心源组 | 1,464,020 |
| 8 线程 2,048 组（seed 46） | 109.132 s → 21.67 h |
| 8 线程 2,048 组（seed 47） | 142.858 s → 28.37 h |
| 4 线程 2,048 组（seed 46） | 134.705 s → 26.75 h |
| 12 线程 2,048 组（seed 46） | 104.045 s → 20.66 h |
| 最大双连通块 | 1,443,769 节点 / 10,830,987 边（98.57% / 99.70%） |
| BCC 工作指标下降 | 1.8% |
| degree-2 节点 | 259,370（17.71%） |

## 12. 参考与检索线索

- Brandes, U. *A Faster Algorithm for Betweenness Centrality*. Journal of Mathematical Sociology, 2001. [DOI](https://doi.org/10.1080/0022250X.2001.9990249)
- Bader, D. A., Madduri, K. *Parallel Algorithms for Evaluating Centrality Indices in Real-world Networks*. ICPP 2006. [ACM](https://dl.acm.org/doi/abs/10.1109/ICPP.2006.57)
- Madduri, K., Ediger, D., Jiang, K., Bader, D. A., Chavarria-Miranda, D. *A Faster Parallel Algorithm and Efficient Multithreaded Implementations for Evaluating Betweenness Centrality on Massive Datasets*. IPDPS 2009.
- Sarıyüce, A. E., Saule, E., Kaya, K., Çatalyürek, Ü. V. *Shattering and Compressing Networks for Betweenness Centrality (BADIOS)*. SDM 2013. [PDF](http://sariyuce.com/papers/sdm13.pdf) / [arXiv:1209.6007](http://export.arxiv.org/pdf/1209.6007v1)
- Then, M., Kaufmann, M., Chirigati, F., et al. *The More the Merrier: Efficient Multi-Source Graph Traversal (MS-BFS)*. VLDB 2014. [PDF](https://db.in.tum.de/~kaufmann/papers/msbfs.pdf)
- Then, M. *Efficient Batched Graph Analytics Through Algorithmic Transformation*. PhD thesis, TU München, 2019. [PDF](https://mediatum.ub.tum.de/doc/1510416/cjdehr8ae5owabcn6l06ay31g.Manuel_Then_Thesis.pdf)
- Beamer, S., Asanović, K., Patterson, D. *Direction-Optimizing Breadth-First Search*. SC 2012.
- FrontOrder: *Frontier-guided Graph Reordering*. PPoPP 2025 poster. [PDF](https://dlnext.acm.org/doi/pdf/10.1145/3710848.3710895)
- HisOrder: *A Historical Frontier-Aware Graph Reordering for Efficient BFS*. 2025. [PDF](https://dlnext.acm.org/doi/pdf/10.1145/3711708.3723445)
- Galliot: *Path Merging Based Betweenness Centrality Algorithm on GPU*. IEEE INFOCOM 2023.
- Bernaschi, M., Carbone, G., Vella, F. *Scalable Betweenness Centrality on Multi-GPU Systems*. Computing Frontiers 2016; [arXiv:1602.00963](https://arxiv.org/abs/1602.00963)
- 项目内部：[as-skitter 导入与全量估算](as-skitter_导入与全量估算.md)、[图结构诊断](RESULTS.md#core-structure)、[BFS 重排实验](RESULTS.md#layout-results)、[树商图](Graph_Optimization.md#tree-quotient)。

> 文献收益均来自不同图、不同实现和不同硬件，不能直接迁移成本项目 as-skitter 的保证值。本文把实测/结构上界/文献值分开标注。
