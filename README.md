# LinkRadar

LinkRadar在无权无向网络中，用自写全起点精确Brandes识别最短路径中转节点，并以同一内核对比串行、静态分批和动态分批的spawn多进程计算。项目还包含数据清洗、全节点正确性验证、预算与日志、固定排名失效实验、真实图表与中文本地Web展示。

## 从哪里开始

- 第一次看项目：[文档入口](docs/README.md) → [整体说明](docs/README.md#project-guide)。
- 想讲清并行：[并行计算原理与真实PID证据](docs/CSR_and_Numba.md#process-parallel)。
- 想看目前做到了什么：[当前结果与Top10报告](docs/RESULTS.md#results)。
- 准备给同学展示：[约8分钟讲稿与操作路线](docs/Run_and_Demo.md#demo)。

Facebook 和 ca-GrQc 已完成全节点计算、验证和结构实验，并分别做过配对复测；ca-GrQc 基线筛查未通过，观察比值按限制解释。YouTube 数据已导入并完成部分源组实验，完整全源计算尚未完成。现有 full 配置不代表完整 full 矩阵已执行，性能结论和限制以当前结果报告为准。

## 安装与短离线演示

在项目根目录执行，Python要求≥3.11，已验收环境为3.12.6。已有.venv时无需重新创建；新环境先运行 `python -m venv .venv`。

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m parallel_bc --help
.\.venv\Scripts\python.exe -m parallel_bc demo --profile smoke
.\.venv\Scripts\python.exe -m pytest -q
```

如本机镜像访问慢，可在pip安装命令追加 `--index-url https://pypi.org/simple`。依赖快照见 [requirements-lock.txt](requirements-lock.txt)。smoke不依赖外网或真实图缓存，输出独立 `results/<RUN_ID>/report.md`。

## 打开现有Web展示

```powershell
.\.venv\Scripts\python.exe -m http.server 8765 --bind 127.0.0.1 --directory web/linkradar/dist
```

访问 http://127.0.0.1:8765/ ，保持终端运行，结束时Ctrl+C。这是本机地址，未部署到外网。网页读取已有CSV/JSON结果，不会在打开时重跑Brandes；支持网络切换、原始ID搜索、介数/度排序、全节点分页、CSV下载、局部节点点选和失效/性能解释。

详细说明与验证限制见 [Web README](web/linkradar/README.md)；全套数据导入、长实验、复测和报告生成命令集中在[展示与运行指南](docs/Run_and_Demo.md#demo)。

## YouTube百万边数据

新增独立CSR缓存与受预算的固定起点短试算入口，支持本地KONECT格式。准备、运行命令及结果边界见[YouTube CSR说明](docs/CSR_and_Numba.md#youtube-csr)。部分起点输出是未缩放贡献，不是全图精确介数。

新增可选Numba单核后端及同源配对比较；安装、数学原理、整数溢出保护和计时边界见[编译版Brandes说明](docs/CSR_and_Numba.md#numba-background)。

编译内核还支持共享只读CSR的源级线程并行，使用`compiled-scale`比较1/2/4线程；调度、归并与指标说明见[编译版多核实现](docs/CSR_and_Numba.md#compiled-parallel)。

新增[带权树商图与原图分数恢复](docs/Graph_Optimization.md#tree-quotient)：保留所有外围树边与节点映射，以端点数量权重计算核心贡献，再恢复全部原始节点；重建图指纹核对保证不是直接丢掉叶节点重算。

全源计划支持[分组检查点与恢复](docs/Pipeline_and_Checkpoints.md#checkpoint)，使用 `quotient-run` 保存已提交的核心源组；未完成时不输出完整原图介数，重启后校验并续算。

## 数据与证据放在哪里

| 目录 | 内容 |
|---|---|
| src/parallel_bc | 自写内核、并行执行、CLI、数据与实验模块 |
| configs | smoke/quick/full、真实图、粒度与诊断配置 |
| tests | 手算图、随机图、清洗、真实spawn、回收、续跑和CLI测试 |
| data/raw、data/processed | 本地原文件与确定性缓存，默认Git忽略 |
| results | 原始运行目录、计时、验证、分数、图表与自动报告，默认Git忽略 |
| docs | 八份主说明与 node_insights 展示附件 |
| web/linkradar | 本地交互报告与使用说明 |

自动生成的阶段性摘要放入results/summaries，docs只维护当前说明与展示附件。原完整实施方案保留，个人协作与未执行范围按事实说明，不将匿名节点排名解释成真实人物重要性。
