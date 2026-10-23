# LinkRadar

LinkRadar在无权无向网络中，用自写全起点精确Brandes识别最短路径中转节点，并以同一内核对比串行、静态分批和动态分批的spawn多进程计算。项目还包含数据清洗、全节点正确性验证、预算与日志、固定排名失效实验、真实图表与中文本地Web展示。

## 从哪里开始

- 第一次看项目：[文档入口](docs/README.md) → [整体说明](docs/PROJECT_GUIDE.md)。
- 想讲清并行：[并行计算原理与真实PID证据](docs/PARALLEL_COMPUTING.md)。
- 想看目前做到了什么：[当前结果与Top10报告](docs/RESULTS.md)。
- 准备给同学展示：[约8分钟讲稿与操作路线](docs/DEMO_GUIDE.md)。

当前两张真实图均已导入并完成全节点计算、验证和结构实验；两张图均已做后续配对复测；ca-GrQc本次基线筛查未通过，观察比值按限制解释。早期合成试跑、冒烟输出与连接探测记录已清理，当前运行产物只保留两张真实图相关内容。现有full配置不代表完整full矩阵已执行，性能结论和限制以当前结果报告为准。

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

详细说明与验证限制见 [Web README](web/linkradar/README.md)；全套数据导入、长实验、复测和报告生成命令集中在[展示与运行指南](docs/DEMO_GUIDE.md)。

## 数据与证据放在哪里

| 目录 | 内容 |
|---|---|
| src/parallel_bc | 自写内核、并行执行、CLI、数据与实验模块 |
| configs | smoke/quick/full、真实图、粒度与诊断配置 |
| tests | 手算图、随机图、清洗、真实spawn、回收、续跑和CLI测试 |
| data/raw、data/processed | 本地原文件与确定性缓存，默认Git忽略 |
| results | 原始运行目录、计时、验证、分数、图表与自动报告，默认Git忽略 |
| docs | 四份主说明、索引与实验展示附件 |
| web/linkradar | 本地交互报告与使用说明 |

自动生成的阶段性摘要放入results/summaries，docs只维护当前说明与展示附件。原完整实施方案保留，个人协作与未执行范围按事实说明，不将匿名节点排名解释成真实人物重要性。
