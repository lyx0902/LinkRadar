# LinkRadar

面向无权无向网络的精确介数中心性计算、并行实验与交互式探索工具。

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Numba optional 0.68.x](https://img.shields.io/badge/Numba-optional%200.68.x-00A3E0)](pyproject.toml)

[项目概览](#项目概览) · [快速开始](#快速开始) · [使用指南](#使用指南) · [项目架构](#项目架构) · [实际性能](#实际性能) · [详细文档](#详细文档)

## 项目概览

LinkRadar 分析**无权无向图**中节点在最短路径上的中转作用。项目实现了精确的 Brandes 介数中心性算法，并提供可复现的串行与并行对比实验。本地静态报告可用于探索已有的 Facebook、ca-GrQc、YouTube 和 as-skitter 计算结果。

项目还提供优化计算链路，包括 CSR 存储、Numba 编译内核、源组级线程并行、无损树商图、边记录和断点续算。四张图均已使用该链路完成全源计算及原图节点结果恢复；两张小图也保留原 Python 多进程实验。

## 功能特性

- **精确中心性计算**：使用全源 Brandes 算法计算原始值和归一化值，并在支持的验证案例中与 NetworkX 对照。
- **并行实验**：比较串行、静态分批和动态分批多进程执行，分别记录执行、启动和归并耗时。
- **大图处理**：构建 CSR 和带权树商图缓存，使用编译版源点并行内核与最短路径边记录优化。
- **断点续算**：交替保存累计快照，中断后可恢复已完成的工作；全部源组处理完成后才发布原图节点的最终分数。
- **网络探索**：在静态报告中查看节点排名、搜索 ID、下载 CSV，并探索局部邻域与固定排名节点移除结果。
- **实验记录**：在运行目录中保存配置、环境、图指纹、耗时、验证结果和报告。

## 项目架构

```mermaid
flowchart LR
    G["无权无向图"] --> P["Python Brandes<br/>串行 / 多进程"]
    P --> V["验证与实验报告<br/>Facebook / ca-GrQc"]
    G --> C["CSR 缓存"]
    C --> Q["带权树商图<br/>与原图节点映射"]
    Q --> N["Numba 加权 Brandes<br/>源组级线程 / 最短路径边记录"]
    N --> S["交替累计快照"]
    S --> R["全量完成后恢复原图贡献<br/>无向修正与归一化"]
    R -.->W["完成全源处理:<br/>四张真实图"]
```

两条计算分支都基于 Brandes 算法：一条直接在原图上使用 Python 串行或多进程计算，另一条在商图上使用加权 Brandes，并以 Numba 编译、线程并行和边记录加速。商图保留外围树及其映射，通过源点权重和结果恢复计入原图节点。未完成的检查点保存的是核心累计贡献，并非最终中心性结果。静态报告独立展示已有结果，不会由商图任务自动生成。

## 实际性能

以下四张无权无向图均已完成全源介数计算，节点数与边数按项目实际处理后的图统计。

| 图名 | 节点数 / 边数 | 采用链路方式 | 实际处理时间 | 原链路处理时间 | 加速比 |
| --- | ---: | --- | ---: | ---: | ---: |
| Facebook | 4,039 / 88,234 | CSR + 树商图 + Numba，4 线程、边记录和双快照 | 2.172 秒 | 77.31 秒 | 约 35.59×（历史比值） |
| ca-GrQc | 5,242 / 14,484 | CSR + 树商图 + Numba，4 线程、边记录和双快照 | 1.528 秒 | 20.95 秒 | 约 13.71×（历史比值） |
| YouTube | 1,134,890 / 2,987,624 | CSR + 树商图 + Numba 编译加权 Brandes，8 线程、边记录和双快照 | 2 小时 30 分 37 秒 | 约 165 天（外推） | 约 1,577×（估算比） |
| as-skitter | 1,696,415 / 11,095,298 | CSR + 树商图 + Numba 编译加权 Brandes，12 线程、边记录和双快照 | 约 13 小时 36 分 31 秒| — | — |

## 快速开始

需要 **Python 3.11 或更高版本**。在本地项目根目录的 Windows PowerShell 中执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,compiled]"
python -m parallel_bc demo --profile smoke
```

离线 smoke 配置会使用种子 42 生成一张 80 节点的图，运行串行和并行案例，并与 NetworkX 比较。完成后终端会输出 `实验状态：complete；报告：results/.../report.md`。运行目录还包含解析后的配置、环境信息和验证记录。`compiled` 可选依赖用于安装大图处理所需的 Numba；smoke 示例本身不依赖 Numba。已记录的依赖快照见 [requirements-lock.txt](requirements-lock.txt)。

其他 Shell 请使用对应平台的方式激活虚拟环境。本地示例不需要 API 密钥或用户自行配置的环境变量。

## 使用指南

### 查看现有报告

访问 [LinkRadar 在线报告](https://lyx0902.github.io/LinkRadar/)可探索已部署的结果；本地页面现已接入 as-skitter，线上需提交并部署更新。该静态报告不会在打开时重新运行算法，也不会自动导入新的计算结果；YouTube 与 as-skitter 的资源分别由导出脚本生成。详见 [Web 使用说明](web/linkradar/README.md)。

### 数据处理命令

以下 PowerShell 命令使用两次全量任务记录的内核、线程数、批次、分块、种子和 CPU 亲和性。运行前需要安装 `compiled` 可选依赖，并分别准备好 CSR 与树商图缓存：`data/processed/youtube_csr_treeq/` 和 `data/processed/as-skitter_csr_treeq/`。YouTube 缓存准备方法见下方的有限规模示例，as-skitter 数据说明见[全量处理报告](docs/as-skitter全量处理报告.md)。

#### YouTube 全量计算

```powershell
python -m parallel_bc quotient-run `
  --dataset youtube_csr `
  --run-dir results/youtube_full_8t_2048_new `
  --kernel dag --checkpoint-mode snapshot `
  --threads 8 --batch-size 4 --chunk-size 2048 --seed 45 `
  --timeout 28800 --reserve-gib 1 `
  --cpu-affinity 0 2 4 6 8 9 10 11
```

#### as-skitter 全量计算

```powershell
python -m parallel_bc quotient-run `
  --dataset as-skitter_csr `
  --run-dir results/as_skitter_full_12t_2048_new `
  --kernel dag --checkpoint-mode snapshot `
  --threads 12 --batch-size 4 --chunk-size 2048 --seed 45 `
  --timeout 28800 --reserve-gib 1 `
  --cpu-affinity 0 2 4 6 8 9 10 11 12 13 14 15
```

示例结果目录须不存在；需要重新运行时可改用其他新目录。CPU 编号是当时机器的配置，其他机器应按实际可用逻辑 CPU 调整。`--timeout 28800` 为示例设置的单次会话 8 小时上限，并非全量耗时承诺；任务可能需要多次会话。

中断或会话超时后，将原命令中的同一运行目录保留，并添加 `--resume` 继续；数据集、内核、线程数、批次、分块、种子及 CPU 亲和性须与原任务一致。终端每提交一块会显示已完成源组数，完整结果在全部源组及原图恢复完成后生成。续算和结果文件说明见[检查点与恢复说明](docs/Pipeline_and_Checkpoints.md)。

### 运行有限规模的 YouTube 实验

单独下载并解压 [YouTube 数据集压缩包](https://github.com/dange-academic/real_network_datasets/blob/main/PRE2020/youtube.zip)，将其中 KONECT 格式的 `youtube.txt` 放在 `data/raw/youtube/youtube.txt`，并查看上游来源的数据使用条款。数据集和生成的缓存不包含在 Git 仓库中。安装 `compiled` 可选依赖后，可按以下步骤准备缓存，并运行一个包含 64 个核心源组的分块：

```powershell
python -m parallel_bc data prepare-csr --dataset youtube --path data/raw/youtube/youtube.txt --format konect
python -m parallel_bc data prepare-quotient --dataset youtube_csr
python -m parallel_bc quotient-run --dataset youtube_csr --run-dir results/youtube_example --threads 4 --batch-size 4 --chunk-size 64 --seed 45 --max-chunks 1 --timeout 1800 --reserve-gib 1
```

请使用新的运行目录，并为预处理和线程工作区预留足够内存。每提交一个分块，终端会输出 `committed <completed>/<total> core groups`；分块计算期间不会逐源输出进度。这个示例只计算**部分源组**，不是全源结果。

如需继续同一任务，重复执行 `quotient-run` 命令并添加 `--resume`，同时保持数据集、内核、线程数、批次大小、分块大小、种子和 CPU 亲和性设置一致。去掉 `--max-chunks 1` 后，任务会继续运行至完成或本次会话超时；超时限制针对单次会话。可通过 `manifest.json` 查看进度和 `exact_all_sources` 状态。只有全部源组计算及最终恢复完成后，才会发布 `bc_raw.npy` 和 `bc_normalized.npy`；节点顺序与商图缓存中的 `original_ids.npy` 一致。启动长任务前，特别是在恢复旧版内核检查点时，请先阅读[检查点与恢复说明](docs/Pipeline_and_Checkpoints.md)。

```powershell
python -m parallel_bc --help
python -m parallel_bc quotient-run --help
python -m pytest -q
```

## 详细文档

| 文档 | 内容 |
| --- | --- |
| [文档入口](docs/README.md) | 项目介绍与阅读路径 |
| [算法](docs/Algorithm.md) | Brandes 推导与中心性定义 |
| [CSR 与 Numba](docs/CSR_and_Numba.md) | 数据表示、编译与并行处理 |
| [图结构优化](docs/Graph_Optimization.md) | 树商图、结果恢复与边记录 |
| [处理链路与检查点](docs/Pipeline_and_Checkpoints.md) | 运行、输出、恢复与兼容性 |
| [实验结果](docs/RESULTS.md) | 实验、验证与限制 |
| [YouTube 全量汇总报告](docs/Youtube全量处理报告.md) | 全源结果、两次会话、正确性核验与限制 |
| [as-skitter 全量处理报告](docs/as-skitter全量处理报告.md) | 全源结果、会话记录、正确性核验与优化实验 |
| [全量耗时估算](docs/Youtube提速历程.md) | 提速历程与有限条件下的外推 |
