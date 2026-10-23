"""Chinese run reports whose quantitative statements are backed by saved rows."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

from .plotting import dataset_dirs, read_json, read_rows


def _fmt(value):
    if value is None or value == "":
        return "未记录"
    if isinstance(value, bool):
        return "通过" if value else "失败"
    if isinstance(value, int) or (isinstance(value, str) and value.lstrip("-").isdigit()):
        return str(value)
    try:
        numeric = float(value)
        if not math.isfinite(numeric):
            return "未记录"
        return f"{numeric:.6g}"
    except (TypeError, ValueError):
        return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers, rows):
    rows = list(rows)
    if not rows:
        return ["未执行或没有可用记录。", ""]
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |",
            *["| " + " | ".join(_fmt(v) for v in row) + " |" for row in rows], ""]


def _formal(row):
    return row.get("status", "ok") == "ok" and str(row.get("warmup", False)).lower() not in {"true", "1"}


def generate_report(run_dir):
    """Build report.md from immutable experiment evidence, without new runs."""
    root = Path(run_dir)
    manifest = read_json(root / "manifest.json")
    environment = read_json(root / "environment.json")
    dirs = dataset_dirs(root)
    title = manifest.get("run_id", root.name)
    lines = [f"# LinkRadar 实际实验报告：{title}", "",
             "本报告由本目录中已经保存的实验记录生成。未测项目不填写预测数值；全部结论仅适用于本次图、参数和运行环境。", "",
             "## 问题、输入与输出", "",
             "输入是保留全部连通分量和孤立节点的无权无向简单图；每个起点精确执行 Brandes，输出全部节点 raw 介数及 normalized 介数、原始 ID 映射、运行计时和固定初始排名的节点移除连通性。端点不计入介数，无向路径重复贡献只在最终归并后除以 2；n≤2 时归一化为零。", "",
             "Facebook 数据表示公开采样的自我网络合并图；ca-GrQc 表示作者合作网络。介数反映所选网络中最短路径的中转位置，不代表平台全部关系、因果重要性或作者贡献。节点移除结果是结构模拟，不能直接推出真实人员行为。", "",
             "## 执行证据与复现", "",
             f"- 开始时间（UTC）：{manifest.get('started_utc', '未记录')}",
             f"- 完成时间（UTC）：{manifest.get('completed_utc', '未记录')}",
             f"- 运行状态：{manifest.get('status', '未记录')}",
             f"- 原始命令：`{manifest.get('command') or '未记录'}`",
             f"- 已解析配置：[config_resolved.yaml](config_resolved.yaml)",
             f"- 清单：[manifest.json](manifest.json)；环境：[environment.json](environment.json)",
             f"- 源码 SHA256：`{manifest.get('source_sha256', environment.get('source_sha256', '未记录'))}`",
             f"- 配置 SHA256：`{manifest.get('config_sha256', '未记录')}`", "",
             "中心性 t_total 包含创建执行器、spawn 启动、执行与归并、归一化和清理；t_steady=t_execute_reduce+t_finalize，排除启动和清理。下载、清洗、NetworkX 对照、外层预算监督进程的启动和绘图不计入算法内部计时，但消耗实验墙钟预算。详细任务日志改变运行开销，应与关闭日志的主实验分别解释。", "",
             "## 算法与调度", "",
             "serial、static_batches、dynamic_batches 共享同一单源 BFS 路径计数与依赖回传内核。static_batches 将同一源批次按 batch_id % p_used 预先分配给工作进程；dynamic_batches 让工作进程从共享队列领取同一批次集合。两种并行模式明确使用 spawn，每个工作进程在本地累加长度 n 的贡献向量，完成全部任务后回传一次，再由父进程归并并统一归一化。", "",
             "## 实际环境", ""]
    fields = [("Python", environment.get("python")), ("操作系统", environment.get("os")),
              ("CPU 型号", environment.get("cpu_model")), ("逻辑核心", environment.get("logical_cpus")),
              ("物理核心", environment.get("physical_cpus")), ("CPU 亲和可用数", environment.get("cpu_affinity_count")),
              ("CPU quota", environment.get("cpu_quota")), ("内存字节", environment.get("memory_bytes")),
              ("启动方式", environment.get("start_method")), ("Git commit", environment.get("git_commit")),
              ("Git 工作区有修改", str(environment.get("git_dirty")))]
    lines += _table(["项目", "记录值"], fields)
    lines += ["包版本：`" + json.dumps(environment.get("packages", {}), ensure_ascii=False) + "`。环境可能受限；未获取字段表示未知，不能据此声称资源无限制。", ""]
    all_timings = []
    summaries = []
    graph_sources = set()
    for directory in dirs:
        relative = directory.relative_to(root).as_posix()
        prefix = "" if relative == "." else relative + "/"
        metadata = read_json(directory / "graph_metadata.json")
        dataset = metadata.get("dataset_id", directory.name)
        graph_sources.add(dataset)
        validation = read_json(directory / "validation.json")
        timing_rows = read_rows(directory / "timings_raw.csv")
        rows = read_rows(directory / "timings_summary.csv")
        summaries += rows
        completed = [r for r in timing_rows if _formal(r)]
        all_timings += completed
        robustness = read_rows(directory / "robustness_summary.csv")
        lines += [f"## 数据集：{dataset}", "", f"图与清洗元信息：[{prefix}graph_metadata.json]({prefix}graph_metadata.json)。", ""]
        lines += _table(["字段", "实际记录"], [(name, metadata.get(name)) for name in
            ("n", "m", "source", "source_url", "raw_sha256", "graph_sha256", "raw_data_lines", "self_loop_lines",
             "duplicate_edge_lines", "skipped_invalid_lines", "explicit_nodes", "connected_components",
             "largest_component_nodes", "average_degree", "cleaning_version", "node_mapping_rule", "t_prepare")])
        if metadata.get("source_page"):
            lines += [f"数据来源页面：[{dataset}]({metadata['source_page']})。", ""]
        if metadata.get("generation_parameters"):
            lines += ["生成参数：`" + json.dumps(metadata["generation_parameters"], ensure_ascii=False) + "`。", ""]
        if metadata.get("official_statistics"):
            lines += ["官方统计用于来源核对，实际测量以上表清洗结果为准：`" + json.dumps(metadata["official_statistics"]) + "`。", ""]
        lines += ["### 正确性", "", f"完整对照记录：[{prefix}validation.json]({prefix}validation.json)。", ""]
        if validation.get("allclose") is not None:
            lines += _table(["对照", "全节点 allclose", "有限值", "最大绝对误差", "误差最大原始 ID"],
                            [(name, validation.get(name, {}).get("allclose"), validation.get(name, {}).get("finite"),
                              validation.get(name, {}).get("max_abs_error"), validation.get(name, {}).get("worst_node"))
                             for name in ("raw", "normalized")])
            lines += [f"总检查：allclose={validation.get('allclose')}；比较节点数={validation.get('compared_nodes', '未记录')}；atol={validation.get('atol', '未记录')}；rtol={validation.get('rtol', '未记录')}；NetworkX 版本={validation.get('reference_version', '未记录')}。", ""]
        else:
            lines += [f"NetworkX 全图对照未完成：状态 `{validation.get('status', '未记录')}`，原因 `{validation.get('reason', validation.get('error', '未记录'))}`。", ""]
        error_groups = {}
        for row in completed:
            if row.get("max_abs_error") not in (None, ""):
                error_groups.setdefault(row["mode"], []).append(float(row["max_abs_error"]))
        lines += _table(["实测模式与串行基线全节点比较", "案例数", "最大 raw 误差"],
                        [(mode, len(errors), max(errors)) for mode, errors in sorted(error_groups.items())])
        statuses = dict(Counter(r.get("status", "unknown") for r in timing_rows))
        lines += ["### 主性能结果", "", f"案例状态（含热身）：`{json.dumps(statuses, ensure_ascii=False)}`；正式统计排除热身、失败、超时和未执行项。时间单位为秒，波动为样本 Q1–Q3；单次 smoke 没有重复波动证据。", ""]
        lines += _table(["模式", "p 实际", "B", "次数", "total 中位数", "total Q1–Q3", "steady 中位数", "S_total", "S_steady", "E_total"],
                        [(r.get("mode"), r.get("p_used"), r.get("batch_size"), r.get("count"), r.get("median_total"),
                          f"{_fmt(r.get('q1_total'))} – {_fmt(r.get('q3_total'))}", r.get("median_steady"),
                          r.get("speedup_total"), r.get("speedup_steady"), r.get("efficiency_total")) for r in rows])
        lines += [f"原始计时：[{prefix}timings_raw.csv]({prefix}timings_raw.csv)；汇总：[{prefix}timings_summary.csv]({prefix}timings_summary.csv)。S_total 与 S_steady 均以同图、同源顺序和日志设置的自写 serial 中位数为分子；NetworkX 只作正确性参照。", ""]
        phase_rows = []
        for mode in sorted({r["mode"] for r in completed}):
            group = [r for r in completed if r["mode"] == mode]
            for p in sorted({r["p_used"] for r in group}, key=float):
                for batch in sorted({r["batch_size"] for r in group if r["p_used"] == p}, key=float):
                    local = [r for r in group if r["p_used"] == p and r["batch_size"] == batch]
                    def median(field):
                        import statistics
                        values = [float(r[field]) for r in local if r.get(field) not in (None, "")]
                        return statistics.median(values) if values else None
                    phase_rows.append((mode, p, batch, median("t_startup"), median("t_execute_reduce"), median("t_finalize"), median("t_cleanup"), median("imbalance_ratio"), median("diagnostic_busy_fraction")))
        lines += _table(["模式", "p", "B", "启动中位数", "执行归并中位数", "finalize 中位数", "cleanup 中位数", "忙碌不均衡比", "诊断忙碌比例"], phase_rows)
        lines += ["执行归并阶段含任务排队、IPC 和父进程归并，当前测量不能将三者分别归因。诊断忙碌比例来自工作进程批次墙钟时间，不是操作系统 CPU 利用率。", ""]
        parallel = [r for r in rows if r.get("mode") != "serial" and r.get("speedup_total") not in (None, "")]
        if parallel:
            fastest = max(parallel, key=lambda r: float(r["speedup_total"]))
            lines += [f"本次已测并行配置中最大的 S_total={_fmt(fastest['speedup_total'])}（{fastest['mode']}，实际 p={fastest['p_used']}，B={fastest['batch_size']}）。该值只是当前已测配置的描述，不能推广为稳定优势或最优参数。", ""]
            if all(float(r["speedup_total"]) < 1 for r in parallel):
                lines += ["所有已测并行配置的总耗时均慢于同核心串行。启动阶段耗时有直接记录；图规模偏小、通信或任务粒度也可能影响表现，但这些因素的独立因果贡献仍需更大图和参数对照实验验证。", ""]
        lines += ["### 固定排名节点失效", "",
                  "每个策略从同一原图开始；介数和度数排名固定于原始图，并列按原始 ID 升序。随机重复使用独立且可复现的子种子，每条曲线按同一排列逐步移除。k=floor(q*n_original) 并去重；主指标始终为 LCC 节点数 / 原始节点数，原图不连通时初值允许小于 1。随机带是 10%–90% 经验分位带，不是置信区间。", ""]
        endpoints = []
        for strategy in sorted({r["strategy"] for r in robustness}):
            group = [r for r in robustness if r["strategy"] == strategy]
            last = max(group, key=lambda r: int(r["k"]))
            endpoints.append((strategy, last["k"], last["removed_fraction"], last["count"], last["lcc_original_mean"], last["lcc_original_p10"], last["lcc_original_p90"], last["num_components_mean"]))
        lines += _table(["策略", "末点 k", "实际移除比例", "次数", "LCC/原始 n 均值", "p10", "p90", "分量数均值"], endpoints)
        lines += [f"完整曲线：[{prefix}robustness_raw.csv]({prefix}robustness_raw.csv)；[{prefix}robustness_summary.csv]({prefix}robustness_summary.csv)。失效曲线只比较当前图的结构响应，不预设介数策略必然造成最大破坏。", ""]
        if robustness:
            k = max(int(r["k"]) for r in robustness)
            vals = {r["strategy"]: float(r["lcc_original_mean"]) for r in robustness if int(r["k"]) == k}
            if "bc" in vals and "degree" in vals:
                lines += [f"在本次共同末点 k={k}，L_bc − L_degree={vals['bc'] - vals['degree']:.6g}；负值仅表示此点介数移除后的最大分量占比更小，零值表示此点相同。", ""]
    lines += ["## 图表与未执行项目", ""]
    figure_records = read_json(root / "figures" / "index.json", [])
    chinese_names = {"correctness": "正确性", "total_time": "总耗时", "speedup_total": "总加速比", "speedup_steady": "稳态加速比", "size_scaling": "规模实验", "batch_sizes": "批次扫描", "robustness": "节点失效", "worker_busy": "工作进程忙碌时长", "task_timeline": "真实任务时间线"}
    for record in figure_records:
        name = chinese_names.get(record["name"], record["name"])
        lines += [f"### {name}", "", f"状态：{'已生成实际记录图' if record['measured'] else '未执行相应实验，图中显示明确占位说明'}。{record.get('reason', '')}", "", f"![{name}]({record['png']})", "", f"[PDF]({record['pdf']})", ""]
    limitations = list(manifest.get("limitations", []))
    for record in figure_records:
        if not record["measured"]:
            limitations.append(f"{chinese_names.get(record['name'], record['name'])}：未执行相应实验（{record.get('reason', '缺少记录')}）。")
    sources = [read_json(directory / "graph_metadata.json") for directory in dirs]
    for real_id, keyword in (("Facebook", "facebook"), ("ca-GrQc", "ca-grqc")):
        if not any(keyword in str(meta.get("source_url", "")).lower() for meta in sources):
            limitations.append(f"本次目录没有 {real_id} 真实全图实验记录；下载/导入入口已提供，不能将合成图结果视为真实网络结论。")
    if any(int(r.get("count", 0)) < 3 for r in summaries):
        limitations.append("至少一组正式计时少于 3 次；只能证明流程运行，重复波动和性能结论仍有限。")
    if not figure_records:
        limitations.append("尚未生成图表；运行 report 命令的绘图流程后检查 figures/index.json。")
    lines += ["## 限制与下一步", ""]
    lines += ["- " + str(value) for value in dict.fromkeys(limitations)] or ["当前记录未附加限制；仍仅对所测配置负责。"]
    lines += ["", "下一步优先在预算允许下补足同图重复计时、固定图强扩展、固定图固定进程数的 B 扫描、同图族规模阶梯及真实网络全起点对照。未执行项应保留状态，不将超时记录纳入正常中位数。", "",
              "报告中的时间、误差和失效指标均来自本目录 CSV/JSON；图表使用英文坐标避免中文字体依赖。报告重新生成不会运行算法或更改原始测量。", ""]
    path = root / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
