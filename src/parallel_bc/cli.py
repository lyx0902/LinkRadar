"""Chinese CLI for graph preparation and reproducible experiments."""
from __future__ import annotations
import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml


def _common(parser):
    parser.add_argument("--data-dir", default=argparse.SUPPRESS, help="数据根目录（默认 data）")
    parser.add_argument("--output-dir", default=argparse.SUPPRESS, help="结果根目录（默认 results）")


def parser():
    p = argparse.ArgumentParser(prog="linkradar", description="LinkRadar：精确介数中心性、spawn 并行与失效实验")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output-dir", default="results")
    subs = p.add_subparsers(dest="command", required=True)
    data = subs.add_parser("data", help="下载、导入、清洗或生成图")
    _common(data)
    ds = data.add_subparsers(dest="data_command", required=True)
    for name in ("download", "prepare", "import", "generate"):
        d = ds.add_parser(name)
        _common(d)
        if name != "generate":
            d.add_argument("--dataset", required=True)
        if name == "download":
            d.add_argument("--timeout", type=float, default=30)
            d.add_argument("--retries", type=int, default=2)
        if name == "import":
            d.add_argument("--path", required=True)
            d.add_argument("--nodes-path")
        if name == "generate":
            d.add_argument("--family", choices=["er", "ba", "sbm"], required=True)
            d.add_argument("--nodes", type=int, required=True)
            d.add_argument("--seed", type=int, default=42)
            d.add_argument("--attach", type=int, default=3)
            d.add_argument("--avg-degree", type=float, default=6)
            d.add_argument("--name", required=True)
    d = ds.add_parser("prepare-csr", help="严格导入文本并生成独立CSR缓存")
    _common(d)
    d.add_argument("--dataset", required=True)
    d.add_argument("--path", required=True)
    d.add_argument("--format", dest="file_format", choices=["snap", "konect"], default="snap")
    d.add_argument("--provenance", help="原文件审计JSON，SHA必须匹配")
    d.add_argument("--timeout", type=float, default=300)
    d.add_argument("--reserve-gib", type=float, default=3)
    d = subs.add_parser("large-pilot", help="完整CSR图上的受预算固定起点试算")
    _common(d)
    d.add_argument("--dataset", required=True, help="CSR目录名，如youtube_csr")
    d.add_argument("--counts", type=int, nargs="+", default=[1, 4, 8])
    d.add_argument("--backend", choices=["python_tuple", "python_csr", "numba_csr"], default="python_tuple")
    d.add_argument("--seed", type=int, default=42)
    d.add_argument("--timeout", type=float, default=120, help="每案例墙钟上限")
    d.add_argument("--total-timeout", type=float, default=1800)
    d.add_argument("--reserve-gib", type=float, default=3)
    d = subs.add_parser("compiled-compare", help="同源配对比较Python与Numba单核CSR")
    _common(d)
    d.add_argument("--dataset", required=True)
    d.add_argument("--sources", type=int, default=8)
    d.add_argument("--repeats", type=int, default=3)
    d.add_argument("--seed", type=int, default=42)
    d.add_argument("--timeout", type=float, default=240)
    d.add_argument("--total-timeout", type=float, default=1200)
    d.add_argument("--reserve-gib", type=float, default=2)
    d = subs.add_parser("compiled-scale", help="同一Numba内核的1/2/4线程扩展实验")
    _common(d)
    d.add_argument("--dataset", required=True)
    d.add_argument("--sources", type=int, default=64)
    d.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    d.add_argument("--batch-size", type=int, default=4)
    d.add_argument("--repeats", type=int, default=3)
    d.add_argument("--seed", type=int, default=42)
    d.add_argument("--timeout", type=float, default=120)
    d.add_argument("--total-timeout", type=float, default=900)
    d.add_argument("--reserve-gib", type=float, default=2)
    d.add_argument("--cpu-affinity", type=int, nargs="+", help="限定计算子进程的逻辑CPU集合；所有线程配置用同一集合")
    d = ds.add_parser("prepare-quotient", help="生成可恢复原图的带权外围树商图缓存")
    _common(d)
    d.add_argument("--dataset", required=True, help="已有CSR缓存ID")
    d.add_argument("--timeout", type=float, default=300)
    d.add_argument("--reserve-gib", type=float, default=2)
    d = subs.add_parser("quotient-compare", help="同原图同源对照原生CSR与无损树商图")
    _common(d)
    d.add_argument("--dataset", required=True)
    d.add_argument("--quotient", help="默认dataset后加_treeq")
    d.add_argument("--sources", type=int, default=256)
    d.add_argument("--threads", type=int, default=4)
    d.add_argument("--batch-size", type=int, default=4)
    d.add_argument("--repeats", type=int, default=3)
    d.add_argument("--seed", type=int, default=42)
    d.add_argument("--timeout", type=float, default=120)
    d.add_argument("--total-timeout", type=float, default=900)
    d.add_argument("--reserve-gib", type=float, default=2)
    d.add_argument("--cpu-affinity", type=int, nargs="+")
    d = subs.add_parser("quotient-run", help="分组检查点运行全源计划，可中断续算")
    _common(d)
    d.add_argument("--dataset", required=True, help="CSR目录名，使用其_treeq缓存")
    d.add_argument("--quotient", help="自定义商图缓存目录名")
    d.add_argument("--run-dir", required=True)
    d.add_argument("--resume", action="store_true")
    d.add_argument("--kernel", choices=("dag", "legacy"), default="dag",
                   help="dag为默认边记录内核；旧检查点续算须指定legacy")
    d.add_argument("--threads", type=int, default=4)
    d.add_argument("--batch-size", type=int, default=4)
    d.add_argument("--chunk-size", type=int, default=1024, help="每次持久化的核心源组数量")
    d.add_argument("--seed", type=int, default=45)
    d.add_argument("--max-chunks", type=int, help="本次最多新增组数；省略则运行至完成或预算耗尽")
    d.add_argument("--timeout", type=float, default=1800, help="本次会话硬墙钟上限，已提交组可续算")
    d.add_argument("--reserve-gib", type=float, default=1)
    d.add_argument("--cpu-affinity", type=int, nargs="+")
    d = subs.add_parser("core-diagnose", help="只读诊断商图核心的桥、割点与双连通分量")
    _common(d)
    d.add_argument("--dataset", required=True, help="CSR目录名，默认使用其_treeq商图")
    d.add_argument("--quotient", help="自定义商图缓存目录名")
    d.add_argument("--protect-run", help="另行哈希保护已有检查点目录；诊断不会修改它")
    d.add_argument("--timeout", type=float, default=180)
    d.add_argument("--reserve-gib", type=float, default=1)
    d = subs.add_parser("layout-tune", help="独立比较BFS重排、线程配置与独立源集合")
    _common(d)
    d.add_argument("--dataset", required=True)
    d.add_argument("--protect-run")
    d.add_argument("--sources", type=int, default=512, help="主实验核心源组数")
    d.add_argument("--confirm-sources", type=int, default=2048)
    d.add_argument("--threads", type=int, nargs="+", default=[4, 6, 8])
    d.add_argument("--repeats", type=int, default=3)
    d.add_argument("--batch-size", type=int, default=4)
    d.add_argument("--seed", type=int, default=46)
    d.add_argument("--confirm-seed", type=int, default=47)
    d.add_argument("--layout-seed", type=int, default=1234)
    d.add_argument("--timeout", type=float, default=180)
    d.add_argument("--total-timeout", type=float, default=1800)
    d.add_argument("--reserve-gib", type=float, default=1)
    for name in ("centrality", "validate", "robustness"):
        d = subs.add_parser(name)
        _common(d)
        d.add_argument("--dataset", required=True)
        if name == "centrality":
            d.add_argument("--mode", choices=["serial", "static_batches", "dynamic_batches"], default="serial")
            d.add_argument("--processes", type=int, default=2)
            d.add_argument("--max-processes", type=int, default=4)
            d.add_argument("--batch-size", type=int, default=32)
            d.add_argument("--timeout", type=float, default=120)
            d.add_argument("--task-logging", action="store_true")
        elif name == "validate":
            d.add_argument("--against", choices=["networkx"], default="networkx")
            d.add_argument("--timeout", type=float, default=120)
        else:
            d.add_argument("--random-repeats", type=int, default=10)
            d.add_argument("--seed", type=int, default=42)
            d.add_argument("--scores", help="centrality 输出目录；不提供则显式重算串行")
            d.add_argument("--timeout", type=float, default=120)
    d = subs.add_parser("benchmark", help="有预算的性能实验")
    _common(d)
    d.add_argument("--config", required=True)
    d.add_argument("--resume", help="仅在源码、图、配置均匹配时续跑")
    d = subs.add_parser("demo", help="不依赖网络的完整演示")
    _common(d)
    d.add_argument("--profile", choices=["smoke", "quick"], default="smoke")
    d = subs.add_parser("report", help="由已有原始结果重新生成图表和中文报告")
    d.add_argument("--run-dir", required=True)
    return p


def _dataset_path(data_dir, dataset):
    if Path(dataset).name != dataset or dataset in {".", ".."} or "\\" in dataset:
        raise ValueError("非法 dataset id")
    return Path(data_dir) / "processed" / dataset


def _new_output(root, label):
    p = Path(root) / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ_") + label)
    p.mkdir(parents=True, exist_ok=False)
    return p


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = parser().parse_args(argv)
    try:
        return dispatch(args)
    except (ValueError, OSError, RuntimeError, TimeoutError, KeyError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("已中断，工作进程已回收；已有实验记录保留。", file=sys.stderr)
        return 130


def dispatch(args):
    from .benchmark import bounded_call, run_benchmark, write_csv, write_json
    from .data import download_dataset, import_dataset, load_cached_graph, prepare_dataset, save_graph
    from .generators import generate_graph
    from .robustness import ranking_rows
    if args.command == "layout-tune":
        from .layout_benchmark import run_layout_tuning
        if not np.isfinite(args.reserve_gib) or args.reserve_gib < 0:
            raise ValueError("reserve-gib必须非负且有限")
        dest = run_layout_tuning(_dataset_path(args.data_dir, args.dataset + "_treeq"), args.output_dir,
            source_count=args.sources, confirm_count=args.confirm_sources, thread_counts=args.threads,
            repeats=args.repeats, seed=args.seed, confirm_seed=args.confirm_seed, layout_seed=args.layout_seed,
            batch_size=args.batch_size, max_case_seconds=args.timeout, max_total_seconds=args.total_timeout,
            min_available_bytes=int(args.reserve_gib * 2**30), protect_run=args.protect_run)
        status = json.loads((dest / "manifest.json").read_text())["status"]
        print(f"重排与线程实验：{status}；报告：{dest / 'report.md'}")
        return 0 if status == "complete" else 2
    if args.command == "core-diagnose":
        from .core_diagnostic import run_core_diagnostic
        if not np.isfinite(args.reserve_gib) or args.reserve_gib < 0:
            raise ValueError("reserve-gib必须非负且有限")
        dest = run_core_diagnostic(_dataset_path(args.data_dir, args.quotient or args.dataset + "_treeq"),
            args.output_dir, seconds=args.timeout, min_available_bytes=int(args.reserve_gib * 2**30),
            protect_run=args.protect_run)
        status = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))["status"]
        print(f"核心结构诊断：{status}；报告：{dest / 'report.md'}")
        return 0 if status == "complete" else 2
    if args.command == "quotient-run":
        if args.kernel == "dag":
            from .quotient_checkpoint_dag import run_checkpoint
        else:
            from .quotient_checkpoint import run_checkpoint
        if not np.isfinite(args.reserve_gib) or args.reserve_gib < 0:
            raise ValueError("reserve-gib必须非负且有限")
        row = run_checkpoint(_dataset_path(args.data_dir, args.quotient or args.dataset + "_treeq"),
            args.run_dir, seconds=args.timeout, min_available_bytes=int(args.reserve_gib * 2**30),
            cpu_affinity=args.cpu_affinity, resume=args.resume, threads=args.threads,
            batch_size=args.batch_size, chunk_size=args.chunk_size, seed=args.seed,
            max_chunks=args.max_chunks)
        print(json.dumps(row, ensure_ascii=False, indent=2))
        return 0 if row["status"] == "ok" else 2
    if args.command == "data" and args.data_command == "prepare-quotient":
        from .quotient_benchmark import run_quotient_preparation
        if not np.isfinite(args.reserve_gib) or args.reserve_gib < 0:
            raise ValueError("reserve-gib必须非负且有限")
        row = run_quotient_preparation(_dataset_path(args.data_dir, args.dataset),
            _dataset_path(args.data_dir, args.dataset + "_treeq"), seconds=args.timeout,
            min_available_bytes=int(args.reserve_gib * 2**30))
        print(json.dumps(row, ensure_ascii=False, indent=2))
        return 0 if row["status"] == "ok" else 2
    if args.command == "quotient-compare":
        from .quotient_benchmark import run_quotient_comparison
        if not np.isfinite(args.reserve_gib) or args.reserve_gib < 0:
            raise ValueError("reserve-gib必须非负且有限")
        dest = run_quotient_comparison(_dataset_path(args.data_dir, args.dataset),
            _dataset_path(args.data_dir, args.quotient or args.dataset + "_treeq"), args.output_dir,
            source_count=args.sources, threads=args.threads, batch_size=args.batch_size,
            repeats=args.repeats, seed=args.seed, max_case_seconds=args.timeout,
            max_total_seconds=args.total_timeout, min_available_bytes=int(args.reserve_gib * 2**30),
            cpu_affinity=args.cpu_affinity)
        status = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))["status"]
        print(f"无损树商图比较：{status}；报告：{dest / 'report.md'}")
        return 0 if status == "complete" else 2
    if args.command == "compiled-scale":
        from .compiled_scaling import run_scaling
        if not np.isfinite(args.reserve_gib) or args.reserve_gib < 0:
            raise ValueError("reserve-gib必须非负且有限")
        dest = run_scaling(_dataset_path(args.data_dir, args.dataset), args.output_dir,
            source_count=args.sources, threads=args.threads, batch_size=args.batch_size,
            repeats=args.repeats, seed=args.seed, max_case_seconds=args.timeout,
            max_total_seconds=args.total_timeout, min_available_bytes=int(args.reserve_gib * 2**30),
            cpu_affinity=args.cpu_affinity)
        status = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))["status"]
        print(f"编译版线程扩展：{status}；报告：{dest / 'report.md'}")
        return 0 if status == "complete" else 2
    if args.command == "compiled-compare":
        from .compiled_benchmark import run_comparison
        if not np.isfinite(args.reserve_gib) or args.reserve_gib < 0:
            raise ValueError("reserve-gib必须非负且有限")
        dest = run_comparison(_dataset_path(args.data_dir, args.dataset), args.output_dir,
            source_count=args.sources, repeats=args.repeats, seed=args.seed,
            max_case_seconds=args.timeout, max_total_seconds=args.total_timeout,
            min_available_bytes=int(args.reserve_gib * 2**30))
        status = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))["status"]
        print(f"编译版配对比较：{status}；报告：{dest / 'report.md'}")
        return 0 if status == "complete" else 2
    if args.command == "large-pilot":
        from .large_benchmark import run_pilot
        if not np.isfinite(args.reserve_gib) or args.reserve_gib < 0:
            raise ValueError("reserve-gib必须非负且有限")
        dest = run_pilot(_dataset_path(args.data_dir, args.dataset), args.output_dir,
            counts=args.counts, backend=args.backend, seed=args.seed,
            max_case_seconds=args.timeout, max_total_seconds=args.total_timeout,
            min_available_bytes=int(args.reserve_gib * 2**30))
        status = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))["status"]
        print(f"固定起点试算状态：{status}；报告：{dest / 'report.md'}")
        return 0 if status == "complete" else 2
    if args.command == "data" and args.data_command == "prepare-csr":
        from .large_benchmark import run_preparation
        if not np.isfinite(args.reserve_gib) or args.reserve_gib < 0:
            raise ValueError("reserve-gib必须非负且有限")
        dest = _dataset_path(args.data_dir, args.dataset + "_csr")
        if dest.exists():
            raise ValueError(f"CSR目标已存在，不覆盖：{dest}")
        row = run_preparation(args.path, dest, dataset_id=args.dataset,
            file_format=args.file_format, provenance_path=args.provenance,
            timeout_seconds=args.timeout, min_available_bytes=int(args.reserve_gib * 2**30))
        print(json.dumps(row, ensure_ascii=False, indent=2))
        return 0 if row["status"] == "ok" else 2
    if args.command == "data":
        if args.data_command == "download":
            print(download_dataset(args.dataset, args.data_dir, timeout=args.timeout, retries=args.retries))
            return 0
        if args.data_command == "prepare":
            graph = prepare_dataset(args.dataset, args.data_dir)
        elif args.data_command == "import":
            graph = import_dataset(args.dataset, args.path, args.data_dir, nodes_path=args.nodes_path)
        else:
            graph = generate_graph(dict(family=args.family, nodes=args.nodes, seed=args.seed, attach=args.attach, avg_degree=args.avg_degree))
            save_graph(graph, _dataset_path(args.data_dir, args.name))
        print(json.dumps(dict(n=graph.n, m=graph.m, graph_sha256=graph.graph_sha256), ensure_ascii=False))
        return 0
    if args.command in {"demo", "benchmark"}:
        path = Path(args.config) if args.command == "benchmark" else Path(__file__).parent / "profiles" / f"{args.profile}.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
        dest = run_benchmark(config, data_dir=args.data_dir, output_dir=args.output_dir, resume=getattr(args, "resume", None), command="python -m parallel_bc " + " ".join(sys.argv[1:]))
        manifest = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
        print(f"实验状态：{manifest['status']}；报告：{dest / 'report.md'}")
        return 0 if manifest["status"] == "complete" else 2
    if args.command == "report":
        from .plotting import plot_results
        from .reporting import generate_report
        plot_results(args.run_dir)
        print(generate_report(args.run_dir))
        return 0
    graph = load_cached_graph(_dataset_path(args.data_dir, args.dataset))
    if args.command == "validate":
        val = dict(enabled=True, max_reference_nodes=graph.n, atol=1e-10, rtol=1e-9)
        status, result, error = bounded_call("preflight", graph, val, args.timeout)
        if status != "ok":
            raise RuntimeError(f"{status}: {error}")
        print(json.dumps(result[1], ensure_ascii=False, indent=2))
        return 0 if result[1]["allclose"] else 1
    dest = _new_output(args.output_dir, f"{args.command}_{args.dataset}")
    write_json(dest / "graph_metadata.json", dict(graph.metadata, graph_sha256=graph.graph_sha256, n=graph.n, m=graph.m, dataset_id=args.dataset))
    if args.command == "centrality":
        options = dict(mode=args.mode, processes=args.processes, max_processes=args.max_processes,
                       batch_size=args.batch_size, timeout_seconds=args.timeout * .9, task_logging=args.task_logging)
        status, scores, error = bounded_call("algorithm", graph, options, args.timeout)
        if status != "ok":
            write_json(dest / "status.json", dict(status=status, error=error))
            raise RuntimeError(f"{status}: {error}")
        write_csv(dest / "centrality.csv", ranking_rows(graph, scores))
        write_json(dest / "centrality_metadata.json", dict(graph_sha256=graph.graph_sha256, exact_all_sources=True,
                   source_order_hash=scores.source_order_hash, timings=scores.timings, mode=scores.mode, p_used=scores.p_used))
        write_csv(dest / "worker_summary.csv", scores.workers)
        if scores.tasks:
            (dest / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in scores.tasks), encoding="utf-8")
        print(f"全起点精确中心性已保存：{dest}")
        return 0
    if args.scores:
        folder = Path(args.scores)
        meta = json.loads((folder / "centrality_metadata.json").read_text(encoding="utf-8"))
        if meta.get("graph_sha256") != graph.graph_sha256 or meta.get("exact_all_sources") is not True:
            raise ValueError("中心性缓存图指纹或全起点标记不匹配")
        with (folder / "centrality.csv").open(encoding="utf-8", newline="") as f:
            rows = sorted(csv.DictReader(f), key=lambda row: int(row["index"]))
        if len(rows) != graph.n or any(int(r["index"]) != i or int(r["original_id"]) != graph.ids[i] for i, r in enumerate(rows)):
            raise ValueError("中心性缓存节点覆盖不匹配")
        scores = SimpleNamespace(raw=np.array([float(r["bc_raw"]) for r in rows]), normalized=np.array([float(r["bc_normalized"]) for r in rows]), graph_sha256=graph.graph_sha256, exact_all_sources=True)
        t_centrality = None
    else:
        print("未提供分数缓存，先执行串行全起点计算（单独计时）。")
        status, scores, error = bounded_call("algorithm", graph, dict(mode="serial", timeout_seconds=args.timeout * .9), args.timeout)
        if status != "ok":
            raise RuntimeError(f"{status}: {error}")
        t_centrality = scores.timings
    options = dict(scores=scores, random_repeats=args.random_repeats, seed=args.seed,
                   fractions=[0, .01, .02, .05, .1, .15, .2], strategies=["bc", "degree", "random"])
    status, result, error = bounded_call("robustness", graph, options, args.timeout)
    if status != "ok":
        raise RuntimeError(f"{status}: {error}")
    write_csv(dest / "robustness_raw.csv", result["raw"])
    write_csv(dest / "robustness_summary.csv", result["summary"])
    write_json(dest / "robustness_metadata.json", dict(result.get("timings", {}), t_centrality=t_centrality))
    print(f"固定排名失效实验已保存：{dest}")
    return 0
