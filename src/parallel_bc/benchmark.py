"""Bounded, auditable experiment execution; algorithm times exclude the supervisor."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from .brandes import brandes_serial, source_order_digest
from .data import load_cached_graph, save_graph
from .generators import generate_graph
from .parallel import brandes_parallel
from .validation import validate_against_networkx


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        if fields:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    tmp.replace(path)


def source_fingerprint():
    h = hashlib.sha256()
    root = Path(__file__).parent
    for p in sorted(root.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def environment():
    info = dict(python=sys.version, os=platform.platform(), cpu_model=platform.processor() or None,
                logical_cpus=os.cpu_count(), physical_cpus=None, memory_bytes=None,
                cpu_affinity_count=None, cpu_quota=None, start_method="spawn",
                source_sha256=source_fingerprint(), packages={})
    for package in ("numpy", "networkx", "pandas", "matplotlib", "PyYAML", "pytest", "psutil", "linkradar"):
        try:
            info["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            info["packages"][package] = None
    try:
        import psutil
        info.update(physical_cpus=psutil.cpu_count(logical=False), memory_bytes=psutil.virtual_memory().total)
        info["cpu_affinity_count"] = len(psutil.Process().cpu_affinity())
    except (ImportError, AttributeError, OSError):
        pass
    try:
        info["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        info["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        info.update(git_commit=None, git_dirty=None)
    return info


def _case_child(conn, kind, graph, params):
    try:
        if kind == "algorithm":
            options = dict(params)
            mode = options.pop("mode")
            if mode == "serial":
                options.pop("processes", None)
                options.pop("max_processes", None)
                result = brandes_serial(graph.adj, **options)
            else:
                result = brandes_parallel(graph.adj, mode=mode, **options)
        elif kind == "preflight":
            result = brandes_serial(graph.adj)
            if params["enabled"] and graph.n <= params["max_reference_nodes"]:
                validation = validate_against_networkx(graph, result, atol=params["atol"], rtol=params["rtol"])
                if not validation["allclose"]:
                    raise ValueError(f"NetworkX 对照失败: {validation}")
            else:
                validation = {"status": "skipped_reference", "allclose": None,
                              "reason": "reference disabled or graph exceeds max_reference_nodes"}
            result = (result, validation)
        elif kind == "robustness":
            from .robustness import run_robustness
            scores = params.pop("scores")
            result = run_robustness(graph, scores, params)
        else:
            raise ValueError(kind)
        if kind == "algorithm":
            result.graph_sha256 = graph.graph_sha256
            result.exact_all_sources = True
        conn.send(("ok", result))
    except BaseException as exc:
        conn.send(("timeout" if isinstance(exc, TimeoutError) else "failed", traceback.format_exc()))
    finally:
        conn.close()


def _stop_tree(process):
    """Snapshot descendants before stopping the supervisor, then reap all of them."""
    import psutil
    try:
        supervisor = psutil.Process(process.pid)
        supervisor.suspend()  # Prevent a new worker being spawned after the snapshot.
        descendants = supervisor.children(recursive=True)
    except psutil.Error:
        descendants = []
    if process.is_alive():
        process.terminate()
    for child in descendants:
        try:
            child.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(descendants, timeout=2)
    for child in alive:
        try:
            child.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(alive, timeout=2)
    process.join(timeout=3)
    if process.is_alive():
        process.kill()
        process.join()


def bounded_call(kind, graph, params, seconds):
    """Hard wall-clock boundary, including outer spawn. Never leaves child workers running."""
    if seconds <= 0:
        return "budget_not_run", None, "实验总预算耗尽"
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_case_child, args=(child, kind, graph, params))
    begin = time.perf_counter()
    try:
        proc.start()
        child.close()
        while time.perf_counter() - begin < seconds:
            if parent.poll(min(.05, max(.001, seconds - (time.perf_counter() - begin)))):
                try:
                    status, value = parent.recv()
                except EOFError:
                    return "failed", None, f"案例子进程异常退出，exitcode={proc.exitcode}"
                proc.join(timeout=max(.01, seconds - (time.perf_counter() - begin)))
                if proc.is_alive():
                    _stop_tree(proc)
                    return "timeout", None, "案例收尾超过预算"
                return status, value if status == "ok" else None, value if status != "ok" else None
            if not proc.is_alive():
                return "failed", None, f"案例子进程未返回结果，exitcode={proc.exitcode}"
        _stop_tree(proc)
        return "timeout", None, f"超过墙钟预算 {seconds:.3f}s，已终止案例及其工作进程"
    except BaseException:
        if proc.pid:
            _stop_tree(proc)
        raise
    finally:
        parent.close()
        child.close()
        if proc.pid and proc.is_alive():
            _stop_tree(proc)
        proc.close()


def summarize(rows):
    groups = {}
    for row in rows:
        if row["status"] == "ok" and not row["warmup"]:
            key = tuple(row[k] for k in ("dataset_id", "graph_sha256", "mode", "p_used", "batch_size", "source_order_hash", "task_logging"))
            groups.setdefault(key, []).append(row)
    output = []
    for key, items in groups.items():
        row = {k: items[0][k] for k in ("dataset_id", "graph_sha256", "mode", "p_used", "batch_size", "n", "m", "source_order_hash", "task_logging")}
        row["count"] = len(items)
        for short, field in (("total", "t_total"), ("steady", "t_steady")):
            vals = np.array([v[field] for v in items])
            for label, value in (("median", np.median(vals)), ("q1", np.quantile(vals, .25)), ("q3", np.quantile(vals, .75)), ("min", vals.min()), ("max", vals.max())):
                row[f"{label}_{short}"] = float(value)
        output.append(row)
    for row in output:
        baseline = next((r for r in output if r["dataset_id"] == row["dataset_id"] and r["mode"] == "serial"
                         and r["source_order_hash"] == row["source_order_hash"] and r["task_logging"] == row["task_logging"]), None)
        for metric in ("total", "steady"):
            denom = row[f"median_{metric}"]
            row[f"speedup_{metric}"] = baseline[f"median_{metric}"] / denom if baseline and denom else None
        row["efficiency_total"] = row["speedup_total"] / row["p_used"] if row["speedup_total"] and row["p_used"] else None
    return output


def _validate_config(config):
    if config.get("start_method", "spawn") != "spawn":
        raise ValueError("本项目只支持 spawn")
    b = config["benchmark"]
    for k in ("repeats", "max_processes"):
        if int(b.get(k, 1)) < 1:
            raise ValueError(f"{k} 必须为正数")
    if int(b.get("warmups", 0)) < 0 or not config.get("datasets"):
        raise ValueError("warmups 必须非负且 datasets 不可为空")
    if any(m not in {"serial", "static_batches", "dynamic_batches"} for m in b["modes"]):
        raise ValueError("未知调度模式")
    if "serial" not in b["modes"]:
        raise ValueError("性能实验必须包含 serial 基线")
    if not b["batch_sizes"] or any(int(v) < 1 for v in b["batch_sizes"] + b["processes"]):
        raise ValueError("进程和批次必须为正数")
    if b["max_case_seconds"] <= 0 or b["max_total_seconds"] <= 0:
        raise ValueError("预算必须为正数")
    if b.get("source_order", "ascending") not in {"ascending", "shuffled"}:
        raise ValueError("source_order 应为 ascending/shuffled")


def _case_key(dataset_id, graph_hash, code_hash, order_hash, benchmark, case):
    identity = dict(dataset_id=dataset_id, graph_sha256=graph_hash, code_hash=code_hash,
                    source_order_hash=order_hash, start_method="spawn",
                    task_logging=benchmark.get("task_logging", False),
                    max_processes=benchmark.get("max_processes", 4), **case)
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def run_benchmark(config, *, data_dir="data", output_dir=None, resume=None, command=None):
    config = dict(config)
    _validate_config(config)
    b = config["benchmark"]
    started = time.perf_counter()
    total_budget = float(b["max_total_seconds"])
    case_budget = float(b["max_case_seconds"])
    def remaining():
        return min(case_budget, total_budget - (time.perf_counter() - started))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    root = Path(resume) if resume else Path(output_dir or config.get("output", {}).get("root", "results")) / run_id
    root.mkdir(parents=True, exist_ok=bool(resume))
    code_hash = source_fingerprint()
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if resume:
        old = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if old["source_sha256"] != code_hash or old["config_sha256"] != config_hash:
            raise ValueError("源码或配置已改变，拒绝复用旧案例；请创建新实验")
        manifest = old
        manifest["resumed_utc"] = datetime.now(timezone.utc).isoformat()
    else:
        manifest = dict(run_id=run_id, started_utc=datetime.now(timezone.utc).isoformat(), command=command,
                        source_sha256=code_hash, config_sha256=config_hash, datasets=[], cases=[], limitations=[])
    (root / "config_resolved.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    if resume:
        write_json(root / ("environment_resume_" + run_id + ".json"), environment())
    else:
        write_json(root / "environment.json", environment())
    all_rows = []
    for spec in config["datasets"]:
        dataset_id = str(spec["id"])
        if Path(dataset_id).name != dataset_id or dataset_id in {".", ".."} or "\\" in dataset_id:
            raise ValueError("非法 dataset id")
        dest = root / dataset_id
        dest.mkdir(exist_ok=True)
        prep_begin = time.perf_counter()
        try:
            graph = generate_graph(dict(spec["generator"], dataset_id=dataset_id)) if "generator" in spec else load_cached_graph(Path(data_dir) / "processed" / dataset_id)
        except (OSError, ValueError) as exc:
            manifest["limitations"].append(f"{dataset_id} 未准备: {exc}")
            continue
        previous = next((d for d in manifest["datasets"] if d["id"] == dataset_id), None)
        if previous and previous["graph_sha256"] != graph.graph_sha256:
            raise ValueError("图指纹改变，拒绝续跑")
        if not previous:
            save_graph(graph, dest / "graph")
            prep_time = time.perf_counter() - prep_begin
            write_json(dest / "graph_metadata.json", dict(graph.metadata, graph_sha256=graph.graph_sha256, dataset_id=dataset_id,
                                                         n=graph.n, m=graph.m, t_prepare=prep_time))
            manifest["datasets"].append(dict(id=dataset_id, path=dataset_id, graph_sha256=graph.graph_sha256, n=graph.n, m=graph.m))
        # Clear a prior attempt's stage limitations; the new attempt records current failures.
        manifest["limitations"] = [x for x in manifest["limitations"] if not x.startswith(dataset_id + " ")]
        order = list(range(graph.n))
        if b.get("source_order") == "shuffled":
            random.Random(config.get("seed", 42)).shuffle(order)
        order_hash = source_order_digest(order)
        cases = []
        for mode in b["modes"]:
            pairs = [(1, b["batch_sizes"][0])] if mode == "serial" else [(p, size) for p in dict.fromkeys(b["processes"]) for size in dict.fromkeys(b["batch_sizes"])]
            for p, size in pairs:
                for rep in range(-b.get("warmups", 0), b["repeats"]):
                    cases.append(dict(mode=mode, processes=int(p), batch_size=int(size), repeat=rep, warmup=rep < 0))
        random.Random(config.get("seed", 42)).shuffle(cases)
        val = dict(enabled=True, max_reference_nodes=1000, atol=1e-10, rtol=1e-9)
        val.update(config.get("validation", {}))
        status, preflight, error = bounded_call("preflight", graph, val, remaining())
        if status != "ok":
            write_json(dest / ("preflight_status.json" if resume else "validation.json"), {"status": status, "error": error})
            manifest["limitations"].append(f"{dataset_id} 正确性预检未完成: {status}: {error}")
            deferred = []
            for case in cases:
                identity = dict(dataset_id=dataset_id, graph_sha256=graph.graph_sha256, code_hash=code_hash,
                                source_order_hash=order_hash, task_logging=b.get("task_logging", False), **case)
                deferred.append(dict(identity, case_key=_case_key(dataset_id, graph.graph_sha256, code_hash, order_hash, b, case),
                                     run_id=manifest["run_id"], n=graph.n, m=graph.m, p_requested=case["processes"],
                                     p_used=None, active_workers=None, start_method="spawn", status="not_run_preflight",
                                     error=f"{status}: {error}", t_startup=None, t_execute_reduce=None, t_finalize=None,
                                     t_cleanup=None, t_total=None, t_steady=None))
            prior = {r["case_key"]: r for r in manifest["cases"] if r["dataset_id"] == dataset_id}
            for row in deferred:
                if prior.get(row["case_key"], {}).get("status") != "ok":
                    prior[row["case_key"]] = row
            retained = list(prior.values())
            manifest["cases"] = [r for r in manifest["cases"] if r["dataset_id"] != dataset_id] + retained
            write_csv(dest / "timings_raw.csv", retained)
            all_rows.extend(retained)
            write_json(root / "manifest.json", manifest)
            continue
        baseline, validation = preflight
        validation.update(graph_sha256=graph.graph_sha256, exact_all_sources=True)
        write_json(dest / "validation.json", validation)
        from .robustness import ranking_rows
        write_csv(dest / "centrality.csv", ranking_rows(graph, baseline))
        write_json(dest / "centrality_metadata.json", dict(graph_sha256=graph.graph_sha256, exact_all_sources=True,
                                                          source_order_hash=baseline.source_order_hash, t_centrality=baseline.timings))
        write_json(dest / "run_plan.json", dict(planned_timed_cases=len(cases), serial_pilot_seconds=baseline.timings["t_total"],
                   approximate_serial_work_seconds=len(cases) * baseline.timings["t_total"],
                   note="粗估只用本图串行pilot；未预测spawn/通信，不能保证完成时间", remaining_total_seconds=max(0., remaining())))
        rows = []
        worker_file = dest / "worker_summary.csv"
        task_file = dest / "tasks.jsonl"
        if resume and worker_file.exists() and worker_file.stat().st_size:
            with worker_file.open(encoding="utf-8", newline="") as f:
                workers = list(csv.DictReader(f))
        else:
            workers = []
        tasks = [json.loads(line) for line in task_file.read_text(encoding="utf-8").splitlines() if line] if resume and task_file.exists() else []
        old_records = {r["case_key"]: r for r in manifest["cases"] if r["dataset_id"] == dataset_id}
        for case in cases:
            params = dict(mode=case["mode"], processes=case["processes"], batch_size=case["batch_size"], source_order=order,
                          task_logging=b.get("task_logging", False), timeout_seconds=max(.01, remaining() * .9),
                          max_processes=b.get("max_processes", 4))
            key = _case_key(dataset_id, graph.graph_sha256, code_hash, order_hash, b, case)
            row = dict(run_id=manifest["run_id"], case_key=key, dataset_id=dataset_id, graph_sha256=graph.graph_sha256,
                       n=graph.n, m=graph.m, mode=case["mode"], p_requested=case["processes"], p_used=None,
                       active_workers=None, batch_size=case["batch_size"], source_order_hash=order_hash,
                       repeat=case["repeat"], warmup=case["warmup"], task_logging=params["task_logging"], start_method="spawn")
            for field in ("t_startup", "t_execute_reduce", "t_finalize", "t_cleanup", "t_total", "t_steady"):
                row[field] = None
            if key in old_records and old_records[key]["status"] == "ok":
                row = old_records[key]
                rows.append(row)
                continue
            workers = [w for w in workers if w.get("case_key") != key]
            tasks = [t for t in tasks if t.get("case_key") != key]
            outer_begin = time.perf_counter()
            status, result, error = bounded_call("algorithm", graph, params, remaining())
            row.update(status=status, error=error, t_supervised_wall=time.perf_counter() - outer_begin)
            if result is not None:
                if not (np.allclose(result.raw, baseline.raw, atol=val["atol"], rtol=val["rtol"]) and
                        np.allclose(result.normalized, baseline.normalized, atol=val["atol"], rtol=val["rtol"])):
                    row.update(status="failed_validation", error="串并行全节点结果不一致")
                row.update(result.timings, p_used=result.p_used, active_workers=result.active_workers,
                           max_abs_error=float(np.max(np.abs(result.raw - baseline.raw))) if graph.n else 0.)
                busy = [w["busy_seconds"] for w in result.workers]
                mean = float(np.mean(busy)) if busy else 0.
                row["imbalance_ratio"] = max(busy) / mean if mean else None
                row["diagnostic_busy_fraction"] = sum(busy) / (result.p_used * result.timings["t_execute_reduce"]) if result.p_used and result.timings["t_execute_reduce"] else None
                workers.extend(dict(w, case_key=key, mode=case["mode"], repeat=case["repeat"], warmup=case["warmup"], p_used=result.p_used, batch_size=case["batch_size"]) for w in result.workers)
                tasks.extend(dict(t, case_key=key, mode=case["mode"]) for t in result.tasks)
            rows.append(row)
            # Publish supporting evidence before the successful case checkpoint.
            write_csv(worker_file, workers)
            task_tmp = task_file.with_suffix(".jsonl.tmp")
            task_tmp.write_text("".join(json.dumps(t) + "\n" for t in tasks), encoding="utf-8")
            task_tmp.replace(task_file)
            manifest["cases"] = [r for r in manifest["cases"] if r["case_key"] != key] + [row]
            write_csv(dest / "timings_raw.csv", rows)
            write_json(root / "manifest.json", manifest)
        all_rows.extend(rows)
        write_csv(dest / "timings_raw.csv", rows)
        write_csv(dest / "timings_summary.csv", summarize(rows))
        if config.get("robustness", {}).get("enabled", True):
            robustness_config = dict(config.get("robustness", {}), scores=baseline, seed=config.get("seed", 42))
            status, result, error = bounded_call("robustness", graph, robustness_config, remaining())
            if status == "ok":
                write_csv(dest / "robustness_raw.csv", result["raw"])
                write_csv(dest / "robustness_summary.csv", result["summary"])
                write_json(dest / "robustness_metadata.json", result.get("timings", {}))
            else:
                manifest["limitations"].append(f"{dataset_id} 失效实验: {status}: {error}")
    write_json(root / "manifest.json", manifest)
    manifest["elapsed_seconds_before_export"] = time.perf_counter() - started
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["status"] = "complete" if manifest["datasets"] and all(r["status"] == "ok" for r in manifest["cases"]) and not manifest["limitations"] else "partial"
    manifest["limitations"] = list(dict.fromkeys(manifest["limitations"]))
    write_json(root / "manifest.json", manifest)
    write_csv(root / "timings_raw.csv", all_rows)
    write_csv(root / "timings_summary.csv", summarize(all_rows))
    from .plotting import plot_results
    from .reporting import generate_report
    export_begin = time.perf_counter()
    plot_results(root)
    generate_report(root)
    manifest["t_export"] = time.perf_counter() - export_begin
    manifest["elapsed_seconds_including_export"] = time.perf_counter() - started
    write_json(root / "manifest.json", manifest)
    return root
