"""Matched layout, physical-core count and independent-source experiments.

No full-source run or checkpoint migration. Global endpoint masses stay fixed;
each sampled core group represents ALL its original vertices as sources.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import operator
import os
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import psutil

from .brandes import source_order_digest
from .compiled_benchmark import compare_vectors
from .core_diagnostic import _snapshot
from .data import _atomic_json
from .large_benchmark import bounded_job, _save_array


def cpu_configurations(thread_counts):
    """Read current topology; prefer one logical CPU per physical P/E core."""
    allowed = psutil.Process().cpu_affinity()
    rows = []
    if os.name == "nt":
        import ctypes
        import struct
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        get = kernel.GetSystemCpuSetInformation
        get.argtypes = (ctypes.c_void_p, wintypes.ULONG, ctypes.POINTER(wintypes.ULONG),
                        wintypes.HANDLE, wintypes.ULONG)
        get.restype = wintypes.BOOL
        length = wintypes.ULONG()
        get(None, 0, ctypes.byref(length), None, 0)
        if not length.value:
            raise RuntimeError("Cannot query CPU Set topology")
        buffer = ctypes.create_string_buffer(length.value)
        if not get(buffer, length.value, ctypes.byref(length), None, 0):
            raise ctypes.WinError(ctypes.get_last_error())
        offset = 0
        while offset < length.value:
            size, kind = struct.unpack_from("<II", buffer.raw, offset)
            if size < 8 or offset + size > length.value:
                raise RuntimeError("Invalid CPU Set information size")
            if kind == 0 and size >= 20:
                ident, group, logical, core, cache, numa, efficiency, flags = struct.unpack_from("<IH6B", buffer.raw, offset + 8)
                rows.append(dict(cpu_set_id=ident, group=group, logical_cpu=logical,
                    core_index=core, efficiency_class=efficiency, flags=flags))
            offset += size
        representatives = {}
        for row in rows:
            if row["group"] == 0 and row["logical_cpu"] in allowed:
                key = (row["group"], row["core_index"])
                if key not in representatives or row["logical_cpu"] < representatives[key]["logical_cpu"]:
                    representatives[key] = row
        ordered = sorted(representatives.values(), key=lambda r: (-r["efficiency_class"], r["core_index"], r["logical_cpu"]))
        candidates = [r["logical_cpu"] for r in ordered]
    else:
        candidates = sorted(allowed)
    if not candidates or max(thread_counts) > len(candidates):
        raise ValueError("Requested threads exceed available physical-core representatives")
    return {p: sorted(candidates[:p]) for p in thread_counts}, dict(
        method="GetSystemCpuSetInformation" if rows else "allowed logical CPUs; topology unavailable",
        allowed=allowed, cpu_sets=rows, selected_order=candidates)


def _digest():
    result = hashlib.sha256()
    for name in ("core_layout.py", "layout_benchmark.py", "quotient_brandes.py", "compiled.py",
                 "compiled_threads.py", "tree_quotient.py", "csr.py", "large_benchmark.py", "cli.py"):
        result.update(name.encode())
        result.update(Path(__file__).with_name(name).read_bytes())
    return result.hexdigest()


def _compare(a, b):
    if a["status"] != "ok" or b["status"] != "ok":
        raise RuntimeError("Cannot compare unfinished cases")
    coverage = all(a[key] == b[key] for key in ("graph_sha256", "root_source_order_hash",
        "source_order_hash", "source_count", "represented_source_count", "core_traversals", "restored_original_nodes"))
    output = compare_vectors(np.load(a["contribution_path"], mmap_mode="r"),
                             np.load(b["contribution_path"], mmap_mode="r"))
    core = compare_vectors(np.load(a["core_contribution_path"], mmap_mode="r"),
                           np.load(b["core_contribution_path"], mmap_mode="r"))
    if not coverage or not output["allclose"] or not core["allclose"]:
        raise RuntimeError(f"Layout/configuration validation mismatch: {coverage}, {output}, {core}")
    return dict(coverage_valid=coverage, original_nodes=output, old_core_indices=core,
        speedup_work=a["t_work"] / b["t_work"],
        speedup_execute=a["t_execute_reduce"] / b["t_execute_reduce"],
        speedup_wall=a["t_supervised_total"] / b["t_supervised_total"])


def _gate(pairs):
    """Engineering screen, NOT a significance test from three observations."""
    ratios = [p["validation"]["speedup_work"] for p in pairs]
    return bool(ratios and min(ratios) > 1.0 and statistics.median(ratios) >= 1.05)


def run_layout_tuning(cache_directory, output_root, *, source_count=512, confirm_count=2048,
                      thread_counts=(4, 6, 8), repeats=3, seed=46, confirm_seed=47,
                      layout_seed=1234, batch_size=4, max_case_seconds=180,
                      max_total_seconds=1800, min_available_bytes=2**30,
                      protect_run=None, cpu_sets=None):
    source_count, confirm_count, repeats, batch_size = map(operator.index,
        (source_count, confirm_count, repeats, batch_size))
    thread_counts = tuple(map(operator.index, thread_counts))
    if (min(source_count, confirm_count, repeats, batch_size) < 1 or not thread_counts
            or thread_counts != tuple(sorted(set(thread_counts))) or min(thread_counts) < 1
            or source_count < max(thread_counts) * batch_size
            or confirm_count < max(thread_counts) * batch_size
            or not np.isfinite(max_case_seconds) or not np.isfinite(max_total_seconds)
            or min(max_case_seconds, max_total_seconds) <= 0 or min_available_bytes < 0
            or seed == confirm_seed or min(seed, confirm_seed, layout_seed) < 0):
        raise ValueError("Invalid layout experiment protocol or budget")
    if cpu_sets is None:
        cpu_sets, topology = cpu_configurations(thread_counts)
    else:
        cpu_sets = {int(p): list(cpus) for p, cpus in cpu_sets.items()}
        topology = dict(method="explicit test/user CPU sets", allowed=psutil.Process().cpu_affinity())
    allowed = set(psutil.Process().cpu_affinity())
    if set(cpu_sets) != set(thread_counts) or any(len(cpus) != p or cpus != sorted(set(cpus))
        or not set(cpus) <= allowed for p, cpus in cpu_sets.items()):
        raise ValueError("Invalid matched CPU sets")
    cache = Path(cache_directory).resolve()
    qmeta = json.loads((cache / "metadata.json").read_text())
    ptr = np.load(cache / "core/indptr.npy", mmap_mode="r", allow_pickle=False)
    eligible = np.flatnonzero(np.diff(ptr) > 0)
    if max(source_count, confirm_count) > len(eligible):
        raise ValueError("Requested core groups exceed eligible groups")
    roots = np.random.default_rng(seed).permutation(eligible)[:source_count].astype(np.int32)
    confirm = np.random.default_rng(confirm_seed).permutation(eligible)[:confirm_count].astype(np.int32)
    root = Path(output_root) / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ_") + "layout_tuning")
    protected = [cache]
    if protect_run:
        protected += [protect_run, *(Path(__file__).with_name(name) for name in (
            "quotient_checkpoint.py", "quotient_brandes.py", "compiled_threads.py", "compiled.py",
            "tree_quotient.py", "csr.py", "brandes.py"))]
    if any(Path(p).is_dir() and root.resolve().is_relative_to(Path(p).resolve()) for p in protected):
        raise ValueError("Output must be outside protected inputs")
    root.mkdir(parents=True, exist_ok=False)
    before = _snapshot(protected)
    _atomic_json(root / "protected_before.json", before)
    _atomic_json(root / "cpu_topology.json", dict(topology, configurations=cpu_sets))
    _save_array(root / "source_groups.npy", roots)
    _save_array(root / "independent_source_groups.npy", confirm)
    base_p = thread_counts[0]
    manifest = dict(status="running", graph_sha256=qmeta["source_graph_sha256"],
        n=qmeta["original_nodes"], m=qmeta["original_edges"], core_nodes=qmeta["core_nodes"],
        core_edges=qmeta["core_edges"], total_core_groups=len(eligible), source_count_groups=source_count,
        confirm_count_groups=confirm_count, seed=seed, confirm_seed=confirm_seed, layout_seed=layout_seed,
        source_groups_hash=source_order_digest(roots), independent_groups_hash=source_order_digest(confirm),
        repeats=repeats, batch_size=batch_size, cpu_sets=cpu_sets, code_sha256=_digest(),
        environment=dict(python=platform.python_version(), os=platform.platform(),
            **{name: version(name) for name in ("numpy", "numba", "llvmlite", "psutil")}),
        min_available_bytes_reserved=min_available_bytes, max_total_seconds=max_total_seconds,
        max_case_seconds=max_case_seconds, result_kind="source_subset_contribution", exact_all_sources=False,
        scaled=False, protocol="matched old core groups with full block source/target masses; restore all original nodes",
        preparations=[], warmups=[], cases=[], layout_pairs=[], thread_pairs=[], confirmation_pairs=[])
    started = time.perf_counter()
    def persist():
        _atomic_json(root / "manifest.json", manifest)
    persist()
    def job(label, layout, p, selected, prepare=False):
        output = root / label
        output.mkdir(parents=True, exist_ok=False)
        seconds = min(max_case_seconds, max_total_seconds - (time.perf_counter() - started))
        if seconds <= 0:
            row = dict(status="budget_not_run", error="Total tuning budget exhausted")
        else:
            params = dict(cache_directory=str(cache), output=str(output.resolve()),
                kind="prepare" if prepare else "case", seed=layout_seed,
                layout_directory=(layout, str((root / "layouts" / layout).resolve())) if prepare
                                 else str((root / "layouts" / layout).resolve()),
                roots=None if prepare else list(map(int, selected)), threads=p, batch_size=batch_size,
                cpu_affinity=cpu_sets[p], timeout_seconds=seconds)
            row = bounded_job("layout_experiment", params, seconds=seconds,
                              min_available_bytes=min_available_bytes)
        row.update(label=label, layout=layout, threads=p)
        _atomic_json(output / "case.json", row)
        manifest["preparations" if prepare else "warmups" if label.startswith("warmup") else "cases"].append(row)
        persist()
        print(f"{label}: {row['status']}; work={row.get('t_work', 0):.3f}s; "
              f"wall={row.get('t_supervised_total',0):.3f}s", flush=True)
        if row["status"] != "ok":
            raise RuntimeError(f"Stopped after {label}: {row['status']}: {row.get('error','')}")
        if not prepare and (row["cpu_affinity_used"] != cpu_sets[p] or row["active_workers"] != p):
            raise RuntimeError("Observed worker/affinity mismatch")
        return row
    try:
        for layout in ("natural", "bfs"):
            job("prepare_" + layout, layout, base_p, roots[:1], prepare=True)
            job("warmup_" + layout, layout, base_p, roots[:max(thread_counts) * batch_size])
        for repeat in range(repeats):
            rows = {}
            for layout in (("natural", "bfs") if repeat % 2 == 0 else ("bfs", "natural")):
                rows[layout] = job(f"layout_r{repeat+1}_{layout}", layout, base_p, roots)
            pair = dict(repeat=repeat+1, validation=_compare(rows["natural"], rows["bfs"]))
            manifest["layout_pairs"].append(pair)
            persist()
        selected_layout = "bfs" if _gate(manifest["layout_pairs"]) else "natural"
        manifest["selected_layout"] = selected_layout
        for repeat in range(repeats):
            rows = {}
            order = thread_counts[repeat % len(thread_counts):] + thread_counts[:repeat % len(thread_counts)]
            for p in order:
                rows[p] = job(f"threads_r{repeat+1}_p{p}", selected_layout, p, roots)
            for p in thread_counts[1:]:
                manifest["thread_pairs"].append(dict(repeat=repeat+1, threads=p,
                    validation=_compare(rows[base_p], rows[p])))
            persist()
        accepted = [p for p in thread_counts[1:] if _gate([r for r in manifest["thread_pairs"] if r["threads"] == p])]
        candidate_p = max(accepted, key=lambda p: statistics.median(
            r["validation"]["speedup_work"] for r in manifest["thread_pairs"] if r["threads"] == p)) if accepted else base_p
        candidate_layout = selected_layout
        eligible_adoption = candidate_layout != "natural" or candidate_p != base_p
        if not eligible_adoption:
            # Confirm the negative layout result instead of rerunning identical configurations.
            candidate_layout = "bfs"
        manifest.update(confirmation_candidate=dict(layout=candidate_layout, threads=candidate_p),
                        main_screen_passed=eligible_adoption)
        for repeat in range(repeats):
            configs = [("baseline", "natural", base_p), ("candidate", candidate_layout, candidate_p)]
            rows = {}
            for label, layout, p in (configs if repeat % 2 == 0 else list(reversed(configs))):
                rows[label] = job(f"confirm_r{repeat+1}_{label}", layout, p, confirm)
            manifest["confirmation_pairs"].append(dict(repeat=repeat+1,
                validation=_compare(rows["baseline"], rows["candidate"])))
            persist()
        adopt = eligible_adoption and _gate(manifest["confirmation_pairs"])
        manifest.update(status="complete", adopt_candidate=adopt,
            recommended_config=dict(layout=candidate_layout, threads=candidate_p) if adopt
                               else dict(layout="natural", threads=base_p),
            selection_rule="all paired work ratios >1 and median >=1.05 in main screen and independent confirmation; not a significance test")
    except BaseException as exc:
        manifest.update(status="incomplete", error=f"{type(exc).__name__}: {exc}")
        if isinstance(exc, KeyboardInterrupt):
            persist()
            raise
    finally:
        after = _snapshot(protected)
        _atomic_json(root / "protected_after.json", after)
        manifest.update(protected_paths_unchanged=before == after, protected_files=len(before),
                        code_unchanged=_digest() == manifest["code_sha256"],
                        t_tuning_wall=time.perf_counter() - started)
        if before != after or not manifest["code_unchanged"]:
            manifest.update(status="incomplete", adopt_candidate=False, error="Protected inputs or experiment code changed")
        persist()
        report_tuning(root, manifest)
    return root


def report_tuning(root, manifest):
    lines = ["# BFS重排与线程配置实验", "", f"状态：{manifest['status']}；保护文件一致：{manifest.get('protected_paths_unchanged')}。", "",
        f"完整原图{manifest['n']:,}节点、{manifest['m']:,}边；核心{manifest['core_nodes']:,}节点。",
        f"主集合{manifest['source_count_groups']}组，独立集合{manifest['confirm_count_groups']}组；每组采用全部原始源数量权重。所有结果恢复为原始节点顺序。", "",
        "| 案例 | 编号 | 线程 | 工作秒 | 核心执行秒 | 总墙钟秒 |", "|---|---|---:|---:|---:|---:|"]
    for row in manifest["cases"]:
        lines.append(f"| {row['label']} | {row['layout']} | {row['threads']} | {row.get('t_work',0):.3f} | {row.get('t_execute_reduce',0):.3f} | {row.get('t_supervised_total',0):.3f} |")
    lines += ["", "工作时间包含源组选择、核心执行/归并、结果反向映射及全原图恢复；总墙钟另含加载校验、JIT、线程工作区建立、保存和回收。", "",
        "一次性重排/缓存准备另列，不在主工作时间中重复收费："]
    for row in manifest["preparations"]:
        lines.append(f"- {row['layout']}：受监督准备 {row.get('t_supervised_total',0):.3f}秒；构建/等价验证 {row.get('t_build_validate',0):.3f}秒。")
    for name, key in (("重排配对", "layout_pairs"), ("线程配对", "thread_pairs"), ("独立集合确认", "confirmation_pairs")):
        lines += ["", f"## {name}", "", "| 重复 | 候选线程 | 工作加速比 | 核心加速比 | 总墙钟加速比 | 原图最大绝对误差 |", "|---|---:|---:|---:|---:|---:|"]
        for pair in manifest[key]:
            v = pair["validation"]
            lines.append(f"| {pair['repeat']} | {pair.get('threads','')} | {v['speedup_work']:.3f} | {v['speedup_execute']:.3f} | {v['speedup_wall']:.3f} | {v['original_nodes']['max_abs_error']} |")
    lines += ["", f"建议配置：{manifest.get('recommended_config', '实验未完成')}；采用候选：{manifest.get('adopt_candidate',False)}。",
        "三次观测和工程筛选不构成统计显著性结论。P/E核心集合变化是线程配置实验的组成因素，不把它当作同一硬件集合的纯线程扩展。",
        "所有输出都是有限原始源集合的未缩放贡献，不是完整全源介数；没有启动全源任务或迁移旧检查点。",
        "原始数据见manifest.json、cpu_topology.json、各案例case.json/tasks.jsonl及贡献向量。旧输入保护哈希见protected_before/after.json。"]
    if manifest.get("error"):
        lines += ["", "```text", manifest["error"], "```"]
    (Path(root) / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
