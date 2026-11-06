"""Read-only iterative CSR block-cut diagnosis; no BC computation or mutation.

Dyads (one bridge edge) count as edge blocks; isolated vertices do not. Block
vertices overlap at articulations, while undirected edges partition exactly.
This module is deliberately outside the frozen checkpoint numerical digest.
"""
from __future__ import annotations

from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from numba import njit

from .data import _atomic_json, _sha256


@njit(cache=True, nogil=True)
def _blocks(indptr, indices, mass):
    """Linear DFS with explicit vertex/edge stacks, including disconnected CSR."""
    n, m = len(mass), len(indices) // 2
    discovered = np.full(n, -1, np.int32)
    low = np.empty(n, np.int32)
    parent = np.full(n, -1, np.int32)
    next_edge = np.empty(n, np.int64)
    children = np.zeros(n, np.int32)
    stack = np.empty(n, np.int32)
    edge_stack = np.empty((m, 2), np.int32)
    edges = np.empty((m, 2), np.int32)
    offsets = np.empty(n + 1, np.int64)
    offsets[0] = 0
    vertices = np.empty(n, np.int32)
    member_mass = np.empty(n, np.int64)
    stamp = np.full(n, -1, np.int32)
    articulation = np.zeros(n, np.bool_)
    subtree_mass = mass.copy()
    labels = np.empty(n, np.int32)
    component_mass = np.empty(n, np.int64)
    timer = block_count = edge_count = component_count = top_edges = 0
    for root in range(n):
        if discovered[root] != -1:
            continue
        discovered[root] = low[root] = timer
        timer += 1
        labels[root] = component_count
        next_edge[root] = indptr[root]
        depth = 1
        stack[0] = root
        while depth:
            v = stack[depth - 1]
            if next_edge[v] < indptr[v + 1]:
                pos = next_edge[v]
                next_edge[v] += 1
                w = indices[pos]
                if discovered[w] == -1:
                    parent[w] = v
                    children[v] += 1
                    edge_stack[top_edges, 0], edge_stack[top_edges, 1] = v, w
                    top_edges += 1
                    discovered[w] = low[w] = timer
                    timer += 1
                    labels[w] = component_count
                    next_edge[w] = indptr[w]
                    stack[depth] = w
                    depth += 1
                elif w != parent[v] and discovered[w] < discovered[v]:
                    low[v] = min(low[v], discovered[w])
                    edge_stack[top_edges, 0], edge_stack[top_edges, 1] = v, w
                    top_edges += 1
            else:
                depth -= 1
                p = parent[v]
                if p == -1:
                    articulation[v] = children[v] > 1
                    component_mass[component_count] = subtree_mass[v]
                    component_count += 1
                    if top_edges:
                        raise RuntimeError("Unclosed DFS edge stack")
                else:
                    subtree_mass[p] += subtree_mass[v]
                    low[p] = min(low[p], low[v])
                    if low[v] >= discovered[p]:
                        if parent[p] != -1:
                            articulation[p] = True
                        count = total_mass = 0
                        while True:
                            top_edges -= 1
                            a, b = edge_stack[top_edges, 0], edge_stack[top_edges, 1]
                            edges[edge_count, 0] = min(a, b)
                            edges[edge_count, 1] = max(a, b)
                            edge_count += 1
                            if stamp[a] != block_count:
                                stamp[a] = block_count
                                count += 1
                                total_mass += mass[a]
                            if stamp[b] != block_count:
                                stamp[b] = block_count
                                count += 1
                                total_mass += mass[b]
                            if a == p and b == v:
                                break
                        vertices[block_count] = count
                        member_mass[block_count] = total_mass
                        block_count += 1
                        offsets[block_count] = edge_count
    if edge_count != m or timer != n:
        raise RuntimeError("DFS graph coverage mismatch")
    return (edges, offsets[:block_count + 1].copy(), vertices[:block_count].copy(),
            member_mass[:block_count].copy(), articulation, parent, subtree_mass,
            labels, component_mass[:component_count].copy())


def diagnose_graph(graph, mass=None):
    """Trusted path for CSR already validated by load_csr/csr_from_edges."""
    if graph.n > np.iinfo(np.int32).max:
        raise ValueError("Diagnostic requires int32 vertex indices")
    if mass is None:
        mass = np.ones(graph.n, np.int64)
    mass = np.asarray(mass)
    if (mass.shape != (graph.n,) or mass.dtype.kind not in "iu" or np.any(mass <= 0)
            or any(int(v) > np.iinfo(np.int64).max for v in mass)):
        raise ValueError("Positive int64 endpoint masses required")
    if sum(map(int, mass)) > np.iinfo(np.int64).max:
        raise ValueError("Total endpoint mass exceeds int64")
    mass = np.ascontiguousarray(mass, dtype=np.int64).view()
    mass.flags.writeable = False
    # Use readonly signatures for both test-built and mmap-loaded graphs.
    ptr, idx = graph.indptr.view(), graph.indices.view()
    ptr.flags.writeable = idx.flags.writeable = False
    started = time.perf_counter()
    tiny_ptr = np.array([0, 1, 2], np.int64)
    tiny_idx = np.array([1, 0], np.int32)
    tiny_mass = np.ones(2, np.int64)
    for array in (tiny_ptr, tiny_idx, tiny_mass):
        array.flags.writeable = False
    _blocks(tiny_ptr, tiny_idx, tiny_mass)
    warmed = time.perf_counter()
    values = _blocks(ptr, idx, mass)
    computed = time.perf_counter()
    names = ("block_edges", "block_offsets", "block_vertex_counts", "block_member_mass",
             "articulation_mask", "dfs_parent", "dfs_subtree_mass", "component_index", "component_mass")
    arrays = dict(zip(names, values))
    edges, offsets, counts, members, art, parents, subtree, labels, cmass = values
    sizes = np.diff(offsets)
    degree = np.diff(ptr)
    # Every returned edge comes from the CSR; m unique returned edges therefore
    # establish the complete partition without constructing a Python graph.
    keys = (edges[:, 0].astype(np.uint64) << np.uint64(32)) | edges[:, 1].astype(np.uint64)
    keys.sort()
    valid = len(keys) == graph.m and (len(keys) < 2 or bool(np.all(keys[1:] > keys[:-1])))
    if not valid or int(sizes.sum()) != graph.m:
        raise RuntimeError("BCC undirected edge partition failed")
    baseline = int(np.count_nonzero(degree)) * (graph.n + graph.m)
    all_blocks = sum(int(v) * (int(v) + int(e)) for v, e in zip(counts, sizes))
    cyclic_blocks = sum(int(v) * (int(v) + int(e)) for v, e in zip(counts, sizes) if e > 1)
    order = sorted(range(len(counts)), key=lambda i: (-int(counts[i]), -int(sizes[i]), i))
    def block_info(i):
        return dict(block_id=int(i), nodes=int(counts[i]), edges=int(sizes[i]),
            node_fraction=counts[i].item() / graph.n if graph.n else 0.,
            edge_fraction=sizes[i].item() / graph.m if graph.m else 0.,
            member_endpoint_mass=int(members[i]))
    largest_edges = max(range(len(sizes)), key=lambda i: (int(sizes[i]), int(counts[i]), -i)) if len(sizes) else None
    summary = dict(n=graph.n, m=graph.m, graph_sha256=graph.graph_sha256,
        endpoint_mass=sum(map(int, mass)), average_degree=float(degree.mean()) if graph.n else 0.,
        connected_components=len(cmass), isolated_vertices=int(np.count_nonzero(degree == 0)),
        articulation_vertices=int(art.sum()), bridges=int(np.count_nonzero(sizes == 1)),
        edge_blocks=len(counts), cyclic_blocks=int(np.count_nonzero(sizes > 1)),
        top_blocks_by_vertices=[block_info(i) for i in order[:10]],
        largest_block_by_edges=block_info(largest_edges) if largest_edges is not None else None,
        top10_edge_fraction=sum(int(sizes[i]) for i in sorted(range(len(sizes)), key=lambda i: -int(sizes[i]))[:10]) / graph.m if graph.m else 0.,
        edge_partition_valid=bool(valid), edge_count_sum=int(sizes.sum()),
        block_vertex_sum=int(counts.sum()), degree2_vertices=int(np.count_nonzero(degree == 2)),
        work_proxy=dict(baseline=baseline, all_blocks=all_blocks, cyclic_blocks_only=cyclic_blocks,
            ratio_all_blocks=all_blocks / baseline if baseline else None,
            ratio_cyclic_blocks_only=cyclic_blocks / baseline if baseline else None,
            formula="active_n*(n+m) versus sum n_block*(n_block+m_block)",
            meaning="structural operation-count proxy, not a measured speedup or runtime prediction"),
        t_jit_warmup=warmed - started, t_dfs_kernel=computed - warmed,
        t_graph_diagnostic=time.perf_counter() - started,
        exact_all_sources=False, result_kind="core_structure_diagnostic")
    return summary, arrays


def _csv(path, fields, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def diagnose_cache(cache_directory, output):
    from .tree_quotient import load_quotient
    from .large_benchmark import _save_array
    started = time.perf_counter()
    q = load_quotient(cache_directory)
    loaded = time.perf_counter()
    mass = q.mass
    summary, arrays = diagnose_graph(q.core, mass)
    if not np.array_equal(np.sort(arrays["component_mass"]), np.sort(q.component_sizes)):
        raise RuntimeError("DFS original component mass mismatch")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    saved = {}
    for name in ("block_edges", "block_offsets", "block_vertex_counts", "block_member_mass"):
        path = output / (name + ".npy")
        _save_array(path, arrays[name])
        saved[path.name] = _sha256(path)
    cuts = np.flatnonzero(arrays["articulation_mask"])
    _csv(output / "articulations.csv", ["core_index", "original_id", "endpoint_mass"],
        (dict(core_index=int(v), original_id=int(q.core.ids[v]), endpoint_mass=int(mass[v])) for v in cuts))
    bridge_rows = []
    for block in np.flatnonzero(np.diff(arrays["block_offsets"]) == 1):
        u, v = map(int, arrays["block_edges"][arrays["block_offsets"][block]])
        child = v if arrays["dfs_parent"][v] == u else u
        side = int(arrays["dfs_subtree_mass"][child])
        total = int(arrays["component_mass"][arrays["component_index"][child]])
        bridge_rows.append(dict(block_id=int(block), core_u=u, core_v=v,
            original_u=int(q.core.ids[u]), original_v=int(q.core.ids[v]),
            side_endpoint_mass=side, component_endpoint_mass=total,
            cross_unordered_pairs=side * (total - side)))
    _csv(output / "bridges.csv", ["block_id", "core_u", "core_v", "original_u", "original_v",
        "side_endpoint_mass", "component_endpoint_mass", "cross_unordered_pairs"], bridge_rows)
    counts = np.column_stack((arrays["block_vertex_counts"], np.diff(arrays["block_offsets"])))
    kinds, frequency = np.unique(counts, axis=0, return_counts=True)
    _csv(output / "block_size_histogram.csv", ["nodes", "edges", "blocks"],
        (dict(nodes=int(pair[0]), edges=int(pair[1]), blocks=int(count)) for pair, count in zip(kinds, frequency)))
    for name in ("articulations.csv", "bridges.csv", "block_size_histogram.csv"):
        saved[name] = _sha256(output / name)
    summary.update(original_nodes=q.n, original_edges=q.metadata["original_edges"],
        source_graph_sha256=q.source_graph_sha256, output_files=saved,
        top_bridges_by_cross_pairs=sorted(bridge_rows, key=lambda r: -r["cross_unordered_pairs"])[:10],
        t_load_validate=loaded - started, t_child_diagnostic=time.perf_counter() - started)
    _atomic_json(output / "diagnostic.json", summary)
    return summary


def _snapshot(paths):
    """Read hashes only; never acquire or alter the protected run's lock."""
    result = {}
    for raw in paths:
        path = Path(raw).resolve()
        if not path.exists():
            raise ValueError(f"Protected path missing: {path}")
        for item in sorted(path.rglob("*")) if path.is_dir() else [path]:
            if item.is_file():
                result[str(item)] = _sha256(item)
    return result


def run_core_diagnostic(cache_directory, output_root, *, seconds=180, min_available_bytes=2**30,
                        protect_run=None):
    from .large_benchmark import bounded_job
    if not np.isfinite(seconds) or seconds <= 0 or min_available_bytes < 0:
        raise ValueError("Invalid diagnostic resource budget")
    protected = [cache_directory]
    if protect_run is not None:
        protected.append(protect_run)
        # Match the frozen checkpoint digest's source list without importing its
        # numerical module or touching any of these files.
        protected.extend(Path(__file__).with_name(name) for name in (
            "quotient_checkpoint.py", "quotient_brandes.py", "compiled_threads.py",
            "compiled.py", "tree_quotient.py", "csr.py", "brandes.py"))
    root = Path(output_root) / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ_") + "core_diagnostic")
    if any(Path(raw).is_dir() and root.resolve().is_relative_to(Path(raw).resolve()) for raw in protected):
        raise ValueError("Diagnostic output must be outside protected cache/run directories")
    root.mkdir(parents=True, exist_ok=False)
    before = _snapshot(protected)
    _atomic_json(root / "protected_before.json", before)
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    row = bounded_job("core_structure", dict(cache_directory=str(Path(cache_directory).resolve()),
        output=str(root.resolve())), seconds=seconds, min_available_bytes=min_available_bytes)
    after = _snapshot(protected)
    _atomic_json(root / "protected_after.json", after)
    manifest = dict(status="complete" if row["status"] == "ok" and before == after else "incomplete",
        code_sha256=digest, cache_directory=str(Path(cache_directory).resolve()),
        protected_paths_unchanged=before == after, protected_file_count=len(before),
        wall_budget_seconds=seconds, min_available_bytes_reserved=min_available_bytes,
        result_kind="core_structure_diagnostic", exact_all_sources=False, case=row)
    _atomic_json(root / "manifest.json", manifest)
    _report(root, manifest)
    return root


def _report(root, manifest):
    row = manifest["case"]
    lines = ["# 核心图结构诊断", "", f"状态：{manifest['status']}；已保护文件哈希一致：{manifest['protected_paths_unchanged']}。", ""]
    if row["status"] != "ok":
        lines += ["诊断未完成，保留资源/错误记录：", "```text", row.get("error", row["status"]), "```"]
    else:
        lines += [f"原始图：{row['original_nodes']:,}节点、{row['original_edges']:,}条无向边。",
            f"核心图：{row['n']:,}节点、{row['m']:,}条边，平均度{row['average_degree']:.3f}。", "",
            f"连通分量{row['connected_components']:,}；割点{row['articulation_vertices']:,}；桥{row['bridges']:,}。",
            f"边块{row['edge_blocks']:,}（包含单桥的二节点块），其中含环块{row['cyclic_blocks']:,}；孤立点{row['isolated_vertices']:,}另计。", "",
            "## 最大的双连通块", "", "| 块ID | 节点 | 边 | 核心节点占比 | 核心边占比 |", "|---|---:|---:|---:|---:|"]
        for block in row["top_blocks_by_vertices"]:
            lines.append(f"| {block['block_id']} | {block['nodes']:,} | {block['edges']:,} | {block['node_fraction']:.3%} | {block['edge_fraction']:.3%} |")
        work = row["work_proxy"]
        ratio = work["ratio_cyclic_blocks_only"]
        lines += ["", "## 分解价值判断", "",
            f"全局工作量指标 active_n×(n+m)={work['baseline']:,}；各块指标之和={work['all_blocks']:,}；排除可解析桥块后={work['cyclic_blocks_only']:,}。",
            f"含环块指标/全局指标：{ratio:.6f}。" if ratio is not None else "没有需要核心BFS的非孤立源组。",
            "这是结构计算量指标，不是实测加速比，也没有包含边界权重处理、块复制、归并及缓存开销。", ""]
        largest = row["top_blocks_by_vertices"][0] if row["top_blocks_by_vertices"] else None
        if largest and largest["node_fraction"] > .9 and largest["edge_fraction"] > .9:
            lines += ["最大块同时保留超过90%的核心节点和边，分解后主要计算仍集中在这个大块。建议先测顶点重排、线程配置和内核优化，再评估完整BCC数值实现的投入。", ""]
        else:
            lines += ["应结合最大块和所有块的工作量指标评估分解机会；本轮仅完成拓扑诊断，尚未实现或计时BCC加速算法。", ""]
        lines += ["## 口径与验收", "",
            f"所有{row['m']:,}条无向核心边恰好归入一个块：{row['edge_partition_valid']}。割点会出现在多个块中，块节点数和成员端点数量不能直接相加当作独立原始节点覆盖。",
            "bridges.csv 的 s×(N−s) 是按原始端点数量计算的桥边无序跨侧点对数，不是桥端点的节点介数。",
            "本轮不生成介数分数、不修改图、不改变既有源组进度。进一步分块计算必须处理块外数量和割点修正。", "",
            f"DFS内核{row['t_dfs_kernel']:.3f}秒；JIT预热/缓存加载{row['t_jit_warmup']:.3f}秒；原图表示加载校验{row['t_load_validate']:.3f}秒。",
            f"受监督总墙钟{row['t_supervised_total']:.3f}秒，子进程峰值RSS {row['child_peak_rss_bytes']/2**20:.2f} MiB。", "",
            "## 原始附件", "", "- diagnostic.json：规模、块占比、结构工作量指标和文件哈希。",
            "- block_edges.npy / block_offsets.npy：按块连续存储的全部无向边，端点为核心索引。",
            "- block_vertex_counts.npy / block_member_mass.npy：块节点数与成员锚点的原始端点数量（割点重复出现）。",
            "- articulations.csv / bridges.csv / block_size_histogram.csv：原始节点ID、桥两侧质量及块规模分布。",
            "- manifest.json / protected_before.json / protected_after.json：预算、内存、代码及输入/检查点保护证据。"]
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
