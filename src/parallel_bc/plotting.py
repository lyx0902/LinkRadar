"""Evidence-only PNG/PDF figures. Missing experiments produce explicit panels."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else ({} if default is None else default)


def read_rows(path):
    path = Path(path)
    if not path.is_file() or not path.stat().st_size:
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def dataset_dirs(run_dir):
    root = Path(run_dir)
    manifest = read_json(root / "manifest.json")
    paths = []
    for item in manifest.get("datasets", []):
        path = root / item.get("path", item.get("id", "."))
        if path.is_dir() and path not in paths:
            paths.append(path)
    if not paths:
        paths = sorted({p.parent for p in root.rglob("graph_metadata.json")})
    return paths or [root]


def _number(row, key, default=np.nan):
    try:
        return float(row[key])
    except (KeyError, ValueError, TypeError):
        return default


def _formal(row):
    return row.get("status", "ok") == "ok" and str(row.get("warmup", False)).lower() not in {"true", "1"}


def _key(row, fields):
    return tuple(row.get(key, "") for key in fields)


def plot_results(run_dir):
    """Export figures from saved real observations, returning output Paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(run_dir)
    output = root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    dirs = dataset_dirs(root)
    summaries = [row for directory in dirs for row in read_rows(directory / "timings_summary.csv")]
    timings = [row for directory in dirs for row in read_rows(directory / "timings_raw.csv") if _formal(row)]
    workers = [row for directory in dirs for row in read_rows(directory / "worker_summary.csv") if _formal(row)]
    metadata = {str(directory): read_json(directory / "graph_metadata.json") for directory in dirs}
    paths, records = [], []

    def blank(ax, reason):
        ax.text(.5, .5, "Not executed\n" + reason, ha="center", va="center", wrap=True, transform=ax.transAxes)
        ax.set_axis_off()

    def save(fig, name, measured, reason=""):
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            path = output / f"{name}.{suffix}"
            fig.savefig(path, dpi=170, bbox_inches="tight")
            paths.append(path)
        plt.close(fig)
        records.append(dict(name=name, measured=bool(measured), reason=reason,
                            png=f"figures/{name}.png", pdf=f"figures/{name}.pdf"))

    # Full-node accuracy: serial vs NetworkX and every measured mode vs serial.
    accuracy = []
    for directory in dirs:
        validation = read_json(directory / "validation.json")
        name = metadata[str(directory)].get("dataset_id", directory.name)
        if validation.get("allclose") is not None:
            error = validation.get("max_abs_error", validation.get("max_abs_error_raw"))
            if error is None and isinstance(validation.get("raw"), dict):
                error = validation["raw"].get("max_abs_error")
            accuracy.append([name, "serial / NetworkX", str(validation.get("allclose")), f"{error:.3g}" if isinstance(error, (int, float)) else "recorded in validation.json"])
    groups = {}
    for row in timings:
        if np.isfinite(_number(row, "max_abs_error")):
            groups.setdefault(_key(row, ("dataset_id", "mode")), []).append(_number(row, "max_abs_error"))
    accuracy.extend([[dataset, f"{mode} / serial", "completed", f"{max(errors):.3g}"] for (dataset, mode), errors in groups.items()])
    fig, ax = plt.subplots(figsize=(10, max(3, .35 * len(accuracy)+1.5)))
    ax.set_axis_off()
    if accuracy:
        table = ax.table(cellText=accuracy, colLabels=["Dataset", "Comparison", "Status", "Max absolute raw error"], loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.4)
        ax.set_title("All-node correctness checks")
    else:
        blank(ax, "No completed all-node validation records")
    save(fig, "correctness", accuracy, "No completed all-node validation records" if not accuracy else "")

    by_graph = {}
    for row in summaries:
        if np.isfinite(_number(row, "median_total")):
            by_graph.setdefault(_key(row, ("dataset_id", "graph_sha256")), []).append(row)
    fig, axes = plt.subplots(max(1, len(by_graph)), 1, figsize=(11, 4 * max(1, len(by_graph))), squeeze=False)
    if not by_graph:
        blank(axes[0, 0], "No completed timing observations")
    for ax, ((dataset, fingerprint), rows) in zip(axes[:, 0], sorted(by_graph.items())):
        rows = sorted(rows, key=lambda r: (r["mode"], _number(r, "p_used"), _number(r, "batch_size")))
        y = np.asarray([_number(r, "median_total") for r in rows])
        low = np.asarray([max(0., _number(r, "median_total")-_number(r, "q1_total")) for r in rows])
        high = np.asarray([max(0., _number(r, "q3_total")-_number(r, "median_total")) for r in rows])
        labels = [f'{r["mode"]}\np={r["p_used"]}, B={r["batch_size"]}\nN={r["count"]}' for r in rows]
        ax.bar(np.arange(len(rows)), y, yerr=[low, high], capsize=4)
        ax.set_xticks(np.arange(len(rows)), labels, fontsize=8)
        ax.set_ylabel("Total wall time (s)")
        ax.set_title(f"{dataset}: median and empirical IQR; actual p used")
        ax.grid(axis="y", alpha=.25)
    save(fig, "total_time", by_graph, "No completed timing observations" if not by_graph else "")

    for metric, note in (("total", "Includes startup and cleanup"), ("steady", "Excludes startup and cleanup")):
        fig, axes = plt.subplots(max(1, len(by_graph)), 1, figsize=(8, 4 * max(1, len(by_graph))), squeeze=False)
        any_measured = False
        for ax, ((dataset, _), rows) in zip(axes[:, 0], sorted(by_graph.items())):
            groups = {}
            for row in rows:
                if row["mode"] != "serial" and np.isfinite(_number(row, f"speedup_{metric}")):
                    groups.setdefault(_key(row, ("mode", "batch_size", "source_order_hash", "task_logging")), []).append(row)
            if not groups:
                blank(ax, "No paired serial/parallel measurements")
                continue
            any_measured = True
            max_p = 1
            for (mode, size, _, logging), group in groups.items():
                group.sort(key=lambda r: _number(r, "p_used"))
                ps = [_number(r, "p_used") for r in group]
                max_p = max(max_p, *ps)
                ax.plot(ps, [_number(r, f"speedup_{metric}") for r in group], "o-", label=f"{mode}, B={size}, log={logging}")
            ax.plot([0, max_p], [0, max_p], "k--", alpha=.4, label="Ideal reference y=p")
            ax.set(xlabel="Processes actually used", ylabel=f"Speedup ({metric})", title=f"{dataset}: {note}")
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
        if not by_graph:
            blank(axes[0, 0], "No completed timing observations")
        save(fig, f"speedup_{metric}", any_measured, "No paired serial/parallel measurements" if not any_measured else "")

    fig, ax = plt.subplots(figsize=(9, 5))
    scale_groups = {}
    for directory in dirs:
        meta = metadata[str(directory)]
        generator = meta.get("generation_parameters", {})
        family = meta.get("family", generator.get("family") if isinstance(generator, dict) else None)
        if not family:
            continue
        for row in read_rows(directory / "timings_summary.csv"):
            fixed = {k: v for k, v in generator.items() if k not in {"nodes", "target_edges", "sizes", "probabilities", "effective_attach"}}
            parameters_key = json.dumps(fixed, sort_keys=True)
            group_key = (family, parameters_key, row.get("mode"), row.get("p_used"), row.get("batch_size"), row.get("task_logging"))
            scale_groups.setdefault(group_key, []).append(row)
    measured = False
    for (family, parameters_key, mode, p, size, _), rows in scale_groups.items():
        if len({_number(r, "n") for r in rows}) < 2:
            continue
        measured = True
        rows.sort(key=lambda r: _number(r, "n"))
        ax.plot([_number(r, "n") for r in rows], [_number(r, "median_total") for r in rows], "o-", label=f"{family}, {mode}, p={p}, B={size}")
        if mode == "serial":
            for row in rows:
                ax.annotate(f'm={row["m"]}', (_number(row, "n"), _number(row, "median_total")), xytext=(3, 3), textcoords="offset points", fontsize=7)
    if measured:
        ax.set(xlabel="Nodes (n)", ylabel="Median total wall time (s)", title="Measured graph-size series; edges annotated on serial baseline")
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))
        ax.grid(alpha=.25)
    else:
        blank(ax, "Requires at least two n values in the same graph family and parameter setting")
    save(fig, "size_scaling", measured, "No comparable graph-size series" if not measured else "")

    fig, axes = plt.subplots(max(1, len(by_graph)), 1, figsize=(9, 4 * max(1, len(by_graph))), squeeze=False)
    measured = False
    for ax, ((dataset, _), rows) in zip(axes[:, 0], sorted(by_graph.items())):
        groups = {}
        panel_measured = False
        for row in rows:
            if row["mode"] != "serial":
                groups.setdefault(_key(row, ("mode", "p_used", "source_order_hash", "task_logging")), []).append(row)
        for (mode, p, _, logging), group in groups.items():
            if len({r["batch_size"] for r in group}) < 2:
                continue
            panel_measured = measured = True
            group.sort(key=lambda r: _number(r, "batch_size"))
            ax.errorbar([_number(r, "batch_size") for r in group], [_number(r, "median_total") for r in group],
                        yerr=[[max(0., _number(r, "median_total")-_number(r, "q1_total")) for r in group],
                              [max(0., _number(r, "q3_total")-_number(r, "median_total")) for r in group]],
                        marker="o", capsize=3, label=f"{mode}, p={p}, log={logging}")
        if panel_measured:
            ax.set(xlabel="Sources per batch (B)", ylabel="Total wall time (s)", title=f"{dataset}: batch-size sweep, median and IQR")
            ax.legend(fontsize=8)
            ax.grid(alpha=.25)
        else:
            blank(ax, f"{dataset}: requires two or more B values at fixed graph and p")
    if not by_graph:
        blank(axes[0, 0], "No completed timing observations")
    save(fig, "batch_sizes", measured, "No comparable batch-size sweep" if not measured else "")

    robust_sets = [(directory, read_rows(directory / "robustness_summary.csv")) for directory in dirs]
    robust_sets = [(directory, rows) for directory, rows in robust_sets if rows]
    fig, axes = plt.subplots(max(1, len(robust_sets)), 1, figsize=(9, 4 * max(1, len(robust_sets))), squeeze=False)
    for ax, (directory, rows) in zip(axes[:, 0], robust_sets):
        for strategy in sorted({r["strategy"] for r in rows}):
            group = sorted([r for r in rows if r["strategy"] == strategy], key=lambda r: _number(r, "removed_fraction"))
            x = [_number(r, "removed_fraction") for r in group]
            ax.plot(x, [_number(r, "lcc_original_mean") for r in group], "o-", label=strategy)
            if strategy == "random":
                ax.fill_between(x, [_number(r, "lcc_original_p10") for r in group], [_number(r, "lcc_original_p90") for r in group], alpha=.2, label="Random empirical 10%-90% range")
        ax.set(xlabel="Removed nodes / original nodes", ylabel="LCC nodes / original nodes", title=f"{metadata[str(directory)].get('dataset_id', directory.name)}: fixed initial rankings", ylim=(-.02, 1.02))
        ax.legend(fontsize=8)
        ax.grid(alpha=.25)
    if not robust_sets:
        blank(axes[0, 0], "No completed node-removal observations")
    save(fig, "robustness", robust_sets, "No node-removal observations" if not robust_sets else "")

    worker_groups = {}
    for row in workers:
        worker_groups.setdefault(row.get("case_key", "case"), []).append(row)
    if worker_groups:
        selected = sorted(worker_groups.items())[:8]
        fig, axes = plt.subplots(len(selected), 1, figsize=(9, 3 * len(selected)), squeeze=False)
        for ax, (case, rows) in zip(axes[:, 0], selected):
            ax.bar([f'{r.get("worker_id")}\nPID {r.get("pid")}' for r in rows], [_number(r, "busy_seconds") for r in rows])
            ax.set(ylabel="Measured busy seconds", title=f'{rows[0].get("mode")}, p={rows[0].get("p_used")}, B={rows[0].get("batch_size")}, case {case[:10]}')
        save(fig, "worker_busy", True, "First eight cases by case key; busy time is not OS CPU utilization")

    task_groups = {}
    for directory in dirs:
        path = directory / "tasks.jsonl"
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    task = json.loads(line)
                    start = _number(task, "start_rel_seconds")
                    end = _number(task, "end_rel_seconds")
                    compute = _number(task, "compute_seconds")
                    if np.isfinite(start) and np.isfinite(end) and end >= start and np.isclose(end-start, compute, atol=1e-6, rtol=1e-4):
                        task_groups.setdefault(task.get("case_key", "diagnostic"), []).append(task)
    if task_groups:
        case, rows = sorted(task_groups.items())[0]
        fig, ax = plt.subplots(figsize=(10, 4))
        ids = sorted({int(r["worker_id"]) for r in rows})
        positions = {worker: i for i, worker in enumerate(ids)}
        for task in rows:
            ax.broken_barh([(_number(task, "start_rel_seconds"), _number(task, "end_rel_seconds")-_number(task, "start_rel_seconds"))], (positions[int(task["worker_id"])]-.35, .7))
        ax.set(yticks=list(positions.values()), yticklabels=[f"worker {worker}" for worker in ids], xlabel="Measured seconds relative to run clock", title=f'Diagnostic measured task timeline: {rows[0].get("mode")}, case {case[:10]}')
        save(fig, "task_timeline", True, "Diagnostic logging changes execution overhead; first case by key")

    (output / "index.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    return paths
