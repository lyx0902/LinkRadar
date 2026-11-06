"""Rebuild concise timing/validation summaries from an existing tuning run."""
import argparse
import json
from pathlib import Path
import statistics

from parallel_bc.data import _atomic_json, _sha256
from parallel_bc.layout_benchmark import _digest


def summarize(root, continuation=None):
    m = json.loads((root / "manifest.json").read_text())
    supplementary = None
    if continuation:
        supplementary = json.loads((continuation / "continuation.json").read_text())
        assert supplementary["status"] == "complete" and supplementary["code_unchanged"] and supplementary["protected_inputs_unchanged"]
        assert Path(supplementary["original_run"]) == root.resolve()
        assert supplementary["original_group_hash"] == m["independent_groups_hash"]
        for row in supplementary["cases"]:
            assert row["root_source_order_hash"] == m["independent_groups_hash"]
            row = dict(row, label="confirm_r3_"+row["label"]+"_retry")
            m["cases"].append(row)
        m["confirmation_pairs"].append(supplementary["third_pair"])
        m.update(status="complete_with_preserved_timeout",adopt_candidate=supplementary["adopt_candidate"],
            recommended_config=m["confirmation_candidate"] if supplementary["adopt_candidate"] else
                dict(layout="natural",threads=min(map(int,m["cpu_sets"]))),
            t_tuning_wall=supplementary["combined_experiment_wall"])
    stages = {}
    for row in m["cases"]:
        if row["status"] != "ok":
            continue
        stage = "confirmation" if row["label"].startswith("confirm") else "layout" if row["label"].startswith("layout") else "threads"
        key = (stage, row["layout"], row["threads"])
        stages.setdefault(key, []).append(row)
    table = []
    for (stage, layout, threads), rows in stages.items():
        item = dict(stage=stage, layout=layout, threads=threads, repeats=len(rows),
            median_work=statistics.median(r["t_work"] for r in rows),
            median_execute=statistics.median(r["t_execute_reduce"] for r in rows),
            median_wall=statistics.median(r["t_supervised_total"] for r in rows),
            work_range=[min(r["t_work"] for r in rows),max(r["t_work"] for r in rows)],
            peak_rss_mib=max(r["child_peak_rss_bytes"] for r in rows)/2**20,
            min_available_gib=min(r["min_available_bytes"] for r in rows)/2**30,
            median_cpu_parallelism=statistics.median(r.get("average_worker_cpu_cores",0) for r in rows),
            median_worker_cpu_seconds=statistics.median(r.get("worker_cpu_seconds",0) for r in rows),
            fallback_sources=sum(r["fallback_sources"] for r in rows))
        if stage == "confirmation":
            item["projected_full_core_hours"] = (sum(r["t_execute_reduce"] for r in rows)
                / sum(r["core_traversals"] for r in rows) * m["total_core_groups"] / 3600)
        table.append(item)
    before = json.loads((root / "protected_before.json").read_text())
    after = json.loads((root / "protected_after.json").read_text())
    pairs = m["layout_pairs"] + m["thread_pairs"] + m["confirmation_pairs"]
    checks = dict(experiment_complete=m["status"] in {"complete","complete_with_preserved_timeout"},
        code_matches=m["code_sha256"]==_digest(), protected_snapshots_equal=before==after,
        protected_files_still_match=all(_sha256(Path(name))==sha for name,sha in after.items()),
        all_pairs_valid=all(r["validation"]["coverage_valid"] and r["validation"]["original_nodes"]["allclose"]
                            and r["validation"]["old_core_indices"]["allclose"] for r in pairs),
        all_source_groups_unscaled=all(not r["exact_all_sources"] and not r["scaled"] for r in m["cases"] if r["status"]=="ok"))
    result = dict(status=m["status"], checks=checks, protected_files=len(after), table=table,
        selected_layout=m.get("selected_layout"), confirmation_candidate=m.get("confirmation_candidate"),
        recommended_config=m.get("recommended_config"), adopt_candidate=m.get("adopt_candidate"),
        max_abs_original_node_error=max((r["validation"]["original_nodes"]["max_abs_error"] or 0 for r in pairs),default=0),
        median_layout_work_speedup=statistics.median(r["validation"]["speedup_work"] for r in m["layout_pairs"]) if m["layout_pairs"] else None,
        median_confirmation_work_speedup=statistics.median(r["validation"]["speedup_work"] for r in m["confirmation_pairs"]) if m["confirmation_pairs"] else None,
        median_confirmation_wall_speedup=statistics.median(r["validation"]["speedup_wall"] for r in m["confirmation_pairs"]) if m["confirmation_pairs"] else None,
        t_tuning_wall=m["t_tuning_wall"],
        preserved_failed_cases=[r for r in m["cases"] if r["status"]!="ok"],
        continuation_directory=str(continuation.resolve()) if continuation else None)
    # A new backend cannot silently reuse frozen checkpoints. Compare continuing
    # their remaining work against restarting a new plan, using this run's own
    # paired confirmation throughput rather than historical timings.
    for name in before:
        if Path(name).name != "manifest.json":
            continue
        old = json.loads(Path(name).read_text())
        if old.get("format") != "LinkRadar-quotient-checkpoint-v1":
            continue
        fraction = old["completed_core_groups"] / old["identity"]["core_traversals"]
        base_p = min(map(int,m["cpu_sets"]))
        baseline = next((r for r in table if r["stage"]=="confirmation" and r["layout"]=="natural" and r["threads"]==base_p),None)
        candidate = next((r for r in table if r["stage"]=="confirmation" and
            {"layout":r["layout"],"threads":r["threads"]}==m.get("confirmation_candidate")),None)
        if baseline and candidate:
            remaining = baseline["projected_full_core_hours"] * (1-fraction)
            restart = candidate["projected_full_core_hours"]
            result["checkpoint_cost_comparison"] = dict(saved_fraction=fraction,
                break_even_speedup_for_restart=1/(1-fraction),
                old_remaining_core_hours=remaining, candidate_restart_core_hours=restart,
                restart_has_lower_projected_core_cost=restart < remaining,
                note="Extrapolation only; code/cache setup, future load and interruption costs excluded. No checkpoint migration performed.")
    _atomic_json(root / "summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--continuation", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.run_dir,args.continuation), indent=2))
