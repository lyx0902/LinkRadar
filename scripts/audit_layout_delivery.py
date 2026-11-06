"""Audit completed layout cases, source masses, mapping, files and OS cleanup."""
import argparse
import json
from pathlib import Path

import numpy as np
import psutil

from parallel_bc.brandes import source_order_digest
from parallel_bc.data import _atomic_json, _sha256
from summarize_layout_tuning import summarize


def audit(root, continuation):
    summary = summarize(root, continuation)
    m = json.loads((root / "manifest.json").read_text())
    c = json.loads((continuation / "continuation.json").read_text())
    protected = json.loads((root / "protected_before.json").read_text())
    cache = next(Path(name).parent for name in protected if Path(name).name=="metadata.json"
        and json.loads(Path(name).read_text()).get("cache_format")=="LinkRadar-tree-quotient-v1")
    anchor_index = np.load(cache / "anchor_index.npy",mmap_mode="r")
    mass = np.load(cache / "subtree_size.npy",mmap_mode="r")[np.load(cache / "anchor_nodes.npy")]
    ids = np.load(cache / "core/original_ids.npy",mmap_mode="r")
    plans = {"main":np.load(root/"source_groups.npy"),"confirm":np.load(root/"independent_source_groups.npy")}
    source_info = {}
    for name, roots in plans.items():
        original_sources = np.flatnonzero(np.isin(anchor_index,roots))
        source_info[name] = dict(groups_hash=source_order_digest(roots),
            original_hash=source_order_digest(original_sources),original_sources=len(original_sources))
        assert len(original_sources) == int(mass[roots].sum())
    mappings = {}
    for kind in ("natural","bfs"):
        path = root/"layouts"/kind
        meta = json.loads((path/"metadata.json").read_text())
        coremeta = json.loads((path/"core/metadata.json").read_text())
        assert all(_sha256(path/name)==sha for name,sha in meta["files"].items())
        assert all(_sha256(path/"core"/name)==sha for name,sha in coremeta["cache_files"].items())
        fwd,inv = (np.load(path/(name+".npy"),mmap_mode="r") for name in ("new_to_old","old_to_new"))
        assert np.array_equal(inv[fwd],np.arange(len(fwd)))
        assert np.array_equal(np.load(path/"mass.npy"),mass[fwd])
        mappings[kind] = inv
    rows = [r for r in m["cases"] if r["status"]=="ok"]+c["cases"]
    hashes, pids, complete_times = {}, set(), {}
    for row in rows:
        group = "main" if row["label"].startswith(("layout_","threads_")) else "confirm"
        expected = source_info[group]
        assert row["root_source_order_hash"] == expected["groups_hash"]
        assert row["source_order_hash"] == expected["original_hash"]
        assert row["source_count"] == row["represented_source_count"] == expected["original_sources"]
        assert row["graph_sha256"] == m["graph_sha256"] and row["restored_original_nodes"]==m["n"]
        assert row["active_workers"]==row["p_used"]==row["threads"]
        assert row["cpu_affinity_used"] == m["cpu_sets"][str(row["threads"])]
        path = Path(row["contribution_path"])
        tasks = [json.loads(line) for line in (path.parent/"tasks.jsonl").read_text().splitlines()]
        assert [t["source_index"] for t in tasks] == plans[group].tolist()
        for task in tasks:
            old = task["source_index"]
            assert task["original_id"]==int(ids[old]) and task["source_multiplicity"]==int(mass[old])
            assert task["layout_source_index"] == mappings[row["layout"]][old]
        for key, size in (("contribution_path",m["n"]),("core_contribution_path",m["core_nodes"])):
            values = np.load(row[key],mmap_mode="r")
            assert values.shape==(size,) and np.isfinite(values).all() and np.all(values>=0)
            hashes[row[key]] = _sha256(Path(row[key]))
        pids.update(w["pid"] for w in row["workers"])
        for worker in row["workers"]:
            pid = worker["pid"]
            complete_times[pid] = max(complete_times.get(pid,0),(path.parent/"case.json").stat().st_mtime)
    alive, reused = [], []
    for pid in pids:
        try:
            proc = psutil.Process(pid)
            created = proc.create_time()
            if created > complete_times[pid]:
                reused.append(dict(pid=pid,current_process=proc.name(),created_at=created,
                                   old_worker_recorded_complete_at=complete_times[pid]))
            else:
                alive.append(pid)
        except psutil.NoSuchProcess:
            pass
    result = dict(status="verified" if all(summary["checks"].values()) and not alive else "needs_attention",
        summary_checks=summary["checks"], successful_measured_cases=len(rows),
        preserved_timeouts=len(summary["preserved_failed_cases"]), independent_pairs=3,
        source_info=source_info, all_layout_files_and_inverse_maps_valid=True,
        all_original_source_masses_ids_and_indices_valid=True,
        all_original_and_core_vectors_finite_nonnegative=True,
        observed_affinities_and_active_workers_valid=True,
        saved_vector_sha256=hashes, recorded_compute_pids=sorted(pids),still_alive=alive,reused_pids=reused,
        recommended_config=summary["recommended_config"],adopt_candidate=summary["adopt_candidate"])
    _atomic_json(root/"delivery_validation.json",result)
    print(json.dumps({k:v for k,v in result.items() if k not in {"saved_vector_sha256","recorded_compute_pids"}},indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir",type=Path)
    parser.add_argument("continuation",type=Path)
    args = parser.parse_args()
    audit(args.run_dir,args.continuation)
