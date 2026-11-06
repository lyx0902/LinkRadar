"""Complete only the failed third pair, retaining the original timeout record.

No original manifest or cache is modified. The original 1800s total computation
budget is shared by the original run and this independent continuation.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

from parallel_bc.data import _atomic_json, _sha256
from parallel_bc.large_benchmark import bounded_job
from parallel_bc.layout_benchmark import _compare, _digest, _gate


def main(original):
    original = Path(original).resolve()
    m = json.loads((original / "manifest.json").read_text())
    assert m["code_sha256"] == _digest() and m["protected_paths_unchanged"]
    assert len(m["confirmation_pairs"]) == 2 and m["cases"][-1]["status"] == "timeout"
    remaining = m["max_total_seconds"] - m["t_tuning_wall"]
    assert remaining > 0
    root = original.parent / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ_") + "layout_confirmation_retry")
    root.mkdir(parents=True, exist_ok=False)
    protected = json.loads((original / "protected_after.json").read_text())
    assert all(_sha256(Path(name)) == sha for name,sha in protected.items())
    groups = np.load(original / "independent_source_groups.npy").tolist()
    baseline_p = min(map(int,m["cpu_sets"]))
    candidate = m["confirmation_candidate"]
    configs = [("baseline", "natural", baseline_p), ("candidate",candidate["layout"],candidate["threads"])]
    receipt = dict(status="running", original_run=str(original), code_sha256=_digest(),
        original_timeout=m["cases"][-1], remaining_budget_seconds=remaining,
        original_group_hash=m["independent_groups_hash"], cases=[])
    started = time.perf_counter()
    _atomic_json(root / "continuation.json", receipt)
    for label, layout, p in configs:
        output = root / label
        output.mkdir()
        seconds = min(300., remaining - (time.perf_counter()-started))
        if seconds <= 0:
            row = dict(status="budget_not_run",error="Original total budget exhausted")
        else:
            row = bounded_job("layout_experiment", dict(cache_directory=m["cache_directory"] if "cache_directory" in m
                else str(Path("data/processed/youtube_csr_treeq").resolve()),
                output=str(output), kind="case", layout_directory=str(original/"layouts"/layout),
                roots=groups, threads=p, batch_size=m["batch_size"], cpu_affinity=m["cpu_sets"][str(p)],
                timeout_seconds=seconds), seconds=seconds, min_available_bytes=m["min_available_bytes_reserved"])
        row.update(label=label,layout=layout,threads=p)
        _atomic_json(output / "case.json", row)
        receipt["cases"].append(row)
        _atomic_json(root / "continuation.json", receipt)
        print(label, {k:row[k] for k in ("status","t_work","t_supervised_total") if k in row},flush=True)
        if row["status"] != "ok":
            break
    if len(receipt["cases"]) == 2 and all(r["status"]=="ok" for r in receipt["cases"]):
        validation = _compare(*receipt["cases"])
        receipt["third_pair"] = dict(repeat=3,validation=validation)
        pairs = m["confirmation_pairs"] + [receipt["third_pair"]]
        receipt["adopt_candidate"] = bool(m["main_screen_passed"] and _gate(pairs))
        receipt["status"] = "complete"
    else:
        receipt["status"] = "incomplete"
    receipt.update(t_continuation_wall=time.perf_counter()-started,
        combined_experiment_wall=m["t_tuning_wall"]+time.perf_counter()-started,
        protected_inputs_unchanged=all(_sha256(Path(name))==sha for name,sha in protected.items()),
        code_unchanged=_digest()==m["code_sha256"])
    _atomic_json(root / "continuation.json", receipt)
    print(root,flush=True)
    return 0 if receipt["status"]=="complete" and receipt["protected_inputs_unchanged"] and receipt["code_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
