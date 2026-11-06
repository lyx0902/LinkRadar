"""Full-source layout regression on the two previously validated real graphs."""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from parallel_bc.compiled_benchmark import compare_vectors
from parallel_bc.data import _atomic_json
from parallel_bc.large_benchmark import bounded_job
from parallel_bc.layout_benchmark import cpu_configurations


def main():
    previous = Path("results/20261024T090238_578498Z_checkpoint_real_validation")
    root = Path("results") / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ_") + "layout_real_validation")
    root.mkdir(parents=True, exist_ok=False)
    cpus, audit = cpu_configurations((4,))
    cases = []
    for name in ("facebook", "ca_grqc"):
        cache = previous / (name + "_q")
        meta = json.loads((cache / "metadata.json").read_text())
        reference_meta = json.loads((previous / name / "manifest.json").read_text())
        assert reference_meta["exact_all_sources"] and meta["source_graph_sha256"] == reference_meta["identity"]["graph_sha256"]
        reference = np.load(previous / name / "contributions.npy")
        for kind in ("natural", "bfs"):
            output = root / (name + "_" + kind)
            output.mkdir()
            layout = output / "layout"
            common = dict(cache_directory=str(cache.resolve()), output=str(output.resolve()),
                          cpu_affinity=cpus[4])
            prep = bounded_job("layout_experiment", dict(common, kind="prepare",
                layout_directory=(kind, str(layout.resolve()))), seconds=90, min_available_bytes=2**30)
            assert prep["status"] == "ok", prep
            row = bounded_job("layout_experiment", dict(common, kind="case", layout_directory=str(layout.resolve()),
                roots=list(range(meta["core_nodes"])), threads=4, batch_size=4, timeout_seconds=90),
                seconds=90, min_available_bytes=2**30)
            assert row["status"] == "ok", row
            check = compare_vectors(reference, np.load(row["contribution_path"]))
            assert row["exact_all_sources"] and check["allclose"], (row, check)
            cases.append(dict(dataset=name, layout=kind, validation=check, case=row, preparation=prep))
            _atomic_json(root / "validation.json", dict(status="complete" if len(cases)==4 else "running", cases=cases))
            print(name, kind, check, flush=True)
    print(root.resolve(), flush=True)


if __name__ == "__main__":
    main()
