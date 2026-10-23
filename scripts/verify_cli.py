"""Repeat the documented CLI flow and save exact command outcomes."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    base = Path(__file__).resolve().parents[1]
    records = []
    commands = [
        ["--help"],
        ["data", "generate", "--family", "ba", "--nodes", "500", "--attach", "3", "--seed", "42", "--name", "ba_500"],
        ["centrality", "--dataset", "ba_500", "--mode", "serial"],
        ["centrality", "--dataset", "ba_500", "--mode", "static_batches", "--processes", "4", "--batch-size", "32"],
        ["centrality", "--dataset", "ba_500", "--mode", "dynamic_batches", "--processes", "4", "--batch-size", "32"],
        ["validate", "--dataset", "ba_500", "--against", "networkx"],
        ["robustness", "--dataset", "ba_500", "--random-repeats", "10", "--seed", "42"],
    ]
    for args in commands:
        command = [sys.executable, "-m", "parallel_bc", *args]
        result = subprocess.run(command, cwd=base, capture_output=True, text=True, encoding="utf-8", timeout=180,
                                env=dict(os.environ, PYTHONUTF8="1"))
        records.append(dict(command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
        print(args[0], result.returncode, flush=True)
        if result.returncode:
            break
    (base / "docs" / "cli_verification.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return int(any(r["returncode"] for r in records))

if __name__ == "__main__":
    raise SystemExit(main())
