"""Index the latest frozen-source runs, without inventing missing measurements."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import sys
import yaml
from parallel_bc.benchmark import source_fingerprint

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="生成独立的合成实验阶段摘要，不覆盖主说明")
    parser.add_argument("--output", type=Path, default=ROOT / "results/summaries/synthetic_results.md")
    args = parser.parse_args()
    target = args.output.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = source_fingerprint()
    candidates = sorted(p for p in ROOT.glob("results/*/manifest.json") if (p.parent / "config_resolved.yaml").is_file())
    chosen = {}
    for profile in ("smoke", "quick", "granularity", "diagnostic"):
        desired = yaml.safe_load((ROOT / "configs" / f"{profile}.yaml").read_text(encoding="utf-8-sig"))
        matches = [p.parent for p in candidates if read(p).get("source_sha256") == fingerprint
                   and read(p).get("status") == "complete"
                   and yaml.safe_load((p.parent / "config_resolved.yaml").read_text(encoding="utf-8-sig")) == desired]
        if not matches:
            raise RuntimeError(f"没有保留匹配的完整 {profile} 实验；如需合成摘要，请先重跑对应配置")
        chosen[profile] = matches[-1]

    out = ["# LinkRadar 合成实验阶段摘要", "", "本页只汇总本地存在且与源码、配置匹配的合成运行。当前真实图结果见[统一结果](RESULTS.md)。", "",
           f"源码指纹：`{fingerprint}`。所有数字取自实际运行文件；不包含pytest或连接探测记录。", "",
           "| 配置 | 图和案例 | 状态 | 实际报告 |", "|---|---|---|---|"]
    for profile, directory in chosen.items():
        manifest = read(directory / "manifest.json")
        ids = ", ".join(f"{d['id']} ({d['n']}/{d['m']})" for d in manifest["datasets"])
        out.append(f"| {profile} | {ids}；{len(manifest['cases'])}案例 | complete | [报告](../results/{directory.name}/report.md) |")
    out += ["", "smoke 每配置1次、随机失效3次；quick 四图每配置3次、p=1/2/4、B=32、随机失效10次；granularity 同一 SBM200、p=2、B=1/8/32/128、每配置3次；diagnostic 单独记录真实任务时间并打乱全部起点，不混入主计时。", "",
            "```powershell", '.\\.venv\\Scripts\\python.exe -m parallel_bc demo --profile smoke',
            '.\\.venv\\Scripts\\python.exe -m parallel_bc benchmark --config configs/quick.yaml',
            '.\\.venv\\Scripts\\python.exe -m parallel_bc benchmark --config configs/granularity.yaml',
            '.\\.venv\\Scripts\\python.exe -m parallel_bc benchmark --config configs/diagnostic.yaml', "```", "",
            "## quick 的实际数值", "", "下表选择同图 p=2、B=32，时间为三次运行的中位数（秒）。S_total 与 S_steady 严格分开。", "",
            "| 图 | 模式 | total | steady | S_total | S_steady |", "|---|---|---:|---:|---:|---:|"]
    quick = chosen["quick"]
    summary = rows(quick / "timings_summary.csv")
    for r in sorted(summary, key=lambda r: (r["dataset_id"], r["mode"])):
        if r["mode"] == "serial" or r["p_used"] == "2":
            out.append("| " + " | ".join([r["dataset_id"], r["mode"], *[f"{float(r[k]):.6f}" for k in ("median_total", "median_steady", "speedup_total", "speedup_steady")]]) + " |")
    maximum = max(float(r["max_abs_error"]) for r in rows(quick / "timings_raw.csv"))
    out += ["", f"quick {len(rows(quick / 'timings_raw.csv'))}个计时案例全部成功，串并行全节点最大 raw 误差为 **{maximum:.6g}**；每张图 NetworkX 参照见各子目录 validation.json。"]
    if all(float(r["speedup_total"]) < 1 for r in summary if r["mode"] != "serial"):
        out += ["所有已测并行配置的 total 均慢于同核心串行。spawn启动耗时有直接记录；稳态部分有加速，但不能据此声称端到端加速。后台负载和温度等因素未独立控制，结论仅适用于本次规模和环境。"]
    out += ["", "合成图和真实图分开解释；total与steady分开报告。详细配置、原始重复值、正确性与失效结果见所选运行目录。", ""]
    for profile, directory in chosen.items():
        out += [f"- {profile} 完整清单：[manifest.json](../results/{directory.name}/manifest.json)，源码指纹一致：`{read(directory/'manifest.json')['source_sha256'] == fingerprint}`。"]
    target.write_text(rebase_links("\n".join(out), target), encoding="utf-8")
    (target.parent / "synthetic_result_index.json").write_text(json.dumps({k: str(v.relative_to(ROOT)) for k,v in chosen.items()}, indent=2), encoding="utf-8")
    print(json.dumps({k: str(v.relative_to(ROOT)) for k,v in chosen.items()}, ensure_ascii=False))



def rebase_links(text, output):
    """Keep old docs-relative links correct for the selected report location."""
    import os, re
    old_base = Path(__file__).resolve().parents[1] / "docs"
    def replace(match):
        url = match.group(2)
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", url) or url.startswith("#"):
            return match.group(0)
        file, mark, anchor = url.partition("#")
        destination = Path(file[3:]) if re.match(r"^\.\./[A-Za-z]:/", file) else old_base / file
        try:
            relative = Path(os.path.relpath(destination.resolve(), output.parent.resolve())).as_posix()
        except ValueError:
            relative = destination.resolve().as_posix()
        return match.group(1) + relative + (mark + anchor if mark else "") + match.group(3)
    return re.sub(r"(!?\[[^\]\n]*\]\()([^)\n]+)(\))", replace, text)

if __name__ == "__main__":
    main()
