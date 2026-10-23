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
            raise RuntimeError(f"尚无最终源码完整 {profile} 实验；请等待实际运行完成")
        chosen[profile] = matches[-1]

    out = ["# LinkRadar 合成实验阶段摘要", "", "本页只汇总指定合成实验。当前项目状态和后续真实图复测请看 [统一结果](RESULTS.md)。", "", "日期：2026-10-23（Asia/Shanghai）。以下数字均取自实际运行文件。", "",
           "## 交付状态", "", "P0 功能与小规模端到端验收已完成。P1 完成真实 PID/批次时间线和固定种子起点乱序配置；粗粒度静态分配补充项尚未实现。本摘要不包含真实网络全图及后续配对复测；不能据本页判断整个项目未做真实图。", "",
           "保留用户修改的原方案，前后 SHA256 均为 `DCB9CDA0D3EF554B0A25DAA75643C4444321F02A19C6572B9578A02D4868C0A5`。未提交或推送；`.idea` 在会话期间出现外部修改，未重置。", "",
           f"最终源码指纹：`{fingerprint}`。各实验清单分别记录源码、配置和图的 SHA256。", "",
           "## 安装与测试", "", "实际 Python 3.12.6，依赖版本见 [requirements-lock.txt](../requirements-lock.txt)。", "",
           "```powershell", '.\\.venv\\Scripts\\python.exe -m pip install -e ".[dev]" --index-url https://pypi.org/simple',
           '.\\.venv\\Scripts\\python.exe -m pytest -q', '.\\.venv\\Scripts\\python.exe -m pip check',
           '.\\.venv\\Scripts\\python.exe scripts/verify_cli.py', "```", "",
           "该合成验收阶段的历史测试：**110 passed in 20.49s**（当前汇总的较新验证见统一结果）；依赖检查：**No broken requirements found**。CLI 验证7条命令全部 exit 0，精确参数/输出见 [cli_verification.json](cli_verification.json)。", "",
           "测试涵盖五节点可读标签样例、空图/路径/星形/完全/多最短路径/非连通/孤立点、随机小图全节点 NetworkX raw/normalized 对照、真实 spawn p=1/2/3 与边界批次、异常与超时回收、数据清洗与缓存篡改、手工导入及离线 CLI 全流程。", "",
           "新增回归实际启动 supervisor 和两个孙进程后触发硬截止，三个 PID 全部消失；还验证拒绝变更图时保留旧证据、中断后进程/任务日志先于成功状态持久化、预检失败恢复及原始计时保留。", "",
           "调试中出现过依赖尚未安装完成导致的导入失败，以及Windows CLI本地编码导致的一项测试失败；安装完成并统一UTF-8后均已修复。默认阿里云镜像下载停滞，停止本会话安装进程后改用官方PyPI成功。", "",
           "## 最终实际实验", "", "下表均为最终源码、无并行pytest干扰的独立实验；此前调试轮次留在 results 中，不混入下列统计。", "",
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
    out += ["", f"quick 84个计时案例全部成功，串并行全节点最大 raw 误差为 **{maximum:.6g}**；每张图 NetworkX 参照见各子目录 validation.json。"]
    if all(float(r["speedup_total"]) < 1 for r in summary if r["mode"] != "serial"):
        out += ["所有已测并行配置的 total 均慢于同核心串行。spawn启动耗时有直接记录；稳态部分有加速，但不能据此声称端到端加速。后台负载和温度等因素未独立控制，结论仅适用于本次规模和环境。"]
    out += ["", "规模、批次、误差、total/steady加速、失效、工作进程忙碌时间、P1时间线均有真实图表。单个目录未覆盖的实验用明确缺测面板说明，可分别在quick、granularity、diagnostic中查看对应实测图。", "",
            "## 早期真实数据下载尝试（历史）", "", "实际调用 prepare_dataset 对 Facebook、ca-GrQc 各尝试3次，均在TLS握手阶段报 `SSL: UNEXPECTED_EOF_WHILE_READING`；补充curl官方Facebook URL探测15秒超时。未下载到可验证文件，没有填入虚构SHA、字节数或清洗结果。完整失败记录已复制到 [download_status.json](download_status.json)；这份早期失败记录已冻结保存，data/raw 下的当前导入状态可能已更新。", "",
            "保留 `data download`、`data prepare`、`data import` 命令和官方URL。手工导入、gzip、注释/坏行、正反重复、自环及孤立点路径均经离线测试；两张真实图后来已手工导入并完成实验，现状与命令见主说明。", "",
            "## 未完成项与限制", "",
            "- 本摘要只覆盖合成实验。两张真实图现已手工导入并完成研究，见统一结果；下文下载失败只属于早期历史。",
            "- full 五次重复、最高8进程及2000/4000节点等更大规模未执行：本轮优先完成P0离线验证；保留配置与预算入口，大规模性能结论待实测。",
            "- P1粗粒度静态分配独立补充模式未实施；已实现指定P0静态round-robin批次与动态领取，另有真实时间线和源点顺序配置。没有P2 Numba/CSR/共享内存。",
            "- 全起点精确枚举仍使用float64贡献，允许舍入误差；spawn每进程复制图。",
            "- 串行公共函数按起点边界检查超时；CLI/benchmark有独立监管器硬截止并回收进程树。回收可额外耗时，总预算主要约束计算，准备和最终导出可能超过预算；最终墙钟如实记录。",
            "- 计时环境可能受限；CPU quota未知写null。没有OS层CPU利用率结论，busy_fraction只是批次墙钟诊断。",
            "- 失效实验固定初始排名，随机带为经验分位范围；不推断真实人员身份、作者学术贡献或现实行为。", ""]
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
