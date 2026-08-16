"""leaderboard 基线导出 —— 公开 leaderboard 的雏形。

从 report.collect()（聚合的唯一正源）取数，产出两份文件：
- <out>/<task>.json：机器可读基线（可入库 git 追踪，回归可 diff）
- <out>/<task>.md：人读表格

与上次基线对比给出增量（新增 run / 消失 run），供评审与回归检测。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .report import collect


def _calibration_offsets(calib_dir: Path) -> Dict[str, Dict[str, float]]:
    """读取校准目录：{judge_b 名称: {维度: 偏移(a-b)}}，只取 n>=3 的维度。

    偏移语义：a = b + offset → 把 judge_b 的评分换算到 judge_a 口径
    （b_cal = b + offset）。
    """
    out: Dict[str, Dict[str, float]] = {}
    for f in sorted(calib_dir.glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(d, dict) or not d.get("judge_b"):
            continue
        dims = {}
        for dim, v in (d.get("dims") or {}).items():
            if isinstance(v, dict) and v.get("n", 0) >= 3:
                dims[dim] = float(v["mean_offset_a_minus_b"])
        if dims:
            out[str(d["judge_b"])] = dims
    return out


def build(base_dir: Path, task: str, calibrate: bool = False,
          calib_dir: Optional[Path] = None) -> Dict[str, Any]:
    """当前基线数据（不落盘）。

    calibrate=True 时：对使用过校准对象裁判的 run，按 calibration 的
    维度偏移（n>=3）把该裁判的评分换算到主裁判口径，标注 calibrated。
    """
    runs = collect(base_dir, task)
    offsets_by_judge = _calibration_offsets(calib_dir) if (calibrate and calib_dir) else {}
    rows: List[Dict[str, Any]] = []
    for r in runs:
        calibrated = False
        scores_cal = dict(r["scores"])
        for judge_b, dim_offsets in offsets_by_judge.items():
            used = any(judge_b in j for j in r.get("judge_adapters", []))
            if not used:
                continue
            for dim, off in dim_offsets.items():
                if dim in scores_cal:
                    scores_cal[dim] = round(scores_cal[dim] + off, 2)
                    calibrated = True
        rows.append({
            "run_id": r["run_id"],
            "bench_cell": r["bench_cell"],
            "chain_mode": r["chain_mode"],
            "models": r["models"],
            "judge_adapters": r["judge_adapters"],
            "scores": r["scores"],
            "stage_scores": r["stage_scores"],
            "calibrated": calibrated,
            "scores_calibrated": scores_cal if calibrated else r["scores"],
            "passed_rate": r["passed_rate"],
            "total_cost_usd": r["total_cost_usd"],
            "total_elapsed_s": r["total_elapsed_s"],
            "local_gpu_hours": r["local_gpu_hours"],
            "created_at": r["created_at"],
            "finished_at": r["finished_at"],
        })
    return {"task": task, "updated_at": datetime.now().isoformat(),
            "run_count": len(rows), "runs": rows}


def _load_baseline(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def export(base_dir: Path, task: str, out_dir: Path, calibrate: bool = False) -> Tuple[Path, Path, Dict[str, Any]]:
    """导出基线并返回 (json 路径, md 路径, 与上次基线的增量 diff)。"""
    data = build(base_dir, task, calibrate=calibrate,
                 calib_dir=out_dir.parent / "calibration")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{task}.json"
    md_path = out_dir / f"{task}.md"

    baseline = _load_baseline(json_path)
    old_ids = {r.get("run_id") for r in baseline.get("runs", [])}
    new_ids = {r["run_id"] for r in data["runs"]}
    diff = {
        "new_runs": sorted(new_ids - old_ids),
        "removed_runs": sorted(old_ids - new_ids),
    }

    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_render_md(data), encoding="utf-8")
    return json_path, md_path, diff


def export_all(base_dir: Path, out_dir: Path, calibrate: bool = False) -> Dict[str, Any]:
    """导出 experiments 下所有任务的基线并渲染 index.html（公开页面雏形）。"""
    exported = {}
    tasks = sorted(d.name for d in Path(base_dir).iterdir()
                   if d.is_dir() and any((d / r / "manifest.json").exists()
                                         for r in d.iterdir() if r.is_dir()))
    for task in tasks:
        _, _, diff = export(base_dir, task, out_dir, calibrate=calibrate)
        exported[task] = diff
    index = render_index(out_dir)
    return {"tasks": exported, "index": str(index)}


def _latest_run(data: Dict[str, Any]) -> Dict[str, Any]:
    rows = sorted(data.get("runs", []), key=lambda r: r.get("finished_at") or "", reverse=True)
    return rows[0] if rows else {}


def _load_calibrations(out_dir: Path) -> List[Dict[str, Any]]:
    calib_dir = out_dir.parent / "calibration"
    cals = []
    for f in sorted(calib_dir.glob("*.json")) if calib_dir.exists() else []:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(d, dict) and "dims" in d:
                cals.append(d)
        except Exception:
            continue
    return cals


def render_index(out_dir: Path) -> Path:
    """聚合总览页：每任务最新 run 概览 + 混用裁判警告 + 校准数据摘要。"""
    import html as _html
    esc = _html.escape
    tasks = sorted(f.stem for f in out_dir.glob("*.json") if f.stem != "index")
    rows = ""
    for task in tasks:
        data = _load_baseline(out_dir / f"{task}.json")
        if not data.get("runs"):
            rows += f"<tr><td>{esc(task)}</td><td colspan=7>（无基线数据）</td></tr>"
            continue
        r = _latest_run(data)
        seg = r.get("stage_scores", {}).get("segments", {})
        cross = r.get("stage_scores", {}).get("cross_consistency", {})
        key = " / ".join(f"{k} {v}" for k, v in {**seg, **cross}.items()) or "-"
        judges = ", ".join(r.get("judge_adapters") or ["-"])
        models = ", ".join(r.get("models") or ["-"])
        rows += (f"<tr><td><a href='{esc(task)}.md'>{esc(task)}</a></td>"
                 f"<td>{data['run_count']}</td><td>{esc(str(r.get('run_id', '-')))}</td>"
                 f"<td>{esc(models)}</td><td>{esc(judges)}</td>"
                 f"<td>{esc(key)}</td>"
                 f"<td>${r.get('total_cost_usd', 0):.4f}</td></tr>")
    # 混用裁判总警告
    mixed = []
    for task in tasks:
        data = _load_baseline(out_dir / f"{task}.json")
        task_judges = {j for r in data.get("runs", []) for j in r.get("judge_adapters", []) if j}
        if len(task_judges) > 1:
            mixed.append(f"{task}: {sorted(task_judges)}")
    mixed_note = ""
    if mixed:
        mixed_note = ("<p>⚠️ 混用裁判的任务：" + esc("；".join(mixed)) +
                      "——评分尺度不可直接对比（E24/E25），请参考校准数据。</p>")
    # 校准摘要
    calib_rows = ""
    for c in _load_calibrations(out_dir):
        for dim, v in c.get("dims", {}).items():
            calib_rows += (f"<tr><td>{esc(c['judge_a'])} vs {esc(c['judge_b'])}</td>"
                           f"<td>{esc(dim)}</td><td>{v['mean_offset_a_minus_b']}</td>"
                           f"<td>n={v['n']}</td></tr>")
    calib_block = ""
    if calib_rows:
        calib_block = ("<h2>跨裁判校准（calibration/）</h2>"
                       "<table><tr><th>裁判对</th><th>维度</th><th>偏移(a-b)</th><th>样本</th></tr>"
                       f"{calib_rows}</table>")
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>VidHarness Leaderboard</title>
<style>
body {{ font-family: system-ui; margin: 2em; }}
table {{ border-collapse: collapse; width: 100%; }}
td, th {{ border: 1px solid #ccc; padding: 6px 10px; font-size: 14px; text-align: center; }}
th {{ background: #f5f5f5; }}
a {{ color: #2563eb; }}
</style></head><body>
<h1>VidHarness Leaderboard</h1>
<p>生成时间：<span id="t"></span>　任务数：{len(tasks)}</p>
{mixed_note}
<h2>任务总览（每任务最新 run）</h2>
<table>
<tr><th>任务</th><th>runs</th><th>最新 run</th><th>模型</th><th>裁判</th><th>关键评分</th><th>成本(USD)</th></tr>
{rows}
</table>
{calib_block}
<script>document.getElementById('t').textContent = new Date().toLocaleString();</script>
</body></html>"""
    out = out_dir / "index.html"
    out.write_text(html, encoding="utf-8")
    return out


def _fmt_scores(scores: Dict[str, float]) -> str:
    return " / ".join(f"{k} {v}" for k, v in scores.items()) or "-"


def _render_md(data: Dict[str, Any]) -> str:
    rows = sorted(data["runs"], key=lambda r: r.get("created_at") or "", reverse=True)
    lines = [
        f"# Leaderboard: {data['task']}",
        "",
        f"更新：{data['updated_at']}　run 数：{data['run_count']}",
        "",
    ]
    judges_used = {j for r in rows for j in r.get("judge_adapters", []) if j}
    if len(judges_used) > 1:
        lines.append(
            f"> ⚠️ 本表混用裁判 {sorted(judges_used)}：评分尺度不可直接对比（E24），"
            f"请参考 calibration/ 校准数据。")
        lines.append("")
    if any(r.get("calibrated") for r in rows):
        lines.append("> 📐 标（校准）的评分已按 calibration/ 维度偏移（n≥3）"
                     "换算到主裁判口径（E30）。")
        lines.append("")
    lines += [
        "| Run | Bench 格 | 衔接 | 模型 | 裁判 | 各维度均分 | 通过率 | 成本(USD) | GPU时 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        cell = r.get("bench_cell") or "-"
        chain = r.get("chain_mode") or "-"
        models = ", ".join(r.get("models") or ["-"])
        judges = ", ".join(r.get("judge_adapters") or ["-"])
        scores = r.get("scores_calibrated", r.get("scores", {}))
        marker = "（校准）" if r.get("calibrated") else ""
        passed = r.get("passed_rate")
        passed = "-" if passed is None else f"{passed * 100:.0f}%"
        lines.append(
            f"| {r['run_id']} | {cell} | {chain} | {models} | {judges} | "
            f"{_fmt_scores(scores)}{marker} | {passed} | "
            f"{r['total_cost_usd']:.4f} | {r.get('local_gpu_hours') or '-'} |")
    lines.append("")
    return "\n".join(lines)
