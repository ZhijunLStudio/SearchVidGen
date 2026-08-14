"""实验报告生成器：扫描 experiments/ 生成对比报告（HTML + 摘要 JSON）。

- 单实验：各阶段产物、评测分数、耗时、成本、重试次数
- 多实验：同任务横向对比（不同模型/参数 → 指标差异），即"基准"的核心产出
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def collect(base_dir: Path, task: str) -> List[Dict[str, Any]]:
    """扫描 <base>/<task>/*/manifest.json，汇总为实验列表。"""
    task_dir = Path(base_dir) / task
    runs = []
    if not task_dir.exists():
        return runs
    for run_dir in sorted(task_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        manifest = _load_json(run_dir / "manifest.json")
        if not manifest:
            continue
        evals = {}
        for f in (run_dir / "eval").glob("*.json"):
            evals[f.stem] = _load_json(f)
        # 汇总评测分数
        scores: Dict[str, List[float]] = {}
        passed = 0
        total = 0
        for stage, records in evals.items():
            for rec in records if isinstance(records, list) else []:
                if not isinstance(rec, dict):
                    continue
                for k, v in rec.get("scores", {}).items():
                    scores.setdefault(k, []).append(v)
                total += 1
                if rec.get("passed"):
                    passed += 1
        runs.append({
            "run_id": manifest.get("run_id", run_dir.name),
            "dir": str(run_dir),
            "created_at": manifest.get("created_at"),
            "finished_at": manifest.get("finished_at"),
            "total_elapsed_s": manifest.get("total_elapsed_s", 0),
            "total_cost_usd": manifest.get("total_cost_usd", 0),
            "stages": {k: len(v) for k, v in manifest.get("stages", {}).items()},
            "retries": manifest.get("retries", {}),
            "scores": {k: round(sum(v) / len(v), 2) for k, v in scores.items() if v},
            "passed_rate": round(passed / total, 2) if total else None,
            "final_video": str(Path(run_dir) / "final" / "final_video.mp4"),
        })
    return runs


def render_html(runs: List[Dict[str, Any]], out: Path) -> Path:
    rows = ""
    for r in runs:
        score_cells = "".join(
            f"<td>{k}<br><b>{v}</b></td>" for k, v in r.get("scores", {}).items())
        rows += f"""
        <tr>
          <td>{r['run_id']}</td>
          <td>{r['created_at']}</td>
          <td>{r['total_elapsed_s'] / 60:.1f} min</td>
          <td>${r['total_cost_usd']:.4f}</td>
          <td>{r['passed_rate']}</td>
          <td>{json.dumps(r.get('retries', {}), ensure_ascii=False)}</td>
          {score_cells}
          <td><a href="{r['dir']}">产物目录</a></td>
        </tr>"""
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>VidHarness 实验报告</title>
<style>
body {{ font-family: system-ui; margin: 2em; }}
table {{ border-collapse: collapse; width: 100%; }}
td, th {{ border: 1px solid #ccc; padding: 6px 10px; font-size: 14px; text-align: center; }}
th {{ background: #f5f5f5; }}
</style></head><body>
<h2>VidHarness 实验对比</h2>
<p>生成时间：<span id="t"></span>　实验数：{len(runs)}</p>
<table>
<tr><th>Run</th><th>创建时间</th><th>总耗时</th><th>API 成本</th><th>通过率</th><th>重试</th><th>各维度均分</th><th>产物</th></tr>
{rows}
</table>
<script>document.getElementById('t').textContent = new Date().toLocaleString();</script>
</body></html>"""
    out.write_text(html, encoding="utf-8")
    return out


def report(base_dir: Path, task: str, out_html: Path) -> Dict[str, Any]:
    runs = collect(base_dir, task)
    render_html(runs, out_html)
    return {"task": task, "runs": len(runs), "html": str(out_html)}
