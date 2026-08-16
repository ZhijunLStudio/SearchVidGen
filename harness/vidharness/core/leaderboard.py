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
from typing import Any, Dict, List, Tuple

from .report import collect


def build(base_dir: Path, task: str) -> Dict[str, Any]:
    """当前基线数据（不落盘）。"""
    runs = collect(base_dir, task)
    rows: List[Dict[str, Any]] = []
    for r in runs:
        rows.append({
            "run_id": r["run_id"],
            "bench_cell": r["bench_cell"],
            "chain_mode": r["chain_mode"],
            "models": r["models"],
            "judge_adapters": r["judge_adapters"],
            "scores": r["scores"],
            "stage_scores": r["stage_scores"],
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


def export(base_dir: Path, task: str, out_dir: Path) -> Tuple[Path, Path, Dict[str, Any]]:
    """导出基线并返回 (json 路径, md 路径, 与上次基线的增量 diff)。"""
    data = build(base_dir, task)
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
    judges_used = {j for r in rows for j in r.get("judge_adapters", [])}
    if len(judges_used) > 1:
        lines.append(
            f"> ⚠️ 本表混用裁判 {sorted(judges_used)}：评分尺度不可直接对比（E24），"
            f"请参考 calibration/ 校准数据。")
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
        passed = r.get("passed_rate")
        passed = "-" if passed is None else f"{passed * 100:.0f}%"
        lines.append(
            f"| {r['run_id']} | {cell} | {chain} | {models} | {judges} | "
            f"{_fmt_scores(r.get('scores', {}))} | {passed} | "
            f"{r['total_cost_usd']:.4f} | {r.get('local_gpu_hours') or '-'} |")
    lines.append("")
    return "\n".join(lines)
