"""成本报表 —— 护城河"成本统计"的跨任务聚合视图。

从 report.collect()（聚合唯一正源）取数，按任务汇总：
API 成本（total_cost_usd）、GPU 卡时、估算 GPU 成本、总成本。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .report import collect


def build_cost_report(base_dir: Path, gpu_price_usd_per_hour: float = 1.2) -> Dict[str, Any]:
    # 注意：d.iterdir() 返回全路径，r 已是 run 目录本身（勿再 d / r）
    tasks = sorted(d.name for d in Path(base_dir).iterdir()
                   if d.is_dir() and any((r / "manifest.json").exists()
                                         for r in d.iterdir() if r.is_dir()))
    rows: List[Dict[str, Any]] = []
    for task in tasks:
        runs = collect(base_dir, task)
        if not runs:
            continue
        api_cost = round(sum(r["total_cost_usd"] for r in runs), 4)
        gpu_hours = round(sum(r.get("local_gpu_hours") or 0 for r in runs), 3)
        gpu_cost = round(gpu_hours * gpu_price_usd_per_hour, 3)   # 与 finalize 同口径
        rows.append({
            "task": task,
            "runs": len(runs),
            "api_cost_usd": api_cost,
            "gpu_hours": gpu_hours,
            "gpu_cost_usd_est": gpu_cost,
            "total_usd_est": round(api_cost + gpu_cost, 3),
        })
    rows.sort(key=lambda r: -r["total_usd_est"])
    return {"updated_at": datetime.now().isoformat(),
            "tasks": rows,
            "totals": {
                "api_cost_usd": round(sum(r["api_cost_usd"] for r in rows), 3),
                "gpu_hours": round(sum(r["gpu_hours"] for r in rows), 3),
                "total_usd_est": round(sum(r["total_usd_est"] for r in rows), 3),
            }}


def render_cost_table(data: Dict[str, Any]) -> str:
    lines = ["| 任务 | runs | API 成本(USD) | GPU 卡时 | GPU 成本估(USD) | 总计估(USD) |",
             "|---|---|---|---|---|---|"]
    for r in data["tasks"]:
        lines.append(f"| {r['task']} | {r['runs']} | {r['api_cost_usd']:.4f} | "
                     f"{r['gpu_hours']} | {r['gpu_cost_usd_est']:.3f} | {r['total_usd_est']:.3f} |")
    t = data["totals"]
    lines.append(f"| **总计** | | {t['api_cost_usd']:.4f} | {t['gpu_hours']} | | "
                 f"**{t['total_usd_est']:.3f}** |")
    return "\n".join(lines)


def report_costs(base_dir: Path, out: Path, gpu_price_usd_per_hour: float = 1.2) -> Dict[str, Any]:
    data = build_cost_report(base_dir, gpu_price_usd_per_hour)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data
