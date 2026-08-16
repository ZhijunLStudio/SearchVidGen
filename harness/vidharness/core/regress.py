"""变体回归套件 —— check 任务的统一状态视图与执行器。

状态视图（vh regress）：对套件内每个任务给出
- 最新 run（manifest.finished_at 的最近者）及其完成态；
- 关键评分（段级/跨段维度均分）；
- **配置漂移检测**：run 的 config.yaml 快照 vs 当前任务文件——文件改动后
  旧 run 即失配，漂移是"需要重跑"的硬信号。

执行模式（vh regress --run）：逐任务跑（跳过已完成、续跑未完成，
与 bench 格级断点续跑同语义）；ref2va 需 h3int8 环境、其余需 torch 环境
（分两轮跑，见 tasks/regression.yaml 注释）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_regression_list(spec_path: Path) -> List[str]:
    import yaml
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    tasks = (spec or {}).get("tasks") or []
    if not isinstance(tasks, list) or not tasks:
        raise RuntimeError(f"{spec_path} 缺少 tasks 列表")
    # 路径约定：先按写入值解析；不存在则相对 spec 文件解析（两种惯例兼容）
    out = []
    for t in tasks:
        p = Path(t)
        out.append(str(p if p.exists() else spec_path.parent / p))
    return out


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _latest_finished_run(base_dir: Path, task_name: str,
                        match_config: Any = None) -> Optional[Dict[str, Any]]:
    """最新完成 run；match_config 给定时优先返回配置快照与之一致的 run
    （bench 格的配置含矩阵覆写，不是 check 任务的回归对象；--label 的
    普通 run 配置一致，仍算回归对象）。"""
    import yaml
    task_dir = base_dir / task_name
    if not task_dir.exists():
        return None
    best: Optional[Dict[str, Any]] = None
    best_matching: Optional[Dict[str, Any]] = None
    for run_dir in task_dir.iterdir():
        if not run_dir.is_dir():
            continue
        m = _load_json(run_dir / "manifest.json")
        if not isinstance(m, dict) or not m.get("finished_at"):  # type: ignore[union-attr]
            continue
        cand = {"run_id": run_dir.name, "dir": run_dir, "manifest": m}
        if best is None or (m.get("finished_at") or "") > (best["manifest"].get("finished_at") or ""):  # type: ignore[union-attr]
            best = cand
        if match_config is not None:
            try:
                snap = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
            except Exception:
                snap = None
            if snap == match_config and (
                    best_matching is None or (m.get("finished_at") or "") >
                    (best_matching["manifest"].get("finished_at") or "")):  # type: ignore[union-attr]
                best_matching = cand
    return best_matching or best


def _key_scores(run_dir: Path) -> Dict[str, Any]:
    scores: Dict[str, Any] = {}
    for f in (run_dir / "eval").glob("*.json"):
        data = _load_json(f)
        if not isinstance(data, list):
            continue
        dims: Dict[str, List[float]] = {}
        for r in data:
            if not isinstance(r, dict):
                continue
            for k, v in r.get("scores", {}).items():
                dims.setdefault(k, []).append(float(v))
        if dims:
            scores[f.stem] = {k: round(sum(v) / len(v), 2) for k, v in dims.items() if v}
    return scores


def config_drifted(run_dir: Path, task_file: Path) -> Optional[str]:
    """run 快照 vs 当前任务文件：一致返回 None，漂移返回说明。"""
    import yaml
    snap = run_dir / "config.yaml"
    if not snap.exists():
        return "无快照（8-16 前旧 run）"
    if not task_file.exists():
        return "任务文件缺失"
    try:
        if yaml.safe_load(snap.read_text(encoding="utf-8")) == \
                yaml.safe_load(task_file.read_text(encoding="utf-8")):
            return None
    except Exception:
        pass
    return "配置漂移（快照 ≠ 当前任务文件，需重跑）"


def status(base_dir: Path, spec_path: Path) -> List[Dict[str, Any]]:
    """套件状态表（每任务一行）。"""
    rows = []
    for task_file in load_regression_list(spec_path):
        task_path = Path(task_file)
        try:
            import yaml
            task_name = (yaml.safe_load(task_path.read_text(encoding="utf-8"))
                         or {}).get("task_name", task_path.stem)
        except Exception:
            task_name = task_path.stem
        try:
            import yaml
            task_cfg = yaml.safe_load(task_path.read_text(encoding="utf-8"))
        except Exception:
            task_cfg = None
        run = _latest_finished_run(base_dir, task_name, match_config=task_cfg)
        row: Dict[str, Any] = {
            "task_file": task_file, "task_name": task_name,
            "run_id": run["run_id"] if run else None,
            "scores": _key_scores(run["dir"]) if run else {},
            "drift": config_drifted(run["dir"], task_path) if run else None,
        }
        rows.append(row)
    return rows


def render_status(rows: List[Dict[str, Any]]) -> str:
    lines = ["| 任务 | 最新 run | 关键评分 | 配置 |", "|---|---|---|---|"]
    for r in rows:
        if not r["run_id"]:
            lines.append(f"| {r['task_file']} | **未跑过** | - | - |")
            continue
        seg = r["scores"].get("segments", {})
        cross = r["scores"].get("cross_consistency", {})
        key = " / ".join(f"{k} {v}" for k, v in {**seg, **cross}.items()) or "-"
        drift = "✅ 一致" if not r["drift"] else f"⚠️ {r['drift']}"
        lines.append(f"| {r['task_file']} | {r['run_id']} | {key} | {drift} |")
    return "\n".join(lines)
