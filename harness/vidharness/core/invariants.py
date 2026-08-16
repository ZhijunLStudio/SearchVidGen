"""运行时不变量 —— 对齐 deepseek-harness 的 package-owned invariant 原则。

只断言**多个观测之间的关系**（manifest ↔ 文件系统 ↔ 事件流），不断言方法/类
存在性。证据完整性的底线：任何一方与另一方不一致都说明实验不可信，
必须响亮报出而不是静默容忍。

检查清单（check_experiment）：
1. manifest 总额 == 各产物 meta 累计（成本/耗时）；
2. manifest 记录的每个产物文件真实存在；
3. eval/*.json 可解析且每条记录是 dict；
4. manifest.config_file 指向的配置快照存在且 sha256 与事件一致；
5. retries 计数非负整数；
6. 事件流完整时：重放投影与 manifest 一致（产物条目/总额/重试/query/配置）；
7. 事件流完整时：重放评测明细与 eval/*.json 集合一致。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from .experiment import _read_events, replay_events, replay_eval_records

_EPS = 1e-6


def _load_manifest(root: Path) -> Dict[str, Any]:
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))


def _load_eval_file(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    return [r for r in data if isinstance(r, dict)]


def check_experiment(root: Path) -> List[str]:
    """返回违规描述列表（空列表 = 通过）。"""
    root = Path(root)
    v: List[str] = []
    mf = root / "manifest.json"
    if not mf.exists():
        return [f"缺少 manifest.json: {mf}"]
    try:
        manifest = _load_manifest(root)
    except Exception as e:
        return [f"manifest.json 无法解析: {e}"]

    # 1) 总额 == 产物累计（关系不变量：投影与其来源）
    total_cost = sum(
        float(a.get("meta", {}).get("cost_usd", 0.0))
        for arts in manifest.get("stages", {}).values() for a in arts)
    total_elapsed = sum(
        float(a.get("meta", {}).get("elapsed_s", 0.0))
        for arts in manifest.get("stages", {}).values() for a in arts)
    if abs(total_cost - float(manifest.get("total_cost_usd", 0.0))) > _EPS:
        v.append(f"total_cost_usd={manifest.get('total_cost_usd')} ≠ 产物累计 {total_cost}")
    if abs(total_elapsed - float(manifest.get("total_elapsed_s", 0.0))) > _EPS:
        v.append(f"total_elapsed_s={manifest.get('total_elapsed_s')} ≠ 产物累计 {total_elapsed}")

    # 2) 每个产物文件真实存在（关系不变量：投影与其指向的文件）
    for stage, arts in manifest.get("stages", {}).items():
        for i, a in enumerate(arts):
            p = a.get("path", "")
            if not p:
                v.append(f"{stage}[{i}] 产物条目缺少 path")
            elif not Path(p).exists():
                v.append(f"产物文件缺失: {p}")

    # 3) eval 文件可解析、记录是 dict（judge_* 原始输出是旧版布局，给迁移提示）
    eval_dir = root / "eval"
    for f in sorted(eval_dir.glob("*.json")) if eval_dir.exists() else []:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            v.append(f"评测文件无法解析: {f.name}: {e}")
            continue
        if not isinstance(data, list) or any(not isinstance(r, dict) for r in data):
            hint = ("（旧版裁判原始输出；新版已迁至 artifacts/judge/）"
                    if f.name.startswith("judge_") else "")
            v.append(f"评测文件记录必须是 dict 列表: {f.name}{hint}")

    # 4) 配置快照存在且未被动过
    if manifest.get("config_file"):
        cfg_path = root / manifest["config_file"]
        if not cfg_path.exists():
            v.append(f"配置快照缺失: {cfg_path}")
        elif manifest.get("config_sha256"):
            sha = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
            if sha != manifest["config_sha256"]:
                v.append(f"配置快照被修改（sha256 不一致）: {cfg_path}")

    # 5) retries 计数合法性
    for stage, n in (manifest.get("retries") or {}).items():
        if not isinstance(n, int) or n < 0:
            v.append(f"retries['{stage}'] 应为非负整数，得到 {n!r}")

    # 6/7) 事件流完整时的重放一致性
    events = root / "events.jsonl"
    if events.exists():
        proj = replay_events(events)
        if proj is not None:
            raw_events = _read_events(events)
            # 阶段生命周期配对：已 finalize 的 run，每个 stage.started 必须有 finished
            if proj.get("finished_at"):
                started = {ev.get("stage") for ev in raw_events
                           if ev.get("type") == "stage.started"}
                finished = {ev.get("stage") for ev in raw_events
                            if ev.get("type") == "stage.finished"}
                for s in sorted(started - finished):
                    v.append(f"阶段 '{s}' 有 stage.started 但无 stage.finished")
            for stage, arts in proj.get("stages", {}).items():
                marts = manifest.get("stages", {}).get(stage, [])
                if len(arts) != len(marts):
                    v.append(f"事件重放 {stage} 条目数 {len(arts)} ≠ manifest {len(marts)}")
                    continue
                for i, (pa, ma) in enumerate(zip(arts, marts)):
                    if pa.get("path") != ma.get("path"):
                        v.append(f"事件重放 {stage}[{i}] 路径与 manifest 不一致")
            if abs(float(proj.get("total_cost_usd", 0)) - total_cost) > _EPS:
                v.append("事件重放 total_cost_usd ≠ manifest")
            if abs(float(proj.get("total_elapsed_s", 0)) - total_elapsed) > _EPS:
                v.append("事件重放 total_elapsed_s ≠ manifest")
            if proj.get("retries", {}) != manifest.get("retries", {}):
                v.append("事件重放 retries ≠ manifest")
            if proj.get("query") != manifest.get("query"):
                v.append("事件重放 query ≠ manifest")
            # 评测明细：重放集合 == 文件集合（坏文件已在 #3 报告，这里跳过）
            replay_records = replay_eval_records(events)
            for f in sorted((root / "eval").glob("*.json")) if eval_dir.exists() else []:
                try:
                    file_records = _load_eval_file(f)
                except Exception:
                    continue
                stage = f.stem
                replayed = replay_records.get(stage, [])
                file_set = {json.dumps(r, ensure_ascii=False, sort_keys=True) for r in file_records}
                replay_set = {json.dumps(r, ensure_ascii=False, sort_keys=True) for r in replayed}
                if file_set != replay_set:
                    v.append(f"评测明细重放与文件不一致: {f.name}")
    return v
