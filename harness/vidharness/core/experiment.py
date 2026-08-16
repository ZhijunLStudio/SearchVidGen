"""实验管理：每次运行 = 一个 Experiment。

目录结构：
experiments/<task>/<run_id>/
├── events.jsonl           # 追加式事件流（权威记录：崩溃后可重放重建 manifest）
├── manifest.json          # 事件流的投影（当前状态，供脚本/报告快速读取）
├── config.yaml            # 有效任务配置快照（可重建）
├── artifacts/<stage>/     # 各阶段产物（含 .meta.json）
├── eval/<stage>.json      # 评测明细
└── final/                # 成片与报告

事件溯源（对齐 deepseek-harness 的"模型可见 ⟺ 日志"）：
一切影响实验状态的变更先以 append-only 事件落盘，manifest 只是投影。
进程在任何点崩溃，重放 events.jsonl 都能恢复出等价状态。
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..seams import Artifact, ArtifactMeta

EVENT_VERSION = 1


def _read_events(path: Path) -> List[Dict[str, Any]]:
    """读取事件流；坏行跳过（追加式文件的尾部可能因崩溃截断）。"""
    events: List[Dict[str, Any]] = []
    if not path.exists():
        return events
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
            if isinstance(rec, dict) and "type" in rec:
                events.append(rec)
        except Exception:
            continue
    return events


def _merge_eval_records(existing: List[Dict[str, Any]],
                        incoming: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """按记录内容去重合并评测记录；返回 (合并结果, 本次新增)。"""
    def key(r: Dict[str, Any]) -> str:
        return json.dumps(r, ensure_ascii=False, sort_keys=True)
    existing_keys = {key(r) for r in existing if isinstance(r, dict)}
    merged = list(existing)
    added: List[Dict[str, Any]] = []
    for r in incoming:
        if isinstance(r, dict) and key(r) not in existing_keys:
            merged.append(r)
            existing_keys.add(key(r))
            added.append(r)
    return merged, added


def replay_events(path: Path) -> Optional[Dict[str, Any]]:
    """重放事件流，重建 manifest 投影（不含评测明细——评测明细在 eval/*.json）。

    返回 None 表示事件流不完整（缺少 run.created 事件）——此时 manifest
    仍是权威，调用方不得用重放结果覆盖它。
    """
    events = _read_events(path)
    if not events or events[0].get("type") != "run.created":
        return None
    head = events[0]
    manifest: Dict[str, Any] = {
        "task": head.get("task", ""),
        "run_id": head.get("run_id", ""),
        "created_at": head.get("created_at", ""),
        "stages": {},
        "total_cost_usd": 0.0,
        "total_elapsed_s": 0.0,
    }
    retries: Dict[str, int] = {}
    for ev in events[1:]:
        t = ev.get("type")
        if t == "artifact.saved":
            stage = ev["stage"]
            entry = ev["entry"]
            manifest["stages"].setdefault(stage, []).append(entry)
            meta = entry.get("meta", {})
            manifest["total_cost_usd"] += float(meta.get("cost_usd", 0.0))
            manifest["total_elapsed_s"] += float(meta.get("elapsed_s", 0.0))
        elif t == "retry":
            retries[ev["stage"]] = retries.get(ev["stage"], 0) + 1
        elif t == "query.bound":
            manifest["query"] = ev["query"]
        elif t == "config.snapshotted":
            manifest["config_file"] = ev.get("path", "config.yaml")
            manifest["config_sha256"] = ev.get("sha256", "")
        elif t == "manifest.set":
            manifest[ev["key"]] = ev["value"]
        elif t == "finalized":
            for k in ("finished_at", "local_gpu_hours", "local_gpu_cost_usd_est",
                      "total_cost_usd_all"):
                if k in ev:
                    manifest[k] = ev[k]
    if retries:
        manifest["retries"] = retries
    return manifest


def replay_eval_records(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """重放事件流，重建各 stage 的评测明细（供不变量比对 eval/*.json）。"""
    records: Dict[str, List[Dict[str, Any]]] = {}
    for ev in _read_events(path):
        if ev.get("type") == "eval.saved":
            stage = ev["stage"]
            merged, _ = _merge_eval_records(records.get(stage, []), [ev["record"]])
            records[stage] = merged
    return records


class Experiment:
    def __init__(self, task: str, base_dir: Path, run_id: Optional[str] = None):
        self.task = task
        self.base_dir = Path(base_dir)
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        self.root = Path(base_dir) / task / self.run_id
        self.artifacts_dir = self.root / "artifacts"
        self.eval_dir = self.root / "eval"
        self.final_dir = self.root / "final"
        for d in (self.artifacts_dir, self.eval_dir, self.final_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.events_path = self.root / "events.jsonl"
        self.events_complete = False   # 事件流是否自 run.created 起完整（可作权威）

        # 1) 事件流完整 → 重放是权威，manifest 只是投影（崩溃恢复）
        if self.events_path.exists():
            proj = replay_events(self.events_path)
            if proj is not None:
                self.manifest = proj
                self.events_complete = True
                return
        # 2) 否则读 manifest（老 run 无事件流，或事件流不完整）
        existing = self.root / "manifest.json"
        if existing.exists():
            try:
                self.manifest = json.loads(existing.read_text(encoding="utf-8"))
                # 老 run（有 manifest 无完整事件流）：manifest 保持权威
                self.events_complete = False
                return
            except Exception:
                pass
        # 3) 全新 run：事件流从 run.created 开始，此后为权威
        self.manifest: Dict[str, Any] = {
            "task": task,
            "run_id": self.run_id,
            "created_at": datetime.now().isoformat(),
            "stages": {},
            "total_cost_usd": 0.0,
            "total_elapsed_s": 0.0,
        }
        self._emit("run.created", task=task, run_id=self.run_id,
                   created_at=self.manifest["created_at"])
        self.events_complete = True

    # ---- 事件流 ----
    def _emit(self, type: str, **payload: Any) -> None:
        """追加一条事件（事件先于投影落盘：崩溃后重放可恢复）。"""
        rec = {"ts": time.time(), "type": type, "v": EVENT_VERSION, **payload}
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def set_meta(self, key: str, value: Any) -> None:
        """写入 manifest 元信息（经事件流，重放可恢复）。"""
        self._emit("manifest.set", key=key, value=value)
        self.manifest[key] = value

    def record_retry(self, stage: str) -> None:
        """记录一次失败重试（经事件流，重放可恢复计数）。"""
        self._emit("retry", stage=stage)
        self.manifest.setdefault("retries", {})
        self.manifest["retries"][stage] = self.manifest["retries"].get(stage, 0) + 1

    # ---- 产物存取 ----
    def save_artifact(self, stage: str, artifact: Artifact, name: Optional[str] = None) -> Path:
        """把产物落进 artifacts/<stage>/ 并记录 manifest。"""
        stage_dir = self.artifacts_dir / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        if name:
            target = stage_dir / f"{name}{artifact.path.suffix or ''}"
        else:
            target = stage_dir / artifact.path.name
        if artifact.path.resolve() != target.resolve():
            shutil.copy2(artifact.path, target)
        artifact.path = target
        (target.parent / (target.name + ".meta.json")).write_text(
            artifact.meta.to_json(), encoding="utf-8"
        )
        self._emit("artifact.saved", stage=stage, entry=artifact.asdict())
        entry = self.manifest["stages"].setdefault(stage, [])
        entry.append(artifact.asdict())
        self.manifest["total_cost_usd"] += artifact.meta.cost_usd
        self.manifest["total_elapsed_s"] += artifact.meta.elapsed_s
        self._flush()
        return target

    def save_eval(self, stage: str, results: List[Dict[str, Any]]):
        """按 (stage, artifact, attempt) 合并写入，避免多段评测互相覆盖。"""
        path = self.eval_dir / f"{stage}.json"
        merged: List[Dict[str, Any]] = []
        if path.exists():
            try:
                merged = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                merged = []
        merged, added = _merge_eval_records(merged, results)
        for r in added:
            self._emit("eval.saved", stage=stage, record=r)
        path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        self._flush()

    def find_existing(self, stage: str, name: Optional[str] = None) -> Optional[Artifact]:
        """断点续跑：按 stage+name 找已有产物。"""
        stage_dir = self.artifacts_dir / stage
        if not stage_dir.exists():
            return None
        if name is not None:
            candidates = list(stage_dir.glob(f"{name}*"))
            candidates = [c for c in candidates if not str(c).endswith(".meta.json")]
        else:
            candidates = [c for c in stage_dir.iterdir() if not str(c).endswith(".meta.json")]
        if not candidates:
            return None
        path = candidates[0]
        meta_path = Path(str(path) + ".meta.json")
        meta = ArtifactMeta()
        if meta_path.exists():
            meta = ArtifactMeta(**json.loads(meta_path.read_text(encoding="utf-8")))
        kind = path.suffix.lstrip(".")
        payload = {}
        if kind == "json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
        return Artifact(kind=kind, path=path, meta=meta, payload=payload)

    def bind_query(self, query: str) -> None:
        """记录实验变量 query，并守卫续跑一致性。

        剧本按 query 生成；续跑换 query 会静默复用旧剧本（stage_script 的
        断点缓存），等于把两个实验混在一个 run 里 —— 这里拒绝。
        """
        prev = self.manifest.get("query")
        if prev is not None and prev != query:
            raise RuntimeError(
                f"续跑 query 不一致（快照 {prev!r} vs 本次 {query!r}）——"
                f"剧本按 query 生成，请开新实验。")
        self._emit("query.bound", query=query)
        self.manifest["query"] = query

    def bind_label(self, label: str) -> None:
        """记录实验标签（bench 矩阵格标签等），供报告分组对比。"""
        self.set_meta("bench_cell", label)

    def snapshot_config(self, cfg: Dict[str, Any]) -> Path:
        """把本次运行的有效任务配置冻结进实验目录（可重建性）。

        实验证据的原则是"模型可见 ⟺ 日志"：manifest 只记元信息，
        完整配置必须能从实验目录重建 —— 没有快照的对比脚本只能硬编码
        run_id 猜衔接模式（2026-08-16 修复 Bug#4 前的 compare_chains 现状）。
        续跑时 config.yaml 与快照不一致 = 两个不同的实验，拒绝混跑。
        """
        import yaml
        path = self.root / "config.yaml"
        text = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
        if path.exists():
            existing = yaml.safe_load(path.read_text(encoding="utf-8"))
            if existing != cfg:
                raise RuntimeError(
                    f"续跑配置与快照不一致（{path}）。配置变化应开新实验，"
                    f"而不是混跑续用旧产物。")
        path.write_text(text, encoding="utf-8")
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        self._emit("config.snapshotted", path="config.yaml", sha256=sha)
        self.manifest["config_file"] = "config.yaml"
        self.manifest["config_sha256"] = sha
        self._flush()
        return path

    def _flush(self):
        (self.root / "manifest.json").write_text(
            json.dumps(self.manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def finalize(self, gpu_price_usd_per_hour: Optional[float] = None):
        """收尾：汇总统计（含本地 GPU 时间成本估算），并校验运行时不变量。

        成本口径（对齐"提供者声明能力，harness 消费声明"）：
        - API 产物：提供者在 ArtifactMeta.cost_usd 里声明实际/估算费用；
        - 本地产物：backend="local" 能力声明的提供者，其耗时计为 GPU 时间，
          按 gpu_price_usd_per_hour（任务配置 cost 段或调用参数，默认 1.2）
          折算成本。
        """
        self.manifest["finished_at"] = datetime.now().isoformat()
        price = gpu_price_usd_per_hour if gpu_price_usd_per_hour is not None else 1.2
        # 本地 GPU 时间：按提供者声明的 backend 能力识别（不再按名字嗅探 "local"）
        gpu_s = 0.0
        for arts in self.manifest.get("stages", {}).values():
            for a in arts:
                meta = a.get("meta", {})
                adapter = str(meta.get("adapter", ""))
                backend = ""
                if adapter:
                    from ..core.registry import capabilities as _caps
                    try:
                        backend = _caps(adapter).get("backend", "")
                    except Exception:
                        backend = ""
                if backend == "local":
                    gpu_s += float(meta.get("elapsed_s", 0.0))
        local_gpu_hours = round(gpu_s / 3600, 3)
        local_gpu_cost = round(gpu_s / 3600 * price, 3)
        total_all = round(self.manifest["total_cost_usd"] + local_gpu_cost, 4)
        self._emit("finalized", finished_at=self.manifest["finished_at"],
                   local_gpu_hours=local_gpu_hours,
                   local_gpu_cost_usd_est=local_gpu_cost,
                   total_cost_usd_all=total_all)
        self.manifest["local_gpu_hours"] = local_gpu_hours
        self.manifest["local_gpu_cost_usd_est"] = local_gpu_cost
        self.manifest["total_cost_usd_all"] = total_all
        self._flush()
        # 证据完整性：收尾时断言 manifest↔文件系统↔事件流的关系
        from .invariants import check_experiment
        violations = check_experiment(self.root)
        if violations:
            raise RuntimeError(
                "实验不变量校验失败（证据完整性受损）:\n" +
                "\n".join(f"  - {v}" for v in violations))
        return self.root


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self
    def __exit__(self, *a):
        self.elapsed = time.time() - self.t0
