"""实验管理：每次运行 = 一个 Experiment。

目录结构：
experiments/<task>/<run_id>/
├── manifest.json         # 全量元信息：模型/参数/seed/耗时/成本/评分
├── artifacts/<stage>/    # 各阶段产物（含 .meta.json）
├── eval/<stage>.json     # 评测明细
└── final/                # 成片与报告
"""
from __future__ import annotations

import json
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..seams import Artifact, ArtifactMeta


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
        existing = self.root / "manifest.json"
        if existing.exists():
            try:
                self.manifest = json.loads(existing.read_text(encoding="utf-8"))
                return
            except Exception:
                pass
        self.manifest: Dict[str, Any] = {
            "task": task,
            "run_id": self.run_id,
            "created_at": datetime.now().isoformat(),
            "stages": {},
            "total_cost_usd": 0.0,
            "total_elapsed_s": 0.0,
        }

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
        def key(r):
            return json.dumps(r, ensure_ascii=False, sort_keys=True)
        existing = {key(r) for r in merged if isinstance(r, dict)}
        for r in results:
            if key(r) not in existing:
                merged.append(r)
                existing.add(key(r))
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
        self.manifest["query"] = query

    def snapshot_config(self, cfg: Dict[str, Any]) -> Path:
        """把本次运行的有效任务配置冻结进实验目录（可重建性）。

        实验证据的原则是"模型可见 ⟺ 日志"：manifest 只记元信息，
        完整配置必须能从实验目录重建 —— 没有快照的对比脚本只能硬编码
        run_id 猜衔接模式（2026-08-16 修复 Bug#4 前的 compare_chains 现状）。
        续跑时 config.yaml 与快照不一致 = 两个不同的实验，拒绝混跑。
        """
        import yaml
        path = self.root / "config.yaml"
        if path.exists():
            existing = yaml.safe_load(path.read_text(encoding="utf-8"))
            if existing != cfg:
                raise RuntimeError(
                    f"续跑配置与快照不一致（{path}）。配置变化应开新实验，"
                    f"而不是混跑续用旧产物。")
        path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                        encoding="utf-8")
        self.manifest["config_file"] = "config.yaml"
        self._flush()
        return path

    def _flush(self):
        (self.root / "manifest.json").write_text(
            json.dumps(self.manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def finalize(self, gpu_price_usd_per_hour: Optional[float] = None):
        """收尾：汇总统计（含本地 GPU 时间成本估算）。

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
        self.manifest["local_gpu_hours"] = round(gpu_s / 3600, 3)
        self.manifest["local_gpu_cost_usd_est"] = round(gpu_s / 3600 * price, 3)
        self.manifest["total_cost_usd_all"] = round(
            self.manifest["total_cost_usd"] + self.manifest["local_gpu_cost_usd_est"], 4)
        self._flush()
        return self.root


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self
    def __exit__(self, *a):
        self.elapsed = time.time() - self.t0
