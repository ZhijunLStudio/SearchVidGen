"""经验记忆（Experience Memory）—— harness 从环境反馈中学习的核心。

原则：不写领域模板。质量经验来自环境反馈（裁判评分/用户意见），
在记忆中累积；重复出现的同类问题提升为"经验"，自动注入未来生成，
跨任务、跨领域泛化。

存储：JSONL（experiments/_memory.jsonl），每条经验可追溯来源与时间。
提升规则：同一规范化 complaint 出现 >= promote_threshold 次 → 成为经验
（进入生成提示的"经验教训"区）。
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _normalize(text: str) -> str:
    t = re.sub(r"[\s，。！？、,.!?…—\-]", "", text)
    return t[:60]


class ExperienceMemory:
    def __init__(self, path: Path, promote_threshold: int = 1):
        self.path = Path(path)
        self.promote_threshold = promote_threshold
        self._items: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                self._items.append(json.loads(line))
            except Exception:
                continue

    def _flush(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            "\n".join(json.dumps(i, ensure_ascii=False) for i in self._items) + "\n",
            encoding="utf-8")

    def add(self, complaint: str, source: str, kind: str = "feedback") -> None:
        """记录一条环境反馈（裁判负面反馈/用户意见）。"""
        key = _normalize(complaint)
        for item in self._items:
            if item["key"] == key:
                item["count"] += 1
                item["last_at"] = time.time()
                item["sources"].append(source)
                self._flush()
                return
        self._items.append({
            "key": key,
            "complaint": complaint.strip(),
            "kind": kind,
            "count": 1,
            "sources": [source],
            "first_at": time.time(),
            "last_at": time.time(),
            "promoted": False,
        })
        self._flush()

    def add_experience(self, lesson: str, source: str) -> None:
        """直接沉淀一条经验（来自已证实的实验发现 E 系列等环境证据）。"""
        self._items.append({
            "key": _normalize(lesson),
            "complaint": lesson.strip(),
            "kind": "experience",
            "count": self.promote_threshold + 1,
            "sources": [source],
            "first_at": time.time(),
            "last_at": time.time(),
            "promoted": True,
        })
        self._flush()

    def experience_lines(self) -> List[str]:
        """当前生效的经验教训（提升后的条目），供注入生成提示。"""
        out = []
        for item in self._items:
            if item.get("kind") == "experience" or \
               (item.get("count", 0) >= self.promote_threshold and item.get("promoted")):
                out.append(item["complaint"])
        return out

    def recent_feedback(self, n: int = 3) -> List[str]:
        """最近未提升的负面反馈（供局部重试上下文）。"""
        fresh = [i for i in self._items if not i.get("promoted") and i.get("kind") == "feedback"]
        fresh.sort(key=lambda i: i.get("last_at", 0), reverse=True)
        return [i["complaint"] for i in fresh[:n]]
