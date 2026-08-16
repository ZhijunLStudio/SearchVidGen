"""经验记忆（Experience Memory）—— harness 从环境反馈中学习的核心。

原则：不写领域模板。质量经验来自环境反馈（裁判评分/用户意见），
在记忆中累积；重复出现的同类问题提升为"经验"，自动注入未来生成，
跨任务、跨领域泛化。

存储：JSONL（experiments/_memory.jsonl），每条经验可追溯来源与时间。
提升规则：同一规范化 complaint 出现 >= promote_threshold 次 → 提升为经验
（进入生成提示的"经验教训"区）；提升发生在 add() 内（到达阈值即置位，
2026-08-16 修复：此前 add 只涨 count 从不置 promoted，提升机制形同虚设）。

记录格式：v=MEMORY_FORMAT_VERSION；旧文件（无 v 字段）按 v0 兼容读取，
下次 flush 统一升级。坏行跳过并记入 load_warnings（内存是辅助数据，
不整体拒绝加载，但损坏必须可观测）。
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List

MEMORY_FORMAT_VERSION = 1
_MAX_SOURCES = 5          # 每条经验的来源回溯上限（防无限增长）


def _normalize(text: str) -> str:
    t = re.sub(r"[\s，。！？、,.!?…—\-]", "", text)
    return t[:60]


class ExperienceMemory:
    def __init__(self, path: Path, promote_threshold: int = 1):
        self.path = Path(path)
        self.promote_threshold = promote_threshold
        self._items: List[Dict[str, Any]] = []
        self.load_warnings: List[str] = []
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        for i, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except Exception:
                self.load_warnings.append(f"第 {i} 行无法解析，已跳过")
                continue
            if not isinstance(item, dict) or "key" not in item or "complaint" not in item:
                self.load_warnings.append(f"第 {i} 行缺必需字段，已跳过")
                continue
            v = item.get("v")
            if v is not None and v != MEMORY_FORMAT_VERSION:
                self.load_warnings.append(
                    f"第 {i} 行记录版本 v={v} 未知（当前 {MEMORY_FORMAT_VERSION}），已跳过")
                continue
            # 无 v 字段 = v0 旧格式：兼容读取，flush 时升级
            item.setdefault("v", MEMORY_FORMAT_VERSION)
            item.setdefault("promoted", False)
            item.setdefault("sources", [])
            self._items.append(item)

    def _flush(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            "\n".join(json.dumps(i, ensure_ascii=False) for i in self._items) + "\n",
            encoding="utf-8")

    def add(self, complaint: str, source: str, kind: str = "feedback") -> None:
        """记录一条环境反馈（裁判负面反馈/用户意见）。

        到达 promote_threshold 即提升（promoted=True + promoted_at），
        提升后的条目进入 experience_lines() 注入生成提示。
        """
        key = _normalize(complaint)
        now = time.time()
        for item in self._items:
            if item["key"] == key:
                item["count"] = int(item.get("count", 0)) + 1
                item["last_at"] = now
                item["sources"] = (item.get("sources") or [])[-_MAX_SOURCES + 1:] + [source]
                if item["count"] >= self.promote_threshold and not item.get("promoted"):
                    item["promoted"] = True
                    item["promoted_at"] = now
                self._flush()
                return
        self._items.append({
            "v": MEMORY_FORMAT_VERSION,
            "key": key,
            "complaint": complaint.strip(),
            "kind": kind,
            "count": 1,
            "sources": [source],
            "first_at": now,
            "last_at": now,
            "promoted": self.promote_threshold <= 1,
        })
        self._flush()

    def add_experience(self, lesson: str, source: str) -> None:
        """直接沉淀一条经验（来自已证实的实验发现 E 系列等环境证据）。"""
        now = time.time()
        self._items.append({
            "v": MEMORY_FORMAT_VERSION,
            "key": _normalize(lesson),
            "complaint": lesson.strip(),
            "kind": "experience",
            "count": self.promote_threshold + 1,
            "sources": [source],
            "first_at": now,
            "last_at": now,
            "promoted": True,
        })
        self._flush()

    def experience_lines(self) -> List[str]:
        """当前生效的经验教训（提升后的条目），供注入生成提示。"""
        out = []
        for item in self._items:
            if item.get("kind") == "experience" or item.get("promoted"):
                out.append(item["complaint"])
        return out
