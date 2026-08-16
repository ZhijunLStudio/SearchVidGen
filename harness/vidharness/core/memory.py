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


def clean_feedback_text(feedback: str) -> str:
    """清洗反馈文本（E32：真实记忆审计发现两类噪声）。

    1. 整体或片段是 JSON（含 feedback 字段）→ 取内层 feedback（历史上
       unparseable_feedback 的原文片段会把整段/多段 JSON 混入记忆——
       多对象拼接形态用正则取首个 feedback）；
    2. 解析失败指令（"评分解析失败…"）是基础设施噪声，不入质量记忆
       → 返回空串由调用方跳过。
    """
    text = (feedback or "").strip()
    if not text:
        return ""
    if text.startswith("评分解析失败"):
        return ""
    # 多 JSON 对象拼接/整体 JSON：取首个 feedback 字段值
    m = re.search(r'\{"[^"]*":\s*\d+(?:\.\d+)?,\s*"feedback":\s*"((?:[^"\\]|\\.)*)"', text)
    if m:
        inner = m.group(1).replace("\\n", "\n").strip()
        if inner:
            return inner
    return text


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
        raw_items: List[Dict[str, Any]] = []
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
            # E32 迁移：JSON 包装取内层 feedback、解析噪声丢弃、键重算
            cleaned = clean_feedback_text(str(item.get("complaint", "")))
            if not cleaned:
                self.load_warnings.append(f"第 {i} 行 complaint 为解析噪声，已跳过")
                continue
            item["v"] = MEMORY_FORMAT_VERSION
            item["complaint"] = cleaned
            item["key"] = _normalize(cleaned)
            item.setdefault("promoted", False)
            item.setdefault("sources", [])
            raw_items.append(item)
        # 同键合并（历史语义近重复 → count 累加 → 补全提升，E14 语义）
        merged: Dict[str, Dict[str, Any]] = {}
        for item in raw_items:
            key = item["key"]
            if key in merged:
                m = merged[key]
                m["count"] = int(m.get("count", 0)) + int(item.get("count", 1))
                m["sources"] = (m.get("sources") or [])[-_MAX_SOURCES + 1:] + \
                    (item.get("sources") or [])[-_MAX_SOURCES:]
                m["promoted"] = bool(m.get("promoted") or item.get("promoted"))
                m["last_at"] = max(float(m.get("last_at", 0)), float(item.get("last_at", 0)))
            else:
                merged[key] = dict(item)
        self._items = list(merged.values())
        for item in self._items:
            if item.get("kind") != "experience" and \
                    int(item.get("count", 0)) >= self.promote_threshold:
                item["promoted"] = True

    def consolidate(self, canonicalize) -> Dict[str, Any]:
        """语义聚类合并（E33：E32 的语义近重复缺口）。

        canonicalize: callable(complaint) -> 规范短语（LLM 或规则实现）。
        对未提升条目：按规范短语归并（count 累加、sources 截断），
        达到 promote_threshold 即提升；写回并返回统计。
        无标签的条目原样保留（不丢弃数据）；本方法纯逻辑、无 LLM 依赖。
        """
        groups: Dict[str, Dict[str, Any]] = {}
        unlabeled: List[Dict[str, Any]] = []
        # 已提升/experience 项按 complaint 索引（新组归并进同名旧项，防重复）
        existing_by_label = {str(i["complaint"]): i for i in self._items
                             if i.get("kind") == "experience" or i.get("promoted")}
        for item in self._items:
            if item.get("kind") == "experience" or item.get("promoted"):
                continue
            label = str(canonicalize(item["complaint"]) or "").strip()
            if not label:
                unlabeled.append(item)
                continue
            if label in groups:
                m = groups[label]
                m["count"] = int(m.get("count", 0)) + int(item.get("count", 1))
                m["sources"] = (m.get("sources") or [])[-_MAX_SOURCES + 1:] + \
                    (item.get("sources") or [])[-_MAX_SOURCES:]
            else:
                m = dict(item)
                m["key"] = _normalize(label)
                m["complaint"] = label
                groups[label] = m
        changed = 0
        promoted = 0
        merged_into_existing = 0
        new_groups: List[Dict[str, Any]] = []
        for label, m in groups.items():
            if label in existing_by_label:
                target = existing_by_label[label]
                target["count"] = int(target.get("count", 0)) + int(m.get("count", 1))
                target["sources"] = (target.get("sources") or [])[-_MAX_SOURCES + 1:] + \
                    (m.get("sources") or [])[-_MAX_SOURCES:]
                merged_into_existing += 1
                continue
            if int(m.get("count", 0)) >= self.promote_threshold:
                m["promoted"] = True
                m["promoted_at"] = time.time()
                promoted += 1
            changed += 1
            new_groups.append(m)
        # 归并结果替换未提升条目；无标签条目与已提升/experience 原样保留
        kept = [i for i in self._items
                if i.get("kind") == "experience" or i.get("promoted")]
        self._items = kept + unlabeled + new_groups
        self._flush()
        return {"before": len(self._items), "after": len(self._items),
                "groups": changed, "promoted": promoted,
                "merged_into_existing": merged_into_existing,
                "unlabeled": len(unlabeled)}

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
        """当前生效的经验教训（提升后的条目），供注入生成提示（去重防御）。"""
        out = []
        seen = set()
        for item in self._items:
            if (item.get("kind") == "experience" or item.get("promoted")) \
                    and item["complaint"] not in seen:
                seen.add(item["complaint"])
                out.append(item["complaint"])
        return out
