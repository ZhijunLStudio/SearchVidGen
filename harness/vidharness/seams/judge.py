"""评测能力缝（Service Definition）—— harness 的护城河。

模型不会给自己做质检；生成越"一体化"，独立验证越稀缺。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Protocol, runtime_checkable


@runtime_checkable
class Judge(Protocol):
    """多模态评测器：媒体 + 标准 -> 结构化评分与反馈。"""
    name: str
    modalities: List[str]      # ["image","video","audio","text"]

    def judge(self, media: List[Path], criteria: Dict[str, str], workdir: Path, **kw):
        """返回 Artifact(kind='scores')，payload: {scores, score, passed, feedback}。"""
        ...


@dataclass
class JudgeCriteria:
    """一条评测维度。aliases：正则兜底解析时的可匹配别名（如 ["一致性"]）。"""
    name: str
    question: str
    weight: float = 1.0
    min_score: float = 6.0
    scale: float = 10.0
    aliases: list = None


@dataclass
class RetryPolicy:
    """失败重试策略（评测不通过 → 反馈注入 → 重新生成）。"""
    max_attempts: int = 2
    inject_feedback: bool = True
    feedback_prefix: str = "请修正以下问题后重新生成："
