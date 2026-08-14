"""剧本能力缝（Service Definition）。

故事规划目前未被视频模型吸收（模型只生成 ≤15s 片段），保留独立能力。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Protocol, runtime_checkable


@runtime_checkable
class ScriptGenerator(Protocol):
    """故事规划：query + 模板 -> 分镜计划（每段生成指令 + 旁白文本）。"""
    name: str

    def generate(self, query: str, template: Dict[str, Any], workdir: Path, **kw):
        """返回 Artifact(kind='script')，payload 为分镜计划 JSON。"""
        ...
