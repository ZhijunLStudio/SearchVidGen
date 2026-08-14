"""转写能力缝（Service Definition）。

音频→文本转写：验证"模型生成了什么声音"的唯一客观手段。
H3 原生音频里的对白/旁白与剧本旁白是否一致，靠它验证。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Protocol, runtime_checkable


@runtime_checkable
class Transcriber(Protocol):
    """音频/视频转写器。"""
    name: str
    capabilities: Dict[str, Any]     # {"language": "zh+multi", "device": "cpu"}

    def transcribe(self, media: Path, workdir: Path, **kw):
        """返回 Artifact(kind='transcript')，payload: {text, segments, meta}。"""
        ...
