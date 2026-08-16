"""生成能力缝（Service Definition）。

对应 deepseek-harness 的"Service Definition"角色：只声明协议与数据结构。
提供者见 providers/，消费者见 consumers/。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class ArtifactMeta:
    """产物元信息：可复现、成本统计与对比的依据。"""
    adapter: str = ""
    model: str = ""
    version: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None
    elapsed_s: float = 0.0
    cost_usd: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class Artifact:
    """产物：视频/音频/图像/脚本/评分。"""
    kind: str
    path: Path
    meta: ArtifactMeta = field(default_factory=ArtifactMeta)
    payload: Dict[str, Any] = field(default_factory=dict)

    def asdict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["path"] = str(self.path)
        return d


@dataclass
class GenRequest:
    """一次生成请求：文本指令 + 可选多模态上下文。

    - refs: 参考图（角色/风格锚点）
    - first_frame / last_frame: 首/尾帧条件（跨段衔接）
    - seed: 逐请求随机种子（None=提供者构造级种子/缺省随机）
    """
    text: str
    refs: List[Path] = field(default_factory=list)
    first_frame: Optional[Path] = None
    last_frame: Optional[Path] = None
    duration: Optional[int] = None
    ratio: Optional[str] = None
    style: Dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None


@runtime_checkable
class MediaGenerator(Protocol):
    """全模态媒体生成器：文字(+上下文) -> 视频(可选原生音频)。

    capabilities 声明能力，harness 据此路由/降级/校验（fail loud）：
      {"max_duration_s": 15, "audio": True, "refs": 9,
       "first_last_frame": True, "resolution": "768p", "backend": "local"|"api"}
    """
    name: str
    capabilities: Dict[str, Any]

    def generate(self, req: GenRequest, workdir: Path, **kw) -> Artifact:
        """返回 kind='video' 的产物（可能自带音轨）。"""
        ...
