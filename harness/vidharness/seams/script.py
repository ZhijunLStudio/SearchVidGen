"""剧本能力缝（Service Definition）。

故事规划目前未被视频模型吸收（模型只生成 ≤15s 片段），保留独立能力。

本文件拥有 script 缝的**约定**：
- `ScriptGenerator` 协议（generate(query, template, workdir) -> Artifact）；
- `build_script_prompt(query, template)` —— 提示契约（协议骨架 + 用户目标 +
  经验注入，无领域模板）。所有 script 提供者共用同一提示约定，
  换提供者不换面向模型的语言；
- `parse_script_json(content)` —— 输出契约（JSON 兜底解析）。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Protocol, runtime_checkable


@runtime_checkable
class ScriptGenerator(Protocol):
    """故事规划：query + 模板 -> 分镜计划（每段生成指令 + 旁白文本）。"""
    name: str

    def generate(self, query: str, template: Dict[str, Any], workdir: Path, **kw):
        """返回 Artifact(kind='script')，payload 为分镜计划 JSON。"""
        ...


def build_script_prompt(query: str, template: Dict[str, Any]) -> str:
    """组装剧本生成提示（协议契约：提供者共享，换提供者不换面向模型的语言）。

    template 字段：brief（补充要求）、segments（分镜数）、experience（经验教训）。
    """
    segments = int(template.get("segments", 4))
    parts = [f"目标：{query}（共 {segments} 个分镜）"]
    brief = template.get("brief")
    if brief:
        parts.append(f"补充要求：{brief}")
    experience = template.get("experience", [])
    if experience:
        parts.append("经验教训（务必遵守）：\n" + "\n".join(f"- {e}" for e in experience))
    parts.append(
        "输出 JSON（不要其他文字）：\n"
        '{"segments": [{"video_prompt": "...", "narration": "...", "duration": 8}]}'
    )
    return "\n\n".join(parts)


def parse_script_json(content: str) -> Dict[str, Any]:
    """剧本输出解析：优先 ```json 代码块，其次首尾花括号；失败返回可诊断的错误对象。"""
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
    raw = m.group(1) if m else content
    try:
        return json.loads(raw)
    except Exception:
        m2 = re.search(r"\{[\s\S]*\}", raw)
        if m2:
            try:
                return json.loads(m2.group(0))
            except Exception:
                pass
        return {"error": "JSON 解析失败", "raw": content[:500]}
