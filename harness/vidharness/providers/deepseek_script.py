"""DeepSeek V4 剧本生成适配器（OpenAI 兼容 API）。"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict

from openai import OpenAI

from ..seams import Artifact, ArtifactMeta
from ..core.registry import register


def _load_token() -> str:
    # 兼容多种存放位置；用户环境里在 ~/.claude/settings.json
    env = os.environ.get("DEEPSEEK_API_KEY")
    if env:
        return env
    settings = Path.home() / ".claude" / "settings.json"
    if settings.exists():
        try:
            cfg = json.loads(settings.read_text(encoding="utf-8"))
            tok = cfg.get("env", {}).get("ANTHROPIC_AUTH_TOKEN")
            if tok:
                return tok
        except Exception:
            pass
    raise RuntimeError("未找到 DeepSeek API key（DEEPSEEK_API_KEY 或 ~/.claude/settings.json）")


@register("script.deepseek-v4-flash")
class DeepSeekScriptGenerator:
    """把业务模板 + 搜索词交给 LLM，产出结构化 JSON 剧本。"""

    name = "script.deepseek-v4-flash"
    capabilities = {"language": "zh+en", "json_output": True}

    def __init__(self, api_key: str | None = None, base_url: str = "https://api.deepseek.com",
                 model: str = "deepseek-chat", temperature: float = 0.7, max_tokens: int = 8192):
        self.client = OpenAI(api_key=api_key or _load_token(), base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, query: str, template: Dict[str, Any], workdir: Path, **kw) -> Artifact:
        import time as _t
        t0 = _t.time()
        workdir.mkdir(parents=True, exist_ok=True)
        # 通用提示组装：协议契约 + 用户目标(brief) + 环境经验，无领域模板
        segments = template.get("segments", 4)
        system = (
            "你是资深影视导演。把用户的目标拆成 8-15 秒一镜的分镜计划，"
            "每镜含画面指令(video_prompt，中文50-90字：镜头运动/主体动作/环境/情绪)"
            "与旁白(narration)。画面指令末尾写音频要求（环境音与旁白朗读）。"
            "各镜之间机位/景别要有变化。只输出 JSON。"
        )
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
        user = "\n\n".join(parts)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        content = resp.choices[0].message.content
        data = self._parse_json(content, template)
        path = workdir / "script.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        meta = ArtifactMeta(
            adapter=self.name, model=resp.model,
            params={"temperature": self.temperature, "max_tokens": self.max_tokens,
                    "brief": brief, "n_experience": len(experience),
                    # 可重建：完整输入模板落盘（对齐"模型可见 ⟺ 日志"）
                    "template": template},
            elapsed_s=_t.time() - t0,
            cost_usd=_estimate_cost(resp.usage.prompt_tokens, resp.usage.completion_tokens, self.model),
        )
        return Artifact(kind="script", path=path, meta=meta, payload=data)

    @staticmethod
    def _parse_json(content: str, template: Dict[str, Any]) -> Dict[str, Any]:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
        raw = m.group(1) if m else content
        try:
            return json.loads(raw)
        except Exception:
            # 兜底：截取首尾花括号再试
            m2 = re.search(r"\{[\s\S]*\}", raw)
            if m2:
                try:
                    return json.loads(m2.group(0))
                except Exception:
                    pass
            return {"error": "JSON 解析失败", "raw": content[:500]}


def _estimate_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    """DeepSeek 官方定价近似（USD/1M tokens），仅用于成本统计。
    实际计费以官方为准；价格可能变化，这里做数量级估算。"""
    # deepseek-chat(v3.x/v4-flash) 输入约 $0.07/1M，输出约 $0.28/1M（历史档位，允许配置覆盖）
    price_in = float(os.environ.get("DEEPSEEK_PRICE_IN", "0.07"))
    price_out = float(os.environ.get("DEEPSEEK_PRICE_OUT", "0.28"))
    return prompt_tokens / 1e6 * price_in + completion_tokens / 1e6 * price_out
