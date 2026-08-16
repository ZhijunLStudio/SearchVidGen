"""DeepSeek V4 剧本生成适配器（OpenAI 兼容 API）。"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from openai import OpenAI

from ..seams import (Artifact, ArtifactMeta,
                     build_script_prompt, parse_script_json)
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
    param_schema = {
        "api_key": {"type": "secret", "default": None, "help": "DeepSeek API key（缺省读环境）"},
        "base_url": {"type": "str", "default": "https://api.deepseek.com"},
        "model": {"type": "str", "default": "deepseek-chat"},
        "temperature": {"type": "float", "default": 0.7},
        "max_tokens": {"type": "int", "default": 8192},
        "json_mode": {"type": "bool", "default": True,
                      "help": "response_format=json_object（E25 同源：解析健壮性）"},
    }

    def __init__(self, api_key: str | None = None, base_url: str = "https://api.deepseek.com",
                 model: str = "deepseek-chat", temperature: float = 0.7,
                 max_tokens: int = 8192, json_mode: bool = True):
        # 凭据延迟到第一次生成调用解析（规划期 dry-run 不依赖 key）
        self._api_key_param = api_key
        self.client: Optional[OpenAI] = None
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.json_mode = json_mode

    def generate(self, query: str, template: Dict[str, Any], workdir: Path, **kw) -> Artifact:
        import time as _t
        t0 = _t.time()
        workdir.mkdir(parents=True, exist_ok=True)
        # 通用提示组装：协议契约 + 用户目标(brief) + 环境经验，无领域模板
        # （提示契约由 script 缝的 build_script_prompt 拥有，见 seams/script.py）
        # system 默认是导演/分镜规划人格；变换任务（标题/归纳，E33/E43）
        # 经 kw.system 覆盖——变换任务的系统指令归任务自身拥有，
        # 否则 provider 人格会压倒用户指令（E43 实测：temperature 1.0 仍返回分镜）
        system = kw.get("system") or (
            "你是资深影视导演。把用户的目标拆成 8-15 秒一镜的分镜计划，"
            "每镜含画面指令(video_prompt，中文50-90字：镜头运动/主体动作/环境/情绪)"
            "与旁白(narration)。画面指令末尾写音频要求（环境音与旁白朗读）。"
            "各镜之间机位/景别要有变化。只输出 JSON。"
        )
        user = build_script_prompt(query, template)
        brief = template.get("brief")
        experience = template.get("experience", [])
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.json_mode:
            # E25 同源修复：JSON 模式保证剧本输出可解析（提示含 "JSON" 满足前置条件）
            kwargs["response_format"] = {"type": "json_object"}
        # 生成时温度覆盖（E26：优化器候选多样性的 seam 级能力）
        effective_temperature = kw.get("temperature", self.temperature)
        if effective_temperature is not None:
            kwargs["temperature"] = effective_temperature
        if self.client is None:
            self.client = OpenAI(api_key=self._api_key_param or _load_token(),
                                 base_url=self.base_url)
        resp = self.client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content
        data = parse_script_json(content)
        path = workdir / "script.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        meta = ArtifactMeta(
            adapter=self.name, model=resp.model,
            params={"temperature": effective_temperature, "max_tokens": self.max_tokens,
                    "brief": brief, "n_experience": len(experience),
                    # 可重建：完整输入模板 + 有效系统人格落盘
                    # （对齐"模型可见 ⟺ 日志"；E43 变换任务的 system 覆盖可审计）
                    "system": system, "template": template},
            elapsed_s=_t.time() - t0,
            cost_usd=_estimate_cost(resp.usage.prompt_tokens, resp.usage.completion_tokens, self.model),
        )
        return Artifact(kind="script", path=path, meta=meta, payload=data)


def _estimate_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    """DeepSeek 官方定价近似（USD/1M tokens），仅用于成本统计。
    实际计费以官方为准；价格可能变化，这里做数量级估算。"""
    # deepseek-chat(v3.x/v4-flash) 输入约 $0.07/1M，输出约 $0.28/1M（历史档位，允许配置覆盖）
    price_in = float(os.environ.get("DEEPSEEK_PRICE_IN", "0.07"))
    price_out = float(os.environ.get("DEEPSEEK_PRICE_OUT", "0.28"))
    return prompt_tokens / 1e6 * price_in + completion_tokens / 1e6 * price_out
