"""通用 OpenAI 兼容剧本提供者 —— script 缝的第二实现（孪生适配器）。

script.deepseek-v4-flash 绑定 DeepSeek 官方（token 加载 + 官方计费）；
本提供者面向**任意 OpenAI 兼容端点**（本地 vLLM Qwen、第三方 API 等），
base_url/model/api_key 显式声明。

计费口径（对齐"提供者声明成本"）：price_in/out 未配置时 cost_usd=0 且
billing="unpriced"——不编造单价；配置后按 token 计费。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from openai import OpenAI

from ..seams import Artifact, ArtifactMeta, build_script_prompt, parse_script_json
from ..core.registry import register


@register("script.openai-compat")
class OpenAICompatScriptGenerator:
    """script 协议实现：任意 OpenAI 兼容 chat completions 端点。"""

    name = "script.openai-compat"
    capabilities = {"language": "zh+en", "json_output": True}
    param_schema = {
        "base_url": {"type": "str", "required": True, "help": "OpenAI 兼容端点"},
        "model": {"type": "str", "required": True, "help": "模型名"},
        "api_key": {"type": "secret", "default": "EMPTY"},
        "temperature": {"type": "float", "default": 0.7},
        "max_tokens": {"type": "int", "default": 8192},
        "price_in_usd_per_1m": {"type": "float", "default": 0.0,
                                "help": "输入单价 USD/1M tokens（0=不计费）"},
        "price_out_usd_per_1m": {"type": "float", "default": 0.0,
                                 "help": "输出单价 USD/1M tokens（0=不计费）"},
    }

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY",
                 temperature: float = 0.7, max_tokens: int = 8192,
                 price_in_usd_per_1m: float = 0.0, price_out_usd_per_1m: float = 0.0):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.price_in = price_in_usd_per_1m
        self.price_out = price_out_usd_per_1m

    def generate(self, query: str, template: Dict[str, Any], workdir: Path, **kw) -> Artifact:
        import time as _t
        t0 = _t.time()
        workdir.mkdir(parents=True, exist_ok=True)
        system = (
            "你是资深影视导演。把用户的目标拆成 8-15 秒一镜的分镜计划，"
            "每镜含画面指令(video_prompt，中文50-90字：镜头运动/主体动作/环境/情绪)"
            "与旁白(narration)。画面指令末尾写音频要求（环境音与旁白朗读）。"
            "各镜之间机位/景别要有变化。只输出 JSON。"
        )
        user = build_script_prompt(query, template)   # 提示契约来自 seam
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        content = resp.choices[0].message.content or ""
        data = parse_script_json(content)             # 输出契约来自 seam
        path = workdir / "script.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        billing = "priced" if (self.price_in or self.price_out) else "unpriced"
        usage = resp.usage
        cost = ((usage.prompt_tokens if usage else 0) / 1e6 * self.price_in +
                (usage.completion_tokens if usage else 0) / 1e6 * self.price_out)
        meta = ArtifactMeta(
            adapter=self.name, model=resp.model,
            params={"temperature": self.temperature, "max_tokens": self.max_tokens,
                    "billing": billing,
                    "template": template},            # 可重建（模型可见 ⟺ 日志）
            elapsed_s=_t.time() - t0,
            cost_usd=round(cost, 6),
        )
        return Artifact(kind="script", path=path, meta=meta, payload=data)
