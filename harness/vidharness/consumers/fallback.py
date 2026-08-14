"""生成器降级链（Fallback Consumer）。

面向未来：模型换代/宕机/超时是常态，流水线不应绑定单一提供者。
按优先级依次尝试多个 MediaGenerator 提供者，失败自动切换下一档，
每个尝试都记录到 manifest（fallback 事件），最终使用第一个成功的产物。
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..core.registry import resolve
from ..seams import GenRequest


class FallbackGenerator:
    """MediaGenerator 包装器：providers 按序尝试。

    用法（任务 YAML）：
      generator:
        adapter: fallback
        params:
          chain: [generator.minimax-h3-local, generator.minimax-h3-api]
    """

    def __init__(self, chain: List[str], providers: dict | None = None):
        providers = providers or {}
        self.chain = [resolve(name)(**providers.get(name, {})) for name in chain]
        self.name = f"fallback[{','.join(chain)}]"
        # 能力 = 各档的并集（取最优值）
        caps: Dict[str, Any] = {}
        for p in self.chain:
            for k, v in getattr(p, "capabilities", {}).items():
                if isinstance(v, (int, float)) and isinstance(caps.get(k), (int, float)):
                    caps[k] = max(caps[k], v)
                elif v:
                    caps[k] = v
        self.capabilities = caps

    def generate(self, req: GenRequest, workdir, **kw):
        errors = []
        for provider in self.chain:
            try:
                art = provider.generate(req=req, workdir=workdir, **kw)
                art.meta.params["fallback_used"] = provider.name
                return art
            except Exception as e:
                errors.append(f"{provider.name}: {type(e).__name__}: {str(e)[:200]}")
        raise RuntimeError("生成器降级链全部失败:\n" + "\n".join(errors))
