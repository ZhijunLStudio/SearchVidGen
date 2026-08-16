"""注册生成器降级链为可配置提供者（adapter: generator.fallback）。"""
from __future__ import annotations

from ..consumers.fallback import FallbackGenerator
from ..core.registry import register

try:
    register("generator.fallback")(FallbackGenerator)
except ValueError:
    pass    # 已注册（load_builtin_adapters 幂等）
