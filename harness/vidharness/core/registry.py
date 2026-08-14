"""适配器注册表。

所有模型适配器通过 @register("namespace.name") 登记，
流水线 YAML 按名字引用 —— 新模型 = 新文件 + 一行注册，核心零改动。
"""
from __future__ import annotations

from typing import Any, Dict, Type, Union

_REGISTRY: Dict[str, Any] = {}


def register(name: str):
    """装饰器：注册适配器类/实例。"""
    def _wrap(obj: Any):
        if name in _REGISTRY:
            raise ValueError(f"adapter '{name}' 已注册")
        _REGISTRY[name] = obj
        return obj
    return _wrap


def get(name: str) -> Any:
    if name not in _REGISTRY:
        raise KeyError(
            f"未知适配器 '{name}'。已注册: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


def resolve(adapter: Union[str, Any]) -> Any:
    """按名取注册的适配器；传对象则原样返回（测试/注入用）。"""
    if isinstance(adapter, str):
        return get(adapter)
    return adapter


def capabilities(adapter: Union[str, Any]) -> Dict[str, Any]:
    obj = resolve(adapter)
    if isinstance(obj, type):
        obj = obj()
    return getattr(obj, "capabilities", {})


def check_capabilities(adapter: Union[str, Any], required: Dict[str, Any], context: str = "") -> Dict[str, Any]:
    """配置错误响亮失败（fail loud）：任务要求超出提供者能力时直接报错。

    required 形如 {"audio": True, "max_duration_s": 12}：键缺失、数值超上限、
    布尔要求不支持均报错，不做语义推断。
    """
    caps = capabilities(adapter)
    for k, v in required.items():
        if k not in caps:
            raise RuntimeError(
                f"[{context}] 提供者 '{adapter}' 未声明能力 '{k}'（已声明: {sorted(caps)})"
            )
        if isinstance(v, (int, float)) and isinstance(caps[k], (int, float)) and v > caps[k]:
            raise RuntimeError(
                f"[{context}] 要求 {k}={v} 超过提供者 '{adapter}' 上限 {caps[k]}"
            )
        if isinstance(v, bool) and v and not caps[k]:
            raise RuntimeError(
                f"[{context}] 任务要求 {k}，但提供者 '{adapter}' 不支持"
            )
    return caps


def list_adapters() -> Dict[str, str]:
    return {k: type(v).__name__ for k, v in _REGISTRY.items()}


# 保证所有内置适配器被加载（副作用：注册）
def load_builtin_adapters():
    import vidharness.providers  # noqa: F401
