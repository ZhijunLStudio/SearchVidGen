"""适配器注册表 —— 对齐 deepseek-harness 的"声明式提供者目录"。

所有模型适配器通过 @register("namespace.name") 登记，
流水线 YAML 按名字引用 —— 新模型 = 新文件 + 一行注册，核心零改动。

三条 fail-loud 纪律：
1. 注册时校验 capabilities 键符合所属 seam 的能力 schema（能力词汇是协议，
   不允许自由发明键 —— 拼错的能力键会静默绕过能力校验）；
2. instantiate 校验任务配置传给适配器的参数（未知参数/缺必需参数都在最早点报错）；
3. resolve_provider 按能力路由，多候选时不替用户做决定。
"""
from __future__ import annotations

import inspect
from typing import Any, Dict, List, Union

_REGISTRY: Dict[str, Any] = {}

# 每 seam 的能力 schema：允许的键 -> 允许的值类型。
# seam 由适配器名的第一段决定（generator.* / judge.* / script.* / transcribe.*）。
# 新增能力键必须同时更新此表（协议演进走这里，不走自由 dict）。
SEAM_CAPABILITY_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "generator": {
        "max_duration_s": (int, float),
        "audio": bool,
        "refs": (int, float),
        "first_last_frame": bool,
        "resolution": str,
        "backend": str,          # "local" | "api"（成本模型与计费依据）
        "cost_rates_usd_per_s": dict,  # API 单价声明（如 {"768P": 0.042}），规划预估用
    },
    "judge": {
        "frame_sampling": bool,
    },
    "script": {
        "language": str,
        "json_output": bool,
    },
    "transcribe": {
        "language": str,
        "device": str,
        "emotion_tags": bool,
    },
}


def _validate_capability_schema(name: str, caps: Dict[str, Any]) -> None:
    """注册时校验能力键与类型（fail loud：拼错能力键 = 静默绕过校验）。"""
    if not isinstance(caps, dict):
        raise TypeError(f"adapter '{name}' 的 capabilities 必须是 dict")
    seam = name.split(".")[0]
    schema = SEAM_CAPABILITY_SCHEMAS.get(seam)
    if schema is None:
        raise ValueError(
            f"adapter '{name}' 的前缀 '{seam}' 不在已知 seam 中: "
            f"{sorted(SEAM_CAPABILITY_SCHEMAS)}")
    for k, v in caps.items():
        if k not in schema:
            raise ValueError(
                f"adapter '{name}' 声明了未知能力键 '{k}'（{seam} seam 允许: "
                f"{sorted(schema)}）。新能力请先登记 SEAM_CAPABILITY_SCHEMAS。")
        if not isinstance(v, schema[k]):
            raise TypeError(
                f"adapter '{name}' 的能力 '{k}' 类型应为 {schema[k]}, "
                f"得到 {type(v).__name__}")


def register(name: str):
    """装饰器：注册适配器类/实例，并在注册点校验能力 schema。"""
    def _wrap(obj: Any):
        if name in _REGISTRY:
            raise ValueError(f"adapter '{name}' 已注册")
        caps = getattr(obj, "capabilities", None)
        if caps is not None:
            _validate_capability_schema(name, caps)
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


def _type_ok(typ: str, v: Any) -> bool:
    """参数声明类型检查（YAML 进来的值：bool 不当作 int）。"""
    if typ in ("str", "path", "secret"):
        return isinstance(v, str)
    if typ == "int":
        return isinstance(v, int) and not isinstance(v, bool)
    if typ == "float":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if typ == "bool":
        return isinstance(v, bool)
    if typ == "list":
        return isinstance(v, list)
    return True   # 未知类型声明不拦（前向兼容）


def _check_params_schema(name: str, params: Dict[str, Any],
                         schema: Dict[str, Any], context: str) -> None:
    """按提供者声明的参数目录校验（fail loud，提供者拥有 params 的语义）。"""
    if not isinstance(schema, dict):
        raise TypeError(f"adapter '{name}' 的 param_schema 必须是 dict")
    for k, v in params.items():
        if k not in schema:
            raise RuntimeError(
                f"[{context}] 适配器 '{name}' 不接受参数 '{k}'（声明: {sorted(schema)}）")
        spec = schema[k]
        if not isinstance(spec, dict):
            raise TypeError(f"adapter '{name}' 参数 '{k}' 的声明必须是 dict")
        typ = spec.get("type")
        if typ and not _type_ok(typ, v):
            raise RuntimeError(
                f"[{context}] 参数 '{k}' 类型应为 {typ}，得到 {type(v).__name__}（{v!r}）")
        if "choices" in spec and v not in spec["choices"]:
            raise RuntimeError(
                f"[{context}] 参数 '{k}' 只允许 {list(spec['choices'])}，得到 {v!r}")
    for k, spec in schema.items():
        if spec.get("required") and k not in params:
            raise RuntimeError(
                f"[{context}] 适配器 '{name}' 缺少必需参数 '{k}'"
                f"（{spec.get('help', '')}）".rstrip())


def instantiate(name: str, params: Dict[str, Any] | None = None,
                context: str = "", cache: Dict[str, Any] | None = None) -> Any:
    """实例化注册的适配器，并校验任务配置给的参数（fail loud）。

    校验顺序：提供者声明的 param_schema（声明目录，权威）→ 构造签名
    内省兜底（未声明 schema 的适配器）。未知参数/类型错误/缺必需参数
    都在最早点报错。

    cache：可选实例复用缓存（键 = name+params 归一化）。由调用方拥有
    复用范围（bench 逐格执行用它避免每格重载模型）；不进全局状态。
    """
    if cache is not None:
        import json as _json
        key = (name, _json.dumps(params or {}, sort_keys=True, ensure_ascii=False))
        if key in cache:
            return cache[key]
        obj = instantiate(name, params, context=context)
        cache[key] = obj
        return obj
    obj = resolve(name)
    if not isinstance(obj, type):
        return obj
    params = dict(params or {})
    schema = getattr(obj, "param_schema", None)
    if schema:
        _check_params_schema(name, params, schema, context)
        return obj(**params)
    sig = inspect.signature(obj.__init__)
    allowed = [p for p in sig.parameters
               if p not in ("self",) and sig.parameters[p].kind
               not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)]
    has_var_kw = any(sig.parameters[p].kind == inspect.Parameter.VAR_KEYWORD
                     for p in sig.parameters)
    if not has_var_kw:
        for k in params:
            if k not in allowed:
                raise RuntimeError(
                    f"[{context}] 适配器 '{name}' 不接受参数 '{k}'（可接受: {sorted(allowed)}）")
    missing = [p for p in allowed
               if sig.parameters[p].default is inspect.Parameter.empty and p not in params]
    if missing:
        raise RuntimeError(f"[{context}] 适配器 '{name}' 缺少必需参数: {missing}")
    return obj(**params)


def resolve_provider(seam: str, required: Dict[str, Any], context: str = "") -> str:
    """按能力路由：从已注册提供者中选出满足 required 的唯一适配器名。

    对应 deepseek-harness 的 provider-routed 模式（请求按能力选择提供者）。
    多个候选同时满足时拒绝替用户决定（fail loud），要求显式指定 adapter。
    """
    candidates = [n for n in _REGISTRY if n == seam or n.startswith(seam + ".")]
    if not candidates:
        raise RuntimeError(f"[{context}] 没有注册任何 {seam} 提供者")
    satisfied: List[str] = []
    reasons: Dict[str, str] = {}
    for n in sorted(candidates):
        try:
            check_capabilities(n, required)
            satisfied.append(n)
        except RuntimeError as e:
            reasons[n] = str(e)
    if not satisfied:
        detail = "; ".join(f"{n}: {r}" for n, r in reasons.items())
        raise RuntimeError(
            f"[{context}] 没有 {seam} 提供者满足 {required}。{detail}")
    if len(satisfied) > 1:
        raise RuntimeError(
            f"[{context}] 多个 {seam} 提供者满足 {required}: {satisfied}。"
            f"请显式指定 adapter（或调整要求缩小范围）。")
    return satisfied[0]


def capabilities(adapter: Union[str, Any]) -> Dict[str, Any]:
    obj = resolve(adapter)
    # 类读类属性、实例读实例属性（fallback 等合成提供者的能力是实例级并集）
    return getattr(obj, "capabilities", {})


def check_capabilities(adapter: Union[str, Any], required: Dict[str, Any], context: str = "") -> Dict[str, Any]:
    """配置错误响亮失败（fail loud）：任务要求超出提供者能力时直接报错。

    required 形如 {"audio": True, "max_duration_s": 12}：键缺失、数值超上限、
    布尔要求不支持均报错，不做语义推断。
    """
    caps = capabilities(adapter)
    display = getattr(resolve(adapter), "name", adapter)
    for k, v in required.items():
        if k not in caps:
            raise RuntimeError(
                f"[{context}] 提供者 '{display}' 未声明能力 '{k}'（已声明: {sorted(caps)})"
            )
        if isinstance(v, (int, float)) and isinstance(caps[k], (int, float)) and v > caps[k]:
            raise RuntimeError(
                f"[{context}] 要求 {k}={v} 超过提供者 '{display}' 上限 {caps[k]}"
            )
        if isinstance(v, bool) and v and not caps[k]:
            raise RuntimeError(
                f"[{context}] 任务要求 {k}，但提供者 '{display}' 不支持"
            )
    return caps


def list_adapters() -> Dict[str, str]:
    return {k: type(v).__name__ for k, v in _REGISTRY.items()}


# 保证所有内置适配器被加载（副作用：注册）
def load_builtin_adapters():
    import vidharness.providers  # noqa: F401
