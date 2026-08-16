"""任务配置校验 —— 对齐 deepseek-harness 的配置平面纪律。

任务 YAML 是 harness 的配置平面：拼错的键名、写错的衔接策略、畸形的
评测维度，都必须响亮失败（fail loud），而不是被 `cfg.get(...)` 静默吞成默认值。
配置的归属边界：
- harness 拥有：任务结构（pipeline 装配、评测维度、重试、记忆、成本口径）；
- 适配器拥有：params 里的字段（由 registry.instantiate 按构造签名校验）；
- 用户/实验拥有：--query / --brief / --segments 运行时覆盖。
"""
from __future__ import annotations

from typing import Any, Dict, List

CHAIN_MODES = ("none", "hard", "ref")

CRITERIA_FIELDS = ("name", "question", "weight", "min_score", "aliases")
RETRY_FIELDS = ("max_attempts", "inject_feedback", "feedback_prefix")

# 顶层/管线允许键（未知键 fail loud）
TASK_KEYS = {
    "task_name", "segments", "brief", "pipeline", "judge",
    "script_optimize", "script_judge", "script_retry",
    "segment_judge", "segment_retry", "cross_judge",
    "audio_verify", "memory", "cost",
}
PIPELINE_KEYS = {"script", "generator", "context"}
CONTEXT_KEYS = {"chain_mode", "anchor_refs"}

# 允许 adapter / route 二选一的组件块
_ADAPTER_BLOCK_KEYS = ("adapter", "params", "route")


class ConfigError(RuntimeError):
    """任务配置不合法（消息已带定位路径，可直接展示给用户）。"""


def _expect_type(cfg: Any, path: str, typ: type) -> None:
    if not isinstance(cfg, typ):
        raise ConfigError(f"{path}: 应为 {typ.__name__}，得到 {type(cfg).__name__}")


def _expect_keys(cfg: Dict[str, Any], path: str, allowed: Any) -> None:
    unknown = [k for k in cfg if k not in allowed]
    if unknown:
        raise ConfigError(
            f"{path}: 未知配置键 {unknown}（允许: {sorted(allowed)}）——"
            f"拼写错误会被静默忽略，这里拒绝启动")


def _check_adapter_block(block: Dict[str, Any], path: str, need_route_ok: bool = False) -> None:
    _expect_type(block, path, dict)
    allowed = set(_ADAPTER_BLOCK_KEYS) if need_route_ok else {"adapter", "params"}
    _expect_keys(block, path, allowed)
    if "adapter" not in block and "route" not in block:
        raise ConfigError(f"{path}: 缺少 adapter（或 route）")
    if "adapter" in block and "route" in block:
        raise ConfigError(f"{path}: adapter 与 route 只能二选一")
    if "adapter" in block:
        _expect_type(block["adapter"], f"{path}.adapter", str)
    if "params" in block:
        _expect_type(block["params"], f"{path}.params", dict)
    if "route" in block:
        _expect_type(block["route"], f"{path}.route", dict)


def _check_criteria(items: Any, path: str) -> None:
    _expect_type(items, path, list)
    for i, c in enumerate(items):
        p = f"{path}[{i}]"
        _expect_type(c, p, dict)
        _expect_keys(c, p, CRITERIA_FIELDS)
        if "name" not in c or not isinstance(c["name"], str) or not c["name"]:
            raise ConfigError(f"{p}: 缺少非空 name")
        if "question" not in c or not isinstance(c["question"], str):
            raise ConfigError(f"{p}: 缺少 question")
        for f in ("weight", "min_score"):
            if f in c and not isinstance(c[f], (int, float)):
                raise ConfigError(f"{p}.{f}: 应为数值")
        if "aliases" in c and not isinstance(c["aliases"], list):
            raise ConfigError(f"{p}.aliases: 应为字符串列表")


def _check_retry(r: Any, path: str) -> None:
    _expect_type(r, path, dict)
    _expect_keys(r, path, RETRY_FIELDS)
    if "max_attempts" in r and not isinstance(r["max_attempts"], int):
        raise ConfigError(f"{path}.max_attempts: 应为整数")
    if "inject_feedback" in r and not isinstance(r["inject_feedback"], bool):
        raise ConfigError(f"{path}.inject_feedback: 应为布尔")


def validate_task(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """校验任务配置；通过则原样返回，不合法则抛 ConfigError（fail loud）。"""
    _expect_type(cfg, "task", dict)
    _expect_keys(cfg, "task", TASK_KEYS)
    if "task_name" in cfg:
        _expect_type(cfg["task_name"], "task.task_name", str)
    if "segments" in cfg:
        _expect_type(cfg["segments"], "task.segments", int)

    _expect_type(cfg.get("pipeline", {}), "pipeline", dict)
    _expect_keys(cfg["pipeline"], "pipeline", PIPELINE_KEYS)
    _check_adapter_block(cfg["pipeline"].get("script", {}), "pipeline.script")
    _check_adapter_block(cfg["pipeline"].get("generator", {}), "pipeline.generator",
                         need_route_ok=True)

    ctx = cfg["pipeline"].get("context", {})
    _expect_type(ctx, "pipeline.context", dict)
    _expect_keys(ctx, "pipeline.context", CONTEXT_KEYS)
    if "chain_mode" in ctx:
        if ctx["chain_mode"] not in CHAIN_MODES:
            raise ConfigError(
                f"pipeline.context.chain_mode: 未知衔接策略 '{ctx['chain_mode']}'"
                f"（允许: {list(CHAIN_MODES)}；hard 已实测产生冻结帧，见 E3/E8）")
    if "anchor_refs" in ctx:
        _expect_type(ctx["anchor_refs"], "pipeline.context.anchor_refs", list)

    _check_adapter_block(cfg.get("judge", {}), "judge")

    if "script_optimize" in cfg:
        so = cfg["script_optimize"]
        _expect_type(so, "script_optimize", dict)
        _expect_keys(so, "script_optimize", ("rounds", "candidates", "target_score"))

    for key in ("script_judge", "segment_judge", "cross_judge"):
        if key in cfg:
            _check_criteria(cfg[key], key)
    for key in ("script_retry", "segment_retry"):
        if key in cfg:
            _check_retry(cfg[key], key)

    if "audio_verify" in cfg:
        av = cfg["audio_verify"]
        _expect_type(av, "audio_verify", dict)
        _expect_keys(av, "audio_verify", ("adapter",))
        _expect_type(av.get("adapter"), "audio_verify.adapter", str)

    if "memory" in cfg:
        m = cfg["memory"]
        _expect_type(m, "memory", dict)
        _expect_keys(m, "memory", ("path", "promote_threshold"))

    if "cost" in cfg:
        c = cfg["cost"]
        _expect_type(c, "cost", dict)
        _expect_keys(c, "cost", ("gpu_price_usd_per_hour",))
        if "gpu_price_usd_per_hour" in c and not isinstance(
                c["gpu_price_usd_per_hour"], (int, float)):
            raise ConfigError("cost.gpu_price_usd_per_hour: 应为数值")
    return cfg
