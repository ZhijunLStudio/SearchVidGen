"""基准矩阵（benchmark matrix）—— harness 护城河的执行层。

对齐 deepseek-harness 的"组合 = 显式有序列表"与"配置错误在最早可解析点
响亮失败"：基准把"一次只变一个变量"的对比实验制度化。

bench spec（YAML）：
  bench:
    base: tasks/story.yaml          # 基础任务配置
    matrix:                          # 变量轴（点路径 -> 取值列表）
      - context.chain_mode: [none, hard, ref]
      - generator.params.steps: [20, 30]
    local_min_per_seg: 12            # 本地 GPU 每段时间（分钟，E4 规划常数，可覆盖）

展开 = 笛卡尔积；每格在规划期完成全部校验（配置 schema / 适配器存在 /
参数声明 / 能力要求），任何一格不合法即整体失败——不花一分钟 GPU 就发现
配置错误。
"""
from __future__ import annotations

import copy
import itertools
from typing import Any, Dict, List, Optional, Tuple

from .config import validate_task, ConfigError
from .registry import capabilities, check_capabilities, instantiate, resolve, resolve_provider

# E4 实测：A800 双卡 30 步 8s@768p 单段 ~12-18 分钟；12 为规划下限常数
_DEFAULT_LOCAL_MIN_PER_SEG = 12.0

BENCH_KEYS = ("base", "matrix", "local_min_per_seg")


class BenchError(RuntimeError):
    """基准规格不合法（消息带定位）。"""


def _set_dotted(cfg: Dict[str, Any], path: str, value: Any) -> None:
    """按点路径覆写嵌套 dict（如 generator.params.steps）。"""
    parts = path.split(".")
    node = cfg
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            raise BenchError(f"矩阵路径 '{path}' 不可写：'{p}' 不是 dict（配置结构不匹配）")
        node = node[p]
    node[parts[-1]] = value


def expand_matrix(base: Dict[str, Any], matrix: List[Dict[str, list]]) -> List[Tuple[str, Dict[str, Any]]]:
    """笛卡尔积展开：返回 [(格标签, 覆写后的完整配置), ...]。

    格标签 = 各轴取值拼接（如 none.t2va.20），用于 manifest.bench_cell 分组对比。
    """
    if not isinstance(matrix, list) or not matrix:
        raise BenchError("bench.matrix 必须是非空列表（每项 = 一个变量轴 {路径: [取值...]}）")
    axes: List[List[Tuple[str, Any]]] = []
    for axis in matrix:
        if not isinstance(axis, dict) or len(axis) != 1:
            raise BenchError(f"变量轴必须是单键 dict（{axis!r}）")
        (path, values), = axis.items()
        if not isinstance(values, list) or not values:
            raise BenchError(f"变量轴 '{path}' 的取值必须是非空列表")
        axes.append([(path, v) for v in values])
    cells: List[Tuple[str, Dict[str, Any]]] = []
    for combo in itertools.product(*axes):
        cfg = copy.deepcopy(base)
        label_parts: List[str] = []
        for path, value in combo:
            _set_dotted(cfg, path, value)
            label_parts.append(str(value))
        cells.append((".".join(label_parts), cfg))
    return cells


def _generator_requirements(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """按 chain_mode 推导生成能力要求（与 SegmentDirector 同一口径）。"""
    chain = cfg.get("pipeline", {}).get("context", {}).get("chain_mode", "none")
    required: Dict[str, Any] = {}
    if chain == "hard":
        required["first_last_frame"] = True
    elif chain == "ref":
        required["refs"] = 1
    if cfg.get("audio_verify"):
        required["audio"] = True
    return required


def _generator_name(cfg: Dict[str, Any]) -> str:
    gen = cfg.get("pipeline", {}).get("generator", {})
    if "route" in gen:
        return resolve_provider("generator", gen["route"], context="bench.generator")
    return gen["adapter"]


def validate_cell(label: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """规划期校验：配置 schema + 提供者实例化 + 能力要求。

    返回能力声明（预估用）。任何失败抛 ConfigError/BenchError/RuntimeError，
    消息带格标签。
    """
    ctx = f"bench[{label}]"
    try:
        validate_task(cfg)
    except ConfigError as e:
        raise BenchError(f"{ctx}: {e}") from e
    script_cfg = cfg["pipeline"]["script"]
    instantiate(script_cfg["adapter"], script_cfg.get("params", {}), context=ctx)
    gen_name = _generator_name(cfg)
    gen = instantiate(gen_name, cfg["pipeline"]["generator"].get("params", {}), context=ctx)
    judge_cfg = cfg["judge"]
    instantiate(judge_cfg["adapter"], judge_cfg.get("params", {}), context=ctx)
    required = _generator_requirements(cfg)
    caps = check_capabilities(gen, required, context=ctx)
    return caps


def estimate_cost(cfg: Dict[str, Any], caps: Dict[str, Any],
                  local_min_per_seg: float) -> Dict[str, Any]:
    """规划成本预估（显式假设，非结算口径；结算口径在 finalize）。

    返回 {segments, seconds_per_seg, cost_usd_est, basis}；basis 说明假设来源。
    """
    segments = int(cfg.get("segments", 4))
    gen = cfg["pipeline"]["generator"]
    backend = caps.get("backend", "")
    if backend == "api":
        rates = caps.get("cost_rates_usd_per_s") or {}
        resolution = str((gen.get("params") or {}).get("resolution", "768P"))
        rate = rates.get(resolution)
        if rate is None:
            raise BenchError(
                f"提供者未声明 {resolution} 的 cost_rates_usd_per_s，无法预估（已声明: {sorted(rates)}）")
        seconds = segments * int((gen.get("params") or {}).get("duration", 8))
        return {"segments": segments, "seconds": seconds,
                "cost_usd_est": round(seconds * rate, 2),
                "basis": f"API 单价 {resolution}={rate} USD/s（能力声明，规划口径）"}
    if backend == "local":
        gpu_price = float((cfg.get("cost") or {}).get("gpu_price_usd_per_hour", 1.2))
        hours = segments * local_min_per_seg / 60.0
        return {"segments": segments,
                "gpu_hours_est": round(hours, 2),
                "cost_usd_est": round(hours * gpu_price, 2),
                "basis": f"本地 GPU：每段 {local_min_per_seg} 分钟（E4 规划常数）× {gpu_price} USD/卡时"}
    return {"segments": segments, "cost_usd_est": None,
            "basis": f"backend={backend!r} 无成本口径，无法预估"}


def plan(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """规划基准：返回每格 {label, cfg, caps, estimate}；任一格不合法整体失败。"""
    bench = spec.get("bench")
    if not isinstance(bench, dict):
        raise BenchError("缺少 bench 段（{bench: {base, matrix}}）")
    unknown = [k for k in bench if k not in BENCH_KEYS]
    if unknown:
        raise BenchError(f"bench 未知键 {unknown}（允许: {list(BENCH_KEYS)}）")
    import yaml
    from pathlib import Path
    base_path = Path(bench["base"])
    if not base_path.exists():
        raise BenchError(f"base 任务配置不存在: {base_path}")
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    local_min = float(bench.get("local_min_per_seg", _DEFAULT_LOCAL_MIN_PER_SEG))
    cells = expand_matrix(base, bench["matrix"])
    plan_rows: List[Dict[str, Any]] = []
    for label, cfg in cells:
        caps = validate_cell(label, cfg)
        est = estimate_cost(cfg, caps, local_min)
        plan_rows.append({"label": label, "cfg": cfg, "caps": caps, "estimate": est})
    return plan_rows


def bench_cell_status(base_dir: Any, task_name: str, label: str,
                      cfg: Dict[str, Any], query: str = "") -> Dict[str, Any]:
    """bench 格的断点续跑状态：在已有 run 中找与当前格匹配的对象。

    格身份 = bench_cell 标签 + config.yaml 快照 + query（三者同才算同一格：
    换 query 是不同实验，不得跳过/续跑旧格）。
    返回 {"run_id": 最新匹配 run 或 None, "finished": bool}。
    """
    import yaml
    from pathlib import Path
    task_dir = Path(base_dir) / task_name
    if not task_dir.exists():
        return {"run_id": None, "finished": False}
    best: Optional[Dict[str, Any]] = None
    for run_dir in sorted(task_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        mf = run_dir / "manifest.json"
        if not mf.exists():
            continue
        try:
            import json as _json
            manifest = _json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if manifest.get("bench_cell") != label:
            continue
        if query and manifest.get("query") != query:
            continue
        cfg_file = run_dir / "config.yaml"
        if not cfg_file.exists():
            continue
        try:
            if yaml.safe_load(cfg_file.read_text(encoding="utf-8")) != cfg:
                continue
        except Exception:
            continue
        best = {"run_id": manifest.get("run_id", run_dir.name),
                "finished": bool(manifest.get("finished_at"))}
    if best is None:
        return {"run_id": None, "finished": False}
    return best
