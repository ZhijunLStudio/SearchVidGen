"""实验报告生成器：扫描 experiments/ 生成对比报告（HTML + 摘要 JSON）。

- 单实验：各阶段产物、评测分数、耗时、成本、重试次数
- 多实验：同任务横向对比（不同模型/参数 → 指标差异），即"基准"的核心产出
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        # 审计修复：损坏的 JSON 不再静默跳过——报告/leaderboard 少掉
        # 整个 run 或评测时必须有可见提示（模型可见⟺日志）
        print(f"⚠️ 跳过无法解析的 JSON（{e}）: {path}", file=sys.stderr)
        return {}


def collect(base_dir: Path, task: str) -> List[Dict[str, Any]]:
    """扫描 <base>/<task>/*/manifest.json，汇总为实验列表。"""
    task_dir = Path(base_dir) / task
    runs: List[Dict[str, Any]] = []
    if not task_dir.exists():
        return runs
    for run_dir in sorted(task_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        manifest = _load_json(run_dir / "manifest.json")
        if not manifest:
            continue
        # 完整性：有 finished_at（新口径）或成片存在（旧 run 兼容）
        if not manifest.get("finished_at") and \
           not (run_dir / "final" / "final_video.mp4").exists():
            continue
        evals = {}
        for f in (run_dir / "eval").glob("*.json"):
            evals[f.stem] = _load_json(f)
        # 汇总评测分数（全局 + 分 stage 两个口径；消费方只从这里取聚合）
        scores: Dict[str, List[float]] = {}
        stage_scores: Dict[str, Dict[str, float]] = {}
        stage_passed: Dict[str, Dict[str, int]] = {}
        passed = 0
        total = 0
        for stage, records in evals.items():
            if not isinstance(records, list):
                continue
            dims: Dict[str, List[float]] = {}
            sp = st = 0
            for rec in records:
                if not isinstance(rec, dict):
                    continue
                for k, v in rec.get("scores", {}).items():
                    scores.setdefault(k, []).append(v)
                    dims.setdefault(k, []).append(v)
                total += 1
                st += 1
                if rec.get("passed"):
                    passed += 1
                    sp += 1
            stage_scores[stage] = {k: round(sum(v) / len(v), 2) for k, v in dims.items() if v}
            stage_passed[stage] = {"passed": sp, "total": st}
        # 分阶段耗时/成本分解（manifest stages 的 meta 累计）
        stages_cost: Dict[str, float] = {}
        stages_elapsed: Dict[str, float] = {}
        models: List[str] = []
        gen_models: Dict[str, str] = {}
        for stage, arts in manifest.get("stages", {}).items():
            stages_cost[stage] = round(
                sum(float(a.get("meta", {}).get("cost_usd", 0.0)) for a in arts), 4)
            stages_elapsed[stage] = round(
                sum(float(a.get("meta", {}).get("elapsed_s", 0.0)) for a in arts), 1)
            if stage == "segments":
                for a in arts:
                    m = a.get("meta", {})
                    gen_models.setdefault(str(m.get("adapter", "?")),
                                          str(m.get("model", "?")))
        models = [f"{k}:{v}" for k, v in sorted(gen_models.items())]
        # 裁判来源（judge 产物 meta.adapter；混用裁判时的口径标注，E24）
        judge_adapters = sorted({a.get("meta", {}).get("adapter", "?")
                                 for a in manifest.get("stages", {}).get("judge", [])})
        # 旧 run（8-16 前，E12 布局）：裁判原始输出在 eval/judge_*.json 且
        # 未记录 adapter——该时代裁判是 judge.openai-compat 的同名旧版本
        # （E42 实测：同视频 8-14 记 5.0，今日裁判 10.0，跨期不可比），
        # 推断标注必须带上版本未知的口径警示
        if not judge_adapters and any(
                f.name.startswith("judge_") for f in (run_dir / "eval").glob("judge_*.json")):
            judge_adapters = ["judge.openai-compat（推断：旧布局未记录，裁判版本未知，跨期不可比）"]
        runs.append({
            "run_id": manifest.get("run_id", run_dir.name),
            "dir": str(run_dir),
            "bench_cell": manifest.get("bench_cell"),
            "chain_mode": manifest.get("chain_mode"),
            "query": manifest.get("query"),
            "title": manifest.get("title"),
            "models": models,
            "judge_adapters": judge_adapters,
            "created_at": manifest.get("created_at"),
            "finished_at": manifest.get("finished_at"),
            "total_elapsed_s": manifest.get("total_elapsed_s", 0),
            "total_cost_usd": manifest.get("total_cost_usd", 0),
            "local_gpu_hours": manifest.get("local_gpu_hours"),
            "stages": {k: len(v) for k, v in manifest.get("stages", {}).items()},
            "stages_cost_usd": stages_cost,
            "stages_elapsed_s": stages_elapsed,
            "retries": manifest.get("retries", {}),
            "scores": {k: round(sum(v) / len(v), 2) for k, v in scores.items() if v},
            "stage_scores": stage_scores,
            "stage_passed": stage_passed,
            "passed_rate": round(passed / total, 2) if total else None,
            "final_video": str(Path(run_dir) / "final" / "final_video.mp4"),
        })
    return runs


def render_html(runs: List[Dict[str, Any]], out: Path) -> Path:
    import html as _html
    has_cell = any(r.get("bench_cell") for r in runs)
    cell_header = "<th>Bench 格</th>" if has_cell else ""
    has_title = any(r.get("title") for r in runs)
    title_header = "<th>标题</th>" if has_title else ""
    rows = ""
    for r in runs:
        cell = f"<td>{r['bench_cell']}</td>" if has_cell else ""
        title = f"<td>{_html.escape(str(r.get('title') or '-'))}</td>" if has_title else ""
        score_cells = "".join(
            f"<td>{k}<br><b>{v}</b></td>" for k, v in r.get("scores", {}).items())
        rows += f"""
        <tr>
          <td>{r['run_id']}</td>
          {title}
          {cell}
          <td>{r['created_at']}</td>
          <td>{r['total_elapsed_s'] / 60:.1f} min</td>
          <td>${r['total_cost_usd']:.4f}</td>
          <td>{r['passed_rate']}</td>
          <td>{json.dumps(r.get('retries', {}), ensure_ascii=False)}</td>
          {score_cells}
          <td><a href="{r['dir']}">产物目录</a></td>
        </tr>"""
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>VidHarness 实验报告</title>
<style>
body {{ font-family: system-ui; margin: 2em; }}
table {{ border-collapse: collapse; width: 100%; }}
td, th {{ border: 1px solid #ccc; padding: 6px 10px; font-size: 14px; text-align: center; }}
th {{ background: #f5f5f5; }}
</style></head><body>
<h2>VidHarness 实验对比</h2>
<p>生成时间：<span id="t"></span>　实验数：{len(runs)}</p>
<table>
<tr><th>Run</th>{title_header}{cell_header}<th>创建时间</th><th>总耗时</th><th>API 成本</th><th>通过率</th><th>重试</th><th>各维度均分</th><th>产物</th></tr>
{rows}
</table>
<script>document.getElementById('t').textContent = new Date().toLocaleString();</script>
</body></html>"""
    out.write_text(html, encoding="utf-8")
    return out


def report(base_dir: Path, task: str, out_html: Path) -> Dict[str, Any]:
    runs = collect(base_dir, task)
    render_html(runs, out_html)
    return {"task": task, "runs": len(runs), "html": str(out_html)}


def _event_summary(ev: Dict[str, Any]) -> str:
    """事件流的一行摘要（详情页展示）。"""
    t = ev.get("type", "?")
    if t == "artifact.saved":
        return f"{t}: {ev.get('stage')}/{Path(ev.get('entry', {}).get('path', '?')).name}"
    if t == "eval.saved":
        return f"{t}: {ev.get('stage')}"
    if t == "retry":
        return f"{t}: {ev.get('stage')}"
    if t == "query.bound":
        return f"{t}: {ev.get('query', '')[:60]}"
    if t == "config.snapshotted":
        return f"{t}: sha256={ev.get('sha256', '')[:12]}…"
    if t == "manifest.set":
        return f"{t}: {ev.get('key')}"
    if t == "finalized":
        return (f"{t}: gpu_hours={ev.get('local_gpu_hours')} "
                f"cost_all=${ev.get('total_cost_usd_all')}")
    return t


def render_run_html(run_dir: Path, out: Path) -> Path:
    """单 run 详情页：概览 / 配置 / 产物 / 评测明细 / 事件流。"""
    import html as _html
    run_dir = Path(run_dir)
    manifest = _load_json(run_dir / "manifest.json")
    if not manifest:
        raise RuntimeError(f"缺少 manifest.json: {run_dir}")
    esc = _html.escape

    def kv(k: str, v: Any) -> str:
        return f"<tr><td>{esc(k)}</td><td>{esc(str(v))}</td></tr>"

    # ---- 概览 ----
    summary = "".join(kv(k, v) for k, v in [
        ("run_id", manifest.get("run_id")),
        ("title", manifest.get("title")),
        ("query", manifest.get("query")),
        ("bench_cell", manifest.get("bench_cell")),
        ("chain_mode", manifest.get("chain_mode")),
        ("generator_capabilities", manifest.get("generator_capabilities")),
        ("created_at", manifest.get("created_at")),
        ("finished_at", manifest.get("finished_at")),
        ("total_elapsed_s", manifest.get("total_elapsed_s")),
        ("total_cost_usd", manifest.get("total_cost_usd")),
        ("local_gpu_hours", manifest.get("local_gpu_hours")),
        ("total_cost_usd_all", manifest.get("total_cost_usd_all")),
        ("retries", manifest.get("retries")),
    ] if v is not None)

    # ---- 配置快照 ----
    cfg_file = run_dir / "config.yaml"
    cfg_text = esc(cfg_file.read_text(encoding="utf-8")) \
        if cfg_file.exists() else "（无快照：2026-08-16 前旧 run）"

    # ---- 产物表 ----
    art_rows = ""
    for stage, arts in manifest.get("stages", {}).items():
        for a in arts:
            meta = a.get("meta", {})
            name = Path(a.get("path", "?")).name
            art_rows += (
                f"<tr><td>{esc(stage)}</td><td>{esc(name)}</td>"
                f"<td>{esc(str(meta.get('adapter', '-')))}</td>"
                f"<td>{esc(str(meta.get('model', '-')))}</td>"
                f"<td>{meta.get('elapsed_s', 0):.1f}s</td>"
                f"<td>${meta.get('cost_usd', 0):.4f}</td>"
                f"<td>{esc(str(meta.get('seed')))}</td></tr>")

    # ---- 评测明细 ----
    eval_blocks = ""
    eval_dir = run_dir / "eval"
    for f in sorted(eval_dir.glob("*.json")) if eval_dir.exists() else []:
        data = _load_json(f)
        if isinstance(data, list):
            body = "".join(
                f"<pre>{esc(json.dumps(r, ensure_ascii=False, indent=2))}</pre>"
                for r in data if isinstance(r, dict))
        else:
            body = f"<pre>{esc(json.dumps(data, ensure_ascii=False, indent=2))}</pre>"
        eval_blocks += f"<h3>eval/{f.name}</h3>{body}"

    # ---- 事件流（末尾 20 条）----
    events_file = run_dir / "events.jsonl"
    event_rows = ""
    event_note = ""
    timeline_rows = ""
    if events_file.exists():
        events = [json.loads(line) for line in events_file.read_text(encoding="utf-8").splitlines()
                  if line.strip()]
        event_note = f"共 {len(events)} 条事件（显示末尾 20 条）"
        for ev in events[-20:]:
            event_rows += (f"<tr><td>{esc(str(ev.get('ts', '')))}</td>"
                           f"<td>{esc(_event_summary(ev))}</td></tr>")
        # 阶段时间线：stage.started/finished 配对算时长
        started_ts: Dict[str, Any] = {}
        for ev in events:
            if ev.get("type") == "stage.started":
                started_ts[ev["stage"]] = ev.get("ts", 0)
            elif ev.get("type") == "stage.finished" and ev.get("stage") in started_ts:
                dur = ev.get("ts", 0) - started_ts.pop(ev["stage"])
                timeline_rows += (f"<tr><td>{esc(ev['stage'])}</td>"
                                  f"<td>{dur:.1f}s</td></tr>")
        for s in started_ts:
            timeline_rows += f"<tr><td>{esc(s)}</td><td>（未结束：中断/运行中）</td></tr>"
    else:
        event_note = "无事件流（2026-08-16 前旧 run）"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Run 详情 {manifest.get('run_id')}</title>
<style>
body {{ font-family: system-ui; margin: 2em; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 1em; }}
td, th {{ border: 1px solid #ccc; padding: 5px 10px; font-size: 13px; text-align: left; }}
th {{ background: #f5f5f5; }}
pre {{ background: #fafafa; padding: 8px; border: 1px solid #eee; overflow-x: auto; }}
</style></head><body>
<h2>Run 详情：{esc(str(manifest.get('run_id')))}
  <small>{esc(str(manifest.get('task')))}{' · ' + esc(str(manifest.get('bench_cell'))) if manifest.get('bench_cell') else ''}</small></h2>
<h3>概览</h3><table>{summary}</table>
<h3>配置快照</h3><pre>{cfg_text}</pre>
<h3>产物</h3>
<table><tr><th>Stage</th><th>文件</th><th>适配器</th><th>模型</th><th>耗时</th><th>成本</th><th>Seed</th></tr>
{art_rows}</table>
<h3>评测明细</h3>{eval_blocks or '<p>（无评测记录）</p>'}
<h3>事件流</h3><p>{esc(event_note)}</p>
<h3>阶段时间线</h3>
<table><tr><th>阶段</th><th>时长</th></tr>{timeline_rows or '<tr><td colspan="2">（无记录）</td></tr>'}</table>
<table><tr><th>ts</th><th>事件</th></tr>{event_rows}</table>
</body></html>"""
    out.write_text(html, encoding="utf-8")
    return out
