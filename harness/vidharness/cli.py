"""命令行入口：
  vh run <task.yaml> --query "春天在哪里"
  vh bench <spec.yaml> --query "..." [--dry-run]   # 基准矩阵对比
  vh adapters [--verbose]                          # 列出适配器/参数声明
  vh doctor <run_dir> | --all <dir>                # 运行时不变量体检
  vh leaderboard <task> [--publish dir]            # leaderboard 基线导出
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .core.experiment import Experiment
from .consumers.segment_director import SegmentDirector
from .core.registry import list_adapters, load_builtin_adapters


def run_task(cfg: dict, query: str, output: str,
             resume: str | None = None, label: str | None = None,
             adapters_cache: dict | None = None) -> tuple[Path, Path]:
    """执行一次任务运行（bench 逐格执行复用本函数）。返回 (成片, 实验目录)。"""
    task = cfg.get("task_name", "story")
    if resume:
        exp = Experiment(task=task, base_dir=Path(output), run_id=resume)
        if not exp.root.exists():
            raise SystemExit(f"找不到实验 {resume}（在 {exp.root}）")
        print(f"♻️ 断点续跑: {exp.root}")
    else:
        exp = Experiment(task=task, base_dir=Path(output))
    exp.bind_query(query)      # 记录实验变量 + 续跑一致性守卫
    if label:
        exp.bind_label(label)
    exp.snapshot_config(cfg)   # 冻结有效配置进实验目录：可重建 + 续跑守卫
    director = SegmentDirector(exp, cfg, adapters_cache=adapters_cache)
    final = director.run(query)
    return final, exp.root


def cmd_adapters(args):
    load_builtin_adapters()
    from .core.registry import capabilities as caps_of, get
    for name in sorted(list_adapters()):
        cls = get(name)
        caps = caps_of(name)
        line = f"  {name:28s} -> {cls.__name__}"
        if caps:
            line += f"  能力: {caps}"
        print(line)
        if args.verbose:
            schema = getattr(cls, "param_schema", None)
            if schema:
                print("    参数声明:")
                for k, s in schema.items():
                    req = "*" if s.get("required") else " "
                    typ = s.get("type", "?")
                    default = "" if s.get("default") is None else f"（默认 {s.get('default')!r}）"
                    choices = f" 可选 {list(s['choices'])}" if "choices" in s else ""
                    print(f"    {req} {k} ({typ}){choices}{default}: {s.get('help', '')}")
            else:
                print("    （未声明参数目录，按构造签名校验）")


def cmd_doctor(args):
    from .core.invariants import check_experiment
    if args.all is None and args.run is None:
        raise SystemExit("doctor 需要 run 目录或 --all <experiments_dir>")
    if args.all:
        base = Path(args.all)
        checked = bad = 0
        for task_dir in sorted(base.iterdir()):
            if not task_dir.is_dir():
                continue
            for run_dir in sorted(task_dir.iterdir()):
                if not (run_dir / "manifest.json").exists():
                    continue
                checked += 1
                violations = check_experiment(run_dir)
                if violations:
                    bad += 1
                    print(f"❌ {run_dir}: {len(violations)} 条违规")
                    for v in violations[:3]:
                        print(f"   - {v}")
        print(f"体检 {checked} 个 run，{bad} 个违规")
        raise SystemExit(1 if bad else 0)
    path = Path(args.run)
    if not (path / "manifest.json").exists():
        raise SystemExit(f"找不到 manifest.json: {path}")
    violations = check_experiment(path)
    if violations:
        print(f"❌ 不变量违规 {len(violations)} 条:")
        for v in violations:
            print(f"  - {v}")
        raise SystemExit(1)
    has_events = (path / "events.jsonl").exists()
    print(f"✅ 不变量通过: {path}")
    print(f"   事件流: {'有（可重放）' if has_events else '无（2026-08-16 前的旧 run）'}")


def cmd_regress(args):
    from .core.regress import render_status, status
    if not args.run:
        rows = status(Path(args.output), Path(args.spec))
        print(render_status(rows))
        drift = [r["task_file"] for r in rows if r["drift"]]
        missing = [r["task_file"] for r in rows if not r["run_id"]]
        print(f"\n未跑过: {len(missing)}　配置漂移: {len(drift)}")
        if drift:
            print("漂移任务（需重跑）:", ", ".join(drift))
        raise SystemExit(0 if not (drift or missing) else 2)
    # 执行套件（跳过已完成、续跑未完成；环境按任务分两轮：ref2va 用 h3int8）
    from .core.regress import load_regression_list
    for task_file in load_regression_list(Path(args.spec)):
        import yaml
        cfg = yaml.safe_load(Path(task_file).read_text(encoding="utf-8"))
        task_name = cfg.get("task_name", Path(task_file).stem)
        task_dir = Path(args.output) / task_name
        latest = None
        if task_dir.exists():
            cands = []
            for d in task_dir.iterdir():
                m = d / "manifest.json"
                if not m.exists():
                    continue
                try:
                    man = json.loads(m.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if man.get("finished_at"):
                    cands.append((man.get("finished_at"), d.name))
            if cands:
                latest = sorted(cands)[-1][1]
        if latest:
            print(f"⏭ {task_file}: 已完成（{latest}），跳过")
            continue
        print(f"▶ {task_file}")
        final, root = run_task(cfg, args.query, args.output)
        print(f"  完成: {final}")


def cmd_scaffold(args):
    from .core.scaffold import scaffold_provider
    path = scaffold_provider(args.seam, args.name, Path(args.out))
    print(f"✓ 提供者骨架: {path}")
    print("  下一步（cookbook: docs/cookbook/adding-a-provider.md）:")
    print("  1. 实现协议方法 + 填写 capabilities/param_schema")
    print("  2. providers/__init__.py 加 import（加载即注册）")
    print(f"  3. vh adapters 确认注册；任务 YAML 引用 {args.seam}.{args.name}")


def cmd_feedback(args):
    from .core.memory import ExperienceMemory
    mem = ExperienceMemory(Path(args.output) / "_memory.jsonl")
    mem.add_experience(args.text, source="用户反馈")
    print(f"已写入经验记忆（当前 {len(mem.experience_lines())} 条经验）")


def cmd_report(args):
    from .core.report import report
    if args.run:
        from .core.report import render_run_html
        run_dir = Path(args.output) / args.task / args.run
        if not run_dir.exists():
            raise SystemExit(f"找不到 run 目录: {run_dir}")
        out = render_run_html(run_dir, run_dir / "report.html")
        print(json.dumps({"html": str(out)}, ensure_ascii=False, indent=2))
        return
    base = Path(args.output)
    html = base / f"report_{args.task}.html"
    result = report(base, args.task, html)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_leaderboard(args):
    from .core.leaderboard import export
    if args.all:
        from .core.leaderboard import export_all
        result = export_all(Path(args.output), Path(args.publish),
                            calibrate=args.calibrated)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    json_path, md_path, diff = export(
        Path(args.output), args.task, out_dir=Path(args.publish),
        calibrate=args.calibrated)
    print(json.dumps({
        "json": str(json_path), "md": str(md_path),
        "diff": diff,
    }, ensure_ascii=False, indent=2))


def cmd_run(args):
    load_builtin_adapters()
    cfg = yaml.safe_load(Path(args.task).read_text(encoding="utf-8"))
    # 配置校验（fail loud）：拼错的键/策略在启动时拒绝，而不是静默吞默认值
    from .core.config import validate_task
    validate_task(cfg)
    if args.brief:
        cfg["brief"] = args.brief
    if args.segments:
        cfg["segments"] = args.segments
    final, root = run_task(cfg, args.query, args.output, resume=args.resume,
                           label=args.label)
    print(json.dumps({"final": str(final), "experiment": str(root)},
                     ensure_ascii=False, indent=2))


def cmd_bench(args):
    from .core.bench import plan
    load_builtin_adapters()
    spec = yaml.safe_load(Path(args.spec).read_text(encoding="utf-8"))
    # 规划期校验全部格子：任何一格配置不合法即整体失败，不花一分钟 GPU
    rows = plan(spec)
    print(f"基准规划：{len(rows)} 格")
    total = 0.0
    for r in rows:
        est = r["estimate"]
        cost = est.get("cost_usd_est")
        if cost is not None:
            total += cost
        print(f"  [{r['label']:24s}] backend={r['caps'].get('backend')} "
              f"预估 ${cost}（{est['basis']}）")
    print(f"预估总成本 ≈ ${round(total, 2)}（规划口径；实际以各 run 的 manifest 结算为准）")
    if args.dry_run:
        print("dry-run：未执行生成")
        return
    # 逐格执行；相同参数的适配器跨格复用（避免每格重载生成模型，E19 已知成本）
    # 格级断点续跑：已完成格跳过、未完成格续跑（长矩阵崩溃不重来）
    from .core.bench import bench_cell_status
    adapters_cache: dict = {}
    for r in rows:
        task_name = r["cfg"].get("task_name", "story")
        status = bench_cell_status(Path(args.output), task_name, r["label"],
                                   r["cfg"], query=args.query)
        if status["run_id"] and status["finished"]:
            print(f"\n⏭ bench 格 [{r['label']}] 已完成（{status['run_id']}），跳过")
            continue
        resume = status["run_id"]
        note = f"（续跑 {resume}）" if resume else ""
        print(f"\n▶ bench 格 [{r['label']}]{note}")
        final, root = run_task(r["cfg"], args.query, args.output, resume=resume,
                               label=r["label"], adapters_cache=adapters_cache)
        print(f"  完成: {final}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="vidharness", description="视频生成流水线 Harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="运行任务")
    pr.add_argument("task", help="任务配置 YAML")
    pr.add_argument("--query", required=True, help="目标（故事主题/产品/想法）")
    pr.add_argument("--brief", default=None, help="补充要求（可选，运行时给出，如风格/受众/时长）")
    pr.add_argument("--segments", type=int, default=None, help="分镜段数（默认4）")
    pr.add_argument("--label", default=None, help="实验标签（manifest.bench_cell，供分组对比）")
    pr.add_argument("--output", default="experiments", help="实验输出根目录")
    pr.add_argument("--resume", default=None, help="续跑指定 run_id（断点续跑）")
    pr.set_defaults(fn=cmd_run)

    pb = sub.add_parser("bench", help="基准矩阵对比（规划期全格校验 + 逐格执行）")
    pb.add_argument("spec", help="bench spec YAML（含 bench: {base, matrix} 段）")
    pb.add_argument("--query", required=True, help="目标（故事主题/产品/想法）")
    pb.add_argument("--output", default="experiments", help="实验输出根目录")
    pb.add_argument("--dry-run", action="store_true", help="只做规划校验与成本预估，不生成")
    pb.set_defaults(fn=cmd_bench)

    pa = sub.add_parser("adapters", help="列出适配器（--verbose 显示参数声明目录）")
    pa.add_argument("--verbose", action="store_true", help="显示能力与参数声明")
    pa.set_defaults(fn=cmd_adapters)

    pd = sub.add_parser("doctor", help="检查实验目录的运行时不变量（manifest↔文件↔事件流）")
    pd.add_argument("run", nargs="?", default=None, help="实验 run 目录（含 manifest.json）")
    pd.add_argument("--all", default=None, metavar="EXPERIMENTS_DIR",
                    help="全量体检：扫描目录下所有任务的所有 run")
    pd.set_defaults(fn=cmd_doctor)

    pl = sub.add_parser("leaderboard", help="导出 leaderboard 基线（JSON+MD，与上次基线 diff）")
    pl.add_argument("task", nargs="?", default=None, help="任务名（--all 时省略）")
    pl.add_argument("--all", action="store_true", help="导出全部任务 + 渲染 index.html")
    pl.add_argument("--calibrated", action="store_true",
                    help="按 calibration/ 维度偏移（n≥3）换算评分到主裁判口径（E30）")
    pl.add_argument("--output", default="experiments", help="实验输出根目录（读取）")
    pl.add_argument("--publish", default="leaderboards", help="基线输出目录（默认可入库追踪）")
    pl.set_defaults(fn=cmd_leaderboard)

    prg = sub.add_parser("regress", help="变体回归套件（状态表 / --run 执行）")
    prg.add_argument("--spec", default="tasks/regression.yaml", help="套件清单")
    prg.add_argument("--query", default="雨夜，一只小猫在旧书店的橱窗前躲雨")
    prg.add_argument("--output", default="experiments", help="实验输出根目录")
    prg.add_argument("--run", action="store_true", help="执行套件（跳过已完成）")
    prg.set_defaults(fn=cmd_regress)

    psc = sub.add_parser("scaffold", help="生成提供者骨架（新模型 = 新文件）")
    psc.add_argument("seam", help="能力缝（generator/judge/script/transcribe）")
    psc.add_argument("name", help="提供者名（注册为 <seam>.<name>）")
    psc.add_argument("--out", default="vidharness/providers", help="输出目录")
    psc.set_defaults(fn=cmd_scaffold)

    pf = sub.add_parser("feedback", help="把用户意见写入经验记忆（环境反馈直达）")
    pf.add_argument("text", help="反馈内容（如：旁白太肉麻，要真实朴素）")
    pf.add_argument("--output", default="experiments", help="实验输出根目录")
    pf.set_defaults(fn=cmd_feedback)

    prp = sub.add_parser("report", help="生成实验对比报告（--run 生成单 run 详情页）")
    prp.add_argument("task", help="任务名（如 story_short）")
    prp.add_argument("--run", default=None, help="run_id：生成单 run 详情页")
    prp.add_argument("--output", default="experiments", help="实验输出根目录")
    prp.set_defaults(fn=cmd_report)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
