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
import sys
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
    json_path, md_path, diff = export(
        Path(args.output), args.task, out_dir=Path(args.publish))
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
    final, root = run_task(cfg, args.query, args.output, resume=args.resume)
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
    adapters_cache: dict = {}
    for r in rows:
        print(f"\n▶ bench 格 [{r['label']}]")
        final, root = run_task(r["cfg"], args.query, args.output, label=r["label"],
                               adapters_cache=adapters_cache)
        print(f"  完成: {final}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="vidharness", description="视频生成流水线 Harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="运行任务")
    pr.add_argument("task", help="任务配置 YAML")
    pr.add_argument("--query", required=True, help="目标（故事主题/产品/想法）")
    pr.add_argument("--brief", default=None, help="补充要求（可选，运行时给出，如风格/受众/时长）")
    pr.add_argument("--segments", type=int, default=None, help="分镜段数（默认4）")
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
    pl.add_argument("task", help="任务名（如 story_short）")
    pl.add_argument("--output", default="experiments", help="实验输出根目录（读取）")
    pl.add_argument("--publish", default="leaderboards", help="基线输出目录（默认可入库追踪）")
    pl.set_defaults(fn=cmd_leaderboard)

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
