"""命令行入口：
  vh run <task.yaml> --query "春天在哪里"
  vh adapters            # 列出已注册适配器
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
    base = Path(args.output)
    html = base / f"report_{args.task}.html"
    result = report(base, args.task, html)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_run(args):
    load_builtin_adapters()
    cfg = yaml.safe_load(Path(args.task).read_text(encoding="utf-8"))
    # 配置校验（fail loud）：拼错的键/策略在启动时拒绝，而不是静默吞默认值
    from .core.config import validate_task
    validate_task(cfg)
    task = cfg.get("task_name", "story")
    if args.brief:
        cfg["brief"] = args.brief
    if args.segments:
        cfg["segments"] = args.segments
    if args.resume:
        exp = Experiment(task=task, base_dir=Path(args.output), run_id=args.resume)
        if not exp.root.exists():
            raise SystemExit(f"找不到实验 {args.resume}（在 {exp.root}）")
        print(f"♻️ 断点续跑: {exp.root}")
    else:
        exp = Experiment(task=task, base_dir=Path(args.output))
    exp.bind_query(args.query)   # 记录实验变量 + 续跑一致性守卫
    # 冻结有效配置进实验目录：可重建 + 续跑一致性守卫
    exp.snapshot_config(cfg)
    director = SegmentDirector(exp, cfg)
    final = director.run(args.query)
    print(json.dumps({"final": str(final), "experiment": str(exp.root)},
                     ensure_ascii=False, indent=2))


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

    pa = sub.add_parser("adapters", help="列出适配器（--verbose 显示参数声明目录）")
    pa.add_argument("--verbose", action="store_true", help="显示能力与参数声明")
    pa.set_defaults(fn=cmd_adapters)

    pd = sub.add_parser("doctor", help="检查实验目录的运行时不变量（manifest↔文件↔事件流）")
    pd.add_argument("run", help="实验 run 目录（含 manifest.json）")
    pd.set_defaults(fn=cmd_doctor)

    pf = sub.add_parser("feedback", help="把用户意见写入经验记忆（环境反馈直达）")
    pf.add_argument("text", help="反馈内容（如：旁白太肉麻，要真实朴素）")
    pf.add_argument("--output", default="experiments", help="实验输出根目录")
    pf.set_defaults(fn=cmd_feedback)

    prp = sub.add_parser("report", help="生成实验对比报告")
    prp.add_argument("task", help="任务名（如 story_short）")
    prp.add_argument("--output", default="experiments", help="实验输出根目录")
    prp.set_defaults(fn=cmd_report)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
