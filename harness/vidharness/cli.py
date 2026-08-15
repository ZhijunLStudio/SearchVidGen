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


def cmd_adapters(_args):
    load_builtin_adapters()
    for name, cls in sorted(list_adapters().items()):
        print(f"  {name:28s} -> {cls}")


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
    task = cfg.get("task_name", "story")
    if args.resume:
        exp = Experiment(task=task, base_dir=Path(args.output), run_id=args.resume)
        if not exp.root.exists():
            raise SystemExit(f"找不到实验 {args.resume}（在 {exp.root}）")
        print(f"♻️ 断点续跑: {exp.root}")
    else:
        exp = Experiment(task=task, base_dir=Path(args.output))
    if args.brief:
        cfg["brief"] = args.brief
    if args.segments:
        cfg["segments"] = args.segments
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

    pa = sub.add_parser("adapters", help="列出适配器")
    pa.set_defaults(fn=cmd_adapters)

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
