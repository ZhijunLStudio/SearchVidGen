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


def cmd_report(args):
    from .core.report import report
    base = Path(args.output)
    html = base / f"report_{args.task}.html"
    result = report(base, args.task, html)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_run(args):
    load_builtin_adapters()
    cfg = yaml.safe_load(Path(args.task).read_text(encoding="utf-8"))
    exp = Experiment(task=cfg.get("task_name", "story"),
                     base_dir=Path(args.output))
    director = SegmentDirector(exp, cfg)
    final = director.run(args.query)
    print(json.dumps({"final": str(final), "experiment": str(exp.root)},
                     ensure_ascii=False, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(prog="vidharness", description="视频生成流水线 Harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="运行任务")
    pr.add_argument("task", help="任务配置 YAML")
    pr.add_argument("--query", required=True, help="搜索词/故事主题")
    pr.add_argument("--output", default="experiments", help="实验输出根目录")
    pr.set_defaults(fn=cmd_run)

    pa = sub.add_parser("adapters", help="列出适配器")
    pa.set_defaults(fn=cmd_adapters)

    prp = sub.add_parser("report", help="生成实验对比报告")
    prp.add_argument("task", help="任务名（如 story_short）")
    prp.add_argument("--output", default="experiments", help="实验输出根目录")
    prp.set_defaults(fn=cmd_report)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
