"""裁判重复评测方差分解：同一视频用同一裁判/同一维度重复评测 N 次。

动机：评测闭环是 harness 的护城河（E2），但其可靠性从未被量化——
E16/E23 的臂间方差无法区分"生成方差 vs 裁判方差"。本脚本只测裁判侧：
同一视频（不重新生成）重复评测，分解出裁判自身的评分波动。

用法：
  python scripts/judge_repeatability.py --run experiments/story_ref2va_check/20260816_124438_f16c96 \
      [--video seg01.mp4] [--repeats 5] [--out experiments]

原则（E13）：裁判端点/参数/维度全部从该 run 的 config.yaml 快照读取，
本脚本不硬编码任何端点或维度。结果落为一次事件溯源实验
（task=judge_repeat：events + manifest + artifacts/judge 原始输出），
最终打印 维度均值/标准差/最小/最大 + 与 run 原记录的对比。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vidharness.core.experiment import Experiment  # noqa: E402
from vidharness.core.registry import load_builtin_adapters, instantiate  # noqa: E402
from vidharness.consumers.judge_loop import run_judge  # noqa: E402
from vidharness.seams import JudgeCriteria  # noqa: E402


def _criteria(cfg: dict) -> list:
    out = []
    for c in cfg.get("segment_judge") or []:
        out.append(JudgeCriteria(name=c["name"], question=c["question"],
                                 weight=float(c.get("weight", 1.0)),
                                 min_score=float(c.get("min_score", 6.0)),
                                 aliases=c.get("aliases")))
    if not out:
        raise SystemExit("run 配置缺少 segment_judge 维度（无法复现评测口径）")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="源 run 目录（评测视频 + 原记录评分）")
    ap.add_argument("--config-run", default=None,
                    help="judge 口径快照来源 run（缺省同 --run；老 run 无 config.yaml 时另给）")
    ap.add_argument("--video", default="seg01.mp4", help="评测视频（artifacts/segments/ 下或绝对路径）")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default="experiments", help="judge_repeat 实验基座目录")
    args = ap.parse_args()

    run_dir = Path(args.run)
    cfg_run = Path(args.config_run) if args.config_run else run_dir
    cfg = yaml.safe_load((cfg_run / "config.yaml").read_text(encoding="utf-8"))
    judge_cfg = cfg.get("judge") or {}
    if not judge_cfg.get("adapter"):
        raise SystemExit("配置缺少 judge 段（无法复现裁判口径）")
    crit = _criteria(cfg)

    seg_dir = run_dir / "artifacts" / "segments"
    video = Path(args.video)
    if video.parent == Path("."):
        video = seg_dir / video          # 裸文件名 → 源 run 的 segments/
    elif not video.is_absolute():
        video = Path.cwd() / video       # 相对路径 → 相对当前目录
    if not video.exists():
        raise SystemExit(f"找不到视频: {video}（artifacts/segments/ 下可选: "
                         f"{sorted(p.name for p in seg_dir.glob('*.mp4'))}）")

    # 该 run 原记录的评分（同视频同维度，对比基线）
    recorded = None
    evals_file = run_dir / "eval" / "segments.json"
    if evals_file.exists():
        evals = json.loads(evals_file.read_text(encoding="utf-8"))
        for r in evals:
            if isinstance(r, dict) and video.name in str(r.get("artifact", "")):
                recorded = r.get("scores")
                break

    load_builtin_adapters()
    judge = instantiate(judge_cfg["adapter"], judge_cfg.get("params") or {},
                        context="judge_repeat")
    print(f"裁判: {judge.name}（配置快照: {cfg_run / 'config.yaml'}）")
    print(f"视频: {video}")
    print(f"重复评测 {args.repeats} 次 | 维度: {[c.name for c in crit]}")
    if recorded:
        print(f"原记录评分: {recorded}")

    exp = Experiment(task="judge_repeat", base_dir=Path(args.out))
    exp.bind_query(f"裁判重复评测方差分解: {run_dir.name}/{video.name} x{args.repeats}")
    exp.snapshot_config({"source_run": str(run_dir), "config_run": str(cfg_run),
                         "video": str(video), "repeats": args.repeats,
                         "judge": judge_cfg,
                         "segment_judge": cfg.get("segment_judge")})
    exp.set_meta("source_run", str(run_dir))
    exp.set_meta("config_run", str(cfg_run))
    exp.set_meta("video", str(video))

    dims: dict = {}
    for r in range(1, args.repeats + 1):
        t0 = time.time()
        verdict = run_judge(judge, [video], crit, exp.artifacts_dir / "judge", exp=exp)
        dt = round(time.time() - t0, 1)
        exp.save_eval("repeat", [{"repeat": r, "elapsed_s": dt, **verdict}])
        for k, v in verdict.get("scores", {}).items():
            dims.setdefault(k, []).append(float(v))
        print(f"  [{r}/{args.repeats}] {verdict.get('scores')} "
              f"passed={verdict.get('passed')} ({dt}s)")

    summary = {}
    for k, vals in dims.items():
        mean = round(sum(vals) / len(vals), 2)
        sd = round((sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5, 2)
        summary[k] = {"mean": mean, "sd": sd, "min": min(vals), "max": max(vals),
                      "vals": vals}
    summary["recorded"] = recorded
    exp.save_eval("summary", [{"dims": summary}])
    summary_path = exp.artifacts_dir / "summary" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    exp.finalize()

    print("\n== 裁判重复性（同视频 ×%d）==" % args.repeats)
    for k, s in summary.items():
        if k == "recorded":
            continue
        print(f"  {k}: mean={s['mean']} sd={s['sd']} "
              f"range=[{s['min']}, {s['max']}]  vals={s['vals']}")
    if recorded:
        print(f"原记录（该 run 当时）: {recorded}")
    print(f"实验目录: {exp.root}")
    print(f"总耗时: {exp.manifest['total_elapsed_s']:.0f}s")


if __name__ == "__main__":
    main()
