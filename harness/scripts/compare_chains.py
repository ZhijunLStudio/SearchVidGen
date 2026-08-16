"""衔接策略对比分析：聚合不同 chain_mode 实验的评测证据。

用法：python scripts/compare_chains.py experiments/story_short
输出：链式策略对比表（跨段一致性/叙事推进/段均分/旁白验证/成本耗时）

衔接模式从每个 run 的 config.yaml 快照读取（2026-08-16 起每次运行都会
冻结有效配置到实验目录）；没有快照的旧实验归入 "?"（不再硬编码 run_id）。
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml


def load_run(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text()) if (run_dir / "manifest.json").exists() else {}
    evals = {}
    for f in (run_dir / "eval").glob("*.json"):
        try:
            evals[f.stem] = json.loads(f.read_text())
        except Exception:
            pass
    return {"manifest": manifest, "evals": evals}


def chain_mode_of(run_dir: Path) -> str:
    """从 run 目录内的配置快照读取 chain_mode；无快照则返回 "?"。"""
    cfg_file = run_dir / "config.yaml"
    if cfg_file.exists():
        try:
            cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
            return str(cfg.get("pipeline", {}).get("context", {}).get("chain_mode", "?"))
        except Exception:
            return "?"
    return "?"


def summarize(run_dir: Path) -> dict:
    d = load_run(run_dir)
    evals = d["evals"]
    seg = evals.get("segments", [])
    seg_scores = [r.get("score") for r in seg if isinstance(r, dict) and r.get("score") is not None]
    cross = evals.get("cross_consistency", [])
    consistency = [r.get("scores", {}).get("跨段一致性") for r in cross if isinstance(r, dict)]
    progression = [r.get("scores", {}).get("叙事推进") for r in cross if isinstance(r, dict)]
    audio = evals.get("audio_verify", [])
    audio_passed = [r.get("passed") for r in audio if isinstance(r, dict)]
    m = d["manifest"]

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 2) if xs else None

    return {
        "mode": chain_mode_of(run_dir),
        "run_id": run_dir.name,
        "segments": len(seg),
        "seg_avg_score": mean(seg_scores),
        "cross_consistency": mean(consistency),
        "narrative_progression": mean(progression),
        "audio_narration_pass": (sum(1 for p in audio_passed if p) if audio_passed else None),
        "audio_total": len(audio_passed),
        "elapsed_min": round(m.get("total_elapsed_s", 0) / 60, 1),
        "gpu_hours": m.get("local_gpu_hours"),
    }


def main(base: str):
    base = Path(base)
    if not base.exists():
        print(f"未找到实验目录: {base}")
        sys.exit(1)

    by_mode = defaultdict(list)
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        row = summarize(d)
        by_mode[row["mode"]].append(row)

    rows = [r for rs in by_mode.values() for r in rs]
    print(json.dumps(rows, ensure_ascii=False, indent=2))

    # markdown 表（每模式多 run 时取均值）
    print("\n| 模式 | runs | 段均分 | 跨段一致性 | 叙事推进 | 旁白验证 | GPU小时 |")
    print("|---|---|---|---|---|---|---|")
    for mode in sorted(by_mode, key=lambda m: (m == "?", m)):
        rs = by_mode[mode]

        def mean(vals):
            vals = [v for v in vals if v is not None]
            return round(sum(vals) / len(vals), 2) if vals else None

        audio_pass = sum(r["audio_narration_pass"] or 0 for r in rs)
        audio_total = sum(r["audio_total"] or 0 for r in rs)
        audio = f"{audio_pass}/{audio_total}" if audio_total else "-"
        print(f"| {mode} | {len(rs)} | {mean([r['seg_avg_score'] for r in rs])} | "
              f"{mean([r['cross_consistency'] for r in rs])} | "
              f"{mean([r['narrative_progression'] for r in rs])} | {audio} | "
              f"{mean([r['gpu_hours'] for r in rs])} |")
    if "?" in by_mode:
        print("\n注：'?' = 旧实验缺少配置快照（2026-08-16 前的 run 未冻结 config.yaml）。")


if __name__ == "__main__":
    main(sys.argv[1])
