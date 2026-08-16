"""衔接策略对比分析：聚合不同 chain_mode 实验的评测证据。

用法：python scripts/compare_chains.py experiments/story_short
输出：链式策略对比表（跨段一致性/叙事推进/段均分/旁白验证/成本耗时）

聚合从 vidharness.core.report.collect() 取（唯一正源，不再本脚本内重复
汇总评测）；衔接模式优先读 manifest.chain_mode，旧 run 回退 config.yaml
快照，两者都无则归入 "?"。
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vidharness.core.report import collect  # noqa: E402


def chain_mode_of(run_dir: Path, manifest_mode) -> str:
    """衔接模式正源链：manifest（新口径）→ config.yaml 快照 → "?"。"""
    if manifest_mode:
        return str(manifest_mode)
    cfg_file = run_dir / "config.yaml"
    if cfg_file.exists():
        try:
            cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
            return str(cfg.get("pipeline", {}).get("context", {}).get("chain_mode", "?"))
        except Exception:
            return "?"
    return "?"


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 2) if vals else None


def main(base: str):
    base = Path(base)
    if not base.exists():
        print(f"未找到实验目录: {base}")
        sys.exit(1)

    runs = collect(base.parent, base.name)
    by_mode = defaultdict(list)
    per_run = []
    for r in runs:
        mode = chain_mode_of(Path(r["dir"]), r.get("chain_mode"))
        seg_scores = r.get("stage_scores", {}).get("segments", {})
        seg_avg = _mean(seg_scores.values())
        cross = r.get("stage_scores", {}).get("cross_consistency", {})
        audio = r.get("stage_passed", {}).get("audio_verify", {})
        row = {
            "mode": mode,
            "run_id": r["run_id"],
            "seg_avg_score": seg_avg,
            "cross_consistency": cross.get("跨段一致性"),
            "narrative_progression": cross.get("叙事推进"),
            "audio_narration_pass": audio.get("passed"),
            "audio_total": audio.get("total"),
            "elapsed_min": round(r["total_elapsed_s"] / 60, 1),
            "gpu_hours": r.get("local_gpu_hours"),
        }
        per_run.append(row)
        by_mode[mode].append(row)

    print(json.dumps(per_run, ensure_ascii=False, indent=2))

    # markdown 表（每模式多 run 时取均值）
    print("\n| 模式 | runs | 段均分 | 跨段一致性 | 叙事推进 | 旁白验证 | GPU小时 |")
    print("|---|---|---|---|---|---|---|")
    for mode in sorted(by_mode, key=lambda m: (m == "?", m)):
        rs = by_mode[mode]
        audio_pass = sum(r["audio_narration_pass"] or 0 for r in rs)
        audio_total = sum(r["audio_total"] or 0 for r in rs)
        audio = f"{audio_pass}/{audio_total}" if audio_total else "-"
        print(f"| {mode} | {len(rs)} | {_mean([r['seg_avg_score'] for r in rs])} | "
              f"{_mean([r['cross_consistency'] for r in rs])} | "
              f"{_mean([r['narrative_progression'] for r in rs])} | {audio} | "
              f"{_mean([r['gpu_hours'] for r in rs])} |")
    if "?" in by_mode:
        print("\n注：'?' = 旧实验缺少 chain_mode 记录与配置快照。")


if __name__ == "__main__":
    main(sys.argv[1])
