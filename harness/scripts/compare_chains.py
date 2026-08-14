"""衔接策略对比分析：聚合 hard/none/ref 三个实验的评测证据。

用法：python scripts/compare_chains.py experiments/story_short
输出：链式策略对比表（跨段一致性/叙事推进/段均分/旁白验证/成本耗时）
"""
import json
import sys
from pathlib import Path


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
    """从任务配置推断：检查 run 目录内是否存了 config（未存则按规则猜）。"""
    cfg_file = run_dir / "config.yaml"
    if cfg_file.exists():
        return json.loads(cfg_file.read_text()).get("pipeline", {}).get("context", {}).get("chain_mode", "?")
    # 目录名/时间推断不了 —— 由调用方传入映射
    return "?"


def summarize(run_dir: Path, mode: str) -> dict:
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
        "mode": mode,
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
    # 模式 → run 目录映射：按创建顺序推断（hard 最早、none 次之、ref 最新）
    runs = sorted([d for d in base.iterdir() if d.is_dir()], key=lambda d: d.name)
    mapping = [
        ("hard", "20260815_000309_b2ddfd"),   # 实验二（hard）
        ("none", "20260815_035745_dc6795"),   # 实验三（none）
        ("ref",  None),                        # 实验四（ref，运行中）
    ]
    rows = []
    for mode, rid in mapping:
        run_dir = None
        if rid:
            run_dir = base / rid
        else:
            cands = [d for d in runs if d.name.startswith("20260815_05")]
            run_dir = sorted(cands)[-1] if cands else None
        if run_dir and run_dir.exists():
            rows.append(summarize(run_dir, mode))
        else:
            rows.append({"mode": mode, "run_id": None, "note": "未找到/未完成"})

    print(json.dumps(rows, ensure_ascii=False, indent=2))
    # markdown 表
    print("\n| 模式 | 段均分 | 跨段一致性 | 叙事推进 | 旁白验证 | GPU小时 |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        if r.get("run_id") is None:
            print(f"| {r['mode']} | - | - | - | - | - |")
            continue
        audio = f"{r['audio_narration_pass']}/{r['audio_total']}" if r["audio_total"] else "-"
        print(f"| {r['mode']} | {r['seg_avg_score']} | {r['cross_consistency']} | "
              f"{r['narrative_progression']} | {audio} | {r['gpu_hours']} |")


if __name__ == "__main__":
    main(sys.argv[1])
