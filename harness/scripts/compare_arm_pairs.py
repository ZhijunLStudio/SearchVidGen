"""配对 A/B 臂间对比分析（E44/E45 通用工具）。

两个 run 目录（同基准任务、唯一变量不同）的完整差异面：
- 剧本：逐段 video_prompt 是否相同 + 段数
- 段视频：逐段像素 MAE（复用 compare_seed_runs 的抽帧口径）
- 评分：segment / cross_consistency 两臂逐条对比
- 汇总：下游主指标差值

用法：
  python scripts/compare_arm_pairs.py --a <run_a> --b <run_b> [--frames 4]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compare_seed_runs import _frames  # noqa: E402


def _mae(pa: Path, pb: Path, frames: int, workdir: Path) -> float:
    import numpy as np
    from PIL import Image
    fa = _frames(pa, frames, workdir / "a")
    fb = _frames(pb, frames, workdir / "b")
    n = min(len(fa), len(fb))
    d = 0.0
    for i in range(n):
        a = np.asarray(Image.open(fa[i]).convert("RGB"), dtype=np.float32)
        b = np.asarray(Image.open(fb[i]).convert("RGB"), dtype=np.float32)
        d += float(np.abs(a - b).mean())
    return round(d / n, 2) if n else 0.0


def _scripts(run: Path) -> dict:
    files = sorted((run / "artifacts" / "script").glob("script*.json"))
    if not files:
        return {}
    return json.loads(files[0].read_text(encoding="utf-8"))


def _evals(run: Path, stage: str) -> list:
    p = run / "eval" / f"{stage}.json"
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    return data if isinstance(data, list) else []


def _fmt(records, stage: str) -> str:
    out = []
    for r in records:
        if not isinstance(r, dict):
            continue
        if r.get("error"):
            out.append(f"error={r['error'][:40]}")
            continue
        scores = r.get("scores", {})
        passed = "✓" if r.get("passed") else "✗"
        out.append(f"{scores} {passed}")
    return " | ".join(out) if out else "（无记录）"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="臂 A run 目录")
    ap.add_argument("--b", required=True, help="臂 B run 目录")
    ap.add_argument("--frames", type=int, default=4)
    args = ap.parse_args()
    a, b = Path(args.a), Path(args.b)

    sa, sb = _scripts(a), _scripts(b)
    segs_a, segs_b = sa.get("segments", []), sb.get("segments", [])
    print(f"臂 A: {a.name}  臂 B: {b.name}")
    print(f"段数: A={len(segs_a)} B={len(segs_b)}")
    same_prompts = [x.get("video_prompt") == y.get("video_prompt")
                    for x, y in zip(segs_a, segs_b)]
    print(f"剧本逐段相同: {same_prompts}")

    seg_dir_a = a / "artifacts" / "segments"
    seg_dir_b = b / "artifacts" / "segments"
    for i in range(min(len(segs_a), len(segs_b))):
        va = sorted(seg_dir_a.glob(f"seg{i + 1:02d}.mp4"))
        vb = sorted(seg_dir_b.glob(f"seg{i + 1:02d}.mp4"))
        if va and vb:
            m = _mae(va[0], vb[0], args.frames,
                     Path("/tmp/vh-armcmp") / a.name / f"seg{i + 1}")
            print(f"  段{i + 1} 视频 MAE: {m}")

    for stage in ("segments", "cross_consistency"):
        print(f"\n[{stage}] A: {_fmt(_evals(a, stage), stage)}")
        print(f"[{stage}] B: {_fmt(_evals(b, stage), stage)}")
        ea, eb = _evals(a, stage), _evals(b, stage)

        def _mean(records, dim):
            vals = [r["scores"][dim] for r in records
                    if isinstance(r, dict) and isinstance(r.get("scores"), dict)
                    and isinstance(r["scores"].get(dim), (int, float))]
            return round(sum(vals) / len(vals), 2) if vals else None

        dims = set()
        for r in ea + eb:
            if isinstance(r, dict) and isinstance(r.get("scores"), dict):
                dims |= set(r["scores"])
        for dim in sorted(dims):
            ma, mb = _mean(ea, dim), _mean(eb, dim)
            if ma is not None or mb is not None:
                fa, fb = (str(ma), str(mb)) if ma is not None and mb is not None \
                    else ((str(ma), "—") if ma is not None else ("—", str(mb)))
                delta = round((ma or 0) - (mb or 0), 2)
                print(f"  维度 '{dim}': A={fa} B={fb} Δ={delta:+}")


if __name__ == "__main__":
    main()
