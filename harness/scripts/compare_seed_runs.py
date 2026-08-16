"""同种子/异种子生成对比：解码两个 run 的生成视频，逐帧像素差（MAE）。

E43 工具：seed 是唯一变量的基准矩阵跑完后，用本脚本量化
同种子对（应≈0）与异种子对（应>0）的像素距离，验证
"种子控制生成 → 生成侧方差可被种子对齐"的配对 A/B 前提。

用法：
  python scripts/compare_seed_runs.py --a <run_a> --b <run_b> [--frames 6]

帧提取用 ffmpeg（1 fps 等距采样），MAE 用 numpy/PIL；
输出逐帧 MAE + 均值（0-255 尺度）。两个 run 都取
artifacts/segments/ 下第一个 mp4（纯生成视频，无总装）。
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _video_of(run_dir: Path) -> Path:
    segs = sorted((run_dir / "artifacts" / "segments").glob("*.mp4"))
    if segs:
        return segs[0]
    final = run_dir / "final" / "final_video.mp4"
    if final.exists():
        return final
    raise SystemExit(f"{run_dir}: 找不到生成视频（segments/ 或 final/）")


def _frames(video: Path, n: int, out_dir: Path) -> list:
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
                    "-vf", "fps=1,scale=448:256", "-frames:v", str(n),
                    str(out_dir / "f_%02d.png")], check=True)
    return sorted(out_dir.glob("f_*.png"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="run A 目录")
    ap.add_argument("--b", required=True, help="run B 目录")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--workdir", default="/tmp/vh-seed-cmp")
    args = ap.parse_args()

    import numpy as np
    from PIL import Image

    va, vb = _video_of(Path(args.a)), _video_of(Path(args.b))
    print(f"A: {va}")
    print(f"B: {vb}")
    fa = _frames(va, args.frames, Path(args.workdir) / "a")
    fb = _frames(vb, args.frames, Path(args.workdir) / "b")
    n = min(len(fa), len(fb))
    mae = []
    for i in range(n):
        a = np.asarray(Image.open(fa[i]).convert("RGB"), dtype=np.float32)
        b = np.asarray(Image.open(fb[i]).convert("RGB"), dtype=np.float32)
        d = float(np.abs(a - b).mean())
        mae.append(d)
        print(f"  帧{i + 1:02d}: MAE={d:.2f} (0-255)")
    mean = sum(mae) / len(mae)
    print(f"平均 MAE: {mean:.2f}（同种子对应≈0，异种子对应>0）")
    print("video_a:", str(va))
    print("video_b:", str(vb))


if __name__ == "__main__":
    main()
