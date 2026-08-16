"""外部工具检查与媒体帧抽取 —— 唯一的媒体工具实现点。

ffmpeg/ffprobe 是 harness 的硬依赖；不在 PATH 时给出明确指引，
而不是让 subprocess 裸崩 FileNotFoundError。

帧抽取的三个消费方（director 首/尾帧、vllm_judge 抽帧、collect_evidence）
共用本模块实现（2026-08-16 去重：此前三处各自实现）。
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import List, Optional


def require_tool(name: str, hint: str = "") -> str:
    """返回工具路径；未找到则响亮失败并给出指引。"""
    path = shutil.which(name)
    if not path:
        raise RuntimeError(
            f"未找到 {name}：请安装或配置 PATH。{hint}".rstrip())
    return path


def require_ffmpeg() -> str:
    return require_tool("ffmpeg", hint="（视频抽帧/拼接/保存都依赖它）")


def require_ffprobe() -> str:
    return require_tool("ffprobe", hint="（读取视频时长依赖它）")


def _duration_s(video: Path) -> float:
    require_ffprobe()
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(video)], capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except Exception:
        raise RuntimeError(f"无法读取视频时长（ffprobe 失败）: {video}: {out.stderr[:200]}")


def extract_frame(video: Path, t: float, out_dir: Path) -> Optional[Path]:
    """抽取视频第 t 秒的一帧（按 stem_t{:.2f}.jpg 缓存）；失败返回 None。

    缓存命中不需要 ffmpeg（缓存检查先于工具检查）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{video.stem}_t{t:.2f}.jpg"
    if dst.exists():
        return dst
    require_ffmpeg()
    try:
        subprocess.run(["ffmpeg", "-y", "-ss", str(t), "-i", str(video),
                        "-frames:v", "1", str(dst)], capture_output=True)
    except Exception:
        return None
    return dst if dst.exists() else None


def extract_last_frame(video: Path, out_dir: Path) -> Optional[Path]:
    """抽取视频末帧（时长-0.5s）；失败返回 None。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{video.stem}_last.jpg"
    if dst.exists():
        return dst
    try:
        t = max(0.0, _duration_s(video) - 0.5)
    except Exception:
        return None
    require_ffmpeg()
    try:
        subprocess.run(["ffmpeg", "-y", "-ss", str(t), "-i", str(video),
                        "-frames:v", "1", str(dst)], capture_output=True)
    except Exception:
        return None
    return dst if dst.exists() else None


def sample_frames(video: Path, n: int, out_dir: Path) -> List[Path]:
    """均匀抽 n 帧（优先 ffmpeg，失败退回 imageio）；返回帧路径列表。"""
    out = out_dir / f"{video.stem}_frames"
    out.mkdir(parents=True, exist_ok=True)
    existing = sorted(out.glob("frame_*.jpg"))
    if len(existing) >= n:
        return existing[:n]
    try:
        require_ffmpeg()
        dur = _duration_s(video)
        for i in range(n):
            t = dur * (i + 1) / (n + 1)
            subprocess.run(["ffmpeg", "-y", "-ss", str(t), "-i", str(video),
                            "-frames:v", "1", str(out / f"frame_{i:02d}.jpg")],
                           capture_output=True, check=True)
        return sorted(out.glob("frame_*.jpg"))
    except Exception:
        # 兜底：imageio
        import imageio
        reader = imageio.get_reader(str(video))
        frames = []
        for f in reader:  # type: ignore[attr-defined]  # imageio Reader 运行时可迭代
            frames.append(f)
        step = max(1, len(frames) // n)
        import imageio.v2 as iio
        for i, f in enumerate(frames[::step][:n]):
            iio.imwrite(str(out / f"frame_{i:02d}.jpg"), f)
        return sorted(out.glob("frame_*.jpg"))
