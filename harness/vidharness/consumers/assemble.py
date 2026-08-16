"""成片总装：FFmpeg 拼接 + 旁白字幕烧录。

两种模式：
- audio_in_video=True：片段自带原生音轨（如 H3），直接 concat + 烧字幕；
- audio_in_video=False：外部音频（未来适配器）与视频对齐后再拼。
旁白文本是剧本自带的 → 字幕按片段时长生成 SRT，无需 ASR（未来处理外部素材时再接 ASR 适配器）。
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import List


def _ffprobe_duration(path: Path) -> float:
    from .tools import require_ffprobe
    require_ffprobe()
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except Exception:
        raise RuntimeError(f"无法读取视频时长（ffprobe 失败）: {path}: {out.stderr[:200]}")


def _gen_srt(narrations: List[str], durations: List[float], out: Path):
    lines = []
    t = 0.0
    for i, (text, dur) in enumerate(zip(narrations, durations), 1):
        if not text:
            continue
        start, end = t, t + max(dur, 1.0)
        lines.append(f"{i}\n{_fmt(start)} --> {_fmt(end)}\n{text}\n")
        t = end
    out.write_text("\n".join(lines), encoding="utf-8")


def _fmt(sec: float) -> str:
    ms = int(sec * 1000)
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def assemble_final(videos: List[Path], audios: List[Path], narrations: List[str],
                   out_dir: Path, audio_in_video: bool = False) -> Path:
    from .tools import require_ffmpeg
    require_ffmpeg()
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / "final_video.mp4"

    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        durations = [_ffprobe_duration(v) for v in videos]

        # 1) 拼接
        lst = td / "list.txt"
        lst.write_text("".join(f"file '{v.resolve()}'\n" for v in videos), encoding="utf-8")
        merged = td / "merged.mp4"
        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                        "-c", "copy", str(merged)], capture_output=True, check=True)

        # 2) 字幕
        if any(narrations):
            srt = td / "sub.srt"
            _gen_srt(narrations, durations, srt)
            sub_ = td / "merged_sub.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(merged), "-vf",
                 f"subtitles={srt}:force_style='FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H80000000'",
                 "-c:a", "copy", str(sub_)],
                capture_output=True, check=True)
            merged = sub_

        # 跨文件系统安全移动（/tmp 与实验目录可能不在同一设备）
        import shutil
        shutil.move(str(merged), str(final))
    return final
