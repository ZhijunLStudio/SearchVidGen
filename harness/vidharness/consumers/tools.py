"""外部工具检查 —— 配置错误响亮失败（fail loud）。

ffmpeg/ffprobe 是 harness 的硬依赖；不在 PATH 时给出明确指引，
而不是让 subprocess 裸崩 FileNotFoundError。
"""
from __future__ import annotations

import shutil


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
