"""音频验证消费者：转写视频原生音频，与剧本旁白比对。

验证对象：全模态模型生成的"原生音频"与剧情文本的一致性
（对白是否说了该说的、旁白是否匹配、是否有异常静默/杂音）。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from ..core.registry import resolve
from ..seams import Artifact


def _norm(text: str) -> str:
    """归一化中文文本用于比对：去标点/空白/常见语气词。"""
    t = re.sub(r"[\s，。！？、,.!?…—\-]", "", text)
    return t


def _char_overlap(a: str, b: str) -> float:
    """字符集合重叠度（简单、可解释的相似度）。"""
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(len(sa), len(sb))


def verify_segment_audio(video: Path, expected_text: str, transcriber: Any,
                         exp, stage: str = "audio_verify") -> Dict[str, Any]:
    """单段音频验证：转写 → 与期望旁白比对 → 记录评测。"""
    trans_obj = resolve(transcriber) if isinstance(transcriber, str) else transcriber
    if isinstance(trans_obj, type):
        trans_obj = trans_obj()          # 提供者注册的是类，实例化
    art: Artifact = trans_obj.transcribe(video, workdir=exp.artifacts_dir / stage)
    payload = art.payload
    exp.save_artifact(stage, art, name=video.stem)

    expected_norm = _norm(expected_text)
    got_norm = _norm(payload["text"])
    overlap = _char_overlap(expected_norm, got_norm)

    # 判定：期望文本为空（纯音效段）→ 检查是否有异常长语音；否则比对重叠度
    if expected_norm:
        passed = overlap >= 0.5
    else:
        passed = len(got_norm) <= 4      # 纯音效段不应有长句

    record = {
        "video": str(video),
        "expected": expected_text,
        "transcript": payload["text"],
        "emotion": payload.get("emotion"),
        "has_bgm": payload.get("has_bgm"),
        "char_overlap": round(overlap, 3),
        "passed": passed,
    }
    exp.save_eval(stage, [record])
    return record


def verify_film(videos: List[Path], narrations: List[str], transcriber: Any, exp) -> List[Dict[str, Any]]:
    """成片级：逐段音频验证。"""
    records = []
    for v, n in zip(videos, narrations):
        records.append(verify_segment_audio(v, n, transcriber, exp))
    return records
