"""SenseVoice ASR 提供者（CPU 可跑，中文友好，输出情绪/BGM 标签）。"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

from ..core.registry import register
from ..seams import Artifact, ArtifactMeta


@register("transcribe.sensevoice-small")
class SenseVoiceTranscriber:
    name = "transcribe.sensevoice-small"
    capabilities = {"language": "zh+multi", "device": "cpu", "emotion_tags": True}

    def __init__(self, device: str = "cpu", language: str = "zh", use_itn: bool = True):
        self.device = device
        self.language = language
        self.use_itn = use_itn
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._ensure_ffmpeg()
            from funasr import AutoModel
            self._model = AutoModel(model="iic/SenseVoiceSmall", device=self.device,
                                    disable_update=True)
        return self._model

    @staticmethod
    def _ensure_ffmpeg():
        """funasr 调子进程 ffmpeg，需在 PATH 上；找不到就补充常见 conda 环境路径。"""
        import os
        import shutil
        if shutil.which("ffmpeg"):
            return
        for cand in [
            "/data/lizhijun/anaconda3/envs/torch/bin",
            "/data/lizhijun/anaconda3/envs/vllm/bin",
        ]:
            if (Path(cand) / "ffmpeg").exists():
                os.environ["PATH"] = cand + os.pathsep + os.environ.get("PATH", "")
                return
        raise RuntimeError("未找到 ffmpeg：请安装或配置 PATH（funasr 依赖）")

    def transcribe(self, media: Path, workdir: Path, **kw) -> Artifact:
        workdir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        model = self._get_model()
        res = model.generate(input=str(media), language=kw.get("language", self.language),
                             use_itn=kw.get("use_itn", self.use_itn))
        raw = res[0]["text"] if res else ""
        # 解析 SenseVoice 标签：<|zh|><|EMO_..|><|BGM|>...
        import re
        tags = re.findall(r"<\|([^|]+)\|>", raw)
        text = re.sub(r"<\|[^|]+\|>", "", raw).strip()
        payload = {
            "text": text,
            "tags": tags,
            "emotion": next((t for t in tags if t.startswith("EMO_")), None),
            "has_bgm": "BGM" in tags,
        }
        out = workdir / f"transcript_{int(time.time())}.json"
        import json
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        meta = ArtifactMeta(adapter=self.name, model="iic/SenseVoiceSmall",
                            elapsed_s=time.time() - t0, cost_usd=0.0)
        return Artifact(kind="transcript", path=out, meta=meta, payload=payload)
