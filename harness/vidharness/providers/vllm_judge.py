"""通用 LLM/VLM Judge 适配器（OpenAI 兼容端点）。

- 文本评测：任意 OpenAI 兼容 LLM（DeepSeek API、本地 vLLM）
- 图像评测：OpenAI 兼容的多模态端点（本地 vLLM 服务的 Qwen3.5-27B 等）
- 视频评测：抽帧为图像序列（frame_sampling），逐帧+整体双问询

DeepSeek V4 官方 API 当前不支持图像输入，因此图像/视频评测默认应指向本地 vLLM。
"""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

from ..seams import Artifact, ArtifactMeta
from ..core.registry import register


def _img_b64(path: Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


def _extract_frames(video: Path, n: int = 4, workdir: Path = Path(".")) -> List[Path]:
    """从视频抽 n 帧（唯一实现点在 consumers/tools.sample_frames）。"""
    from ..consumers.tools import sample_frames
    return sample_frames(video, n, workdir)


@register("judge.openai-compat")
class OpenAICompatJudge:
    """judge 协议实现：OpenAI 兼容 chat completions（支持图像 base64）。"""

    name = "judge.openai-compat"
    modalities = ["text", "image", "video"]
    capabilities = {"frame_sampling": True}
    param_schema = {
        "base_url": {"type": "str", "required": True, "help": "OpenAI 兼容端点（本地 vLLM）"},
        "model": {"type": "str", "required": True, "help": "served-model-name"},
        "api_key": {"type": "secret", "default": "EMPTY"},
        "temperature": {"type": "float", "default": 0.0},
        "max_tokens": {"type": "int", "default": 4096},
        "frame_samples": {"type": "int", "default": 4, "help": "视频抽帧数"},
        "disable_thinking": {"type": "bool", "default": True, "help": "关闭思考防 token 燃烧"},
    }

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY",
                 temperature: float = 0.0, max_tokens: int = 4096,
                 frame_samples: int = 4, disable_thinking: bool = True):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.disable_thinking = disable_thinking
        self.max_tokens = max_tokens
        self.frame_samples = frame_samples

    def judge(self, media: List[Path], criteria: Dict[str, str], workdir: Path, **kw) -> Artifact:
        t0 = time.time()
        # 媒体归一化：视频抽帧，图像原样
        images: List[Path] = []
        for m in media:
            if m.suffix.lower() in (".mp4", ".mov", ".mkv", ".avi", ".webm"):
                images.extend(_extract_frames(m, self.frame_samples, workdir))
            elif m.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
                images.append(m)

        prompt_lines = [
            "你是一个严格的视频生成质量评审员。请依据以下维度给媒体内容打分（每项 0-10 分，10 为完美）。",
            "媒体内容以图像序列给出（可能是一张图，也可能是从同一视频抽出的多个帧）。",
            "",
            "评分纪律（必须遵守）：",
            "- 10 分只给无可挑剔的完美输出；9 分给极轻微瑕疵；存在任何明显缺陷（畸形、崩坏、",
            "  文字乱码、风格割裂、主体不符、冻结帧）最多 6 分；严重缺陷给 1-3 分。",
            "- 平均分不应超过 8 分，除非每一帧都确实完美。",
            "",
            "评分维度：",
        ]
        for i, (name, spec_v) in enumerate(criteria.items(), 1):
            q = spec_v.get("question", "") if isinstance(spec_v, dict) else spec_v
            prompt_lines.append(f"{i}. {name}: {q}")
        prompt_lines += [
            "",
            "请先输出一个 JSON 对象（这是唯一需要的内容），格式：",
            '{"<维度名>": <分数0-10>, "feedback": "<若不达标，用一句中文说明最需要修正的问题；达标则写 pass>"}',
            "禁止输出思考过程、分析或任何其他文字，只输出 JSON。",
        ]
        text = "\n".join(prompt_lines)

        content: List[Dict[str, Any]] = []
        for img in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{_img_b64(img)}"},
            })
        content.append({"type": "text", "text": text})

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.disable_thinking:
            # Qwen3 系思考开关（vLLM chat_template_kwargs 透传）
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        resp = self.client.chat.completions.create(**kwargs)
        out = resp.choices[0].message.content or ""

        from ..seams import spec_to_criteria
        from ..consumers.judge_loop import parse_scores, unparseable_feedback
        crits = spec_to_criteria(criteria)
        scores, feedback = parse_scores(out, crits)
        if not scores:
            feedback = unparseable_feedback(out)   # 可操作反馈（E21）

        # 可重建：raw 输出 + 输入规格 + 媒体清单全部落盘（对齐"模型可见⟺日志"）
        path = workdir / f"judge_{int(time.time())}.json"
        path.write_text(json.dumps(
            {"raw": out, "criteria": criteria, "scores": scores, "feedback": feedback,
             "media": [str(m) for m in media],
             "frames": [str(f) for f in images]},
            ensure_ascii=False, indent=2), encoding="utf-8")
        meta = ArtifactMeta(adapter=self.name, model=self.model,
                            elapsed_s=time.time() - t0,
                            params={"frame_samples": self.frame_samples})
        return Artifact(kind="scores", path=path, meta=meta,
                        payload={"scores": scores, "feedback": feedback})
