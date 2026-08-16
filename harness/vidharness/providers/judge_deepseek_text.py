"""DeepSeek 文本裁判 —— judge 缝的第二实现（孪生适配器）。

judge.openai-compat（本地 vLLM VLM）负责图像/视频评测；本提供者负责
**纯文本评测**（剧本评审等 media=[] 场景），走 DeepSeek 官方 API：
文本裁判不需要视觉，官方 API 更便宜、不占本地 GPU、且不依赖 vLLM 服务存活。

DeepSeek V4 官方 API 不支持图像输入（E2/RUNBOOK 记录），因此本提供者声明
modalities=["text"]；消费者侧 run_judge 会在媒体评测时拒绝它（fail loud），
这是孪生适配器暴露的 seam 词汇缺口（modalities 此前不可强制）的落地点。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

from ..seams import Artifact, ArtifactMeta, spec_to_criteria
from ..core.registry import register
from ..consumers.judge_loop import parse_scores, unparseable_feedback
from .deepseek_script import _estimate_cost, _load_token


@register("judge.deepseek-text")
class DeepSeekTextJudge:
    """judge 协议实现：DeepSeek 官方 API（文本-only）。"""

    name = "judge.deepseek-text"
    modalities = ["text"]
    capabilities = {"frame_sampling": False}
    param_schema = {
        "api_key": {"type": "secret", "default": None, "help": "DeepSeek API key（缺省读环境）"},
        "base_url": {"type": "str", "default": "https://api.deepseek.com"},
        "model": {"type": "str", "default": "deepseek-chat"},
        "temperature": {"type": "float", "default": 0.0},
        "max_tokens": {"type": "int", "default": 4096},
    }

    def __init__(self, api_key: str | None = None, base_url: str = "https://api.deepseek.com",
                 model: str = "deepseek-chat", temperature: float = 0.0,
                 max_tokens: int = 4096):
        self.client = OpenAI(api_key=api_key or _load_token(), base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def judge(self, media: List[Path], criteria: Dict[str, Any], workdir: Path, **kw) -> Artifact:
        """文本评测（media 应为空；媒体评测由消费者侧模态守卫拒绝）。"""
        workdir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        crits = spec_to_criteria(criteria)
        lines = [
            "你是一个严格的评审员。请依据以下维度给文本内容打分（每项 0-10 分，10 为完美）。",
            "评分纪律（必须遵守）：",
            "- 10 分只给无可挑剔的完美输出；9 分给极轻微瑕疵；存在明显问题最多 6 分；"
            "严重缺陷给 1-3 分。",
            "- 平均分不应超过 8 分，除非每一项都确实完美。",
            "",
            "评分维度：",
        ]
        for i, c in enumerate(crits, 1):
            lines.append(f"{i}. {c.name}: {c.question}")
        lines += [
            "",
            "请先输出一个 JSON 对象（这是唯一需要的内容），格式：",
            '{"<维度名>": <分数0-10>, "feedback": "<若不达标，用一句中文说明最需要修正的问题；达标则写 pass>"}',
            "禁止输出思考过程、分析或任何其他文字，只输出 JSON。",
        ]
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": "\n".join(lines)}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            # JSON 模式（E25）：校准实测文本裁判解析失败率 40%，
            # DeepSeek 官方 JSON 输出从源头保证可解析（提示含 "JSON" 满足前置条件）
            response_format={"type": "json_object"},
        )
        out = resp.choices[0].message.content or ""
        scores, feedback = parse_scores(out, crits)
        if not scores:
            feedback = unparseable_feedback(out)   # 可操作反馈（E21）

        # 可重建：raw 输出 + 输入规格全部落盘（对齐"模型可见 ⟺ 日志"）
        path = workdir / f"judge_ds_{int(time.time())}.json"
        path.write_text(json.dumps(
            {"raw": out, "criteria": criteria, "scores": scores, "feedback": feedback},
            ensure_ascii=False, indent=2), encoding="utf-8")
        usage = resp.usage
        meta = ArtifactMeta(adapter=self.name, model=resp.model,
                            elapsed_s=time.time() - t0,
                            cost_usd=_estimate_cost(
                                usage.prompt_tokens if usage else 0,
                                usage.completion_tokens if usage else 0, self.model))
        return Artifact(kind="scores", path=path, meta=meta,
                        payload={"scores": scores, "feedback": feedback})
