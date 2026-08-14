"""Judge 闭环：生成 → 评测 → 失败反馈重试。

这是 harness 与普通流水线的本质区别 —— 质量验证是一等公民，
而不是"生成完就交付"。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from ..seams import JudgeCriteria, RetryPolicy
from ..core.registry import resolve


def parse_judge_output(text: str, criteria: List[JudgeCriteria]) -> Dict[str, Any]:
    """从 judge 文本回复中提取结构化评分。

    优先解析 JSON；否则用正则匹配 "维度名: 分数" 或 "score: N" 模式。
    """
    text = text.strip()
    # 1) 尝试整段 JSON
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            import json
            data = json.loads(m.group(0))
            scores, feedback = {}, ""
            if isinstance(data, dict):
                for c in criteria:
                    v = data.get(c.name) or data.get(c.name.lower())
                    if isinstance(v, (int, float)):
                        scores[c.name] = float(v)
                feedback = data.get("feedback", "") or ""
                if scores:
                    return _finalize(scores, feedback, criteria)
        except Exception:
            pass
    # 2) 正则兜底："维度名[:：]\s*(\d+(\.\d+)?)"；支持别名（如"一致性"匹配"与指令一致性"）
    scores, feedback = {}, text
    terms = []
    for c in criteria:
        aliases = getattr(c, "aliases", None) or [c.name]
        if c.name not in aliases:
            aliases = [c.name] + list(aliases)
        terms.append((c, aliases))
    for c, aliases in terms:
        for term in aliases:
            m = re.search(rf"{re.escape(term)}\s*[:：]?\s*(\d+(?:\.\d+)?)", text)
            if m:
                scores[c.name] = float(m.group(1))
                break
    if not scores:
        # 任何 "N/10" 或 "N分" 视作第一维度的分数
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:/10|分)", text)
        if m and criteria:
            scores[criteria[0].name] = float(m.group(1))
    return _finalize(scores, feedback, criteria)


def _finalize(scores: Dict[str, float], feedback: str, criteria: List[JudgeCriteria]) -> Dict[str, Any]:
    total, weight_sum = 0.0, 0.0
    passed = True
    for c in criteria:
        s = scores.get(c.name)
        if s is None:
            s = 0.0
            passed = False  # 缺失维度视为未通过
        total += s * c.weight
        weight_sum += c.weight
    score = total / weight_sum if weight_sum else 0.0
    if any(scores.get(c.name, 0) < c.min_score for c in criteria):
        passed = False
    return {
        "scores": scores,
        "score": round(score, 2),
        "passed": passed,
        "feedback": feedback,
    }


def run_with_judge(
    adapter: Any,
    judge: Any,
    criteria: List[JudgeCriteria],
    retry: RetryPolicy,
    generate_inputs: Dict[str, Any],
    media_collector,
    exp,
    stage: str,
    name: Optional[str] = None,
    **gen_kw,
) -> Tuple[Any, List[Dict[str, Any]]]:
    """带评测闭环的单产物生成。

    generate_inputs: 传给 adapter.generate 的输入 dict；
    media_collector: callable(artifact) -> List[Path] 取评测媒体（图像/帧/音频）；
    返回 (最终产物, 评测历史)。
    """
    judge_obj = resolve(judge) if isinstance(judge, str) else judge
    history: List[Dict[str, Any]] = []
    feedback = ""
    last: Any = None

    for attempt in range(1, retry.max_attempts + 1):
        if attempt > 1:
            exp.manifest.setdefault("retries", {})
            exp.manifest["retries"].setdefault(stage, 0)
            exp.manifest["retries"][stage] += 1

        # 1) 生成
        inputs = dict(generate_inputs)
        if retry.inject_feedback and feedback:
            req = inputs.get("req")
            if req is not None:
                req.text = f"{req.text}\n{retry.feedback_prefix}{feedback}"
            elif "prompt" in inputs:
                inputs["prompt"] = f"{inputs['prompt']}\n{retry.feedback_prefix}{feedback}"
            gen_kw["feedback"] = feedback
        last = adapter.generate(workdir=exp.artifacts_dir / stage, **inputs, **gen_kw)
        exp.save_artifact(stage, last, name=name)

        # 2) 评测
        if not criteria or judge_obj is None:
            return last, history
        media = media_collector(last)
        if not media:
            return last, history
        res_artifact = judge_obj.judge(
            media=media,
            criteria={c.name: c.question for c in criteria},
            workdir=exp.eval_dir,
        )
        verdict = res_artifact.payload
        record = {
            "attempt": attempt,
            "artifact": str(last.path),
            **verdict,
        }
        history.append(record)
        exp.save_eval(stage, history)
        if verdict.get("passed"):
            return last, history
        feedback = verdict.get("feedback", "")

    return last, history
