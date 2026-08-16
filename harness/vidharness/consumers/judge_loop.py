"""Judge 闭环：生成 → 评测 → 失败反馈重试。

这是 harness 与普通流水线的本质区别 —— 质量验证是一等公民，
而不是"生成完就交付"。

职责边界（对齐 deepseek-harness 的"显式 > 隐式"）：
- parse_scores：从 judge 文本回复提取原始评分（解析归提供者侧能力，输入兼容性由
  规格里的 aliases 兜底）。
- finalize_verdict：加权/阈值判定。评测策略（weight/min_score）是任务配置的
  拥有物，由消费者统一结算 —— 提供者不得替消费者算总分，否则 YAML 的权重
  在协议传递中被静默丢弃（2026-08-16 修复的 Bug#1）。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from ..seams import JudgeCriteria, RetryPolicy, criteria_to_spec
from ..core.registry import resolve


def parse_scores(text: str, criteria: List[JudgeCriteria]) -> Tuple[Dict[str, float], str]:
    """从 judge 文本回复中提取原始评分（不做加权/阈值判定）。

    返回 (scores, feedback)。优先解析 JSON；否则用正则匹配
    "维度名: 分数" 或 "score: N" 模式（别名兜底）。
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
                    return scores, feedback
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
    return scores, feedback


def finalize_verdict(scores: Dict[str, float], feedback: str,
                     criteria: List[JudgeCriteria]) -> Dict[str, Any]:
    """加权结算与阈值判定 —— 评测策略的唯一归属点。

    缺失维度计 0 分并判未通过（宁严勿松：解析失败不得静默通过）。
    """
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


def parse_judge_output(text: str, criteria: List[JudgeCriteria]) -> Dict[str, Any]:
    """兼容封装：parse_scores + finalize_verdict（旧调用点/外部脚本使用）。"""
    scores, feedback = parse_scores(text, criteria)
    return finalize_verdict(scores, feedback, criteria)


def run_judge(judge: Any, media, criteria: List[JudgeCriteria],
              workdir, exp=None) -> Dict[str, Any]:
    """统一评测调用：传完整规格（含权重/阈值/别名），回来统一 finalize。

    所有消费者（judge_loop / script / cross_consistency / optimizer）
    都经此结算，保证评测策略只在一处生效。
    传 exp 时裁判原始输出作为产物存档（artifacts/judge/，经事件流，
    对齐"模型可见 ⟺ 日志"——每次裁判调用可重建）。
    """
    art = judge.judge(media=media, criteria=criteria_to_spec(criteria), workdir=workdir)
    if exp is not None:
        exp.save_artifact("judge", art)
    payload = art.payload or {}
    return finalize_verdict(payload.get("scores", {}), payload.get("feedback", ""), criteria)


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
            exp.record_retry(stage)

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
        verdict = run_judge(judge_obj, media, criteria,
                            exp.artifacts_dir / "judge", exp=exp)
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
