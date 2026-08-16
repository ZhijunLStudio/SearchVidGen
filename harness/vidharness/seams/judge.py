"""评测能力缝（Service Definition）—— harness 的护城河。

模型不会给自己做质检；生成越"一体化"，独立验证越稀缺。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Protocol, runtime_checkable


@runtime_checkable
class Judge(Protocol):
    """多模态评测器：媒体 + 标准 -> 原始评分与反馈。

    协议约定（提供者契约）：
    - criteria 接收 criteria_to_spec() 产出的完整规格 dict（name -> 字段 dict），
      兼容旧式 name -> question 字符串。
    - 返回 Artifact(kind='scores')，payload 只含原始数据：
        {"scores": {维度名: 分数}, "feedback": str}
      加权/阈值判定（评测策略）属于任务配置，由消费者 finalize_verdict 完成，
      提供者不得替消费者计算总分 —— 防止 YAML 的 weight/min_score 在提供者侧丢失。
    """
    name: str
    modalities: List[str]      # ["image","video","audio","text"]

    def judge(self, media: List[Path], criteria: Dict[str, Any], workdir: Path, **kw):
        """返回 Artifact(kind='scores')，payload: {scores, feedback}。"""
        ...


@dataclass
class JudgeCriteria:
    """一条评测维度。aliases：正则兜底解析时的可匹配别名（如 ["一致性"]）。"""
    name: str
    question: str
    weight: float = 1.0
    min_score: float = 6.0
    scale: float = 10.0
    aliases: list = None

    def to_spec(self) -> Dict[str, Any]:
        """本维度的完整规格（传给 judge 提供者的字段 dict）。"""
        return {
            "question": self.question,
            "weight": self.weight,
            "min_score": self.min_score,
            "aliases": self.aliases,
        }


def criteria_to_spec(criteria: List[JudgeCriteria]) -> Dict[str, Dict[str, Any]]:
    """评测维度 -> judge 协议规格 dict（权重/阈值/别名随协议传递，不丢失）。"""
    return {c.name: c.to_spec() for c in criteria}


def spec_to_criteria(spec: Dict[str, Any]) -> List[JudgeCriteria]:
    """judge 协议规格 -> JudgeCriteria 列表。

    兼容两种值形态：旧协议的裸字符串问题（question），或 to_spec 产出的字段 dict。
    字段 dict 里的未知键忽略（前向兼容），关键字段类型错误则响亮失败。
    """
    out: List[JudgeCriteria] = []
    for name, v in spec.items():
        if isinstance(v, str):
            out.append(JudgeCriteria(name=name, question=v))
        elif isinstance(v, dict):
            weight = v.get("weight", 1.0)
            min_score = v.get("min_score", 6.0)
            if not isinstance(weight, (int, float)) or not isinstance(min_score, (int, float)):
                raise TypeError(
                    f"评测维度 '{name}' 的 weight/min_score 必须是数值，得到 {v!r}")
            aliases = v.get("aliases")
            if aliases is not None and not isinstance(aliases, list):
                raise TypeError(f"评测维度 '{name}' 的 aliases 必须是列表，得到 {aliases!r}")
            out.append(JudgeCriteria(
                name=name,
                question=str(v.get("question", "")),
                weight=float(weight),
                min_score=float(min_score),
                aliases=aliases,
            ))
        else:
            raise TypeError(
                f"评测维度 '{name}' 的规格必须是字符串或字段 dict，得到 {type(v).__name__}")
    return out


@dataclass
class RetryPolicy:
    """失败重试策略（评测不通过 → 反馈注入 → 重新生成）。"""
    max_attempts: int = 2
    inject_feedback: bool = True
    feedback_prefix: str = "请修正以下问题后重新生成："
