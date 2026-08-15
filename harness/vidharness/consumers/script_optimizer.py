"""剧本自主优化（Script Optimizer）—— harness 在环境中持续学习的核心循环。

不是"生成一次、评测一次"，而是多轮进化：
  每轮生成 K 个候选（不同温度/视角）→ 全部交裁判评分 → 反馈入经验记忆
  → 淘汰低分、以最高分候选为种子（带反馈）进入下一轮 → 直到达标或轮次用尽。

完全领域无关：对故事/广告/口播同样适用（目标与质量维度由环境决定）。
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from ..seams import JudgeCriteria


class ScriptOptimizer:
    def __init__(self, script_adapter, judge, memory, exp, rounds: int = 2,
                 candidates: int = 2, target_score: float = 7.5):
        self.script_adapter = script_adapter
        self.judge = judge
        self.memory = memory
        self.exp = exp
        self.rounds = rounds
        self.candidates = candidates
        self.target_score = target_score

    def _judge_script(self, art, criteria: List[JudgeCriteria]) -> Dict[str, Any]:
        q = {c.name: f"{c.question}\n\n剧本内容：\n{json.dumps(art.payload, ensure_ascii=False)}"
             for c in criteria}
        verdict_art = self.judge.judge(media=[], criteria=q, workdir=self.exp.eval_dir)
        return verdict_art.payload

    def optimize(self, query: str, brief: str, criteria: List[JudgeCriteria],
                 workdir) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """返回 (最优剧本, 进化记录)。"""
        history: List[Dict[str, Any]] = []
        best_art, best_score = None, -1.0
        from pathlib import Path
        Path(workdir).mkdir(parents=True, exist_ok=True)

        for rnd in range(1, self.rounds + 1):
            round_records = []
            for k in range(self.candidates):
                template = {
                    "brief": brief,
                    "segments": int(self.exp.manifest.get("segments", 4)),
                    "experience": self.memory.experience_lines(),
                }
                art = self.script_adapter.generate(
                    query=query, template=template, workdir=workdir)
                self.exp.save_artifact("script", art, name=f"script_r{rnd}c{k + 1}")
                try:
                    verdict = self._judge_script(art, criteria)
                except Exception as e:
                    verdict = {"score": 0.0, "passed": False,
                               "feedback": f"评测不可用: {type(e).__name__}"}
                score = verdict.get("score", 0.0)
                fb = verdict.get("feedback", "")
                if fb and fb.strip() and "pass" not in fb[:4].lower():
                    kind = "feedback" if not verdict.get("passed") else "suggestion"
                    self.memory.add(fb, source=f"{self.exp.run_id}/opt-r{rnd}", kind=kind)
                rec = {"round": rnd, "candidate": k + 1, "artifact": str(art.path),
                       "score": score, "passed": verdict.get("passed"),
                       "feedback": fb[:300]}
                round_records.append(rec)
                history.append(rec)
                print(f"   [r{rnd}c{k + 1}] 评分 {score} {'✓' if verdict.get('passed') else '✗'}")
                if score > best_score:
                    best_score = score
                    best_art = art
                    best_fb = fb
            self.exp.save_eval("script_optimize", history)
            if best_score >= self.target_score:
                print(f"   达标（≥{self.target_score}），停止进化")
                break
            # 以最优候选的反馈引导下一轮
            brief = f"{brief}\n上一轮最优稿的改进点：{best_fb}".strip()
        return best_art.payload if best_art else {"segments": []}, history
