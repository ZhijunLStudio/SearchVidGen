"""经验记忆的因果效应 A/B（E34 的机制级验证，纯 API 无 GPU）。

设计：同一 query、同一裁判、同一生成器——唯一变量是注入的经验列表。
A 臂 = 真实经验记忆（experiments/_memory.jsonl 的提升条目）；
B 臂 = 空经验。对比剧本评分（叙事完整/旁白自然/可生成性）。

用法：python scripts/compare_memory_effect.py --query "..." [--trials 5]
注意：这是"经验→剧本质量"的最近端因果；跨段一致性等下游效应需 GPU
实验（bench repeats 基建已就绪）。
"""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vidharness.core.registry import load_builtin_adapters, instantiate  # noqa: E402
from vidharness.seams import JudgeCriteria  # noqa: E402
from vidharness.consumers.judge_loop import run_judge  # noqa: E402
from vidharness.core.memory import ExperienceMemory  # noqa: E402

CRITERIA = [
    JudgeCriteria(name="叙事完整", question="分镜是否构成完整有起伏的故事，段间有因果与推进？", min_score=6),
    JudgeCriteria(name="旁白自然", question="旁白是否像真人说话（口语化、有情绪、长短句结合），而不是口号式短句或播音腔？", weight=1.2, min_score=6, aliases=["自然度", "口语化"]),
    JudgeCriteria(name="可生成性", question="画面指令是否具体可执行（主体/动作/镜头明确）？", min_score=6),
]


def single_trial(script_adapter, judge, query, experiences, workdir):
    template = {"brief": "", "segments": 4, "experience": experiences}
    art = script_adapter.generate(query=query, template=template, workdir=workdir)
    crit = [replace(c, question=f"{c.question}\n\n剧本内容：\n"
                                 f"{json.dumps(art.payload, ensure_ascii=False)}")
            for c in CRITERIA]
    return run_judge(judge, [], crit, workdir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--memory", default="experiments/_memory.jsonl")
    ap.add_argument("--workdir", default="/tmp/memory-ab")
    args = ap.parse_args()

    load_builtin_adapters()
    script_adapter = instantiate("script.deepseek-v4-flash",
                                 {"model": "deepseek-chat", "temperature": 0.9})
    judge = instantiate("judge.deepseek-text", {"model": "deepseek-chat"})
    experiences = ExperienceMemory(Path(args.memory)).experience_lines()
    print(f"A 臂经验数: {len(experiences)}")
    for e in experiences:
        print("   -", e[:40])

    rows = []
    for arm, exp_lines in (("A(经验注入)", experiences), ("B(空记忆)", [])):
        for trial in range(1, args.trials + 1):
            workdir = Path(args.workdir) / f"{arm[0]}{trial}"
            workdir.mkdir(parents=True, exist_ok=True)
            verdict = single_trial(script_adapter, judge, args.query, exp_lines, workdir)
            rows.append({"arm": arm, "trial": trial, "score": verdict.get("score"),
                         "scores": verdict.get("scores"), "passed": verdict.get("passed")})
            print(f"  [{arm} t{trial}] score={verdict.get('score')} "
                  f"passed={verdict.get('passed')}")

    def mean(xs):
        return round(sum(xs) / len(xs), 2) if xs else None

    summary = {}
    for arm in ("A(经验注入)", "B(空记忆)"):
        rs = [r for r in rows if r["arm"] == arm]
        dims = {}
        for c in CRITERIA:
            vals = [r["scores"].get(c.name) for r in rs if r["scores"].get(c.name) is not None]
            dims[c.name] = mean(vals)
        summary[arm] = {"n": len(rs), "mean_score": mean([r["score"] for r in rs]),
                        "passed": sum(1 for r in rs if r["passed"]), "dims": dims}
    print(json.dumps({"rows": rows, "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
