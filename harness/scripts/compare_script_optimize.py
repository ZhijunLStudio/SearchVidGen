"""剧本优化闭环量化：script_optimize 开/关的剧本评分对比（纯 API，无 GPU）。

用法：python scripts/compare_script_optimize.py --query "..." [--trials 3]
输出：JSON 明细 + markdown 表（两模式的 均分/通过率/LLM 调用数/成本）。

设计（控制变量）：同一 query、同一剧本裁判（judge.deepseek-text）、
同一评测维度；off = 单次生成，on = ScriptOptimizer(2轮×2候选)。
成本口径：exp.manifest.total_cost_usd（剧本+裁判调用都入产物计费）。
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
from vidharness.consumers.script_optimizer import ScriptOptimizer  # noqa: E402
from vidharness.core.memory import ExperienceMemory  # noqa: E402
from vidharness.core.experiment import Experiment  # noqa: E402

CRITERIA = [
    JudgeCriteria(name="叙事完整", question="分镜是否构成完整有起伏的故事，段间有因果与推进？", min_score=6),
    JudgeCriteria(name="旁白自然", question="旁白是否像真人说话（口语化、有情绪、长短句结合），而不是口号式短句或播音腔？", weight=1.2, min_score=6, aliases=["自然度", "口语化"]),
    JudgeCriteria(name="可生成性", question="画面指令是否具体可执行（主体/动作/镜头明确）？", min_score=6),
]


def _embed(art, criteria):
    return [replace(c, question=f"{c.question}\n\n剧本内容：\n"
                                 f"{json.dumps(art.payload, ensure_ascii=False)}")
            for c in criteria]


def single_shot(script_adapter, judge, query, exp, workdir, trial):
    template = {"brief": "", "segments": 4, "experience": []}
    art = script_adapter.generate(query=query, template=template, workdir=workdir)
    exp.save_artifact("script", art, name=f"off_t{trial}")
    verdict = run_judge(judge, [], _embed(art, CRITERIA),
                        exp.artifacts_dir / "judge", exp=exp)
    return verdict, 2   # 1 生成 + 1 裁判


def optimized(script_adapter, judge, query, exp, workdir, rounds=2, candidates=2):
    mem = ExperienceMemory(workdir.parent / "_memory_bench.jsonl")
    opt = ScriptOptimizer(script_adapter, judge, mem, exp,
                          rounds=rounds, candidates=candidates,
                          target_score=8.0, segments=4)
    best, history = opt.optimize(query, "", CRITERIA, workdir)
    best_rec = max(history, key=lambda r: r.get("score", 0))
    return best_rec, len(history)   # 调用数 = 生成+裁判总数


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--candidates", type=int, default=2)
    ap.add_argument("--workdir", default="/tmp/script-opt-bench")
    args = ap.parse_args()

    load_builtin_adapters()
    script_adapter = instantiate("script.deepseek-v4-flash",
                                 {"model": "deepseek-chat", "temperature": 0.9})
    judge = instantiate("judge.deepseek-text", {"model": "deepseek-chat"})

    rows = []
    for mode, fn in (("off", single_shot), ("on", optimized)):
        for trial in range(1, args.trials + 1):
            base = Path(args.workdir)
            exp = Experiment(task="script_opt_bench", base_dir=base)
            workdir = base / f"{mode}_t{trial}"
            workdir.mkdir(parents=True, exist_ok=True)
            if mode == "off":
                verdict, calls = fn(script_adapter, judge, args.query, exp, workdir, trial)
            else:
                verdict, calls = fn(script_adapter, judge, args.query, exp, workdir,
                                     rounds=args.rounds, candidates=args.candidates)
            rows.append({
                "mode": mode, "trial": trial,
                "score": verdict.get("score"), "passed": verdict.get("passed"),
                "scores": verdict.get("scores"),
                "calls": calls,
                "cost_usd": round(exp.manifest["total_cost_usd"], 5),
            })
            print(f"  [{mode} t{trial}] score={verdict.get('score')} "
                  f"passed={verdict.get('passed')} calls={calls}")

    def mean(xs):
        return round(sum(xs) / len(xs), 2) if xs else None

    summary = {}
    for mode in ("off", "on"):
        rs = [r for r in rows if r["mode"] == mode]
        summary[mode] = {
            "n": len(rs),
            "mean_score": mean([r["score"] for r in rs]),
            "passed": sum(1 for r in rs if r["passed"]),
            "total_calls": sum(r["calls"] for r in rs),
            "total_cost_usd": round(sum(r["cost_usd"] for r in rs), 5),
        }
    print(json.dumps({"rows": rows, "summary": summary}, ensure_ascii=False, indent=2))
    print("\n| 模式 | n | 剧本均分 | 通过率 | LLM 调用 | API 成本(USD) |")
    print("|---|---|---|---|---|---|")
    for mode in ("off", "on"):
        s = summary[mode]
        print(f"| {mode} | {s['n']} | {s['mean_score']} | {s['passed']}/{s['n']} | "
              f"{s['total_calls']} | {s['total_cost_usd']} |")


if __name__ == "__main__":
    main()
