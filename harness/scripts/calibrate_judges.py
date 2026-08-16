"""跨裁判校准（E24 警报的回应）：同一批剧本用两个裁判打分，计算维度级偏移。

用法：python scripts/calibrate_judges.py --k 5 [--judge-a judge.openai-compat] [--judge-b judge.deepseek-text]
输出：harness/calibration/<a>_vs_<b>.json（维度级均差/样本对 + 时间戳）

动机（E24）：vLLM 裁判与 deepseek-text 对同一批剧本的评分尺度显著不同
（≈9-10 vs 2.65-5.46）。校准数据用于：①报告/leaderboard 混用裁判时的
口径标注；②人工解读跨裁判对比（自动换算在样本量充足前不启用）。

样本：从真实实验的 script.json 产物中取 K 个（同一批剧本双裁判打分）。
"""
import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vidharness.core.registry import load_builtin_adapters, instantiate  # noqa: E402
from vidharness.seams import JudgeCriteria  # noqa: E402
from vidharness.consumers.judge_loop import run_judge  # noqa: E402

CRITERIA = [
    JudgeCriteria(name="叙事完整", question="分镜是否构成完整有起伏的故事，段间有因果与推进？", min_score=6),
    JudgeCriteria(name="旁白自然", question="旁白是否像真人说话（口语化、有情绪、长短句结合），而不是口号式短句或播音腔？", weight=1.2, min_score=6, aliases=["自然度", "口语化"]),
    JudgeCriteria(name="可生成性", question="画面指令是否具体可执行（主体/动作/镜头明确）？", min_score=6),
]

# 两个裁判的默认参数（与 story.yaml / story_smoke.yaml 同口径）
JUDGE_SPECS = {
    "judge.openai-compat": {"base_url": "http://127.0.0.1:8030/v1",
                            "model": "judge-qwen3.5-27b",
                            "frame_samples": 2, "disable_thinking": True},
    "judge.deepseek-text": {"model": "deepseek-chat"},
}


def collect_scripts(experiments_dir: Path, k: int):
    """取真实实验的 K 个剧本产物（去重按内容）。"""
    seen, scripts = set(), []
    for f in sorted(experiments_dir.glob("*/*/artifacts/script/script.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        key = json.dumps(data, ensure_ascii=False, sort_keys=True)
        if key in seen or "segments" not in data:
            continue
        seen.add(key)
        scripts.append({"source": str(f), "payload": data})
        if len(scripts) >= k:
            break
    return scripts


def score_script(judge, payload, workdir: Path, max_attempts: int = 2):
    """用裁判给剧本打分（文本评测：剧本内容嵌入维度问题）。

    与真实使用同口径的重试：解析失败时注入"务必只输出 JSON"指令 +
    上次反馈再试（E22 的可操作反馈机制）。
    返回 (verdict, 尝试次数)。
    """
    crit = [replace(c, question=f"{c.question}\n\n剧本内容：\n"
                                 f"{json.dumps(payload, ensure_ascii=False)}")
            for c in CRITERIA]
    for attempt in range(1, max_attempts + 1):
        verdict = run_judge(judge, [], crit, workdir)
        if verdict["scores"]:
            return verdict, attempt
        fb = verdict.get("feedback") or "请只输出 JSON"
        crit = [replace(c, question=f"{c.question}\n\n（上次解析失败，务必只输出"
                                    f" JSON 对象，不要任何其他文字）\n{fb}")
                for c in crit]
    return verdict, max_attempts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--judge-a", default="judge.openai-compat")
    ap.add_argument("--judge-b", default="judge.deepseek-text")
    ap.add_argument("--experiments", default="experiments")
    args = ap.parse_args()

    load_builtin_adapters()
    judges = {}
    for name in (args.judge_a, args.judge_b):
        spec = JUDGE_SPECS.get(name, {})
        judges[name] = instantiate(name, spec, context=f"calibrate.{name}")

    scripts = collect_scripts(Path(args.experiments), args.k)
    if len(scripts) < 2:
        print(f"剧本样本不足（{len(scripts)} < 2）：请先跑过实验生成 script.json")
        sys.exit(1)

    pairs = {c.name: {"a": [], "b": []} for c in CRITERIA}
    rows = []
    attempts_used = {args.judge_a: 0, args.judge_b: 0}
    scored = {args.judge_a: 0, args.judge_b: 0}
    for i, s in enumerate(scripts, 1):
        workdir = Path("/tmp/judge-calib")
        workdir.mkdir(parents=True, exist_ok=True)
        va, ta = score_script(judges[args.judge_a], s["payload"], workdir)
        vb, tb = score_script(judges[args.judge_b], s["payload"], workdir)
        attempts_used[args.judge_a] += ta
        attempts_used[args.judge_b] += tb
        scored[args.judge_a] += 1 if va["scores"] else 0
        scored[args.judge_b] += 1 if vb["scores"] else 0
        row = {"i": i, "source": s["source"], args.judge_a: va["scores"],
               args.judge_b: vb["scores"]}
        rows.append(row)
        for c in CRITERIA:
            pairs[c.name]["a"].append(va["scores"].get(c.name))
            pairs[c.name]["b"].append(vb["scores"].get(c.name))
        print(f"  [{i}] a={va['scores']} b={vb['scores']}")

    dims = {}
    for name, p in pairs.items():
        paired = [(a, b) for a, b in zip(p["a"], p["b"])
                  if a is not None and b is not None]
        if paired:
            offsets = [a - b for a, b in paired]
            dims[name] = {
                "n": len(paired),
                "mean_a": round(sum(a for a, _ in paired) / len(paired), 2),
                "mean_b": round(sum(b for _, b in paired) / len(paired), 2),
                "mean_offset_a_minus_b": round(sum(offsets) / len(offsets), 2),
                "pairs_a_b": [[round(a, 2), round(b, 2)] for a, b in paired],
            }
    result = {
        "judge_a": args.judge_a, "judge_b": args.judge_b,
        "created_at": datetime.now().isoformat(),
        "n_scripts": len(scripts),
        "scored": scored,
        "attempts_used": attempts_used,
        "parse_failure_rate": {
            name: round(1 - scored[name] / len(scripts), 2) for name in scored},
        "dims": dims, "rows": rows,
    }
    out_dir = Path("calibration")
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{args.judge_a.replace('.', '_')}__vs__{args.judge_b.replace('.', '_')}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n校准数据 → {out}")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'pairs_a_b'}
                      for k, v in dims.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
