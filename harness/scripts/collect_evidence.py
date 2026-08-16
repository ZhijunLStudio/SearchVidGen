"""实验证据收集：用裁判描述每段视频首帧 + 汇总评测与性能指标。

用法：python scripts/collect_evidence.py <experiment_dir>
输出：evidence.json（每段首帧描述 + 评测记录 + 性能成本）

裁判适配器与参数从 run 的 config.yaml 快照读取（配置正源是实验快照，
脚本不硬编码端点/模型名）；旧 run 无快照则响亮失败并给出指引。
"""
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vidharness.core.registry import load_builtin_adapters, instantiate


def load_judge_from_run(run_dir: Path):
    """从 run 的配置快照实例化裁判（fail loud：无快照/缺配置直接报错）。"""
    cfg_file = run_dir / "config.yaml"
    if not cfg_file.exists():
        raise RuntimeError(
            "run 缺少配置快照 config.yaml（2026-08-16 前的旧 run）。"
            "重新运行该任务生成快照后再收集证据，保证证据与运行同口径。")
    cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    judge_cfg = cfg.get("judge") or {}
    name = judge_cfg.get("adapter")
    if not name:
        raise RuntimeError("配置快照缺少 judge.adapter")
    return instantiate(name, judge_cfg.get("params", {}), context="collect_evidence")


def frame_describe(judge, img: Path, workdir: Path) -> str:
    art = judge.judge(
        media=[img],
        criteria={"画面描述": "用一句话客观描述画面内容（主体/动作/场景/风格）。"},
        workdir=workdir,
    )
    return art.payload.get("feedback", "") or ""


def main(exp_dir: str):
    exp_dir = Path(exp_dir)
    load_builtin_adapters()
    judge = load_judge_from_run(exp_dir)

    manifest = json.loads((exp_dir / "manifest.json").read_text())
    evidence = {"run_id": manifest.get("run_id"), "segments": []}

    from vidharness.consumers.tools import _duration_s, extract_frame
    seg_dir = exp_dir / "artifacts" / "segments"
    workdir = exp_dir / "evidence"
    workdir.mkdir(exist_ok=True)
    for seg in sorted(seg_dir.glob("seg*.mp4")):
        first = extract_frame(seg, 0.5, workdir)
        if first is None:
            evidence["segments"].append({"segment": seg.stem, "error": "首帧抽取失败"})
            continue
        desc = frame_describe(judge, first, workdir)
        # 时长
        try:
            dur = f"{_duration_s(seg):.2f}"
        except Exception:
            dur = "?"
        evidence["segments"].append({
            "segment": seg.stem, "duration_s": dur, "first_frame_desc": desc,
            "first_frame": str(first),
        })

    # 评测汇总
    evals = {}
    for f in (exp_dir / "eval").glob("*.json"):
        if f.stem in ("segments", "cross_consistency", "audio_verify"):
            evals[f.stem] = json.loads(f.read_text())
    evidence["evals"] = evals
    evidence["metrics"] = {
        "total_elapsed_s": manifest.get("total_elapsed_s"),
        "api_cost_usd": manifest.get("total_cost_usd"),
        "local_gpu_hours": manifest.get("local_gpu_hours"),
        "retries": manifest.get("retries", {}),
    }
    out = exp_dir / "evidence.json"
    out.write_text(json.dumps(evidence, ensure_ascii=False, indent=2))
    print(json.dumps(evidence, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
