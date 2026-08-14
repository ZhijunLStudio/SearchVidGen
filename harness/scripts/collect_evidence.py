"""实验证据收集：用裁判描述每段视频首帧 + 汇总评测与性能指标。

用法：python scripts/collect_evidence.py <experiment_dir>
输出：evidence.json（每段首帧描述 + 评测记录 + 性能成本）
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vidharness.core.registry import load_builtin_adapters, get


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
    judge = get("judge.openai-compat")(
        base_url="http://127.0.0.1:8030/v1", model="judge-qwen3.5-27b")

    manifest = json.loads((exp_dir / "manifest.json").read_text())
    evidence = {"run_id": manifest.get("run_id"), "segments": []}

    seg_dir = exp_dir / "artifacts" / "segments"
    workdir = exp_dir / "evidence"
    workdir.mkdir(exist_ok=True)
    for seg in sorted(seg_dir.glob("seg*.mp4")):
        first = workdir / f"{seg.stem}_first.jpg"
        subprocess.run(["ffmpeg", "-y", "-ss", "0.5", "-i", str(seg),
                        "-frames:v", "1", str(first)], capture_output=True)
        desc = frame_describe(judge, first, workdir)
        # 时长
        dur = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                              "-of", "csv=p=0", str(seg)], capture_output=True, text=True).stdout.strip()
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
