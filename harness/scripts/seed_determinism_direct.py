"""种子确定性直测（E43 第二阶段）：固定提示 + 同进程 + 逐调用种子覆盖。

bench 矩阵版（tasks/bench_seed.yaml）的残留混淆：DeepSeek 剧本 API 在
temperature 0 下四次调用仍产出四个略有差异的剧本（音频措辞不同），
同种子 MAE≈13-26 无法区分"提示差异 vs 内核不确定性"。

本脚本用 E43 新增的逐调用种子覆盖（kw.seed > req.seed > 构造参数）：
同一进程、同一已加载模型、逐字相同的提示，直接分解生成侧确定性。

用法：
  python scripts/seed_determinism_direct.py \
      --config tasks/story_seed_check.yaml \
      [--repeats 2] [--out experiments]

生成 2 种子 × repeats 个视频（产物经事件溯源入账），
输出同种子/异种子对的逐帧 MAE 分解表。
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vidharness.core.experiment import Experiment  # noqa: E402
from vidharness.core.registry import load_builtin_adapters, instantiate  # noqa: E402
from vidharness.seams import GenRequest  # noqa: E402
from compare_seed_runs import _frames  # noqa: E402


def _mae(pa: Path, pb: Path, frames: int, workdir: Path) -> float:
    import numpy as np
    from PIL import Image
    fa = _frames(pa, frames, workdir / "a")
    fb = _frames(pb, frames, workdir / "b")
    n = min(len(fa), len(fb))
    d = 0.0
    for i in range(n):
        a = np.asarray(Image.open(fa[i]).convert("RGB"), dtype=np.float32)
        b = np.asarray(Image.open(fb[i]).convert("RGB"), dtype=np.float32)
        d += float(np.abs(a - b).mean())
    return round(d / n, 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="任务 YAML（读 generator 参数，E13）")
    ap.add_argument("--prompt", default=None, help="固定提示（缺省用内置固定提示）")
    ap.add_argument("--seeds", default="20260816,20260817")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", default="experiments")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    gen_cfg = cfg["pipeline"]["generator"]
    prompt = args.prompt or ("镜头从窗台侧面缓慢推近，橘猫蜷缩在窗台上，尾巴轻轻摆动，"
                             "窗外雨滴打在玻璃上，形成水痕。环境音：雨声淅沥。")
    seeds = [int(s) for s in args.seeds.split(",")]

    load_builtin_adapters()
    gen = instantiate(gen_cfg["adapter"], gen_cfg.get("params") or {},
                      context="seed_determinism_direct")
    print(f"生成器: {gen.name}（单实例，逐调用覆盖种子）")
    print(f"提示（逐字相同）: {prompt}")

    exp = Experiment(task="seed_direct", base_dir=Path(args.out))
    exp.bind_query(f"种子确定性直测: 固定提示 x{args.repeats} "
                   f"seeds={seeds}")
    exp.snapshot_config({"generator": gen_cfg, "prompt": prompt,
                         "seeds": seeds, "repeats": args.repeats})

    videos = {}
    for s in seeds:
        for rep in range(1, args.repeats + 1):
            name = f"seed{s}_r{rep}"
            req = GenRequest(text=prompt, duration=5, ratio="16:9")
            art = gen.generate(req, exp.artifacts_dir / "segments", seed=s)
            exp.save_artifact("segments", art, name=name)
            exp.save_eval("gen", [{"seed": s, "repeat": rep,
                                   "name": name, "path": str(art.path),
                                   "meta_seed": art.meta.seed}])
            videos[name] = art.path
            print(f"  [{name}] {art.path.name} (meta.seed={art.meta.seed})")

    # 同种子 vs 异种子 分解
    pairs_same, pairs_cross = [], []
    for s in seeds:
        pairs_same.append((f"seed{s}_r1", f"seed{s}_r2"))
    for i in range(args.repeats):
        pairs_cross.append((f"seed{seeds[0]}_r{i + 1}", f"seed{seeds[1]}_r{i + 1}"))

    print("\n== 同种子对 ==")
    same_maes = []
    for a, b in pairs_same:
        m = _mae(videos[a], videos[b], 5, Path("/tmp/vh-seed-d") / f"{a}_{b}")
        same_maes.append(m)
        print(f"  {a} vs {b}: MAE={m}")
    print("== 异种子对 ==")
    cross_maes = []
    for a, b in pairs_cross:
        m = _mae(videos[a], videos[b], 5, Path("/tmp/vh-seed-d") / f"{a}_{b}")
        cross_maes.append(m)
        print(f"  {a} vs {b}: MAE={m}")

    summary = {"same_seed_mae": same_maes, "cross_seed_mae": cross_maes,
               "same_mean": round(sum(same_maes) / len(same_maes), 2),
               "cross_mean": round(sum(cross_maes) / len(cross_maes), 2)}
    exp.save_eval("summary", [{"dims": summary}])
    exp.finalize()
    print(f"\n同种子均值 MAE={summary['same_mean']}　异种子均值 MAE={summary['cross_mean']}")
    print(f"实验目录: {exp.root}")


if __name__ == "__main__":
    main()
