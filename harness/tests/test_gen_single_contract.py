"""vh gen-single JSON 子进程契约测试（mock 生成器 + ffmpeg 小样，无 GPU）。

验证 dsh-video-provider local 后端依赖的 stdout 协议：
- 单行 JSON、video.path 真实存在、backend=local、judge 透传/缺省为 null；
- 失败路径（text 缺失）SystemExit 且非零；
- runDir 内 manifest 可读（证据层完整）。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vidharness.core.registry import register  # noqa: E402
from vidharness.seams import Artifact, ArtifactMeta  # noqa: E402

_FFMPEG_DIR = Path("/data/lizhijun/anaconda3/envs/torch/bin")


@pytest.fixture(autouse=True)
def ffmpeg_path(monkeypatch):
    """契约测试需要真实 ffmpeg/ffprobe（成片总装用），torch 环境自带。"""
    if _FFMPEG_DIR.is_dir():
        monkeypatch.setenv(
            "PATH", str(_FFMPEG_DIR) + os.pathsep + os.environ.get("PATH", ""))
    yield


@register("generator.mock-contract")
class MockContractGenerator:
    """写出一段真实 1s mp4 的假生成器（无 GPU、无网络）。"""
    name = "generator.mock-contract"
    capabilities = {
        "max_duration_s": 15,
        "audio": True,
        "refs": 9,
        "first_last_frame": False,
        "resolution": "768p",
        "backend": "local",
    }

    def __init__(self, **params):
        self.params = params

    def generate(self, req, workdir, **kw):
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        vid = workdir / "seg.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=blue:s=128x128:d=1",
             "-c:v", "libx264", str(vid)],
            capture_output=True, check=True)
        return Artifact(kind="video", path=vid, meta=ArtifactMeta(
            adapter=self.name, params=self.params, elapsed_s=1.0, cost_usd=0.01))


def _run_gen_single(spec: dict, tmp_path: Path):
    """进程内调用 cmd_gen_single，返回 (stdout_text, exit_code)。"""
    from vidharness.cli import cmd_gen_single

    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")

    class Args:
        pass

    args = Args()
    args.spec = str(spec_path)

    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    code = 0
    try:
        with redirect_stdout(buf):
            cmd_gen_single(args)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    return buf.getvalue(), code


def _base_spec(tmp_path: Path, **overrides) -> dict:
    spec = {
        "text": "test prompt",
        "generator": {"adapter": "generator.mock-contract", "params": {"steps": 4}},
        "out": str(tmp_path / "out"),
    }
    spec.update(overrides)
    return spec


class TestGenSingleContract:
    def test_stdout_single_line_json_with_video(self, tmp_path):
        out, code = _run_gen_single(_base_spec(tmp_path), tmp_path)
        assert code == 0
        lines = out.strip().splitlines()
        assert len(lines) == 1, "stdout 必须只有一行 JSON，得到: " + repr(out)
        result = json.loads(lines[0])
        assert result["video"]["backend"] == "local"
        assert result["video"]["model"] == "generator.mock-contract"
        assert Path(result["video"]["path"]).exists()
        assert result["judge"] is None
        assert isinstance(result["costUsd"], (int, float))
        assert isinstance(result["elapsedS"], (int, float))
        run_dir = Path(result["runDir"])
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["total_cost_usd"] == pytest.approx(0.01)
        assert (run_dir / "final" / "final_video.mp4").exists()

    def test_seed_and_duration_land_in_params_and_result(self, tmp_path):
        out, code = _run_gen_single(
            _base_spec(tmp_path, seed=42, duration=8), tmp_path)
        assert code == 0
        result = json.loads(out)
        assert result["video"]["params"]["seed"] == 42
        assert result["video"]["durationS"] == 8
        assert result["video"]["seed"] == 42

    def test_missing_text_fails_loud(self, tmp_path):
        spec = _base_spec(tmp_path)
        del spec["text"]
        out, code = _run_gen_single(spec, tmp_path)
        assert code != 0
        assert out == ""

    def test_ffmpeg_dir_prepends_to_path(self, tmp_path, monkeypatch):
        """本机模型环境（h3int8）无 ffmpeg：ffmpeg_dir 注入 PATH 后成片总装可用。"""
        monkeypatch.setenv("PATH", "/nonexistent-dir")
        spec = _base_spec(tmp_path)
        spec["ffmpeg_dir"] = str(_FFMPEG_DIR)
        out, code = _run_gen_single(spec, tmp_path)
        assert code == 0, "ffmpeg_dir 注入失败: " + out
        result = json.loads(out)
        assert Path(result["video"]["path"]).exists()

    def test_unknown_generator_fails_nonzero(self, tmp_path):
        spec = _base_spec(tmp_path)
        spec["generator"]["adapter"] = "generator.does-not-exist"
        out, code = _run_gen_single(spec, tmp_path)
        assert code != 0
        assert out == ""
