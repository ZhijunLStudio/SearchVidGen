"""核心逻辑单元测试：注册表、评测解析、实验缓存。

运行：python -m pytest tests/ -v  （在 harness/ 目录下）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vidharness.core.registry import register, get, check_capabilities  # noqa: E402
from vidharness.consumers.judge_loop import parse_judge_output  # noqa: E402
from vidharness.seams import JudgeCriteria, RetryPolicy, Artifact, ArtifactMeta  # noqa: E402
from vidharness.core.experiment import Experiment  # noqa: E402


class TestRegistry:
    def test_register_and_get(self):
        @register("test.dummy")
        class Dummy:
            capabilities = {"audio": True}

        assert get("test.dummy") is Dummy

    def test_unknown_adapter_fails_loud(self):
        with pytest.raises(KeyError):
            get("does.not.exist")

    def test_capability_check_fails_loud(self):
        @register("test.audio-only")
        class AudioOnly:
            capabilities = {"audio": True, "max_duration_s": 10}

        with pytest.raises(RuntimeError):
            check_capabilities("test.audio-only", {"video": True})
        with pytest.raises(RuntimeError):
            check_capabilities("test.audio-only", {"max_duration_s": 11})
        # 满足能力则通过
        caps = check_capabilities("test.audio-only", {"audio": True, "max_duration_s": 8})
        assert caps["audio"] is True


class TestJudgeParsing:
    def test_json_block(self):
        out = '```json\n{"与指令一致性": 8, "画面质量": 7, "feedback": "ok"}\n```'
        crit = [JudgeCriteria(name="与指令一致性", question="q", min_score=6),
                JudgeCriteria(name="画面质量", question="q", min_score=6)]
        v = parse_judge_output(out, crit)
        assert v["passed"] is True
        assert v["scores"]["与指令一致性"] == 8

    def test_below_threshold_fails(self):
        out = '{"与指令一致性": 4, "画面质量": 8, "feedback": "主体崩坏"}'
        crit = [JudgeCriteria(name="与指令一致性", question="q", min_score=6),
                JudgeCriteria(name="画面质量", question="q", min_score=6)]
        v = parse_judge_output(out, crit)
        assert v["passed"] is False
        assert "主体崩坏" in v["feedback"]

    def test_fallback_score_pattern(self):
        out = "总体评分：7/10，画面尚可"
        crit = [JudgeCriteria(name="与指令一致性", question="q", min_score=6)]
        v = parse_judge_output(out, crit)
        assert v["scores"]["与指令一致性"] == 7


class TestExperiment:
    def test_artifact_caching_and_resume(self, tmp_path):
        exp = Experiment(task="t", base_dir=tmp_path, run_id="r1")
        art = Artifact(kind="video", path=Path(tmp_path) / "v.mp4",
                       meta=ArtifactMeta(adapter="x", elapsed_s=1.0, cost_usd=0.1))
        Path(tmp_path, "v.mp4").write_bytes(b"fake")
        exp.save_artifact("segments", art, name="seg01")
        # 断点续跑：能找到
        found = exp.find_existing("segments", "seg01")
        assert found is not None and found.path.name.startswith("seg01")
        # manifest 记录了成本
        m = json.loads((exp.root / "manifest.json").read_text())
        assert m["total_cost_usd"] == pytest.approx(0.1)
        assert len(m["stages"]["segments"]) == 1


class TestJudgeAliases:
    def test_alias_fallback(self):
        """思考模型常省略维度前缀（'与指令一致性'→'一致性'），别名兜底应能解析。"""
        out = "分析：一致性: 8，画面质量: 7"
        crit = [JudgeCriteria(name="与指令一致性", question="q", min_score=6, aliases=["一致性"]),
                JudgeCriteria(name="画面质量", question="q", min_score=6)]
        v = parse_judge_output(out, crit)
        assert v["scores"]["与指令一致性"] == 8
        assert v["scores"]["画面质量"] == 7


class TestFallback:
    def test_fallback_switches_on_failure(self):
        from vidharness.consumers.fallback import FallbackGenerator
        from vidharness.seams import GenRequest, Artifact, ArtifactMeta
        from pathlib import Path

        class Boom:
            name = "boom"
            capabilities = {"audio": False}
            def generate(self, req, workdir, **kw):
                raise RuntimeError("boom 不可用")

        class Ok:
            name = "ok"
            capabilities = {"audio": True}
            def generate(self, req, workdir, **kw):
                return Artifact(kind="video", path=Path(workdir) / "v.mp4",
                                meta=ArtifactMeta(adapter="ok"))

        fb = FallbackGenerator.__new__(FallbackGenerator)   # 绕过注册表直接注入
        fb.chain = [Boom(), Ok()]
        fb.name = "fallback[boom,ok]"
        fb.capabilities = {"audio": True}
        art = fb.generate(GenRequest(text="t"), workdir=Path("."))
        assert art.meta.params["fallback_used"] == "ok"

    def test_fallback_all_fail_raises(self):
        from vidharness.consumers.fallback import FallbackGenerator
        from vidharness.seams import GenRequest
        from pathlib import Path

        class Boom:
            name = "boom"
            capabilities = {}
            def generate(self, req, workdir, **kw):
                raise RuntimeError("不可用")

        fb = FallbackGenerator.__new__(FallbackGenerator)
        fb.chain = [Boom()]
        fb.name = "fallback[boom]"
        fb.capabilities = {}
        import pytest
        with pytest.raises(RuntimeError):
            fb.generate(GenRequest(text="t"), workdir=Path("."))


class TestScriptOptimizer:
    def test_optimizer_selects_best(self, tmp_path):
        import sys, json
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from vidharness.consumers.script_optimizer import ScriptOptimizer
        from vidharness.seams import Artifact, ArtifactMeta, JudgeCriteria
        from vidharness.core.memory import ExperienceMemory
        from vidharness.core.experiment import Experiment

        class FakeScriptAdapter:
            name = "fake"
            def __init__(self):
                self.n = 0
            def generate(self, query, template, workdir, **kw):
                self.n += 1
                payload = {"segments": [{"video_prompt": "p", "narration": f"旁白{self.n}",
                                         "duration": 8}]}
                path = Path(workdir) / f"s{self.n}.json"
                path.write_text(json.dumps(payload))
                return Artifact(kind="script", path=path, meta=ArtifactMeta(), payload=payload)

        class FakeJudge:
            def __init__(self):
                self.n = 0
            def judge(self, media, criteria, workdir, **kw):
                self.n += 1
                score = min(9.0, 5.0 + self.n * 1.5)   # 越来越好
                return Artifact(kind="scores", path=Path(workdir)/"j.json",
                                meta=ArtifactMeta(),
                                payload={"score": score, "passed": score >= 6,
                                         "feedback": "再真实一点" if score < 8 else "pass"})

        exp = Experiment(task="t", base_dir=tmp_path, run_id="r1")
        mem = ExperienceMemory(tmp_path / "_memory.jsonl")
        opt = ScriptOptimizer(FakeScriptAdapter(), FakeJudge(), mem, exp,
                              rounds=2, candidates=2, target_score=9.5)
        crit = [JudgeCriteria(name="旁白自然", question="q", min_score=6)]
        best, history = opt.optimize("目标", "brief", crit, tmp_path / "s")
        assert best["segments"][0]["narration"] == "旁白3"   # 首个最高分候选
        assert len(history) == 4                              # 两轮跑满
        assert max(r["score"] for r in history) == 9.0
