"""核心逻辑单元测试：注册表、评测解析、实验缓存、配置校验、能力路由。

运行：python -m pytest tests/ -v  （在 harness/ 目录下）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vidharness.core.registry import (register, get, check_capabilities,  # noqa: E402
                                      instantiate, resolve_provider,
                                      SEAM_CAPABILITY_SCHEMAS)
from vidharness.consumers.judge_loop import (parse_judge_output, parse_scores,  # noqa: E402
                                             finalize_verdict, run_judge)
from vidharness.seams import (JudgeCriteria, RetryPolicy, Artifact, ArtifactMeta,  # noqa: E402
                              criteria_to_spec, spec_to_criteria)
from vidharness.core.experiment import Experiment  # noqa: E402
from vidharness.core.config import validate_task, ConfigError  # noqa: E402


class TestRegistry:
    def test_register_and_get(self):
        @register("generator.test-dummy")
        class Dummy:
            capabilities = {"audio": True, "backend": "local"}

        assert get("generator.test-dummy") is Dummy

    def test_unknown_adapter_fails_loud(self):
        with pytest.raises(KeyError):
            get("does.not.exist")

    def test_capability_check_fails_loud(self):
        @register("generator.audio-only")
        class AudioOnly:
            capabilities = {"audio": True, "max_duration_s": 10, "backend": "local"}

        with pytest.raises(RuntimeError):
            check_capabilities("generator.audio-only", {"first_last_frame": True})
        with pytest.raises(RuntimeError):
            check_capabilities("generator.audio-only", {"max_duration_s": 11})
        # 满足能力则通过
        caps = check_capabilities("generator.audio-only", {"audio": True, "max_duration_s": 8})
        assert caps["audio"] is True

    def test_register_validates_capability_schema(self):
        """能力词汇是协议：拼错的能力键必须注册时响亮失败，而不是静默绕过校验。"""
        with pytest.raises(ValueError):
            @register("generator.bad-caps")
            class BadCaps:
                capabilities = {"max_duraton_s": 15}   # 拼写错误

        with pytest.raises(TypeError):
            @register("generator.bad-type")
            class BadType:
                capabilities = {"audio": "yes"}        # 类型错误

        with pytest.raises(ValueError):
            @register("nosuchseam.x")
            class UnknownSeam:
                capabilities = {}

    def test_instantiate_param_validation(self):
        """任务 YAML 给适配器的参数：未知参数/缺必需参数都在最早点报错。"""
        @register("script.param-check")
        class NeedsKey:
            capabilities = {"language": "zh", "json_output": True}
            def __init__(self, api_key: str, temperature: float = 0.7):
                self.api_key = api_key

        with pytest.raises(RuntimeError, match="不接受参数"):
            instantiate("script.param-check", {"api_key": "k", "apikey": "oops"})
        with pytest.raises(RuntimeError, match="缺少必需参数"):
            instantiate("script.param-check", {})
        obj = instantiate("script.param-check", {"api_key": "k"})
        assert obj.api_key == "k"

    def test_resolve_provider_by_capability(self):
        # 用独立的测试 seam，避免被其他测试注册的 generator.* 污染路由
        SEAM_CAPABILITY_SCHEMAS["bench"] = {
            "max_duration_s": (int, float), "audio": bool, "refs": (int, float),
            "first_last_frame": bool, "resolution": str, "backend": str,
        }

        @register("bench.route-a")
        class A:
            capabilities = {"max_duration_s": 15, "audio": True, "refs": 9,
                            "first_last_frame": True, "resolution": "768p",
                            "backend": "local"}

        @register("bench.route-b")
        class B:
            capabilities = {"max_duration_s": 10, "audio": False, "refs": 0,
                            "first_last_frame": False, "resolution": "1080p",
                            "backend": "api"}

        # 只有 A 满足
        assert resolve_provider("bench", {"audio": True}) == "bench.route-a"
        # 无人满足 → 报错并说明各候选不满足原因（A 的 refs 只有 9，B 无首尾帧）
        with pytest.raises(RuntimeError, match="没有 bench 提供者满足"):
            resolve_provider("bench", {"first_last_frame": True, "refs": 10})
        # 多候选同时满足 → 拒绝替用户做决定
        with pytest.raises(RuntimeError, match="多个 bench 提供者满足"):
            resolve_provider("bench", {"max_duration_s": 8})


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

    def test_parse_scores_returns_raw(self):
        scores, fb = parse_scores('{"旁白自然": 5, "feedback": "太口号化"}',
                                  [JudgeCriteria(name="旁白自然", question="q")])
        assert scores == {"旁白自然": 5.0}
        assert "口号" in fb


class TestJudgeWeightsPreserved:
    """Bug#1 回归：YAML 的 weight/min_score/aliases 必须生效于最终判定，
    而不是在提供者侧被默认值替换。"""

    class _NewContractJudge:
        """按新协议返回原始 scores+feedback 的假裁判。"""
        name = "judge.fake"
        modalities = ["text"]
        def judge(self, media, criteria, workdir, **kw):
            # 验证收到的是完整规格（含 weight/min_score）
            for spec_v in criteria.values():
                assert isinstance(spec_v, dict) and "weight" in spec_v
            return Artifact(kind="scores", path=Path(workdir) / "j.json",
                            meta=ArtifactMeta(),
                            payload={"scores": {"旁白自然": 5.0, "可生成性": 9.0},
                                     "feedback": "口号式短句"})

    def test_weight_and_min_score_apply(self, tmp_path):
        crit = [JudgeCriteria(name="旁白自然", question="q", weight=1.2, min_score=5.5),
                JudgeCriteria(name="可生成性", question="q", weight=0.8, min_score=6.0)]
        verdict = run_judge(self._NewContractJudge(), [], crit, tmp_path)
        # 加权分 = (5.0*1.2 + 9.0*0.8) / 2.0 = 6.6
        assert verdict["score"] == 6.6
        # 旁白自然 5.0 < 5.5 → 不通过
        assert verdict["passed"] is False
        assert verdict["scores"]["旁白自然"] == 5.0

    def test_spec_roundtrip(self):
        crit = [JudgeCriteria(name="旁白自然", question="q", weight=1.2,
                              min_score=5.5, aliases=["自然度"])]
        spec = criteria_to_spec(crit)
        back = spec_to_criteria(spec)
        assert back[0].weight == 1.2
        assert back[0].min_score == 5.5
        assert back[0].aliases == ["自然度"]
        # 旧协议裸字符串仍兼容
        back2 = spec_to_criteria({"旁白自然": "问题文本"})
        assert back2[0].question == "问题文本"

    def test_missing_dimension_fails(self, tmp_path):
        class PartialJudge:
            def judge(self, media, criteria, workdir, **kw):
                return Artifact(kind="scores", path=Path(workdir) / "j.json",
                                meta=ArtifactMeta(),
                                payload={"scores": {"叙事完整": 9.0}, "feedback": ""})
        crit = [JudgeCriteria(name="叙事完整", question="q", min_score=6),
                JudgeCriteria(name="旁白自然", question="q", min_score=6)]
        verdict = run_judge(PartialJudge(), [], crit, tmp_path)
        assert verdict["passed"] is False   # 缺失维度判未通过


class TestJudgeAliases:
    def test_alias_fallback(self):
        """思考模型常省略维度前缀（'与指令一致性'→'一致性'），别名兜底应能解析。"""
        out = "分析：一致性: 8，画面质量: 7"
        crit = [JudgeCriteria(name="与指令一致性", question="q", min_score=6, aliases=["一致性"]),
                JudgeCriteria(name="画面质量", question="q", min_score=6)]
        v = parse_judge_output(out, crit)
        assert v["scores"]["与指令一致性"] == 8
        assert v["scores"]["画面质量"] == 7


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

    def test_snapshot_config_and_guard(self, tmp_path):
        exp = Experiment(task="t", base_dir=tmp_path, run_id="r1")
        exp.snapshot_config({"task_name": "t", "pipeline": {"context": {"chain_mode": "none"}}})
        assert (exp.root / "config.yaml").exists()
        # 相同配置幂等
        exp.snapshot_config({"task_name": "t", "pipeline": {"context": {"chain_mode": "none"}}})
        # 不同配置 = 不同实验 → 拒绝混跑
        with pytest.raises(RuntimeError, match="续跑配置与快照不一致"):
            exp.snapshot_config({"task_name": "t", "pipeline": {"context": {"chain_mode": "hard"}}})

    def test_bind_query_guard(self, tmp_path):
        exp = Experiment(task="t", base_dir=tmp_path, run_id="r1")
        exp.bind_query("春天在哪里")
        # 同 query 幂等
        exp.bind_query("春天在哪里")
        # 续跑换 query = 剧本缓存会错位 → 拒绝
        with pytest.raises(RuntimeError, match="续跑 query 不一致"):
            exp.bind_query("冬天在哪里")
        # 新 run 不受影响
        exp2 = Experiment(task="t", base_dir=tmp_path, run_id="r2")
        exp2.bind_query("冬天在哪里")

    def test_finalize_costs_by_declared_backend(self, tmp_path):
        @register("generator.cost-local")
        class Local:
            capabilities = {"max_duration_s": 15, "audio": True, "refs": 9,
                            "first_last_frame": True, "resolution": "768p",
                            "backend": "local"}

        @register("generator.cost-api")
        class Api:
            capabilities = {"max_duration_s": 15, "audio": True, "refs": 9,
                            "first_last_frame": True, "resolution": "2K",
                            "backend": "api"}

        exp = Experiment(task="t", base_dir=tmp_path, run_id="r1")
        (Path(tmp_path) / "a.mp4").write_bytes(b"fake")
        (Path(tmp_path) / "b.mp4").write_bytes(b"fake")
        exp.save_artifact("segments", Artifact(
            kind="video", path=Path(tmp_path) / "a.mp4",
            meta=ArtifactMeta(adapter="generator.cost-local", elapsed_s=3600.0)))
        exp.save_artifact("segments", Artifact(
            kind="video", path=Path(tmp_path) / "b.mp4",
            meta=ArtifactMeta(adapter="generator.cost-api", elapsed_s=3600.0, cost_usd=1.5)))
        exp.finalize(gpu_price_usd_per_hour=2.0)
        m = json.loads((exp.root / "manifest.json").read_text())
        assert m["local_gpu_hours"] == 1.0
        assert m["local_gpu_cost_usd_est"] == 2.0
        assert m["total_cost_usd"] == pytest.approx(1.5)


class TestConfigValidation:
    def _base(self):
        return {
            "task_name": "story_short",
            "segments": 4,
            "pipeline": {
                "script": {"adapter": "script.x", "params": {}},
                "generator": {"adapter": "generator.x", "params": {}},
                "context": {"chain_mode": "none", "anchor_refs": []},
            },
            "judge": {"adapter": "judge.x", "params": {}},
            "script_judge": [{"name": "叙事完整", "question": "q?", "weight": 1.0, "min_score": 6}],
            "script_retry": {"max_attempts": 2, "inject_feedback": True},
            "segment_judge": [{"name": "与指令一致性", "question": "q?"}],
            "segment_retry": {"max_attempts": 2},
            "cross_judge": [{"name": "跨段一致性", "question": "q?"}],
            "audio_verify": {"adapter": "transcribe.x"},
            "memory": {"path": "_memory.jsonl", "promote_threshold": 2},
        }

    def test_valid_config_passes(self):
        assert validate_task(self._base()) is not None

    def test_unknown_key_fails_loud(self):
        cfg = self._base()
        cfg["segmant"] = 4                    # 拼写错误
        with pytest.raises(ConfigError, match="未知配置键"):
            validate_task(cfg)
        cfg = self._base()
        cfg["pipeline"]["genrator"] = {}      # 拼写错误
        with pytest.raises(ConfigError, match="未知配置键"):
            validate_task(cfg)

    def test_bad_chain_mode(self):
        cfg = self._base()
        cfg["pipeline"]["context"]["chain_mode"] = "hardcore"
        with pytest.raises(ConfigError, match="未知衔接策略"):
            validate_task(cfg)

    def test_bad_criteria(self):
        cfg = self._base()
        cfg["script_judge"][0].pop("name")
        with pytest.raises(ConfigError, match="缺少非空 name"):
            validate_task(cfg)
        cfg = self._base()
        cfg["script_judge"][0]["weight"] = "heavy"
        with pytest.raises(ConfigError, match="应为数值"):
            validate_task(cfg)

    def test_adapter_block_requires_adapter(self):
        cfg = self._base()
        cfg["pipeline"]["script"] = {"params": {}}
        with pytest.raises(ConfigError, match="缺少 adapter"):
            validate_task(cfg)


class TestFallback:
    def test_fallback_switches_on_failure(self):
        from vidharness.consumers.fallback import FallbackGenerator
        from vidharness.seams import GenRequest

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

        class Boom:
            name = "boom"
            capabilities = {}
            def generate(self, req, workdir, **kw):
                raise RuntimeError("不可用")

        fb = FallbackGenerator.__new__(FallbackGenerator)
        fb.chain = [Boom()]
        fb.name = "fallback[boom]"
        fb.capabilities = {}
        with pytest.raises(RuntimeError):
            fb.generate(GenRequest(text="t"), workdir=Path("."))


class TestScriptOptimizer:
    def test_optimizer_selects_best(self, tmp_path):
        from vidharness.consumers.script_optimizer import ScriptOptimizer
        from vidharness.core.memory import ExperienceMemory

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
                return Artifact(kind="scores", path=Path(workdir) / "j.json",
                                meta=ArtifactMeta(),
                                payload={"scores": {"旁白自然": score},
                                         "feedback": "再真实一点" if score < 8 else "pass"})

        exp = Experiment(task="t", base_dir=tmp_path, run_id="r1")
        mem = ExperienceMemory(tmp_path / "_memory.jsonl")
        opt = ScriptOptimizer(FakeScriptAdapter(), FakeJudge(), mem, exp,
                              rounds=2, candidates=2, target_score=9.5, segments=3)
        crit = [JudgeCriteria(name="旁白自然", question="q", min_score=6)]
        best, history = opt.optimize("目标", "brief", crit, tmp_path / "s")
        assert best["segments"][0]["narration"] == "旁白3"   # 首个最高分候选
        assert len(history) == 4                              # 两轮跑满
        assert max(r["score"] for r in history) == 9.0

    def test_optimizer_uses_configured_segments(self, tmp_path):
        """Bug#2 回归：优化器段数必须来自任务配置，而非 manifest 里不存在的字段。"""
        from vidharness.consumers.script_optimizer import ScriptOptimizer
        from vidharness.core.memory import ExperienceMemory

        seen = {}

        class FakeScriptAdapter:
            name = "fake"
            def generate(self, query, template, workdir, **kw):
                seen["segments"] = template["segments"]
                payload = {"segments": [{"video_prompt": "p", "narration": "n", "duration": 8}]}
                path = Path(workdir) / "s.json"
                path.write_text(json.dumps(payload))
                return Artifact(kind="script", path=path, meta=ArtifactMeta(), payload=payload)

        class FakeJudge:
            def judge(self, media, criteria, workdir, **kw):
                return Artifact(kind="scores", path=Path(workdir) / "j.json",
                                meta=ArtifactMeta(),
                                payload={"scores": {"旁白自然": 9.0}, "feedback": "pass"})

        exp = Experiment(task="t", base_dir=tmp_path, run_id="r1")
        opt = ScriptOptimizer(FakeScriptAdapter(), FakeJudge(),
                              ExperienceMemory(tmp_path / "_memory.jsonl"), exp,
                              rounds=1, candidates=1, segments=6)
        opt.optimize("目标", "brief",
                     [JudgeCriteria(name="旁白自然", question="q", min_score=6)],
                     tmp_path / "s")
        assert seen["segments"] == 6


# ---------------------------------------------------------------------------
# SegmentDirector 能力校验用的假适配器（模块级注册一次；registry 是进程全局的）
# ---------------------------------------------------------------------------

@register("script.cap-test")
class _FakeScript:
    name = "script.cap-test"
    capabilities = {"language": "zh", "json_output": True}
    def __init__(self):
        pass
    def generate(self, query, template, workdir, **kw):
        payload = {"segments": [{"video_prompt": "p", "narration": "n", "duration": 5}]}
        path = Path(workdir) / "s.json"
        path.write_text(json.dumps(payload))
        return Artifact(kind="script", path=path, meta=ArtifactMeta(), payload=payload)


@register("judge.cap-test")
class _FakeJudge:
    name = "judge.cap-test"
    capabilities = {"frame_sampling": True}
    modalities = ["text"]
    def __init__(self):
        pass
    def judge(self, media, criteria, workdir, **kw):
        return Artifact(kind="scores", path=Path(workdir) / "j.json",
                        meta=ArtifactMeta(),
                        payload={"scores": {}, "feedback": "pass"})


@register("generator.text-only")
class _TextOnlyGen:
    name = "generator.text-only"
    capabilities = {"max_duration_s": 15, "audio": True, "refs": 0,
                    "first_last_frame": False, "resolution": "768p",
                    "backend": "local"}
    def __init__(self):
        pass
    def generate(self, req, workdir, **kw):
        return Artifact(kind="video", path=Path(workdir) / "v.mp4",
                        meta=ArtifactMeta(adapter=self.name))


class TestSegmentDirectorCapabilities:
    """Bug#3 回归：能力校验必须按 chain_mode 推导，而非硬编码 first_last_frame。"""

    def _make_director(self, tmp_path, chain_mode):
        from vidharness.consumers.segment_director import SegmentDirector

        cfg = {
            "task_name": "t",
            "pipeline": {
                "script": {"adapter": "script.cap-test", "params": {}},
                "generator": {"adapter": "generator.text-only", "params": {}},
                "context": {"chain_mode": chain_mode, "anchor_refs": []},
            },
            "judge": {"adapter": "judge.cap-test", "params": {}},
        }
        exp = Experiment(task="t", base_dir=tmp_path)
        return SegmentDirector(exp, cfg)

    def test_chain_none_needs_no_frame_capability(self, tmp_path):
        d = self._make_director(tmp_path, "none")
        assert d.generator.name == "generator.text-only"

    def test_chain_hard_demands_first_last_frame(self, tmp_path):
        with pytest.raises(RuntimeError, match="first_last_frame"):
            self._make_director(tmp_path, "hard")

    def test_chain_ref_demands_refs(self, tmp_path):
        with pytest.raises(RuntimeError, match="refs"):
            self._make_director(tmp_path, "ref")

    def test_fallback_instance_capabilities_checked(self, tmp_path):
        """fallback 的能力是实例级并集：校验必须发生在实例化之后。"""
        from vidharness.consumers.segment_director import SegmentDirector
        from vidharness.consumers.fallback import FallbackGenerator
        try:
            register("generator.fallback")(FallbackGenerator)   # 测试进程未加载 providers，补注册
        except ValueError:
            pass                                                # 已注册则幂等跳过

        @register("generator.fb-ok")
        class FBOK:
            name = "generator.fb-ok"
            capabilities = {"max_duration_s": 15, "audio": True, "refs": 9,
                            "first_last_frame": True, "resolution": "768p",
                            "backend": "local"}
            def __init__(self):
                pass
            def generate(self, req, workdir, **kw):
                return Artifact(kind="video", path=Path(workdir) / "v.mp4",
                                meta=ArtifactMeta(adapter=self.name))

        cfg = {
            "task_name": "t",
            "pipeline": {
                "script": {"adapter": "script.cap-test", "params": {}},
                "generator": {"adapter": "generator.fallback",
                              "params": {"chain": ["generator.fb-ok"]}},
                "context": {"chain_mode": "none", "anchor_refs": []},
            },
            "judge": {"adapter": "judge.cap-test", "params": {}},
            "audio_verify": {"adapter": "transcribe.x"},   # 触发 audio 能力要求
        }
        exp = Experiment(task="t", base_dir=tmp_path)
        d = SegmentDirector(exp, cfg)
        assert d.generator.capabilities["audio"] is True   # 并集能力
        assert exp.manifest["generator_capabilities"]["audio"] is True
