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
from vidharness.core.experiment import Experiment, replay_events  # noqa: E402
from vidharness.core.invariants import check_experiment  # noqa: E402
from vidharness.core.config import validate_task, ConfigError  # noqa: E402


def _build_exp(tmp_path: Path) -> Experiment:
    """构造一个有事件流、产物、评测、重试、配置快照的最小实验。"""
    exp = Experiment(task="t", base_dir=tmp_path, run_id="r1")
    exp.bind_query("q")
    exp.snapshot_config({"task_name": "t"})
    (Path(tmp_path) / "v.mp4").write_bytes(b"fake")
    exp.save_artifact("segments", Artifact(
        kind="video", path=Path(tmp_path) / "v.mp4",
        meta=ArtifactMeta(adapter="x", elapsed_s=2.0, cost_usd=0.5)), name="seg01")
    exp.save_eval("segments", [{"attempt": 1, "score": 8.0}])
    exp.record_retry("segments")
    return exp


def _event_types(exp: Experiment):
    return [json.loads(l)["type"]
            for l in exp.events_path.read_text(encoding="utf-8").splitlines()]


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
        workdir.mkdir(parents=True, exist_ok=True)
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
        workdir.mkdir(parents=True, exist_ok=True)
        path = Path(workdir) / "j.json"
        path.write_text(json.dumps({"scores": {}, "feedback": "pass"}), encoding="utf-8")
        return Artifact(kind="scores", path=path,
                        meta=ArtifactMeta(adapter=self.name),
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
        workdir.mkdir(parents=True, exist_ok=True)
        p = Path(workdir) / "v.mp4"
        p.write_bytes(b"fake")
        return Artifact(kind="video", path=p,
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


class TestEventSourcing:
    def test_events_emitted(self, tmp_path):
        exp = _build_exp(tmp_path)
        types = _event_types(exp)
        assert types[0] == "run.created"
        for t in ("query.bound", "config.snapshotted", "artifact.saved",
                  "eval.saved", "retry"):
            assert t in types, types

    def test_crash_recovery_rebuilds_manifest(self, tmp_path):
        exp = _build_exp(tmp_path)
        (exp.root / "manifest.json").unlink()   # 模拟崩溃丢失投影
        exp2 = Experiment(task="t", base_dir=tmp_path, run_id="r1")
        assert exp2.events_complete is True
        assert exp2.manifest["query"] == "q"
        assert exp2.manifest["total_cost_usd"] == 0.5
        assert exp2.manifest["retries"] == {"segments": 1}
        assert len(exp2.manifest["stages"]["segments"]) == 1

    def test_legacy_run_keeps_manifest_authority(self, tmp_path):
        exp = Experiment(task="t", base_dir=tmp_path, run_id="r1")
        exp.snapshot_config({"task_name": "t"})   # 先把 manifest 落盘
        (exp.root / "events.jsonl").unlink()      # 模拟老 run：无事件流
        exp2 = Experiment(task="t", base_dir=tmp_path, run_id="r1")
        assert exp2.events_complete is False
        assert exp2.manifest["task"] == "t"

    def test_replay_matches_manifest_after_finalize(self, tmp_path):
        exp = _build_exp(tmp_path)
        exp.finalize()                          # finalize 内含不变量校验
        proj = replay_events(exp.events_path)
        m = json.loads((exp.root / "manifest.json").read_text(encoding="utf-8"))
        assert proj["total_cost_usd"] == m["total_cost_usd"]
        assert proj["total_elapsed_s"] == m["total_elapsed_s"]
        assert proj["retries"] == m["retries"]
        assert list(proj["stages"]) == list(m["stages"])
        assert len(proj["stages"]["segments"]) == len(m["stages"]["segments"])


class TestInvariants:
    def test_healthy_run_passes(self, tmp_path):
        exp = _build_exp(tmp_path)
        exp.finalize()
        assert check_experiment(exp.root) == []

    def test_cost_mismatch_detected(self, tmp_path):
        exp = _build_exp(tmp_path)
        m = json.loads((exp.root / "manifest.json").read_text(encoding="utf-8"))
        m["total_cost_usd"] = 999.0
        (exp.root / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
        v = check_experiment(exp.root)
        assert any("total_cost_usd" in x for x in v)

    def test_missing_artifact_detected(self, tmp_path):
        exp = _build_exp(tmp_path)
        art = exp.artifacts_dir / "segments" / "seg01.mp4"
        art.unlink()
        v = check_experiment(exp.root)
        assert any("产物文件缺失" in x for x in v)

    def test_tampered_config_detected(self, tmp_path):
        exp = _build_exp(tmp_path)
        cfg = exp.root / "config.yaml"
        cfg.write_text(cfg.read_text(encoding="utf-8") + "# 篡改\n", encoding="utf-8")
        v = check_experiment(exp.root)
        assert any("配置快照被修改" in x for x in v)

    def test_bad_eval_file_detected(self, tmp_path):
        exp = _build_exp(tmp_path)
        (exp.eval_dir / "bad.json").write_text("not json", encoding="utf-8")
        v = check_experiment(exp.root)
        assert any("无法解析" in x for x in v)

    def test_event_divergence_detected(self, tmp_path):
        exp = _build_exp(tmp_path)
        # 手写一条与 manifest 不一致的事件（多出一件高价产物）
        with open(exp.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": 0, "type": "artifact.saved", "v": 1,
                                "stage": "segments",
                                "entry": {"kind": "video", "path": "/x.mp4",
                                          "meta": {"cost_usd": 5.0, "elapsed_s": 0.0}}},
                               ensure_ascii=False) + "\n")
        v = check_experiment(exp.root)
        assert any("事件重放" in x for x in v)


class TestParamSchema:
    def test_schema_type_choices_required(self):
        @register("generator.schema-test")
        class S:
            name = "generator.schema-test"
            capabilities = {"max_duration_s": 15, "audio": True, "refs": 9,
                            "first_last_frame": True, "resolution": "768p",
                            "backend": "local"}
            param_schema = {
                "model_path": {"type": "path", "required": True, "help": "权重目录"},
                "steps": {"type": "int", "default": None},
                "variant": {"type": "str", "choices": ["t2va", "fl2va"], "default": "t2va"},
            }
            def __init__(self, model_path, steps=None, variant="t2va"):
                self.model_path = model_path
                self.steps = steps

        with pytest.raises(RuntimeError, match="缺少必需参数 'model_path'"):
            instantiate("generator.schema-test", {})
        with pytest.raises(RuntimeError, match="类型应为 int"):
            instantiate("generator.schema-test", {"model_path": "/x", "steps": "30"})
        with pytest.raises(RuntimeError, match="只允许"):
            instantiate("generator.schema-test", {"model_path": "/x", "variant": "fl2va2"})
        with pytest.raises(RuntimeError, match="不接受参数"):
            instantiate("generator.schema-test", {"model_path": "/x", "modelpath": "/y"})
        o = instantiate("generator.schema-test", {"model_path": "/x", "steps": 30})
        assert o.steps == 30

    def test_builtin_schemas_match_constructors(self):
        """声明目录必须与构造签名一致（防 drift 的元测试）。"""
        import inspect
        from vidharness.core.registry import load_builtin_adapters, list_adapters, get
        load_builtin_adapters()
        checked = 0
        for name in list_adapters():
            cls = get(name)
            schema = getattr(cls, "param_schema", None)
            if schema is None:
                continue    # 未声明目录的适配器走签名内省兜底
            sig = inspect.signature(cls.__init__)
            ctor = {p for p in sig.parameters if p != "self"}
            assert set(schema) == ctor, \
                f"{name}: 声明 {sorted(schema)} ≠ 构造签名 {sorted(ctor)}"
            checked += 1
        assert checked >= 4   # 4 个内置提供者都声明了目录


class TestTools:
    def test_require_tool_fails_loud(self, monkeypatch):
        import shutil
        from vidharness.consumers.tools import require_tool
        monkeypatch.setattr(shutil, "which", lambda name: None)
        with pytest.raises(RuntimeError, match="未找到 ffmpeg"):
            require_tool("ffmpeg")

    def test_require_tool_found(self, monkeypatch):
        import shutil
        from vidharness.consumers.tools import require_tool
        monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
        assert require_tool("ffprobe") == "/usr/bin/ffprobe"


@register("generator.bench-fake")
class _BenchFakeGen:
    name = "generator.bench-fake"
    capabilities = {"max_duration_s": 15, "audio": True, "refs": 9,
                    "first_last_frame": True, "resolution": "768p",
                    "backend": "api",
                    "cost_rates_usd_per_s": {"768P": 0.1, "2K": 0.2}}
    param_schema = {"steps": {"type": "int", "default": 30}}
    def __init__(self, steps=30):
        self.steps = steps

    def generate(self, req, workdir, **kw):
        raise NotImplementedError("规划假提供者，不实际生成")


class TestBench:
    def _base_cfg(self):
        return {
            "task_name": "t", "segments": 4,
            "pipeline": {
                "script": {"adapter": "script.cap-test", "params": {}},
                "generator": {"adapter": "generator.bench-fake",
                              "params": {"steps": 30}},
                "context": {"chain_mode": "none", "anchor_refs": []}},
            "judge": {"adapter": "judge.cap-test", "params": {}},
        }

    def test_matrix_expansion(self):
        from vidharness.core.bench import expand_matrix
        cells = expand_matrix(self._base_cfg(), [
            {"pipeline.generator.params.steps": [20, 30]},
            {"pipeline.context.chain_mode": ["none", "hard"]},
        ])
        assert [c[0] for c in cells] == ["20.none", "20.hard", "30.none", "30.hard"]
        # 深拷贝隔离：改一格不影响其他格
        cells[0][1]["pipeline"]["generator"]["params"]["steps"] = 999
        assert cells[2][1]["pipeline"]["generator"]["params"]["steps"] == 30

    def test_matrix_bad_path_fails(self):
        from vidharness.core.bench import expand_matrix, BenchError
        with pytest.raises(BenchError, match="不可写"):
            expand_matrix(self._base_cfg(), [{"nope.steps": [1]}])
        with pytest.raises(BenchError, match="非空列表"):
            expand_matrix(self._base_cfg(), [{"pipeline.generator.params.steps": []}])

    def test_plan_validates_every_cell(self, tmp_path):
        from vidharness.core.bench import plan, BenchError
        base = tmp_path / "base.yaml"
        base.write_text(json.dumps(self._base_cfg(), ensure_ascii=False), encoding="utf-8")
        spec = {"bench": {"base": str(base),
                          "matrix": [{"pipeline.generator.params.steps": [20, "bad"]}]}}
        # 第二格 steps="bad" 违反参数声明 → 规划期整体失败（不花 GPU）
        with pytest.raises(Exception):
            plan(spec)

    def test_plan_estimate_api(self, tmp_path):
        from vidharness.core.bench import plan
        base = tmp_path / "base.yaml"
        base.write_text(json.dumps(self._base_cfg(), ensure_ascii=False), encoding="utf-8")
        spec = {"bench": {"base": str(base),
                          "matrix": [{"pipeline.generator.params.steps": [20]}]}}
        rows = plan(spec)
        assert len(rows) == 1
        est = rows[0]["estimate"]
        # 4 段 × 8 秒 × 0.1 USD/s（768P 声明单价）
        assert est["cost_usd_est"] == 3.2
        assert "768P" in est["basis"]

    def test_plan_estimate_local(self, tmp_path):
        from vidharness.core.bench import plan
        cfg = self._base_cfg()
        cfg["pipeline"]["generator"] = {"adapter": "generator.text-only", "params": {}}
        base = tmp_path / "base.yaml"
        base.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
        spec = {"bench": {"base": str(base),
                          "matrix": [{"segments": [4]}],
                          "local_min_per_seg": 12}}
        rows = plan(spec)
        est = rows[0]["estimate"]
        # 4 段 × 12 分钟 = 0.8 GPU 时 × 1.2 USD/卡时
        assert est["gpu_hours_est"] == 0.8
        assert est["cost_usd_est"] == 0.96

    def test_plan_rejects_unknown_bench_keys(self, tmp_path):
        from vidharness.core.bench import plan, BenchError
        base = tmp_path / "base.yaml"
        base.write_text(json.dumps(self._base_cfg(), ensure_ascii=False), encoding="utf-8")
        with pytest.raises(BenchError, match="未知键"):
            plan({"bench": {"base": str(base), "matrix": [], "matrixx": []}})


class TestReport:
    def test_collect_stage_breakdown_and_completeness(self, tmp_path):
        from vidharness.core.report import collect
        exp = _build_exp(tmp_path)
        assert collect(tmp_path, "t") == []          # 未 finalize：不完整，不进对比
        exp.finalize()
        runs = collect(tmp_path, "t")
        assert len(runs) == 1
        r = runs[0]
        assert r["stages_cost_usd"]["segments"] == 0.5
        assert r["stages_elapsed_s"]["segments"] == 2.0
        assert r["finished_at"]

    def test_collect_legacy_completeness_by_final_video(self, tmp_path):
        from vidharness.core.report import collect
        exp = _build_exp(tmp_path)
        (exp.final_dir / "final_video.mp4").write_bytes(b"fake")   # 旧口径：有成品即完整
        runs = collect(tmp_path, "t")
        assert len(runs) == 1 and runs[0]["finished_at"] is None

    def test_bench_cell_passthrough(self, tmp_path):
        from vidharness.core.report import collect
        exp = _build_exp(tmp_path)
        exp.bind_label("20.none")
        exp.finalize()
        runs = collect(tmp_path, "t")
        assert runs[0]["bench_cell"] == "20.none"


class TestEvidenceScript:
    def _run_dir(self, tmp_path, judge_cfg=None):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        cfg = {"judge": judge_cfg} if judge_cfg else {}
        (run_dir / "config.yaml").write_text(
            json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
        return run_dir

    def test_load_judge_from_snapshot(self, tmp_path):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from collect_evidence import load_judge_from_run
        run_dir = self._run_dir(tmp_path, {"adapter": "judge.cap-test", "params": {}})
        judge = load_judge_from_run(run_dir)
        assert judge.name == "judge.cap-test"

    def test_load_judge_without_snapshot_fails_loud(self, tmp_path):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from collect_evidence import load_judge_from_run
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        with pytest.raises(RuntimeError, match="缺少配置快照"):
            load_judge_from_run(run_dir)

    def test_load_judge_without_judge_key_fails_loud(self, tmp_path):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from collect_evidence import load_judge_from_run
        run_dir = self._run_dir(tmp_path, None)
        with pytest.raises(RuntimeError, match="缺少 judge.adapter"):
            load_judge_from_run(run_dir)


class TestSeamConformance:
    BUILTINS = {
        "generator.fallback", "generator.minimax-h3-api", "generator.minimax-h3-local",
        "judge.openai-compat", "judge.deepseek-text",
        "script.deepseek-v4-flash", "script.openai-compat",
        "transcribe.sensevoice-small",
    }

    def test_registered_providers_conform_to_seam(self):
        """seam 一致性元测试：每个内置提供者都具备其 seam 的协议成员。"""
        from vidharness.core.registry import (load_builtin_adapters, list_adapters,
                                              get, capabilities)
        load_builtin_adapters()
        methods = {"generator": "generate", "judge": "judge",
                   "script": "generate", "transcribe": "transcribe"}
        checked = 0
        for name in list_adapters():
            if name not in self.BUILTINS:
                continue    # 测试假类不在 seam 契约范围内
            cls = get(name)
            seam = name.split(".")[0]
            assert isinstance(getattr(cls, "name", ""), str) and cls.name, name
            assert callable(getattr(cls, methods[seam])), f"{name} 缺 {methods[seam]}"
            assert isinstance(capabilities(name), dict), f"{name} 能力声明缺失"
            if seam == "generator" and "fallback" not in name:
                assert capabilities(name).get("backend") in ("local", "api"), \
                    f"{name} 未声明 backend 成本口径"
            checked += 1
        assert checked == len(self.BUILTINS)


class TestLeaderboard:
    def test_export_and_diff(self, tmp_path):
        from vidharness.core.leaderboard import export
        exp = _build_exp(tmp_path)
        exp.bind_label("20.none")
        exp.save_eval("segments", [{"attempt": 2, "scores": {"与指令一致性": 9.0},
                                    "passed": True, "score": 9.0}])
        exp.finalize()
        out = tmp_path / "leaderboards"
        json_p, md_p, diff = export(tmp_path, "t", out)
        assert json_p.exists() and md_p.exists()
        data = json.loads(json_p.read_text(encoding="utf-8"))
        assert data["run_count"] == 1
        assert data["runs"][0]["bench_cell"] == "20.none"
        assert data["runs"][0]["stage_scores"]["segments"]["与指令一致性"] == 9.0
        assert "new_runs" in diff and data["runs"][0]["run_id"] in diff["new_runs"]
        md = md_p.read_text(encoding="utf-8")
        assert exp.run_id in md and "20.none" in md
        # 再次导出：无增量
        _, _, diff2 = export(tmp_path, "t", out)
        assert diff2["new_runs"] == [] and diff2["removed_runs"] == []

    def test_export_removed_run_detected(self, tmp_path):
        from vidharness.core.leaderboard import export
        exp = _build_exp(tmp_path)
        exp.finalize()
        out = tmp_path / "leaderboards"
        export(tmp_path, "t", out)
        # 基线在，run 消失（模拟目录被清）
        import shutil
        shutil.rmtree(exp.root)
        _, _, diff = export(tmp_path, "t", out)
        assert diff["removed_runs"] == ["r1"]


class TestCollectStageAggregation:
    def test_stage_scores_and_passed(self, tmp_path):
        from vidharness.core.report import collect
        exp = _build_exp(tmp_path)
        exp.save_eval("segments", [{"attempt": 1, "scores": {"与指令一致性": 6.0},
                                    "passed": True, "score": 6.0},
                                   {"attempt": 2, "scores": {"与指令一致性": 8.0},
                                    "passed": False, "score": 8.0}])
        exp.finalize()
        runs = collect(tmp_path, "t")
        r = runs[0]
        assert r["stage_scores"]["segments"]["与指令一致性"] == 7.0
        assert r["stage_passed"]["segments"] == {"passed": 1, "total": 3}
        assert r["passed_rate"] == pytest.approx(0.33)


class TestExperienceMemory:
    def test_promote_on_threshold(self, tmp_path):
        """Bug#5 回归：重复反馈到达阈值必须提升为经验（此前 promoted 从不置位）。"""
        from vidharness.core.memory import ExperienceMemory
        mem = ExperienceMemory(tmp_path / "_memory.jsonl", promote_threshold=2)
        mem.add("旁白太肉麻，要真实朴素", source="run1/judge")
        assert mem.experience_lines() == []          # 第一次不提升
        mem.add("旁白太肉麻，要真实朴素！", source="run2/judge")   # 规范化后同 key
        assert "旁白太肉麻，要真实朴素" in mem.experience_lines()
        assert mem.recent_feedback() == []           # 提升后不再算待提升反馈

    def test_threshold_one_promotes_first_add(self, tmp_path):
        from vidharness.core.memory import ExperienceMemory
        mem = ExperienceMemory(tmp_path / "_memory.jsonl", promote_threshold=1)
        mem.add("镜头晃动", source="r1")
        assert mem.experience_lines() == ["镜头晃动"]

    def test_legacy_lines_load_and_upgrade(self, tmp_path):
        from vidharness.core.memory import ExperienceMemory
        p = tmp_path / "_memory.jsonl"
        # 旧格式：无 v 字段、无 promoted 字段
        p.write_text(json.dumps({"key": "旧经验", "complaint": "旧经验",
                                 "kind": "experience", "count": 3,
                                 "sources": ["x"], "first_at": 1, "last_at": 1})
                     + "\n", encoding="utf-8")
        mem = ExperienceMemory(p)
        assert mem.experience_lines() == ["旧经验"]
        assert mem.load_warnings == []
        # flush 后升级为 v=1
        mem.add("新反馈", source="r1")
        lines = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert all(l.get("v") == 1 for l in lines)

    def test_corrupt_lines_skipped_with_warning(self, tmp_path):
        from vidharness.core.memory import ExperienceMemory
        p = tmp_path / "_memory.jsonl"
        p.write_text("not json\n" + json.dumps({"key": "k", "complaint": "c",
                                                  "kind": "experience", "promoted": True}) + "\n",
                     encoding="utf-8")
        mem = ExperienceMemory(p)
        assert len(mem.load_warnings) == 1 and "无法解析" in mem.load_warnings[0]
        assert mem.experience_lines() == ["c"]

    def test_sources_capped(self, tmp_path):
        from vidharness.core.memory import ExperienceMemory
        mem = ExperienceMemory(tmp_path / "_memory.jsonl", promote_threshold=99)
        for i in range(7):
            mem.add("同一问题", source=f"r{i}")
        item = mem._items[0]
        assert item["count"] == 7
        assert item["sources"] == ["r2", "r3", "r4", "r5", "r6"]   # 只留最后 5 个


class TestRunReport:
    def test_render_run_html(self, tmp_path):
        from vidharness.core.report import render_run_html
        exp = _build_exp(tmp_path)
        exp.save_eval("segments", [{"attempt": 2, "scores": {"与指令一致性": 9.0},
                                    "passed": True, "score": 9.0, "feedback": "主体明确"}])
        exp.finalize()
        out = render_run_html(exp.root, exp.root / "report.html")
        html = out.read_text(encoding="utf-8")
        assert exp.run_id in html
        assert "segments" in html and "config.yaml 存在" or "chain_mode" in html
        assert "与指令一致性" in html          # 评测明细
        assert "artifact.saved" in html        # 事件流
        assert "generator_capabilities" in html or "query" in html

    def test_render_run_html_missing_manifest(self, tmp_path):
        from vidharness.core.report import render_run_html
        with pytest.raises(RuntimeError, match="缺少 manifest"):
            render_run_html(tmp_path, tmp_path / "report.html")


class TestDeepSeekTextJudge:
    """judge 缝的第二实现（孪生适配器）：DeepSeek 官方 API 文本裁判。"""

    def _fake_client(self, monkeypatch, raw='{"叙事完整": 8, "feedback": "pass"}'):
        from types import SimpleNamespace
        captured = {}

        class FakeClient:
            def __init__(self, **kw):
                captured["init"] = kw
            @property
            def chat(self):
                return self
            @property
            def completions(self):
                return self
            def create(self, **kw):
                captured["create"] = kw
                msg = SimpleNamespace(content=raw)
                choice = SimpleNamespace(message=msg)
                return SimpleNamespace(choices=[choice], model="deepseek-chat",
                                       usage=SimpleNamespace(
                                           prompt_tokens=100, completion_tokens=50))

        monkeypatch.setattr("vidharness.providers.judge_deepseek_text.OpenAI", FakeClient)
        return captured

    def test_judge_text_and_cost(self, tmp_path, monkeypatch):
        from vidharness.core.registry import instantiate
        captured = self._fake_client(monkeypatch)
        judge = instantiate("judge.deepseek-text", {"api_key": "k"}, context="t")
        crit = criteria_to_spec([JudgeCriteria(name="叙事完整", question="分镜完整吗？", min_score=6)])
        art = judge.judge(media=[], criteria=crit, workdir=tmp_path)
        assert art.payload["scores"] == {"叙事完整": 8.0}
        assert art.payload["feedback"] == "pass"
        assert art.meta.cost_usd > 0            # deepseek 计费口径
        prompt = captured["create"]["messages"][0]["content"]
        assert "叙事完整" in prompt and "分镜完整吗？" in prompt
        assert art.path.exists()                # 可重建：raw+criteria 落盘
        assert captured["init"]["api_key"] == "k"

    def test_modality_guard_rejects_media_for_text_judge(self, tmp_path, monkeypatch):
        from vidharness.core.registry import instantiate
        from vidharness.consumers.judge_loop import run_judge
        self._fake_client(monkeypatch)
        judge = instantiate("judge.deepseek-text", {"api_key": "k"}, context="t")
        crit = [JudgeCriteria(name="与指令一致性", question="q", min_score=6)]
        with pytest.raises(RuntimeError, match="仅支持"):
            run_judge(judge, [Path(tmp_path) / "x.mp4"], crit, tmp_path)

    def test_none_media_filtered(self, tmp_path):
        from vidharness.consumers.judge_loop import run_judge

        class VLMJudge:
            name = "j"
            modalities = ["text", "image", "video"]
            def judge(self, media, criteria, workdir, **kw):
                self.seen = media
                return Artifact(kind="scores", path=Path(workdir) / "j.json",
                                meta=ArtifactMeta(),
                                payload={"scores": {"与指令一致性": 8.0}, "feedback": "pass"})
        j = VLMJudge()
        crit = [JudgeCriteria(name="与指令一致性", question="q", min_score=6)]
        img = tmp_path / "a.jpg"
        img.write_bytes(b"fake")
        run_judge(j, [None, img], crit, tmp_path)
        assert j.seen == [img]                   # None 被过滤，不传给裁判


class TestCrossConsistencyFrameFailure:
    def test_missing_frames_recorded_not_faked(self, tmp_path, monkeypatch):
        """抽帧失败必须可见：记错误记录，而不是让裁判空评。"""
        from vidharness.consumers.segment_director import SegmentDirector
        director = TestSegmentDirectorCapabilities()._make_director(tmp_path, "none")
        monkeypatch.setattr(SegmentDirector, "_extract_last_frame",
                            staticmethod(lambda video, exp: None))
        monkeypatch.setattr(SegmentDirector, "_extract_frame",
                            staticmethod(lambda video, t, exp: None))
        result = director.stage_cross_consistency(
            [Path("a.mp4"), Path("b.mp4")], {"segments": []})
        assert result["checked"] is True
        assert any("error" in r and "抽帧失败" in r["error"] for r in result["records"])


class TestStageLifecycle:
    def test_stage_events_pair_and_invariants_pass(self, tmp_path, monkeypatch):
        from vidharness.consumers.segment_director import SegmentDirector
        # 抽帧/总装 monkeypatch（测试环境不依赖 ffmpeg 与真实媒体）
        monkeypatch.setattr(SegmentDirector, "_extract_last_frame",
                            staticmethod(lambda video, exp: None))
        monkeypatch.setattr(SegmentDirector, "_extract_frame",
                            staticmethod(lambda video, t, exp: None))
        monkeypatch.setattr(SegmentDirector, "stage_assemble",
                            lambda self, videos, script: Path(tmp_path) / "final.mp4")
        director = TestSegmentDirectorCapabilities()._make_director(tmp_path, "none")
        final = director.run("测试故事")
        assert final == Path(tmp_path) / "final.mp4"
        types = _event_types(director.exp)
        assert types.count("stage.started") == 4        # script/segments/cross/assemble
        assert types.count("stage.finished") == 4
        from vidharness.core.invariants import check_experiment
        assert check_experiment(director.exp.root) == []

    def test_unpaired_stage_detected(self, tmp_path):
        exp = _build_exp(tmp_path)
        exp.stage_started("script")
        exp.stage_finished("script")
        exp.stage_started("segments")                   # 故意不 finish
        with pytest.raises(RuntimeError, match="不变量"):
            exp.finalize()                               # 收尾时应拒绝：配对不变量
        from vidharness.core.invariants import check_experiment
        v = check_experiment(exp.root)
        assert any("stage.started 但无 stage.finished" in x for x in v)


class TestScriptSeamContract:
    def test_build_script_prompt(self):
        from vidharness.seams import build_script_prompt
        p = build_script_prompt("目标", {"segments": 3, "brief": "短", "experience": ["教训1"]})
        assert "共 3 个分镜" in p and "短" in p and "教训1" in p and "输出 JSON" in p

    def test_parse_script_json_variants(self):
        from vidharness.seams import parse_script_json
        d = parse_script_json('```json\n{"segments": []}\n```')
        assert d == {"segments": []}
        d = parse_script_json('前缀 {"segments": []} 后缀')
        assert d == {"segments": []}
        d = parse_script_json("完全不是 JSON")
        assert "error" in d


class TestOpenAICompatScript:
    """script 缝的第二实现（孪生适配器）：通用 OpenAI 兼容端点。"""

    def _fake(self, monkeypatch, raw):
        from types import SimpleNamespace
        captured = {}

        class FakeClient:
            def __init__(self, **kw):
                captured["init"] = kw
            @property
            def chat(self):
                return self
            @property
            def completions(self):
                return self
            def create(self, **kw):
                captured["create"] = kw
                msg = SimpleNamespace(content=raw)
                choice = SimpleNamespace(message=msg)
                return SimpleNamespace(choices=[choice], model="m",
                                       usage=SimpleNamespace(
                                           prompt_tokens=100, completion_tokens=50))

        monkeypatch.setattr("vidharness.providers.script_openai_compat.OpenAI", FakeClient)
        return captured

    def test_generate_contract_and_unpriced(self, tmp_path, monkeypatch):
        raw = '{"segments": [{"video_prompt": "p", "narration": "n", "duration": 8}]}'
        captured = self._fake(monkeypatch, raw)
        gen = instantiate("script.openai-compat", {"base_url": "http://x/v1", "model": "m"})
        art = gen.generate("目标", {"segments": 3, "brief": "短", "experience": ["教训1"]},
                           tmp_path)
        assert art.payload["segments"][0]["narration"] == "n"
        assert art.meta.cost_usd == 0.0 and art.meta.params["billing"] == "unpriced"
        prompt = captured["create"]["messages"][1]["content"]
        assert "共 3 个分镜" in prompt and "短" in prompt and "教训1" in prompt

    def test_priced_billing(self, tmp_path, monkeypatch):
        self._fake(monkeypatch, '{"segments": []}')
        gen = instantiate("script.openai-compat", {
            "base_url": "http://x/v1", "model": "m",
            "price_in_usd_per_1m": 0.1, "price_out_usd_per_1m": 0.2})
        art = gen.generate("q", {}, tmp_path)
        expected = 100 / 1e6 * 0.1 + 50 / 1e6 * 0.2
        assert abs(art.meta.cost_usd - expected) < 1e-9
        assert art.meta.params["billing"] == "priced"


class TestDeepSeekScriptSeamRefactor:
    def test_generate_uses_seam_contract(self, tmp_path, monkeypatch):
        """deepseek_script 重构后仍走 seam 的提示/解析契约（行为不回归）。"""
        from types import SimpleNamespace
        captured = {}

        class FakeClient:
            def __init__(self, **kw):
                pass
            @property
            def chat(self):
                return self
            @property
            def completions(self):
                return self
            def create(self, **kw):
                captured["create"] = kw
                msg = SimpleNamespace(
                    content='```json\n{"segments": [{"video_prompt": "p", '
                            '"narration": "n", "duration": 8}]}\n```')
                choice = SimpleNamespace(message=msg)
                return SimpleNamespace(choices=[choice], model="deepseek-chat",
                                       usage=SimpleNamespace(
                                           prompt_tokens=100, completion_tokens=50))

        monkeypatch.setattr("vidharness.providers.deepseek_script.OpenAI", FakeClient)
        gen = instantiate("script.deepseek-v4-flash", {"api_key": "k"})
        art = gen.generate("目标", {"segments": 2}, tmp_path)
        assert art.payload["segments"][0]["narration"] == "n"
        assert art.meta.cost_usd > 0
        prompt = captured["create"]["messages"][1]["content"]
        assert "共 2 个分镜" in prompt
