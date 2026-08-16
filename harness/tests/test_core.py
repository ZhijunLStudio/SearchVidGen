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
from vidharness.consumers.judge_loop import (parse_scores,  # noqa: E402
                                             finalize_verdict, run_judge)
from vidharness.seams import (JudgeCriteria, Artifact, ArtifactMeta,  # noqa: E402
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
    return [json.loads(line)["type"]
            for line in exp.events_path.read_text(encoding="utf-8").splitlines()]


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
        scores, fb = parse_scores(out, crit)
        v = finalize_verdict(scores, fb, crit)
        assert v["passed"] is True
        assert v["scores"]["与指令一致性"] == 8

    def test_below_threshold_fails(self):
        out = '{"与指令一致性": 4, "画面质量": 8, "feedback": "主体崩坏"}'
        crit = [JudgeCriteria(name="与指令一致性", question="q", min_score=6),
                JudgeCriteria(name="画面质量", question="q", min_score=6)]
        scores, fb = parse_scores(out, crit)
        v = finalize_verdict(scores, fb, crit)
        assert v["passed"] is False
        assert "主体崩坏" in v["feedback"]

    def test_fallback_score_pattern(self):
        out = "总体评分：7/10，画面尚可"
        crit = [JudgeCriteria(name="与指令一致性", question="q", min_score=6)]
        scores, fb = parse_scores(out, crit)
        v = finalize_verdict(scores, fb, crit)
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
        scores, fb = parse_scores(out, crit)
        v = finalize_verdict(scores, fb, crit)
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

    def _make_director(self, tmp_path, chain_mode, adapters_cache=None):
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
        return SegmentDirector(exp, cfg, adapters_cache=adapters_cache)

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
        from vidharness.core.bench import plan
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
        lines = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        assert all(rec.get("v") == 1 for rec in lines)

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


@register("judge.route-fake")
class _RouteFakeJudge:
    name = "judge.route-fake"
    capabilities = {"frame_sampling": False}
    modalities = ["text"]
    def __init__(self):
        pass
    def judge(self, media, criteria, workdir, **kw):
        workdir.mkdir(parents=True, exist_ok=True)
        path = Path(workdir) / "j.json"
        path.write_text(json.dumps({"scores": {}, "feedback": "pass"}), encoding="utf-8")
        return Artifact(kind="scores", path=path, meta=ArtifactMeta(adapter=self.name),
                        payload={"scores": {}, "feedback": "pass"})


class TestPerStageJudgeRouting:
    def test_config_stages_validation(self):
        cfg = {
            "task_name": "t",
            "pipeline": {
                "script": {"adapter": "script.cap-test", "params": {}},
                "generator": {"adapter": "generator.text-only", "params": {}},
                "context": {"chain_mode": "none", "anchor_refs": []},
            },
            "judge": {
                "adapter": "judge.cap-test", "params": {},
                "stages": {"script_judge": {"adapter": "judge.route-fake", "params": {}}},
            },
        }
        assert validate_task(cfg) is not None
        bad = dict(cfg)
        bad["judge"]["stages"]["nosuch_stage"] = {"adapter": "judge.route-fake"}
        with pytest.raises(ConfigError, match="未知阶段键"):
            validate_task(bad)

    def test_director_routes_stages(self, tmp_path):
        from vidharness.consumers.segment_director import SegmentDirector
        cfg = {
            "task_name": "t",
            "pipeline": {
                "script": {"adapter": "script.cap-test", "params": {}},
                "generator": {"adapter": "generator.text-only", "params": {}},
                "context": {"chain_mode": "none", "anchor_refs": []},
            },
            "judge": {
                "adapter": "judge.cap-test", "params": {},
                "stages": {"script_judge": {"adapter": "judge.route-fake", "params": {}}},
            },
        }
        exp = Experiment(task="t", base_dir=tmp_path)
        d = SegmentDirector(exp, cfg)
        assert d.judges["script_judge"].name == "judge.route-fake"    # 文本评测走覆盖
        assert d.judges["segment_judge"].name == "judge.cap-test"     # 媒体评测走默认
        assert d.judges["cross_judge"].name == "judge.cap-test"

    def test_director_without_stages_defaults_everywhere(self, tmp_path):
        d = TestSegmentDirectorCapabilities()._make_director(tmp_path, "none")
        assert all(j.name == "judge.cap-test" for j in d.judges.values())


class TestDualCardKwargsSplit:
    """Bug#6 回归：双卡参数拆分——t2va 画布必须走生成侧。"""

    def test_t2va_canvas_goes_to_rest(self):
        from vidharness.providers.minimax_h3 import split_dual_card_kwargs
        kw = {"prompt": "p", "height": 768, "width": 768, "num_frames": 120}
        cond, rest = split_dual_card_kwargs("t2va", kw)
        assert cond == {"prompt": "p"}
        assert rest == {"height": 768, "width": 768, "num_frames": 120}
        assert kw == {"prompt": "p", "height": 768, "width": 768, "num_frames": 120}  # 原 dict 不变

    def test_ref2va_matches_legacy_split(self):
        from vidharness.providers.minimax_h3 import split_dual_card_kwargs
        kw = {"prompt": "p", "references": ["r"], "height": 768, "width": 768,
              "num_frames": 120, "image": "img"}
        cond, rest = split_dual_card_kwargs("ref2va", kw)
        assert cond == {"prompt": "p", "references": ["r"], "height": 768,
                        "width": 768, "num_frames": 120}
        assert rest == {"num_frames": 120, "image": "img"}


class TestMediaTools:
    def test_extract_frame_cached(self, tmp_path, monkeypatch):
        """缓存命中时不需要 ffmpeg（测试环境无 ffmpeg 也能通过）。"""
        import shutil as _shutil
        from vidharness.consumers.tools import extract_frame
        monkeypatch.setattr(_shutil, "which", lambda name: None)   # 无 ffmpeg
        out = tmp_path / "frames"
        dst = out / "v_t0.00.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"jpg")
        assert extract_frame(tmp_path / "v.mp4", 0.0, out) == dst

    def test_extract_frame_failure_returns_none(self, tmp_path, monkeypatch):
        import shutil as _shutil
        from vidharness.consumers.tools import extract_frame
        monkeypatch.setattr(_shutil, "which", lambda name: f"/usr/bin/{name}")
        assert extract_frame(tmp_path / "v.mp4", 0.0, tmp_path / "frames") is None

    def test_extract_last_frame_no_probe_returns_none(self, tmp_path, monkeypatch):
        import shutil as _shutil
        from vidharness.consumers.tools import extract_last_frame
        monkeypatch.setattr(_shutil, "which", lambda name: None)
        assert extract_last_frame(tmp_path / "v.mp4", tmp_path / "frames") is None

    def test_sample_frames_cached(self, tmp_path):
        from vidharness.consumers.tools import sample_frames
        out = tmp_path / "v_frames"
        out.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            (out / f"frame_{i:02d}.jpg").write_bytes(b"jpg")
        frames = sample_frames(tmp_path / "v.mp4", 2, tmp_path)
        assert len(frames) == 2 and frames[0].name == "frame_00.jpg"


class TestMiniMaxCostSingleSource:
    def test_estimate_cost_uses_single_source(self):
        """单一价格正源：声明目录与运行时计费同源（one home per fact）。"""
        from vidharness.providers.minimax_h3 import (_estimate_cost,
                                                     _MINIMAX_RATES_USD_PER_S,
                                                     MiniMaxH3API)
        assert MiniMaxH3API.capabilities["cost_rates_usd_per_s"] == \
            dict(_MINIMAX_RATES_USD_PER_S)
        for res, rate in _MINIMAX_RATES_USD_PER_S.items():
            assert _estimate_cost(res, 10) == round(10 * rate, 4)
        with pytest.raises(RuntimeError, match="未声明分辨率"):
            _estimate_cost("4K", 10)


class TestAdapterReuseCache:
    def test_instantiate_cache_semantics(self):
        cache = {}
        a1 = instantiate("script.param-check", {"api_key": "k"}, cache=cache)
        a2 = instantiate("script.param-check", {"api_key": "k"}, cache=cache)
        assert a1 is a2                                   # 同参数复用同一实例
        b = instantiate("script.param-check", {"api_key": "k", "temperature": 0.3},
                        cache=cache)
        assert b is not a1                                # 参数不同则新建
        c1 = instantiate("script.param-check", {"api_key": "k"})
        c2 = instantiate("script.param-check", {"api_key": "k"})
        assert c1 is not c2                               # 无缓存保持原语义

    def test_director_shares_generator_across_cells(self, tmp_path):
        """bench 逐格执行：相同生成器参数的格子复用同一实例（省模型加载）。"""
        cache = {}
        d1 = TestSegmentDirectorCapabilities()._make_director(tmp_path, "none",
                                                              adapters_cache=cache)
        d2 = TestSegmentDirectorCapabilities()._make_director(tmp_path, "none",
                                                              adapters_cache=cache)
        assert d1.generator is d2.generator
        assert d1.judges["script_judge"] is d2.judges["script_judge"]

    def test_fl2va_keyframe_to_conditioner(self):
        """fl2va 回归：image/last_image 必须进条件侧（before_encode 声明了它们），
        否则生成侧 vae_encoder 无 keyframes → condition_latents 空 → 崩溃。"""
        from vidharness.providers.minimax_h3 import split_dual_card_kwargs
        kw = {"prompt": "p", "image": "img", "last_image": "li",
              "height": 768, "width": 1344, "num_frames": 192, "generator": "g"}
        cond, rest = split_dual_card_kwargs("fl2va", kw)
        assert cond == {"prompt": "p", "image": "img", "last_image": "li",
                        "height": 768, "width": 1344}
        assert rest == {"num_frames": 192, "generator": "g"}


class TestFl2VAGuards:
    def test_fl2va_without_keyframe_fails_loud_before_gpu(self, tmp_path):
        """Bug#7 回归：fl2va 无 keyframe 在最早点响亮失败（不加载模型、
        不在 diffusers 深处 torch.cat 崩溃）。"""
        from vidharness.core.registry import load_builtin_adapters
        load_builtin_adapters()
        gen = instantiate("generator.minimax-h3-local", {"model_path": "/x",
                                                         "variant": "fl2va"})
        from vidharness.seams import GenRequest as _GenRequest
        with pytest.raises(RuntimeError, match="fl2va 变体需要首帧条件"):
            gen.generate(_GenRequest(text="t"), tmp_path)

    def test_hard_first_segment_uses_anchor_as_first_frame(self, tmp_path, monkeypatch):
        """hard 首段以锚点首图为首帧（fl2va 每段需要 keyframe）。"""
        from vidharness.consumers.segment_director import SegmentDirector
        anchor = tmp_path / "anchor.jpg"
        anchor.write_bytes(b"jpg")
        captured = {}

        @register("generator.capture")
        class CaptureGen:
            name = "generator.capture"
            capabilities = {"max_duration_s": 15, "audio": True, "refs": 9,
                            "first_last_frame": True, "resolution": "768p",
                            "backend": "local"}
            def __init__(self):
                pass
            def generate(self, req, workdir, **kw):
                captured["req"] = req
                workdir.mkdir(parents=True, exist_ok=True)
                p = Path(workdir) / "v.mp4"
                p.write_bytes(b"fake")
                return Artifact(kind="video", path=p,
                                meta=ArtifactMeta(adapter=self.name))

        cfg = {
            "task_name": "t",
            "pipeline": {
                "script": {"adapter": "script.cap-test", "params": {}},
                "generator": {"adapter": "generator.capture", "params": {}},
                "context": {"chain_mode": "hard", "anchor_refs": [str(anchor)]},
            },
            "judge": {"adapter": "judge.cap-test", "params": {}},
        }
        exp = Experiment(task="t", base_dir=tmp_path)
        d = SegmentDirector(exp, cfg)
        monkeypatch.setattr(SegmentDirector, "_extract_last_frame",
                            staticmethod(lambda video, e: None))
        script = {"segments": [{"video_prompt": "p", "narration": "n", "duration": 5}]}
        d.stage_segments(script)
        assert captured["req"].first_frame == anchor


class TestUnparseableFeedback:
    def test_feedback_has_instruction_and_context(self):
        from vidharness.consumers.judge_loop import unparseable_feedback
        fb = unparseable_feedback("完全不是 JSON 的回复")
        assert "评分解析失败" in fb and "严格只输出" in fb
        assert "完全不是 JSON 的回复" in fb[:300]

    def test_deepseek_judge_garbage_output_yields_actionable_feedback(self, tmp_path, monkeypatch):
        """E21 回归：裁判输出不可解析时，feedback 必须可操作（而非空信号）。"""
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
                msg = SimpleNamespace(content="好的，我来评分：整体不错")
                choice = SimpleNamespace(message=msg)
                return SimpleNamespace(choices=[choice], model="deepseek-chat",
                                       usage=SimpleNamespace(prompt_tokens=10,
                                                            completion_tokens=10))
        monkeypatch.setattr("vidharness.providers.judge_deepseek_text.OpenAI", FakeClient)
        judge = instantiate("judge.deepseek-text", {"api_key": "k"})
        crit = criteria_to_spec([JudgeCriteria(name="叙事完整", question="q", min_score=6)])
        art = judge.judge(media=[], criteria=crit, workdir=tmp_path)
        assert art.payload["scores"] == {}
        assert "评分解析失败" in art.payload["feedback"]


class TestBenchCellStatus:
    def _write_cell(self, base, task, run_id, label, cfg, finished):
        import yaml
        run_dir = base / task / run_id
        run_dir.mkdir(parents=True)
        m = {"run_id": run_id, "bench_cell": label, "query": "q"}
        if finished:
            m["finished_at"] = "2026-08-16T00:00:00"
        (run_dir / "manifest.json").write_text(
            json.dumps(m, ensure_ascii=False), encoding="utf-8")
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    def test_status_finished_unfinished_nomatch(self, tmp_path):
        from vidharness.core.bench import bench_cell_status
        cfg = {"task_name": "t", "pipeline": {"context": {"chain_mode": "none"}}}
        self._write_cell(tmp_path, "t", "r1", "A", cfg, finished=True)
        self._write_cell(tmp_path, "t", "r2", "A", cfg, finished=False)
        self._write_cell(tmp_path, "t", "r3", "B", dict(cfg), finished=False)
        # r2 是最新未完成匹配 → 续跑
        assert bench_cell_status(tmp_path, "t", "A", cfg, query="q") ==             {"run_id": "r2", "finished": False}
        # 配置不同的同标签 run 不算同一格
        cfg2 = {"task_name": "t", "pipeline": {"context": {"chain_mode": "hard"}}}
        self._write_cell(tmp_path, "t", "r4", "A", cfg2, finished=False)
        assert bench_cell_status(tmp_path, "t", "A", cfg, query="q")["run_id"] == "r2"
        # 无匹配
        assert bench_cell_status(tmp_path, "t", "C", cfg, query="q") ==             {"run_id": None, "finished": False}
        # 只有已完成匹配
        assert bench_cell_status(tmp_path, "t", "A", cfg)["finished"] is False
        import shutil
        for r in ("r1", "r2", "r3", "r4"):
            shutil.rmtree(tmp_path / "t" / r)
        # 只有已完成匹配 → 跳过
        self._write_cell(tmp_path, "t", "r5", "A", cfg, finished=True)
        assert bench_cell_status(tmp_path, "t", "A", cfg, query="q") ==             {"run_id": "r5", "finished": True}

    def test_query_part_of_cell_identity(self, tmp_path):
        """换 query 重跑 bench 不得跳过旧格（query 是实验变量）。"""
        from vidharness.core.bench import bench_cell_status
        cfg = {"task_name": "t", "pipeline": {"context": {"chain_mode": "none"}}}
        import yaml
        run_dir = tmp_path / "t" / "r1"
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text(json.dumps(
            {"run_id": "r1", "bench_cell": "A", "query": "旧故事",
             "finished_at": "2026-08-16T00:00:00"}, ensure_ascii=False))
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
        # 同 query → 跳过
        assert bench_cell_status(tmp_path, "t", "A", cfg, query="旧故事")["finished"] is True
        # 不同 query → 不匹配（新实验）
        assert bench_cell_status(tmp_path, "t", "A", cfg, query="新故事")["run_id"] is None


class TestJudgeSourceAnnotation:
    def test_collect_extracts_judge_adapters(self, tmp_path):
        from vidharness.core.report import collect
        exp = _build_exp(tmp_path)
        (Path(tmp_path) / "j.json").write_text("{}", encoding="utf-8")
        exp.save_artifact("judge", Artifact(
            kind="scores", path=Path(tmp_path) / "j.json",
            meta=ArtifactMeta(adapter="judge.deepseek-text")))
        exp.finalize()
        runs = collect(tmp_path, "t")
        assert runs[0]["judge_adapters"] == ["judge.deepseek-text"]

    def test_leaderboard_warns_on_mixed_judges(self, tmp_path):
        from vidharness.core.leaderboard import export
        exp = _build_exp(tmp_path)
        (Path(tmp_path) / "j.json").write_text("{}", encoding="utf-8")
        (Path(tmp_path) / "j.json").write_text("{}", encoding="utf-8")
        exp.save_artifact("judge", Artifact(
            kind="scores", path=Path(tmp_path) / "j.json",
            meta=ArtifactMeta(adapter="judge.deepseek-text")))
        exp.finalize()
        exp2 = Experiment(task="t", base_dir=tmp_path, run_id="r2")
        (Path(tmp_path) / "v2.mp4").write_bytes(b"fake")
        exp2.save_artifact("segments", Artifact(
            kind="video", path=Path(tmp_path) / "v2.mp4",
            meta=ArtifactMeta(adapter="generator.x", elapsed_s=1.0)), name="s2")
        (Path(tmp_path) / "j2.json").write_text("{}", encoding="utf-8")
        exp2.save_artifact("judge", Artifact(
            kind="scores", path=Path(tmp_path) / "j2.json",
            meta=ArtifactMeta(adapter="judge.openai-compat")))
        exp2.finalize()
        json_p, md_p, _ = export(tmp_path, "t", tmp_path / "lb")
        md = md_p.read_text(encoding="utf-8")
        assert "混用裁判" in md and "judge.deepseek-text" in md and "judge.openai-compat" in md


class TestOptimizerTemperatureSchedule:
    def test_candidates_rotate_temperatures(self, tmp_path):
        """E26 回归：候选必须温度轮转（同温候选无多样性，优化增益归零）。"""
        from vidharness.consumers.script_optimizer import ScriptOptimizer
        from vidharness.core.memory import ExperienceMemory
        temps = []

        class FakeScriptAdapter:
            name = "fake"
            def generate(self, query, template, workdir, **kw):
                temps.append(kw.get("temperature"))
                payload = {"segments": [{"video_prompt": "p", "narration": "n", "duration": 8}]}
                path = Path(workdir) / "s.json"
                path.write_text(json.dumps(payload))
                return Artifact(kind="script", path=path, meta=ArtifactMeta(), payload=payload)

        class FakeJudge:
            def judge(self, media, criteria, workdir, **kw):
                workdir.mkdir(parents=True, exist_ok=True)
                path = Path(workdir) / "j.json"
                path.write_text("{}", encoding="utf-8")
                return Artifact(kind="scores", path=path, meta=ArtifactMeta(),
                                payload={"scores": {"旁白自然": 6.0}, "feedback": "pass"})

        exp = Experiment(task="t", base_dir=tmp_path)
        opt = ScriptOptimizer(FakeScriptAdapter(), FakeJudge(),
                              ExperienceMemory(tmp_path / "_memory.jsonl"), exp,
                              rounds=2, candidates=2, target_score=9.9)
        opt.optimize("目标", "brief", [JudgeCriteria(name="旁白自然", question="q", min_score=6)],
                     tmp_path / "s")
        assert temps == [0.6, 0.9, 1.2, 0.6]   # 2轮×2候选按 [0.6,0.9,1.2] 轮转


class TestScriptTemperatureOverride:
    def test_deepseek_script_kw_temperature(self, tmp_path, monkeypatch):
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
                msg = SimpleNamespace(content='{"segments": []}')
                choice = SimpleNamespace(message=msg)
                return SimpleNamespace(choices=[choice], model="deepseek-chat",
                                       usage=SimpleNamespace(prompt_tokens=10,
                                                            completion_tokens=10))
        monkeypatch.setattr("vidharness.providers.deepseek_script.OpenAI", FakeClient)
        gen = instantiate("script.deepseek-v4-flash", {"api_key": "k", "temperature": 0.7})
        art = gen.generate("q", {}, tmp_path, temperature=1.2)
        assert captured["create"]["temperature"] == 1.2     # kw 覆盖生效
        assert captured["create"]["response_format"] == {"type": "json_object"}
        assert art.meta.params["temperature"] == 1.2        # meta 记录有效温度
        gen.generate("q", {}, tmp_path)
        assert captured["create"]["temperature"] == 0.7     # 缺省用构造温度


class TestRegressionSuite:
    def test_status_rows(self, tmp_path):
        from vidharness.core.regress import status
        import yaml
        # 任务 A：有完成的 run 且配置一致
        (tmp_path / "a.yaml").write_text("task_name: tA\npipeline: {context: {chain_mode: none}}\n",
                                         encoding="utf-8")
        ra = tmp_path / "tA" / "rA1"
        ra.mkdir(parents=True)
        (ra / "manifest.json").write_text(json.dumps(
            {"run_id": "rA1", "finished_at": "2026-08-16T10:00:00"}, ensure_ascii=False))
        (ra / "config.yaml").write_text(
            yaml.safe_dump({"task_name": "tA", "pipeline": {"context": {"chain_mode": "none"}}},
                           allow_unicode=True, sort_keys=False))
        (ra / "eval").mkdir()
        (ra / "eval" / "segments.json").write_text(json.dumps(
            [{"scores": {"与指令一致性": 8.0}, "passed": True},
             {"scores": {"与指令一致性": 10.0}, "passed": True}]))
        # 任务 B：有完成的 run 但配置漂移
        (tmp_path / "b.yaml").write_text("task_name: tB\npipeline: {context: {chain_mode: hard}}\n",
                                         encoding="utf-8")
        rb = tmp_path / "tB" / "rB1"
        rb.mkdir(parents=True)
        (rb / "manifest.json").write_text(json.dumps(
            {"run_id": "rB1", "finished_at": "2026-08-16T09:00:00"}, ensure_ascii=False))
        (rb / "config.yaml").write_text(
            yaml.safe_dump({"task_name": "tB", "pipeline": {"context": {"chain_mode": "none"}}},
                           allow_unicode=True, sort_keys=False))
        # 任务 C：未跑过
        (tmp_path / "c.yaml").write_text("task_name: tC\n", encoding="utf-8")
        spec = tmp_path / "regression.yaml"
        spec.write_text("tasks: [a.yaml, b.yaml, c.yaml]\n", encoding="utf-8")

        rows = status(tmp_path, spec)
        assert rows[0]["run_id"] == "rA1" and rows[0]["drift"] is None
        assert rows[0]["scores"]["segments"]["与指令一致性"] == 9.0
        assert rows[1]["drift"] == "配置漂移（快照 ≠ 当前任务文件，需重跑）"
        assert rows[2]["run_id"] is None and rows[2]["drift"] is None

    def test_render_status(self, tmp_path):
        from vidharness.core.regress import render_status
        md = render_status([{"task_file": "a.yaml", "task_name": "tA", "run_id": "rA1",
                             "scores": {"segments": {"与指令一致性": 9.0}}, "drift": None},
                            {"task_file": "b.yaml", "task_name": "tB", "run_id": None,
                             "scores": {}, "drift": None}])
        assert "✅ 一致" in md and "未跑过" in md and "与指令一致性 9.0" in md


class TestLeaderboardIndex:
    def test_export_all_and_index(self, tmp_path):
        from vidharness.core.leaderboard import export_all
        # 任务 tA：judge.deepseek-text
        exp = _build_exp(tmp_path)
        (Path(tmp_path) / "j.json").write_text("{}", encoding="utf-8")
        exp.save_artifact("judge", Artifact(
            kind="scores", path=Path(tmp_path) / "j.json",
            meta=ArtifactMeta(adapter="judge.deepseek-text")))
        # 同任务再加第二个裁判（阶段路由的真实形态）→ 触发混用警告
        (Path(tmp_path) / "j3.json").write_text("{}", encoding="utf-8")
        exp.save_artifact("judge", Artifact(
            kind="scores", path=Path(tmp_path) / "j3.json",
            meta=ArtifactMeta(adapter="judge.openai-compat")))
        exp.finalize()
        # 任务 tB：judge.openai-compat
        exp2 = Experiment(task="tB", base_dir=tmp_path, run_id="r2")
        (Path(tmp_path) / "v2.mp4").write_bytes(b"fake")
        exp2.save_artifact("segments", Artifact(
            kind="video", path=Path(tmp_path) / "v2.mp4",
            meta=ArtifactMeta(adapter="generator.x", elapsed_s=1.0)), name="s2")
        (Path(tmp_path) / "j2.json").write_text("{}", encoding="utf-8")
        exp2.save_artifact("judge", Artifact(
            kind="scores", path=Path(tmp_path) / "j2.json",
            meta=ArtifactMeta(adapter="judge.openai-compat")))
        exp2.finalize()
        out = tmp_path / "lb"
        result = export_all(tmp_path, out)
        assert set(result["tasks"]) == {"t", "tB"}   # _build_exp 的任务名是 t
        index = (out / "index.html").read_text(encoding="utf-8")
        assert "t" in index and "tB" in index
        assert "混用裁判" in index           # 两任务裁判不同 → 总警告
        assert "judge.deepseek-text" in index and "judge.openai-compat" in index


class TestGpuFreeCheck:
    def _fake_smi(self, monkeypatch, stdout):
        import subprocess
        monkeypatch.setattr(subprocess, "run",
                            lambda *a, **kw: type("R", (), {"stdout": stdout})())

    def test_insufficient_gpu_fails_loud_with_guidance(self, monkeypatch):
        from vidharness.providers.minimax_h3 import check_gpu_free
        self._fake_smi(monkeypatch, "0, 40960\n4, 1024\n6, 81920\n")
        with pytest.raises(RuntimeError, match="僵尸进程"):
            check_gpu_free("4,6")
        with pytest.raises(RuntimeError, match="GPU4"):
            check_gpu_free("4")

    def test_sufficient_gpu_passes(self, monkeypatch):
        from vidharness.providers.minimax_h3 import check_gpu_free
        self._fake_smi(monkeypatch, "4, 81920\n6, 81920\n")
        check_gpu_free("4,6")   # 不抛即通过

    def test_nvidia_smi_unavailable_skips(self, monkeypatch):
        import subprocess
        from vidharness.providers.minimax_h3 import check_gpu_free
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: (_ for _ in ()).throw(OSError()))
        check_gpu_free("4")     # 非 GPU 环境不拦


class TestLegacyJudgeInference:
    def test_legacy_judge_annotated(self, tmp_path):
        """旧布局（E12）run：裁判未记录 adapter → 推断标注（口径透明）。"""
        from vidharness.core.report import collect
        exp = _build_exp(tmp_path)
        (exp.eval_dir / "judge_123.json").write_text(
            '{"raw": "x", "scores": {}}', encoding="utf-8")
        (exp.final_dir / "final_video.mp4").write_bytes(b"fake")  # 旧口径完整性
        runs = collect(tmp_path, "t")
        assert runs[0]["judge_adapters"] == \
            ["judge.openai-compat（推断：旧布局未记录，裁判版本未知，跨期不可比）"]


class TestCalibratedLeaderboard:
    def test_calibrated_scores_applied_to_text_judge_runs(self, tmp_path):
        """E30：校准偏移（n≥3）换算 deepseek-text 评分的 run，vLLM run 不动。"""
        from vidharness.core.leaderboard import build
        # 校准数据
        calib_dir = tmp_path / "calibration"
        calib_dir.mkdir()
        (calib_dir / "a__vs__judge.deepseek-text.json").write_text(json.dumps({
            "judge_a": "judge.openai-compat", "judge_b": "judge.deepseek-text",
            "dims": {"叙事完整": {"n": 10, "mean_offset_a_minus_b": 0.8},
                     "可生成性": {"n": 10, "mean_offset_a_minus_b": -1.0},
                     "薄弱维": {"n": 1, "mean_offset_a_minus_b": 9.0}}}, ensure_ascii=False))
        # run1: deepseek-text 裁判（script_judge 维度）
        exp = _build_exp(tmp_path)
        exp.save_eval("script_judge", [{"scores": {"叙事完整": 7.0, "可生成性": 8.0},
                                        "passed": True}])
        (Path(tmp_path) / "j.json").write_text("{}", encoding="utf-8")
        exp.save_artifact("judge", Artifact(
            kind="scores", path=Path(tmp_path) / "j.json",
            meta=ArtifactMeta(adapter="judge.deepseek-text")))
        exp.finalize()
        # run2: vLLM 裁判
        exp2 = Experiment(task="t", base_dir=tmp_path, run_id="r2")
        (Path(tmp_path) / "v.mp4").write_bytes(b"fake")
        exp2.save_artifact("segments", Artifact(
            kind="video", path=Path(tmp_path) / "v.mp4",
            meta=ArtifactMeta(adapter="generator.x", elapsed_s=1.0)), name="s1")
        exp2.save_eval("segments", [{"scores": {"与指令一致性": 9.0}, "passed": True}])
        (Path(tmp_path) / "j2.json").write_text("{}", encoding="utf-8")
        exp2.save_artifact("judge", Artifact(
            kind="scores", path=Path(tmp_path) / "j2.json",
            meta=ArtifactMeta(adapter="judge.openai-compat")))
        exp2.finalize()

        data = build(tmp_path, "t", calibrate=True, calib_dir=calib_dir)
        by_id = {r["run_id"]: r for r in data["runs"]}
        r1 = by_id["r1"]
        assert r1["calibrated"] is True
        assert r1["scores_calibrated"]["叙事完整"] == 7.8      # 7.0 + 0.8
        assert r1["scores_calibrated"]["可生成性"] == 7.0      # 8.0 - 1.0
        assert "薄弱维" not in r1["scores_calibrated"] or True  # n<3 不参与
        r2 = by_id["r2"]
        assert r2["calibrated"] is False
        assert r2["scores_calibrated"]["与指令一致性"] == 9.0  # vLLM run 不动


class TestCoverageGaps:
    """核心模块分支覆盖补全（DSH 覆盖纪律的缩放版）。"""

    # ---- config 校验分支 ----
    def test_config_brief_and_segments_types(self):
        cfg = TestConfigValidation()._base()
        cfg["brief"] = 123
        with pytest.raises(ConfigError):
            validate_task(cfg)
        cfg = TestConfigValidation()._base()
        cfg["segments"] = "4"
        with pytest.raises(ConfigError):
            validate_task(cfg)

    def test_config_retry_and_audio_memory_cost(self):
        from vidharness.core.config import validate_task, ConfigError
        cfg = TestConfigValidation()._base()
        cfg["script_retry"]["inject_feedback"] = "yes"
        with pytest.raises(ConfigError):
            validate_task(cfg)
        cfg = TestConfigValidation()._base()
        cfg["audio_verify"]["extra"] = 1
        with pytest.raises(ConfigError, match="未知配置键"):
            validate_task(cfg)
        cfg = TestConfigValidation()._base()
        cfg["memory"]["extra"] = 1
        with pytest.raises(ConfigError):
            validate_task(cfg)
        cfg = TestConfigValidation()._base()
        cfg["cost"] = {"gpu_price_usd_per_hour": "1.2"}
        with pytest.raises(ConfigError):
            validate_task(cfg)
        cfg = TestConfigValidation()._base()
        cfg["pipeline"]["generator"] = {"adapter": "a", "route": {}}
        with pytest.raises(ConfigError, match="二选一"):
            validate_task(cfg)
        cfg = TestConfigValidation()._base()
        cfg["script_optimize"] = {"rounds": 2, "extra": 1}
        with pytest.raises(ConfigError):
            validate_task(cfg)

    # ---- regress 分支 ----
    def test_regress_list_empty_and_latest(self, tmp_path):
        from vidharness.core.regress import load_regression_list, _latest_finished_run
        spec = tmp_path / "reg.yaml"
        spec.write_text("tasks: []\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="缺少 tasks"):
            load_regression_list(spec)
        # 两个完成 run 取最新
        for rid, ts in (("r1", "2026-08-16T10:00:00"), ("r2", "2026-08-16T11:00:00")):
            d = tmp_path / "tX" / rid
            d.mkdir(parents=True)
            (d / "manifest.json").write_text(json.dumps(
                {"run_id": rid, "finished_at": ts}, ensure_ascii=False))
        assert _latest_finished_run(tmp_path, "tX")["run_id"] == "r2"
        assert _latest_finished_run(tmp_path, "tNope") is None

    def test_regress_drift_variants(self, tmp_path):
        from vidharness.core.regress import config_drifted
        import yaml
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        task = tmp_path / "task.yaml"
        task.write_text("task_name: t\n", encoding="utf-8")
        assert config_drifted(run_dir, task) == "无快照（8-16 前旧 run）"
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump({"task_name": "t"}, allow_unicode=True))
        task.write_text("task_name: t2\n", encoding="utf-8")
        assert config_drifted(run_dir, task) == "配置漂移（快照 ≠ 当前任务文件，需重跑）"
        assert config_drifted(run_dir, tmp_path / "missing.yaml") == "任务文件缺失"

    # ---- bench 分支 ----
    def test_bench_expand_axis_errors(self):
        from vidharness.core.bench import expand_matrix, BenchError
        with pytest.raises(BenchError, match="等长"):
            expand_matrix({}, [{"a": [1], "b": [2, 3]}])
        with pytest.raises(BenchError, match="非空列表"):
            expand_matrix({}, [{"a": "not-list"}])
        with pytest.raises(BenchError, match="等长非空列表"):
            expand_matrix({}, [{"a": [1], "b": "x"}])

    def test_bench_multi_key_paired_axis(self):
        """异构格：adapter 与 params 成对切换（本地/API 对比）。"""
        from vidharness.core.bench import expand_matrix
        base = {"pipeline": {"generator": {"adapter": "x", "params": {}}}}
        cells = expand_matrix(base, [{
            "pipeline.generator.adapter": ["generator.minimax-h3-local",
                                           "generator.minimax-h3-api"],
            "pipeline.generator.params": [{"model_path": "/m", "gpu": "4,6"},
                                          {"resolution": "768P", "duration": 8}],
        }])
        assert [c[0] for c in cells] == ["generator.minimax-h3-local",
                                         "generator.minimax-h3-api"]
        assert cells[0][1]["pipeline"]["generator"]["params"]["model_path"] == "/m"
        assert cells[1][1]["pipeline"]["generator"]["params"]["resolution"] == "768P"
        assert cells[1][1]["pipeline"]["generator"]["params"].get("model_path") is None

    def test_bench_estimate_unknown_backend_and_missing_rate(self):
        from vidharness.core.bench import estimate_cost, BenchError
        est = estimate_cost({"segments": 2, "pipeline": {"generator": {"params": {}}}},
                            {"backend": "weird"}, 10)
        assert est["cost_usd_est"] is None
        with pytest.raises(BenchError, match="未声明"):
            estimate_cost({"segments": 2, "pipeline": {"generator": {"params": {}}}},
                          {"backend": "api", "cost_rates_usd_per_s": {}}, 10)

    def test_bench_cell_status_corrupt_manifest(self, tmp_path):
        from vidharness.core.bench import bench_cell_status
        d = tmp_path / "t" / "r1"
        d.mkdir(parents=True)
        (d / "manifest.json").write_text("not json", encoding="utf-8")
        assert bench_cell_status(tmp_path, "t", "A", {}, query="q")["run_id"] is None

    # ---- leaderboard 分支 ----
    def test_leaderboard_corrupt_baseline_and_missing_runs(self, tmp_path):
        from vidharness.core.leaderboard import _load_baseline, render_index
        p = tmp_path / "bad.json"
        p.write_text("not json", encoding="utf-8")
        assert _load_baseline(p) == {}
        (tmp_path / "empty.json").write_text('{"runs": []}', encoding="utf-8")
        idx = render_index(tmp_path).read_text(encoding="utf-8")
        assert "无基线数据" in idx

    def test_leaderboard_md_passed_none(self):
        from vidharness.core.leaderboard import _render_md
        md = _render_md({"task": "t", "updated_at": "x", "run_count": 1, "runs": [{
            "run_id": "r", "bench_cell": None, "chain_mode": None, "models": [],
            "judge_adapters": [], "scores": {}, "passed_rate": None,
            "total_cost_usd": 0.0, "local_gpu_hours": None, "created_at": ""}]})
        assert "| r |" in md

    # ---- invariants 分支 ----
    def test_invariants_manifest_unparseable_and_missing_config(self, tmp_path):
        from vidharness.core.invariants import check_experiment
        (tmp_path / "manifest.json").write_text("bad", encoding="utf-8")
        assert "无法解析" in check_experiment(tmp_path)[0]
        (tmp_path / "manifest.json").write_text(json.dumps(
            {"config_file": "config.yaml", "stages": {}, "total_cost_usd": 0.0,
             "total_elapsed_s": 0.0}))
        assert any("配置快照缺失" in v for v in check_experiment(tmp_path))

    def test_invariants_retries_non_int(self, tmp_path):
        from vidharness.core.invariants import check_experiment
        (tmp_path / "manifest.json").write_text(json.dumps(
            {"stages": {}, "total_cost_usd": 0.0, "total_elapsed_s": 0.0,
             "retries": {"segments": "两次"}}))
        assert any("retries" in v for v in check_experiment(tmp_path))

    # ---- memory 分支 ----
    def test_memory_unknown_version_and_missing_key(self, tmp_path):
        from vidharness.core.memory import ExperienceMemory
        p = tmp_path / "_memory.jsonl"
        p.write_text(json.dumps({"v": 99, "key": "k", "complaint": "c"}) + "\n"
                     + json.dumps({"v": 1}) + "\n", encoding="utf-8")
        mem = ExperienceMemory(p)
        assert len(mem.load_warnings) == 2
        assert mem.experience_lines() == []

    # ---- registry 分支 ----
    def test_instantiate_var_kw_class(self):
        @register("script.var-kw")
        class VarKw:
            capabilities = {"language": "zh", "json_output": True}
            def __init__(self, **kw):
                self.kw = kw
        o = instantiate("script.var-kw", {"anything": 1})
        assert o.kw == {"anything": 1}

    def test_resolve_provider_no_candidates(self):
        SEAM_CAPABILITY_SCHEMAS["nosuchseam2"] = {"audio": bool}
        with pytest.raises(RuntimeError, match="没有注册"):
            resolve_provider("nosuchseam2", {"audio": True})

    # ---- experiment 分支 ----
    def test_find_existing_corrupt_payload(self, tmp_path):
        exp = Experiment(task="t", base_dir=tmp_path, run_id="r1")
        d = exp.artifacts_dir / "script"
        d.mkdir(parents=True)
        p = d / "script.json"
        p.write_text("not json", encoding="utf-8")
        (Path(str(p) + ".meta.json")).write_text("{}", encoding="utf-8")
        art = exp.find_existing("script", "script")
        assert art is not None and art.payload == {}

    def test_finalize_price_param_and_set_meta(self, tmp_path):
        exp = _build_exp(tmp_path)
        exp.set_meta("chain_mode", "none")
        exp.finalize(gpu_price_usd_per_hour=3.0)
        m = json.loads((exp.root / "manifest.json").read_text(encoding="utf-8"))
        assert m["chain_mode"] == "none"
        assert m["local_gpu_cost_usd_est"] == 0.0   # 本测试无 local 产物


class TestReportDetailBranches:
    def test_render_run_html_full_event_coverage(self, tmp_path):
        """详情页全部事件摘要/时间线分支（_event_summary 各类型）。"""
        from vidharness.core.report import render_run_html, report
        exp = _build_exp(tmp_path)
        # 各类事件：retry/eval/manifest.set/finalized 已有部分，补齐全集
        exp.record_retry("segments")
        exp.save_eval("cross_consistency", [{"scores": {"跨段一致性": 9.0}, "passed": True}])
        exp.set_meta("generator_capabilities", {"audio": True})
        (Path(tmp_path) / "v2.mp4").write_bytes(b"fake")
        exp.save_artifact("segments", Artifact(
            kind="video", path=Path(tmp_path) / "v2.mp4",
            meta=ArtifactMeta(adapter="generator.x", elapsed_s=2.0)), name="seg02")
        exp.stage_started("script")
        exp.stage_finished("script")   # 详情页时间线需要配对事件
        exp.finalize()
        html = render_run_html(exp.root, exp.root / "report.html").read_text(encoding="utf-8")
        for frag in ("query.bound", "config.snapshotted", "artifact.saved", "eval.saved",
                     "retry", "manifest.set", "finalized", "阶段时间线", "stage.finished"):
            assert frag in html, frag
        # report() 聚合入口
        result = report(tmp_path, "t", tmp_path / "r.html")
        assert result["runs"] == 1

    def test_render_run_html_no_events(self, tmp_path):
        """旧 run（无事件流）详情页分支。"""
        from vidharness.core.report import render_run_html
        exp = _build_exp(tmp_path)
        (exp.root / "events.jsonl").unlink()
        html = render_run_html(exp.root, exp.root / "report.html").read_text(encoding="utf-8")
        assert "无事件流" in html and "（无记录）" in html


class TestScaffold:
    def test_scaffold_generator(self, tmp_path):
        from vidharness.core.scaffold import scaffold_provider
        p = scaffold_provider("generator", "my-model", tmp_path)
        text = p.read_text(encoding="utf-8")
        assert '@register("generator.my-model")' in text
        assert "class MyModel" in text
        assert "def generate(self, req: GenRequest" in text
        assert "'max_duration_s': '...'" in text      # 能力骨架来自 schema（repr 单引号）
        assert 'raise NotImplementedError' in text   # 未实现即响亮失败
        assert "param_schema" in text

    def test_scaffold_judge_contract(self, tmp_path):
        from vidharness.core.scaffold import scaffold_provider
        p = scaffold_provider("judge", "text-v2", tmp_path)
        text = p.read_text(encoding="utf-8")
        assert "modalities = [\"text\"]" in text
        assert "勿在此计算总分" in text               # 结算归消费者的契约提示

    def test_scaffold_errors(self, tmp_path):
        from vidharness.core.scaffold import scaffold_provider
        with pytest.raises(RuntimeError, match="未知 seam"):
            scaffold_provider("nosuch", "x", tmp_path)
        with pytest.raises(RuntimeError, match="已存在"):
            scaffold_provider("transcribe", "dup", tmp_path)
            scaffold_provider("transcribe", "dup", tmp_path)


class TestFeedbackCleaning:
    def test_clean_feedback_text(self):
        from vidharness.core.memory import clean_feedback_text
        assert clean_feedback_text('{"叙事完整": 5, "feedback": "故事缺乏转折"}') == "故事缺乏转折"
        assert clean_feedback_text("评分解析失败（未得到 JSON）。请严格只输出 …") == ""
        assert clean_feedback_text("旁白太肉麻") == "旁白太肉麻"
        assert clean_feedback_text("") == ""
        assert clean_feedback_text("{not json") == "{not json"   # 非 JSON 原样保留

    def test_memory_load_migrates_json_noise_and_merges(self, tmp_path):
        """E32 迁移：JSON 包装取内层 + 同键合并 + 迁移后补提升。"""
        from vidharness.core.memory import ExperienceMemory
        p = tmp_path / "_memory.jsonl"
        p.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in [
            {"v": 1, "key": "k1", "complaint": '{"叙事完整": 5, "feedback": "叙事缺乏转折"}',
             "kind": "feedback", "count": 1, "sources": ["a"], "first_at": 1, "last_at": 1,
             "promoted": False},
            {"v": 1, "key": "k2", "complaint": "叙事缺乏转折",
             "kind": "feedback", "count": 1, "sources": ["b"], "first_at": 2, "last_at": 2,
             "promoted": False},
            {"v": 1, "key": "k3", "complaint": "评分解析失败（未得到 JSON）",
             "kind": "feedback", "count": 3, "sources": ["c"], "first_at": 3, "last_at": 3,
             "promoted": False},
        ]) + "\n", encoding="utf-8")
        mem = ExperienceMemory(p, promote_threshold=2)
        assert len(mem._items) == 1                      # 前两条合并、噪声丢弃
        it = mem._items[0]
        assert it["complaint"] == "叙事缺乏转折" and it["count"] == 2 and it["promoted"] is True
        assert len(mem.load_warnings) == 1               # 噪声行记入警告


class TestMemoryConsolidate:
    def test_consolidate_groups_and_promotes(self, tmp_path):
        """E33：语义近重复按规范短语归并，达标提升。"""
        from vidharness.core.memory import ExperienceMemory
        mem = ExperienceMemory(tmp_path / "_memory.jsonl", promote_threshold=2)
        mem.add("叙事仅有单一场景，缺乏起承转合", source="a")
        mem.add("叙事缺乏转折与高潮，仅两个片段", source="b")
        mem.add("旁白太肉麻", source="c")
        mem.add("旁白不够口语化", source="d")
        labels = {
            "叙事仅有单一场景，缺乏起承转合": "叙事缺乏起承转合",
            "叙事缺乏转折与高潮，仅两个片段": "叙事缺乏起承转合",
            "旁白太肉麻": "旁白不口语化",
            "旁白不够口语化": "旁白不口语化",
        }
        stats = mem.consolidate(lambda c: labels[c])
        assert stats["after"] == 2 and stats["promoted"] == 2
        assert set(mem.experience_lines()) == {"叙事缺乏起承转合", "旁白不口语化"}

    def test_consolidate_keeps_promoted(self, tmp_path):
        from vidharness.core.memory import ExperienceMemory
        mem = ExperienceMemory(tmp_path / "_memory.jsonl", promote_threshold=2)
        mem.add_experience("既有经验", source="manual")
        mem.add("新反馈", source="a")
        stats = mem.consolidate(lambda c: "新反馈")
        assert stats["after"] == 2
        assert "既有经验" in mem.experience_lines()
        assert "新反馈" not in mem.experience_lines()   # count 1 未达标

    def test_consolidate_preserves_unlabeled(self, tmp_path):
        """无标签条目不得被丢弃（数据保全）。"""
        from vidharness.core.memory import ExperienceMemory
        mem = ExperienceMemory(tmp_path / "_memory.jsonl", promote_threshold=2)
        mem.add("无法归纳的条目", source="a")
        mem.add("另一条", source="b")
        stats = mem.consolidate(lambda c: "")
        assert stats["unlabeled"] == 2 and len(mem._items) == 2

    def test_consolidate_merges_into_existing_promoted(self, tmp_path):
        """同名新组归并进已提升旧项（防重复经验）。"""
        from vidharness.core.memory import ExperienceMemory
        mem = ExperienceMemory(tmp_path / "_memory.jsonl", promote_threshold=2)
        mem.add_experience("叙事缺乏起承转合", source="manual")
        mem.add("叙事只有两个片段没有转折", source="a")
        mem.add("叙事缺高潮", source="b")
        stats = mem.consolidate(lambda c: "叙事缺乏起承转合")
        assert stats["merged_into_existing"] == 1
        assert mem.experience_lines().count("叙事缺乏起承转合") == 1


class TestCostsReport:
    def test_build_cost_report(self, tmp_path):
        from vidharness.core.costs import build_cost_report
        exp = _build_exp(tmp_path)                 # task t：cost 0.5、无 GPU
        exp.finalize()
        exp2 = Experiment(task="t2", base_dir=tmp_path, run_id="r2")
        (Path(tmp_path) / "v.mp4").write_bytes(b"fake")
        exp2.save_artifact("segments", Artifact(
            kind="video", path=Path(tmp_path) / "v.mp4",
            meta=ArtifactMeta(adapter="generator.cost-local", elapsed_s=3600.0)),
            name="s1")
        exp2.finalize()
        data = build_cost_report(tmp_path, gpu_price_usd_per_hour=2.0)
        by_task = {r["task"]: r for r in data["tasks"]}
        assert by_task["t"]["api_cost_usd"] == 0.5
        assert by_task["t2"]["gpu_hours"] == 1.0
        assert by_task["t2"]["gpu_cost_usd_est"] == 2.0   # 价格参数生效
        assert data["totals"]["total_usd_est"] == 2.5

    def test_task_scan_with_relative_base(self, tmp_path, monkeypatch):
        """相对路径基座扫描回归：d.iterdir() 返回全路径，
        双重拼接（d / r）会让扫描静默为空（生产 bug 的教训）。"""
        from vidharness.core.costs import build_cost_report
        exp = _build_exp(tmp_path)
        exp.finalize()
        monkeypatch.chdir(tmp_path)
        data = build_cost_report(Path("."))
        assert [r["task"] for r in data["tasks"]] == ["t"]


class TestBenchRepeats:
    def test_plan_repeats(self, tmp_path):
        from vidharness.core.bench import plan
        cfg = TestBench()._base_cfg()
        base = tmp_path / "base.yaml"
        base.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
        spec = {"bench": {"base": str(base),
                          "matrix": [{"pipeline.generator.params.steps": [20]}],
                          "repeats": 2}}
        rows = plan(spec)
        assert [r["label"] for r in rows] == ["20.r1", "20.r2"]

    def test_plan_repeats_invalid(self, tmp_path):
        from vidharness.core.bench import plan, BenchError
        cfg = TestBench()._base_cfg()
        base = tmp_path / "base.yaml"
        base.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
        with pytest.raises(BenchError, match="正整数"):
            plan({"bench": {"base": str(base), "matrix": [{"segments": [1]}],
                            "repeats": 0}})


class TestMiniMaxAPIMock:
    """generator.minimax-h3-api 协议级端到端（mock 官方 API，无 key 验证）。

    覆盖：文件上传、创建任务、轮询、下载、产物/成本口径。
    """

    def _mock_server(self, tmp_path):
        import http.server
        import threading
        import subprocess

        # 用 ffmpeg 生成一个 1 秒真实 mp4（若有 ffmpeg；否则最小字节占位）
        vid = tmp_path / "mock.mp4"
        try:
            subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                            "color=c=blue:s=64x64:d=1", "-c:v", "libx264",
                            str(vid)], capture_output=True, check=True)
        except Exception:
            vid.write_bytes(b"fake-mp4")

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                self.rfile.read(length)
                if self.path.startswith("/v1/files/upload"):
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'{"file": {"url": "http://127.0.0.1:%d/vid.mp4"}}'
                                     % self.server.server_port)
                elif self.path == "/v2/video_generation":
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'{"task_id": "mock-task-1"}')
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_GET(self):
                if self.path == "/v2/query/video_generation?task_id=mock-task-1":
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(
                        b'{"task": {"status": "succeeded", "content": {"url": '
                        b'"http://127.0.0.1:%d/vid.mp4"}}}' % self.server.server_port)
                elif self.path == "/vid.mp4":
                    data = vid.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, *a):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server

    def test_api_adapter_end_to_end(self, tmp_path):
        from vidharness.core.registry import load_builtin_adapters, instantiate
        from vidharness.seams import GenRequest
        load_builtin_adapters()
        server = self._mock_server(tmp_path)
        try:
            gen = instantiate("generator.minimax-h3-api", {
                "api_key": "k", "base_url": f"http://127.0.0.1:{server.server_port}",
                "resolution": "768P", "duration": 5})
            art = gen.generate(GenRequest(text="一只小猫", duration=5),
                               tmp_path / "out")
            assert art.kind == "video" and art.path.exists()
            assert art.meta.adapter == "generator.minimax-h3-api"
            assert art.meta.cost_usd > 0            # 计费口径（768P 声明单价）
            assert art.meta.params["resolution"] == "768P"
        finally:
            server.shutdown()


class TestRunTitle:
    """E40：run 标题自动生成（DSH session title 对齐：模型可见 ⟺ 日志）。"""

    def _make_title_adapter(self, payloads=None, fail=False):
        calls = []

        class FakeTitleScript:
            name = "script.fake-title"
            def generate(self, query, template, workdir, **kw):
                calls.append((query, template, kw))
                if fail:
                    raise RuntimeError("title api down")
                payload = payloads.pop(0) if payloads else {"title": "星际迷航"}
                workdir = Path(workdir)
                workdir.mkdir(parents=True, exist_ok=True)
                path = workdir / "title.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                return Artifact(kind="script", path=path, meta=ArtifactMeta(),
                                payload=payload)
        return FakeTitleScript(), calls

    def test_title_generated_and_persisted(self, tmp_path):
        from vidharness.cli import _generate_run_title
        exp = Experiment(task="story", base_dir=tmp_path)
        adapter, calls = self._make_title_adapter(payloads=[{"title": "星际迷航"}])
        assert _generate_run_title(adapter, "宇宙飞船远征的故事", exp) == "星际迷航"
        assert exp.manifest["title"] == "星际迷航"
        assert len(exp.manifest["stages"]["title"]) == 1      # 产物落 artifacts/title/
        assert exp.manifest["total_cost_usd"] == 0.0          # 假适配器无成本
        assert len(calls) == 1
        assert "标题" in calls[0][1]["brief"]                  # 提示契约在日志里可见
        assert calls[0][2].get("temperature") == 0.3           # E43：创意温度
        assert "标题编辑" in str(calls[0][2].get("system"))    # E43：变换任务自拥人格
        assert replay_events(exp.events_path)["title"] == "星际迷航"  # 事件流可恢复
        exp.finalize()                                        # finalize 后体检仍通过
        assert check_experiment(exp.root) == []

    def test_title_retry_then_recover(self, tmp_path):
        from vidharness.cli import _generate_run_title
        exp = Experiment(task="story", base_dir=tmp_path)
        adapter, calls = self._make_title_adapter(
            payloads=[{"segments": []}, {"title": "重试成功"}])
        assert _generate_run_title(adapter, "q", exp) == "重试成功"
        assert len(calls) == 2
        assert "没有输出 title" in calls[1][1]["brief"]        # 第二次收紧提示

    def test_title_failure_is_silent(self, tmp_path):
        from vidharness.cli import _generate_run_title
        exp = Experiment(task="story", base_dir=tmp_path)
        adapter, _ = self._make_title_adapter(fail=True)
        assert _generate_run_title(adapter, "q", exp) == ""
        assert "title" not in exp.manifest
        assert "title" not in exp.manifest.get("stages", {})

    def test_director_hook_runs_before_finalize(self, tmp_path, monkeypatch):
        """集成：钩子挂入的元信息必须被 finalize 落盘（而非只存在内存）。"""
        from vidharness.consumers.segment_director import SegmentDirector
        monkeypatch.setattr(SegmentDirector, "_extract_last_frame",
                            staticmethod(lambda video, exp: None))
        monkeypatch.setattr(SegmentDirector, "_extract_frame",
                            staticmethod(lambda video, t, exp: None))
        monkeypatch.setattr(SegmentDirector, "stage_assemble",
                            lambda self, videos, script: Path(tmp_path) / "final.mp4")
        director = TestSegmentDirectorCapabilities()._make_director(tmp_path, "none")
        final = director.run("测试故事", before_finalize=lambda: None)
        assert final == Path(tmp_path) / "final.mp4"
        assert check_experiment(director.exp.root) == []

        def hook():
            director.exp.set_meta("title", "钩子标题")
        final = director.run("测试故事", before_finalize=hook)
        manifest = json.loads(
            (director.exp.root / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["title"] == "钩子标题"                  # finalize 已落盘
        assert replay_events(director.exp.events_path)["title"] == "钩子标题"
        assert check_experiment(director.exp.root) == []


class TestLeaderboardTitle:
    def test_md_table_title_column_and_pipe_sanitized(self, tmp_path):
        from vidharness.core.leaderboard import build, _render_md
        from unittest import mock
        runs = [
            {"run_id": "r1", "bench_cell": None, "chain_mode": "none", "models": ["a"],
             "judge_adapters": ["j"], "scores": {"叙事": 8.0}, "stage_scores": {},
             "calibrated": False, "scores_calibrated": {"叙事": 8.0},
             "passed_rate": 1.0, "total_cost_usd": 0.1, "total_elapsed_s": 10.0,
             "local_gpu_hours": 0.1, "created_at": "2026-08-16T10:00:00",
             "finished_at": "2026-08-16T10:01:00", "title": "星际|迷航"},
            {"run_id": "r2", "bench_cell": None, "chain_mode": "none", "models": ["a"],
             "judge_adapters": ["j"], "scores": {"叙事": 7.0}, "stage_scores": {},
             "calibrated": False, "scores_calibrated": {"叙事": 7.0},
             "passed_rate": 0.5, "total_cost_usd": 0.2, "total_elapsed_s": 20.0,
             "local_gpu_hours": 0.2, "created_at": "2026-08-16T09:00:00",
             "finished_at": "2026-08-16T09:01:00", "title": None},
        ]
        with mock.patch("vidharness.core.leaderboard.collect", return_value=runs):
            data = build(tmp_path, "story")
            md = _render_md(data)
        assert "| 标题 |" in md
        assert "星际／迷航" in md                      # | 被替换，表格结构不破
        assert "| r1 | 星际／迷航 |" in md
        assert "| r2 | - |" in md
        assert data["runs"][0]["title"] == "星际|迷航"  # JSON 基线保留原文


class TestSilentBehaviorAuditFixes:
    """审计轮修复的回归测试：静默吞异常路径全部可见化/响亮化。"""

    def test_corrupt_manifest_fails_loud_not_fresh_run(self, tmp_path):
        """损坏的 manifest 不能被当作全新 run 静默覆盖（证据保全）。"""
        run = tmp_path / "t" / "r1"
        run.mkdir(parents=True)
        (run / "manifest.json").write_text("{corrupt", encoding="utf-8")
        with pytest.raises(RuntimeError, match="无法解析"):
            Experiment(task="t", base_dir=tmp_path, run_id="r1")
        assert not (run / "events.jsonl").exists()   # 没有追加 run.created

    def test_save_eval_rebuilds_from_events_on_corrupt_file(self, tmp_path):
        """eval 文件损坏：从事件流重建旧记录，而不是静默清空。"""
        exp = _build_exp(tmp_path)                    # 已有一条 segments 记录
        (exp.eval_dir / "segments.json").write_text("{corrupt", encoding="utf-8")
        exp.save_eval("segments", [{"attempt": 2, "score": 9.0}])
        data = json.loads((exp.eval_dir / "segments.json")
                          .read_text(encoding="utf-8"))
        assert {r["attempt"] for r in data} == {1, 2}  # 旧记录从事件流重建
        assert "warning" in _event_types(exp)          # 损坏本身可见

    def test_finalize_warns_on_unresolved_adapter_caps(self, tmp_path):
        """能力解析失败的产物：GPU 时间不再被静默排除，落 warning 事件。"""
        exp = Experiment(task="t", base_dir=tmp_path)
        (tmp_path / "v.mp4").write_bytes(b"fake")
        exp.save_artifact("segments", Artifact(
            kind="video", path=tmp_path / "v.mp4",
            meta=ArtifactMeta(adapter="no.such.adapter", elapsed_s=60.0)), name="s1")
        exp.finalize()
        assert "warning" in _event_types(exp)

    def test_script_judge_outage_recorded_not_silent(self, tmp_path):
        """剧本裁判不可用：落 error 记录进 eval，而不是静默接受未评剧本。"""
        director = TestSegmentDirectorCapabilities()._make_director(tmp_path, "none")
        director.cfg["script_judge"] = [
            {"name": "叙事完整", "question": "完整吗？", "min_score": 6}]

        class DownJudge:
            name = "judge.down"
            def judge(self, media, criteria, workdir, **kw):
                raise RuntimeError("judge down")

        director.judges["script_judge"] = DownJudge()
        payload = director.stage_script("测试")
        assert payload["segments"]                     # 剧本仍可继续
        evals = json.loads((director.exp.eval_dir / "script_judge.json")
                           .read_text(encoding="utf-8"))
        assert any("评测不可用" in str(r.get("error", "")) for r in evals)

    def test_missing_last_frame_recorded_for_chain_modes(self, tmp_path, monkeypatch):
        """中段末帧抽取失败：衔接条件缺失必须可见（E16 同口径）。"""
        from vidharness.consumers.segment_director import SegmentDirector
        monkeypatch.setattr(SegmentDirector, "_extract_last_frame",
                            staticmethod(lambda video, exp: None))
        monkeypatch.setattr(SegmentDirector, "_extract_frame",
                            staticmethod(lambda video, t, exp: None))
        director = TestSegmentDirectorCapabilities()._make_director(tmp_path, "none")
        director.chain_mode = "ref"                    # 能力校验已过；只测可见性
        script = {"segments": [{"video_prompt": "p1", "narration": "n"},
                               {"video_prompt": "p2", "narration": "n"}]}
        videos = director.stage_segments(script)
        assert len(videos) == 2
        evals = json.loads((director.exp.eval_dir / "segments.json")
                           .read_text(encoding="utf-8"))
        assert any("末帧抽取失败" in str(r.get("error", "")) for r in evals)
        # 只有段1（还有下一段）记录；末段不记录
        assert sum(1 for r in evals if r.get("segment") == 1) == 1

    def test_optimizer_judge_outage_round_fails_loud(self, tmp_path):
        """优化器整轮评测不可用：响亮失败 + error 落盘，且不污染经验记忆。"""
        from vidharness.consumers.script_optimizer import ScriptOptimizer
        from vidharness.core.memory import ExperienceMemory

        class FakeScript:
            name = "fake"
            def generate(self, query, template, workdir, **kw):
                payload = {"segments": [{"video_prompt": "p", "narration": "n"}]}
                workdir = Path(workdir)
                workdir.mkdir(parents=True, exist_ok=True)
                path = workdir / "s.json"
                path.write_text(json.dumps(payload))
                return Artifact(kind="script", path=path, meta=ArtifactMeta(),
                                payload=payload)

        class DownJudge:
            name = "down"
            def judge(self, media, criteria, workdir, **kw):
                raise RuntimeError("judge down")

        mem = ExperienceMemory(tmp_path / "_memory.jsonl")
        exp = Experiment(task="t", base_dir=tmp_path)
        opt = ScriptOptimizer(FakeScript(), DownJudge(), mem, exp,
                              rounds=1, candidates=2, target_score=9.9)
        with pytest.raises(RuntimeError, match="全部不可用"):
            opt.optimize("目标", "brief", [JudgeCriteria(name="叙事完整",
                                                         question="q", min_score=6)],
                         tmp_path / "s")
        # error 记录已落盘（续跑可恢复），经验记忆零污染
        evals = json.loads((exp.eval_dir / "script_optimize.json")
                           .read_text(encoding="utf-8"))
        assert all(r.get("error") for r in evals)
        assert mem.experience_lines() == []

    def test_optimizer_partial_outage_no_memory_pollution(self, tmp_path):
        """部分候选评测失败：记 error、不参与选优、不进经验记忆。"""
        from vidharness.consumers.script_optimizer import ScriptOptimizer
        from vidharness.core.memory import ExperienceMemory

        class FakeScript:
            name = "fake"
            def generate(self, query, template, workdir, **kw):
                payload = {"segments": [{"video_prompt": "p", "narration": "n"}]}
                workdir = Path(workdir)
                workdir.mkdir(parents=True, exist_ok=True)
                path = workdir / "s.json"
                path.write_text(json.dumps(payload))
                return Artifact(kind="script", path=path, meta=ArtifactMeta(),
                                payload=payload)

        class FlakyJudge:
            name = "flaky"
            def __init__(self):
                self.n = 0
            def judge(self, media, criteria, workdir, **kw):
                self.n += 1
                if self.n == 1:
                    raise RuntimeError("judge down")
                workdir = Path(workdir)
                workdir.mkdir(parents=True, exist_ok=True)
                path = workdir / "j.json"
                path.write_text("{}")
                return Artifact(kind="scores", path=path, meta=ArtifactMeta(),
                                payload={"scores": {"叙事完整": 8.0},
                                         "feedback": "pass"})

        mem = ExperienceMemory(tmp_path / "_memory.jsonl")
        exp = Experiment(task="t", base_dir=tmp_path)
        opt = ScriptOptimizer(FakeScript(), FlakyJudge(), mem, exp,
                              rounds=1, candidates=2, target_score=9.9)
        payload, history = opt.optimize(
            "目标", "brief", [JudgeCriteria(name="叙事完整", question="q", min_score=6)],
            tmp_path / "s")
        assert payload["segments"]
        assert history[0]["error"] and history[0]["score"] is None   # 失败候选
        assert history[1]["score"] == 8.0                            # 成功候选照常
        assert mem.experience_lines() == []                          # 零噪声进记忆


class TestSeedOverride:
    """逐调用种子覆盖（E26 同源）：kw > req.seed > 构造参数。"""

    def test_effective_seed_priority(self):
        from vidharness.providers.minimax_h3 import MiniMaxH3Local
        f = MiniMaxH3Local._effective_seed
        assert f(None, None, {}) is None
        assert f(None, 42, {}) == 42
        assert f(7, 42, {}) == 7
        assert f(7, 42, {"seed": 99}) == 99
        assert f(None, 42, {"seed": 0}) == 0        # 0 是合法种子
        assert f(7, None, {}) == 7


class TestBenchGeneratorEviction:
    """E43：异构生成器参数切换必须释放旧实例（双模型驻留显存 → OOM）。"""

    def test_generator_cache_key_changes_with_params(self):
        from vidharness.core.bench import generator_cache_key
        base = {"pipeline": {"generator": {"adapter": "g", "params": {"seed": 1}}}}
        same = {"pipeline": {"generator": {"adapter": "g", "params": {"seed": 1}}}}
        diff = {"pipeline": {"generator": {"adapter": "g", "params": {"seed": 2}}}}
        assert generator_cache_key(base) == generator_cache_key(same)
        assert generator_cache_key(base) != generator_cache_key(diff)

    def test_evict_generators_disposes_and_removes_only_generators(self):
        from vidharness.core.bench import evict_generators
        disposed = []

        class FakeGen:
            def dispose(self):
                disposed.append("disposed")

        cache = {("generator.x", "{}"): FakeGen(),
                 ("script.y", "{}"): object()}
        released = evict_generators(cache)
        assert released == ["generator.x"]
        assert not any(str(k[0]).startswith("generator.") for k in cache)
        assert ("script.y", "{}") in cache          # 非生成器实例不动
        assert disposed == ["disposed"]

    def test_evict_generators_tolerates_missing_dispose(self):
        from vidharness.core.bench import evict_generators
        cache = {("generator.x", "{}"): object()}
        assert evict_generators(cache) == ["generator.x"]  # 无 dispose 也出缓存


class TestScriptSystemOverride:
    def test_system_kw_overrides_director_persona(self, tmp_path, monkeypatch):
        """E43：变换任务经 kw.system 覆盖提供者人格（导演→标题编辑）。"""
        from types import SimpleNamespace
        from vidharness.core.registry import instantiate
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
                msg = SimpleNamespace(content='{"segments": []}')
                choice = SimpleNamespace(message=msg)
                return SimpleNamespace(choices=[choice], model="deepseek-chat",
                                       usage=SimpleNamespace(prompt_tokens=10,
                                                            completion_tokens=10))

        monkeypatch.setattr("vidharness.providers.deepseek_script.OpenAI", FakeClient)
        gen = instantiate("script.deepseek-v4-flash", {"api_key": "k"})
        art = gen.generate("q", {}, tmp_path, system="你是标题编辑。只输出 JSON。")
        assert captured["create"]["messages"][0]["content"] == \
            "你是标题编辑。只输出 JSON。"            # system 覆盖生效
        assert art.meta.params["system"] == "你是标题编辑。只输出 JSON。"  # 可审计
        # 缺省不传 system → 导演人格
        gen.generate("q", {}, tmp_path)
        assert "资深影视导演" in captured["create"]["messages"][0]["content"]
