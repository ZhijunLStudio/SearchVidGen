"""SegmentDirector —— 跨片段故事编排（模型替代不了的那层）。

单次生成模型上限 ~15s。一部 30-60s 的故事片 = 多次生成 + 上下文携带：
  1. 剧本规划（LLM）：把故事拆成分镜计划（每段指令 + 旁白）
  2. 逐段生成（MediaGenerator）：段 i 的生成请求携带
     - 角色/风格锚点（参考图，来自全片设定）
     - 上一段末帧作为本段首帧条件（首尾帧衔接 → 结构性连续性）
  3. 逐段评测（Judge）：画面-指令一致 / 质量缺陷 / 音画同步
  4. 跨段评测（Judge）：相邻段角色与场景延续性（跨调用一致性验证）
  5. 总装：FFmpeg 拼接 + 旁白字幕（旁白文本是剧本自带的，无需 ASR）
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.experiment import Experiment, Timer
from .judge_loop import run_with_judge, run_judge
from ..seams import GenRequest, JudgeCriteria, MediaGenerator, RetryPolicy
from ..core.registry import check_capabilities, instantiate, resolve


class SegmentDirector:
    def __init__(self, exp: Experiment, config: Dict[str, Any]):
        self.exp = exp
        self.cfg = config
        # 衔接策略：hard=首帧硬条件(fl2va) / ref=末帧作参考图(ref2va) / none=无衔接
        self.chain_mode = config.get("pipeline", {}).get("context", {}).get("chain_mode", "none")

        self.script_adapter = instantiate(
            config["pipeline"]["script"]["adapter"],
            config["pipeline"]["script"].get("params", {}), context="script")

        gen_cfg = config["pipeline"]["generator"]
        if "route" in gen_cfg:
            from ..core.registry import resolve_provider
            gen_name = resolve_provider("generator", gen_cfg["route"], context="generator")
        else:
            gen_name = gen_cfg["adapter"]
        self.generator: MediaGenerator = instantiate(
            gen_name, gen_cfg.get("params", {}), context="generator")

        # 能力校验（fail loud）：按衔接策略推导真实需求，而非硬编码。
        # 校验对象是实例（fallback 等合成提供者的能力是实例级并集，类上没有）。
        # hard 需要首尾帧条件；ref 需要参考图；none 无额外要求。
        required: Dict[str, Any] = {}
        if self.chain_mode == "hard":
            required["first_last_frame"] = True
        elif self.chain_mode == "ref":
            required["refs"] = 1
        if config.get("audio_verify"):
            required["audio"] = True
        caps = check_capabilities(self.generator, required, context="generator")
        exp.set_meta("generator_capabilities", caps)
        exp.set_meta("chain_mode", self.chain_mode)

        self.judge = instantiate(config["judge"]["adapter"],
                                 config["judge"].get("params", {}), context="judge")
        # 阶段级裁判路由（E16 待办①）：文本评测阶段（script_judge/optimize）
        # 可覆盖为 text-only 裁判（如 DeepSeek API），消除对 VLM 服务就绪的依赖；
        # 媒体评测阶段（segment/cross）默认用主裁判。
        self.judges: Dict[str, Any] = {}
        stages_cfg = (config.get("judge") or {}).get("stages") or {}
        for key in ("script_judge", "script_optimize", "segment_judge", "cross_judge"):
            block = stages_cfg.get(key)
            if block:
                self.judges[key] = instantiate(
                    block["adapter"], block.get("params", {}),
                    context=f"judge.stages.{key}")
            else:
                self.judges[key] = self.judge
        # 经验记忆：环境反馈在此积累，跨任务泛化（无领域模板）
        from ..core.memory import ExperienceMemory
        mem_cfg = config.get("memory", {})
        self.memory = ExperienceMemory(
            path=exp.base_dir / mem_cfg.get("path", "_memory.jsonl"),
            promote_threshold=int(mem_cfg.get("promote_threshold", 2)),
        )

    # ---- 1. 剧本（自主优化循环：多轮生成 → 裁判评分 → 反馈入记忆 → 择优进化）----
    def stage_script(self, query: str) -> Dict[str, Any]:
        # 断点续跑：剧本也要缓存（否则重跑的新剧本与已生成片段不对齐）
        existing = self.exp.find_existing("script", "script")
        if existing and existing.payload:
            print("   复用已有剧本（断点续跑）")
            return existing.payload
        brief = self.cfg.get("brief") or ""
        opt_cfg = self.cfg.get("script_optimize")
        if opt_cfg:
            from .script_optimizer import ScriptOptimizer
            opt = ScriptOptimizer(
                self.script_adapter, self.judges["script_judge"], self.memory, self.exp,
                rounds=int(opt_cfg.get("rounds", 2)),
                candidates=int(opt_cfg.get("candidates", 2)),
                target_score=float(opt_cfg.get("target_score", 7.5)),
                segments=int(self.cfg.get("segments", 4)),
            )
            print(f"   剧本自主优化（{opt.rounds}轮×{opt.candidates}候选，目标≥{opt.target_score}）")
            payload, history = opt.optimize(
                query, brief, self._criteria("script_judge"),
                self.exp.artifacts_dir / "script")
            best = max(history, key=lambda r: r.get("score", 0))
            # 保存最终选定剧本为正式产物
            import shutil
            art_path = self.exp.artifacts_dir / "script" / "script.json"
            art_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                encoding="utf-8")
            from ..seams import Artifact, ArtifactMeta
            final_art = Artifact(kind="script", path=art_path, meta=ArtifactMeta(
                adapter=self.script_adapter.name, params={"optimizer": best}))
            self.exp.save_artifact("script", final_art, name="script")
            return payload
        brief = self.cfg.get("brief") or ""
        crit = self._criteria("script_judge")
        retry = self._retry("script_retry")
        feedback = ""
        last = None
        import json as _json
        for attempt in range(1, retry.max_attempts + 1):
            template = {
                "brief": brief,
                "segments": int(self.cfg.get("segments", 4)),
                "experience": self.memory.experience_lines(),
            }
            with Timer():
                art = self.script_adapter.generate(
                    query=query, template=template,
                    workdir=self.exp.artifacts_dir / "script")
                self.exp.save_artifact("script", art, name="script")
            last = art
            if not crit:
                return art.payload
            # 文本评测：把剧本内容嵌入问题交给裁判（完整规格随协议传递，权重不丢失）
            from dataclasses import replace as _replace
            embedded = [_replace(c, question=f"{c.question}\n\n剧本内容：\n"
                                            f"{_json.dumps(art.payload, ensure_ascii=False)}")
                        for c in crit]
            try:
                verdict = run_judge(self.judges["script_judge"], [], embedded,
                                    self.exp.artifacts_dir / "judge", exp=self.exp)
                rec = {"attempt": attempt, "artifact": str(art.path), **verdict}
                self.exp.save_eval("script_judge", [rec])
                feedback = verdict.get("feedback", "")
                # 裁判反馈一律进记忆（环境信号）：通过时的改进建议与失败原因都记录，
                # 重复出现自动提升为跨任务经验
                if feedback and feedback.strip() and "pass" not in feedback[:4].lower():
                    kind = "feedback" if not verdict.get("passed") else "suggestion"
                    self.memory.add(feedback, source=f"{self.exp.run_id}/script", kind=kind)
                if verdict.get("passed"):
                    return art.payload
            except Exception as e:
                print(f"   ⚠️ 剧本评测不可用({type(e).__name__})，跳过")
                return art.payload
            if retry.inject_feedback and feedback:
                brief = f"{brief}\n上一稿的问题：{feedback}".strip()
        return (last.payload if last else {"segments": []})

    # ---- 2/3. 逐段生成 + 逐段评测 ----
    def stage_segments(self, script: Dict[str, Any]) -> List[Path]:
        plans = script.get("segments", [])
        if not plans:
            raise RuntimeError("剧本缺少 segments 字段")
        seg_videos: List[Path] = []
        last_frame: Optional[Path] = None
        anchor_refs = self._anchor_refs()
        chain_mode = self.chain_mode

        for i, plan in enumerate(plans):
            name = f"seg{i + 1:02d}"
            existing = self.exp.find_existing("segments", name)
            if existing:
                seg_videos.append(existing.path)
                last_frame = self._extract_last_frame(existing.path, self.exp)
                continue

            req = GenRequest(
                text=plan.get("video_prompt", plan.get("text", "")),
                refs=anchor_refs,
                duration=plan.get("duration", self._default_duration()),
                ratio=plan.get("ratio", self.cfg.get("pipeline", {}).get("generator", {}).get("params", {}).get("ratio", "16:9")),
            )
            if last_frame is not None:
                if chain_mode == "hard":
                    req.first_frame = last_frame     # 硬衔接：上一段末帧 → 本段首帧
                elif chain_mode == "ref":
                    req.refs = [last_frame] + anchor_refs   # 软衔接：末帧作为参考图
            art, history = run_with_judge(
                self.generator, self.judges["segment_judge"],
                self._criteria("segment_judge"), self._retry("segment_retry"),
                {"req": req},
                lambda a: [a.path],
                self.exp, "segments", name=name,
            )
            seg_videos.append(art.path)
            last_frame = self._extract_last_frame(art.path, self.exp)
        return seg_videos

    # ---- 4. 跨段一致性评测 ----
    def stage_cross_consistency(self, videos: List[Path], script: Dict[str, Any]) -> Dict[str, Any]:
        if len(videos) < 2:
            return {"checked": False}
        records = []
        cross_crit = self._criteria("cross_judge") or [JudgeCriteria(
            name="跨段一致性",
            question="这两帧分别是上一段结尾与下一段开头，请检查：人物外貌/服装是否一致、场景是否自然衔接。不一致请说明差异。")]
        for i in range(1, len(videos)):
            prev_last = self._extract_last_frame(videos[i - 1], self.exp)
            cur_first = self._extract_frame(videos[i], 0.0, self.exp)
            # 抽帧失败必须可见：记录错误而不是让裁判空评（fail-visible）
            if prev_last is None or cur_first is None:
                rec = {"segment_pair": [i, i + 1],
                       "error": f"抽帧失败: last={prev_last}, first={cur_first}"}
                records.append(rec)
                continue
            verdict = run_judge(self.judges["cross_judge"], [prev_last, cur_first],
                                cross_crit, self.exp.artifacts_dir / "judge",
                                exp=self.exp)
            rec = {"segment_pair": [i, i + 1], **verdict}
            records.append(rec)
        self.exp.save_eval("cross_consistency", records)
        return {"checked": True, "records": records}

    # ---- 5. 总装 ----
    def stage_assemble(self, videos: List[Path], script: Dict[str, Any]) -> Path:
        from .assemble import assemble_final
        plans = script.get("segments", [])
        narrations = [p.get("narration", "") for p in plans]
        final = assemble_final(videos=videos, audios=[], narrations=narrations,
                               out_dir=self.exp.final_dir, audio_in_video=True)
        return final

    def run(self, query: str) -> Path:
        def stage(name: str, fn):
            self.exp.stage_started(name)
            try:
                return fn()
            finally:
                self.exp.stage_finished(name)

        print(f"▶ [1/5] 剧本规划 ({self.script_adapter.name})")
        script = stage("script", lambda: self.stage_script(query))
        n = len(script.get("segments", []))
        print(f"   分镜段数: {n}")

        print(f"▶ [2/5] 逐段生成+评测 ({self.generator.name})")
        videos = stage("segments", lambda: self.stage_segments(script))

        print(f"▶ [3/5] 跨段一致性评测 ({self.judges['cross_judge'].name})")
        stage("cross_consistency", lambda: self.stage_cross_consistency(videos, script))

        print("▶ [4/5] 成片总装")
        final = stage("assemble", lambda: self.stage_assemble(videos, script))

        # 音频验证（可选）：转写原生音频，与旁白比对
        audio_cfg = self.cfg.get("audio_verify")
        if audio_cfg:
            print(f"▶ [5/5] 音频验证 ({audio_cfg['adapter']})")
            from .audio_verify import verify_film
            narrations = [p.get("narration", "") for p in script.get("segments", [])]
            stage("audio_verify", lambda: verify_film(
                videos, narrations, audio_cfg["adapter"], self.exp))

        self.exp.finalize(gpu_price_usd_per_hour=float(
            (self.cfg.get("cost") or {}).get("gpu_price_usd_per_hour", 1.2)))
        print(f"\n✅ 完成: {final}")
        print(f"   实验目录: {self.exp.root}")
        print(f"   总耗时: {self.exp.manifest['total_elapsed_s']:.0f}s | "
              f"成本估算: ${self.exp.manifest['total_cost_usd']:.4f}")
        return final

    # ---- 工具 ----
    def _anchor_refs(self) -> List[Path]:
        refs = self.cfg.get("pipeline", {}).get("context", {}).get("anchor_refs", [])
        return [Path(r) for r in refs if Path(r).exists()]

    def _default_duration(self) -> int:
        return int(self.generator.capabilities.get("max_duration_s", 10) * 0.6)

    def _criteria(self, key: str) -> List[JudgeCriteria]:
        out = []
        for c in (self.cfg.get(key) or []):
            out.append(JudgeCriteria(name=c["name"], question=c["question"],
                                     weight=float(c.get("weight", 1.0)),
                                     min_score=float(c.get("min_score", 6.0)),
                                     aliases=c.get("aliases")))
        return out

    def _retry(self, key: str) -> RetryPolicy:
        r = self.cfg.get(key) or {}
        return RetryPolicy(max_attempts=int(r.get("max_attempts", 2)),
                           inject_feedback=bool(r.get("inject_feedback", True)),
                           feedback_prefix=r.get("feedback_prefix", "请修正以下问题后重新生成："))

    @staticmethod
    def _extract_frame(video: Path, t: float, exp: Experiment) -> Optional[Path]:
        from .tools import require_ffmpeg
        require_ffmpeg()
        out = exp.artifacts_dir / "frames"
        out.mkdir(exist_ok=True)
        dst = out / f"{video.stem}_t{t:.2f}.jpg"
        if dst.exists():
            return dst
        import subprocess
        r = subprocess.run(["ffmpeg", "-y", "-ss", str(t), "-i", str(video),
                            "-frames:v", "1", str(dst)], capture_output=True)
        return dst if dst.exists() else None

    def _extract_last_frame(self, video: Path, exp: Experiment) -> Optional[Path]:
        from .tools import require_ffmpeg, require_ffprobe
        require_ffmpeg()
        require_ffprobe()
        out = exp.artifacts_dir / "frames"
        out.mkdir(exist_ok=True)
        dst = out / f"{video.stem}_last.jpg"
        if dst.exists():
            return dst
        import subprocess
        dur = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(video)], capture_output=True, text=True).stdout.strip()
        try:
            t = max(0.0, float(dur) - 0.5)
        except ValueError:
            t = 0.0
        r = subprocess.run(["ffmpeg", "-y", "-ss", str(t), "-i", str(video),
                            "-frames:v", "1", str(dst)], capture_output=True)
        return dst if dst.exists() else None
