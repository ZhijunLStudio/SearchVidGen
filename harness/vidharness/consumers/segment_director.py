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
from .judge_loop import run_with_judge
from ..seams import GenRequest, JudgeCriteria, MediaGenerator, RetryPolicy
from ..core.registry import check_capabilities, resolve


class SegmentDirector:
    def __init__(self, exp: Experiment, config: Dict[str, Any]):
        self.exp = exp
        self.cfg = config
        self.script_adapter = resolve(config["pipeline"]["script"]["adapter"])(
            **config["pipeline"]["script"].get("params", {}))
        gen_name = config["pipeline"]["generator"]["adapter"]
        self.generator: MediaGenerator = resolve(gen_name)(
            **config["pipeline"]["generator"].get("params", {}))
        # 能力校验（fail loud）：任务要求超出提供者能力直接报错
        required = {"first_last_frame": True}
        if config.get("audio_verify"):
            required["audio"] = True
        caps = check_capabilities(gen_name, required, context="generator")
        exp.manifest["generator_capabilities"] = caps
        self.judge = resolve(config["judge"]["adapter"])(**config["judge"].get("params", {}))
        # 经验记忆：环境反馈在此积累，跨任务泛化（无领域模板）
        from ..core.memory import ExperienceMemory
        mem_cfg = config.get("memory", {})
        self.memory = ExperienceMemory(
            path=exp.base_dir / mem_cfg.get("path", "_memory.jsonl"),
            promote_threshold=int(mem_cfg.get("promote_threshold", 2)),
        )

    # ---- 1. 剧本（生成 → 文本评测 → 反馈入记忆 → 重试）----
    def stage_script(self, query: str) -> Dict[str, Any]:
        # 断点续跑：剧本也要缓存（否则重跑的新剧本与已生成片段不对齐）
        existing = self.exp.find_existing("script", "script")
        if existing and existing.payload:
            print("   复用已有剧本（断点续跑）")
            return existing.payload
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
            # 文本评测：把剧本内容嵌入问题交给裁判
            q = {c.name: f"{c.question}\n\n剧本内容：\n{_json.dumps(art.payload, ensure_ascii=False)}"
                 for c in crit}
            try:
                verdict_art = self.judge.judge(media=[], criteria=q,
                                               workdir=self.exp.eval_dir)
                verdict = verdict_art.payload
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
        # 衔接策略：hard=首帧硬条件(fl2va) / ref=末帧作参考图(ref2va) / none=无衔接
        chain_mode = self.cfg.get("pipeline", {}).get("context", {}).get("chain_mode", "hard")

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
                self.generator, self.judge,
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
        crit_dict = {c.name: c.question for c in cross_crit}
        for i in range(1, len(videos)):
            prev_last = self._extract_last_frame(videos[i - 1], self.exp)
            cur_first = self._extract_frame(videos[i], 0.0, self.exp)
            art = self.judge.judge(
                media=[prev_last, cur_first],
                criteria=crit_dict,
                workdir=self.exp.eval_dir,
            )
            rec = {"segment_pair": [i, i + 1], **art.payload}
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
        print(f"▶ [1/5] 剧本规划 ({self.script_adapter.name})")
        script = self.stage_script(query)
        n = len(script.get("segments", []))
        print(f"   分镜段数: {n}")

        print(f"▶ [2/5] 逐段生成+评测 ({self.generator.name})")
        videos = self.stage_segments(script)

        print(f"▶ [3/5] 跨段一致性评测 ({self.judge.name})")
        self.stage_cross_consistency(videos, script)

        print("▶ [4/5] 成片总装")
        final = self.stage_assemble(videos, script)

        # 音频验证（可选）：转写原生音频，与旁白比对
        audio_cfg = self.cfg.get("audio_verify")
        if audio_cfg:
            print(f"▶ [5/5] 音频验证 ({audio_cfg['adapter']})")
            from .audio_verify import verify_film
            narrations = [p.get("narration", "") for p in script.get("segments", [])]
            verify_film(videos, narrations, audio_cfg["adapter"], self.exp)

        self.exp.finalize()
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
