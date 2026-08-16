"""MiniMax H3 适配器：全模态生成（文字+参考 → 视频+原生音频）。

两条后端（同一协议，能力声明不同）：
- 本地 diffusers（H3-Base，768p，开源权重，需 diffusers@minimax-h3 分支）
- 官方 API（完整 H3：Context-IR + 2K Regenerate，托管）

harness 面向未来：换下一个全模态模型 = 新适配器文件，核心零改动。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..seams import Artifact, ArtifactMeta, GenRequest
from ..core.registry import register

# 常用宽高比 → (宽, 高)，短边 768，边长取 32 倍数
# 官方定价大致区间（CN）：768P 约 ¥0.3/秒、2K 约 ¥0.8/秒；按 7.2 汇率近似 USD。
# 单一价格正源：capabilities 声明目录（规划预估）与 _estimate_cost（运行时计费）
# 都读这里——同一事实只在一处维护。
_MINIMAX_RATES_USD_PER_S = {
    "768P": round(0.3 / 7.2, 6),
    "2K": round(0.8 / 7.2, 6),
}

RATIO_CANVAS = {
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "1:1": (768, 768),
    "4:3": (1024, 768),
    "3:4": (768, 1024),
    "21:9": (1568, 672),
}


def split_dual_card_kwargs(variant: str, kwargs: Dict[str, Any]):
    """双卡两段式的参数拆分（按 diffusers 声明的子块输入契约，返回 (cond, rest)）。

    拆分依据是 get_workflow(variant) 各子块的 **inputs 声明**（权威契约）：
    - t2va：text_encoder 只声明 prompt；画布/帧数由生成侧 prepare_layout
      消费（2026-08-16 Bug#6：传给条件侧会被 ignored，画布静默回落 16:9）。
    - fl2va：before_encode 声明 image/last_image/height/width（条件侧）；
      keyframes 经 state 流入生成侧 vae_encoder → condition_latents。
      image 必须进条件侧，否则 vae_encoder 无 keyframes 可编码，
      生成侧 torch.cat 空列表崩溃（2026-08-16 真机回归暴露）。
    - ref2va：before_encode 声明 references/height/width/num_frames（E6）。
    """
    kwargs = dict(kwargs)
    cond = {"prompt": kwargs.pop("prompt")}
    if variant == "fl2va":
        for k in ("image", "last_image", "height", "width"):
            if k in kwargs:
                cond[k] = kwargs.pop(k)
    elif variant == "ref2va":
        for k in ("references", "height", "width"):
            if k in kwargs:
                cond[k] = kwargs.pop(k)
        if "num_frames" in kwargs:
            cond["num_frames"] = kwargs["num_frames"]
    return cond, kwargs


# ==========================================================================
# 本地 H3-Base（diffusers）
# ==========================================================================
def check_gpu_free(gpu_spec: str, min_free_gb: float = 40.0) -> None:
    """加载前 GPU 显存预检（E29 故障演练发现：kill -9 父进程会遗留僵尸
    子进程占住显存，续跑时在 torch 深处 OOM 且报错不可读）。

    对 gpu_spec（物理卡号，如 "4,6"）逐卡检查 nvidia-smi 的空闲显存，
    低于阈值即响亮失败并给出可操作指引。
    """
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
    except Exception:
        return   # nvidia-smi 不可用时不拦（非 GPU 环境/开发机）
    free: Dict[str, float] = {}
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2:
            try:
                free[parts[0]] = float(parts[1]) / 1024.0
            except ValueError:
                continue
    busy = []
    for idx in gpu_spec.split(","):
        idx = idx.strip()
        if idx in free and free[idx] < min_free_gb:
            busy.append(f"GPU{idx} 空闲 {free[idx]:.1f}GB < {min_free_gb}GB")
    if busy:
        raise RuntimeError(
            "目标 GPU 显存不足（" + "；".join(busy) + "）。"
            "可能有僵尸进程占住显存：pkill -f vidharness 清理后重试，"
            "或改用其他空闲卡（nvidia-smi 查看）。")


@register("generator.minimax-h3-local")
class MiniMaxH3Local:
    name = "generator.minimax-h3-local"
    capabilities = {
        "max_duration_s": 15, "audio": True, "refs": 9,
        "first_last_frame": True, "resolution": "768p", "backend": "local",
    }
    # 参数声明目录（配置平面：提供者拥有 params 的语义；instantiate 据此校验）
    param_schema = {
        "model_path": {"type": "path", "required": True, "help": "H3 权重目录（diffusers 子目录）"},
        "gpu": {"type": "str", "default": "6", "help": "CUDA_VISIBLE_DEVICES（如 '4,6' 双卡）"},
        "variant": {"type": "str", "default": "fl2va",
                    "choices": ["t2va", "fl2va", "ref2va"], "help": "生成工作流变体"},
        "num_frames": {"type": "int", "default": None, "help": "帧数（缺省按 duration×24）"},
        "seed": {"type": "int", "default": None, "help": "随机种子"},
        "guidance_scale": {"type": "float", "default": 4.5, "help": "引导系数"},
        "steps": {"type": "int", "default": None, "help": "去噪步数（缺省用管道默认）"},
    }

    def __init__(self, model_path: str, gpu: str = "6", variant: str = "fl2va",
                 num_frames: Optional[int] = None, seed: Optional[int] = None,
                 guidance_scale: float = 4.5, steps: Optional[int] = None):
        self.model_path = model_path
        self.gpu = gpu
        self.variant = variant.lower()          # fl2va | ref2va
        self.num_frames = num_frames
        self.seed = seed
        self.guidance_scale = guidance_scale
        self.steps = steps                      # 去噪步数（None=管道默认，实测约49步）
        self._pipe = None
        self.version = ""

    def _get_pipe(self):
        if self._pipe is None:
            import os
            check_gpu_free(self.gpu)   # 加载前显存预检（E29：僵尸进程占卡时报清晰指引）
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu   # 如 "4,6" 或 "2"
            import torch

            if self.variant == "ref2va":
                # 官方 int8 配方（参考模式单卡方案）：
                # transformer_ref/text_encoder 以 Int8WeightOnly 量化加载（显存减半），
                # transformer 块级流式 offload（use_stream），VAE 常驻 GPU。
                from diffusers import (MiniMaxH3ModularPipeline,
                                       MiniMaxH3Transformer3DModel, TorchAoConfig)
                from diffusers.hooks import apply_group_offloading
                from transformers import Qwen3VLForConditionalGeneration
                from transformers import TorchAoConfig as TransformersTorchAoConfig
                from torchao.quantization import Int8WeightOnlyConfig

                t_not_convert = [
                    "proj_in", "audio_proj_in", "context_embedder", "time_embedder",
                    "time_proj", "token_refiner", "norm_out", "proj_out", "audio_proj_out",
                ]
                te_not_convert = [
                    "model.visual", "model.language_model.embed_tokens",
                    "model.language_model.norm", "lm_head",
                ]
                pipe = MiniMaxH3ModularPipeline.from_pretrained(self.model_path)
                pipe.update_components(
                    transformer_ref=MiniMaxH3Transformer3DModel.from_pretrained(
                        self.model_path, subfolder="transformer_ref",
                        dtype=torch.bfloat16,
                        quantization_config=TorchAoConfig(
                            Int8WeightOnlyConfig(version=2),
                            modules_to_not_convert=t_not_convert),
                        low_cpu_mem_usage=True),
                    text_encoder=Qwen3VLForConditionalGeneration.from_pretrained(
                        self.model_path, subfolder="text_encoder",
                        dtype=torch.bfloat16,
                        quantization_config=TransformersTorchAoConfig(
                            Int8WeightOnlyConfig(version=2),
                            modules_to_not_convert=te_not_convert)),
                )
                pipe.load_components(workflow="ref2va", dtype=torch.bfloat16,
                                     pretrained_model_name_or_path=self.model_path)
                pipe.transformer_ref.requires_grad_(False)
                pipe.text_encoder.requires_grad_(False)
                offload = dict(onload_device=torch.device("cuda"),
                               offload_device=torch.device("cpu"), use_stream=True)
                pipe.transformer_ref.enable_group_offload(
                    offload_type="block_level", num_blocks_per_group=1, **offload)
                apply_group_offloading(pipe.text_encoder.model,
                                       offload_type="leaf_level", **offload)
                pipe.vae.to("cuda")
                pipe.audio_vae.to("cuda")
                pipe._device = torch.device("cuda")   # 显式执行设备兜底
                self._pipe = pipe
                return self._pipe

            from diffusers import ComponentsManager, MiniMaxH3Blocks
            from diffusers.modular_pipelines.modular_pipeline import SequentialPipelineBlocks
            # 双卡方案（fl2va/t2va）：条件侧(~62GB) 一张卡，生成侧(~62GB) 另一张卡。
            workflow = MiniMaxH3Blocks().get_workflow(self.variant)
            cond_blocks = {}
            for key in ("before_encode", "text_encoder"):
                if key in workflow.sub_blocks:
                    cond_blocks[key] = workflow.sub_blocks.pop(key)
            cond_combined = SequentialPipelineBlocks.from_blocks_dict(cond_blocks)

            text_manager = ComponentsManager()
            text_manager.enable_auto_cpu_offload(device="cuda:0")
            conditioner = cond_combined.init_pipeline(
                self.model_path, components_manager=text_manager)
            conditioner.load_components(dtype=torch.bfloat16,
                                        pretrained_model_name_or_path=self.model_path)

            if self.variant == "fl2va":
                # E20：fl2va 条件行额外显存，auto offload 下段 2+ OOM
                # （78.2/79.25GB）。E7 同款配方：生成侧 transformer 块级
                # 流式 offload + VAE 常驻 + 显式执行设备（不用 auto manager，
                # 避免两套放置机制打架）。
                manager = ComponentsManager()
                rest = workflow.init_pipeline(self.model_path, components_manager=manager)
                rest.load_components(dtype=torch.bfloat16,
                                     pretrained_model_name_or_path=self.model_path)
                offload = dict(onload_device=torch.device("cuda:1"),
                               offload_device=torch.device("cpu"), use_stream=True)
                rest.transformer.enable_group_offload(
                    offload_type="block_level", num_blocks_per_group=1, **offload)
                rest.vae.to("cuda:1")
                rest.audio_vae.to("cuda:1")
                rest._device = torch.device("cuda:1")   # 显式执行设备兜底
            else:
                manager = ComponentsManager()
                manager.enable_auto_cpu_offload(device="cuda:1")
                rest = workflow.init_pipeline(self.model_path, components_manager=manager)
                rest.load_components(dtype=torch.bfloat16,
                                     pretrained_model_name_or_path=self.model_path)
            self.version = "diffusers-main"
            self._pipe = (conditioner, rest)
        return self._pipe

    def generate(self, req: GenRequest, workdir: Path, **kw) -> Artifact:
        workdir.mkdir(parents=True, exist_ok=True)
        # fl2va 工作流要求每段都有首/尾帧 keyframe：无 keyframe 时其
        # prepare_condition_latents 无条件执行，condition_latents 为空会在
        # diffusers 深处以 torch.cat 空列表崩溃（Bug#7/E20）。
        # 在最早点响亮失败并给出可操作的指引（首段用 anchor 或换 t2va）。
        if self.variant == "fl2va" and req.first_frame is None and req.last_frame is None:
            raise RuntimeError(
                "fl2va 变体需要首帧条件：chain_mode=hard 的首段没有上一段末帧，"
                "请在 pipeline.context.anchor_refs 提供锚点参考图（首段会以其为"
                "首帧），或对该任务改用 variant=t2va / chain_mode=none。")
        self._get_pipe()
        assert self._pipe is not None
        t0 = time.time()

        fps = 24
        n = max(5, min(req.duration or 8, 15))     # H3 约束：5-15 秒
        num_frames = self.num_frames or n * fps
        num_frames = max(120, min(num_frames, 360))  # 帧数钳制到 VAE 可编码范围
        kwargs: Dict[str, Any] = {
            "prompt": req.text,
            "num_frames": num_frames,
        }
        if req.first_frame is not None:
            from PIL import Image
            kwargs["image"] = Image.open(req.first_frame).convert("RGB")
        if req.last_frame is not None:
            from PIL import Image
            kwargs["last_image"] = Image.open(req.last_frame).convert("RGB")
        if req.refs and self.variant == "ref2va":
            from diffusers.modular_pipelines.minimax_h3.references import MiniMaxH3ImageReference
            refs = []
            for r in req.refs:
                # 参考图默认按 2048 短边编码，视觉 token 数量爆炸导致单卡 OOM；
                # 缩到 768 短边（token 数降 ~7 倍），保留主体特征
                from PIL import Image
                img = Image.open(r).convert("RGB")
                if min(img.size) > 768:
                    s = 768 / min(img.size)
                    img = img.resize((round(img.width * s / 16) * 16,
                                      round(img.height * s / 16) * 16))
                refs.append(MiniMaxH3ImageReference(image=img))
            kwargs["references"] = refs
        # 纯文本模式必须显式指定画布（无首帧可推断尺寸）
        if req.first_frame is None:
            w, h = req.style.get("canvas", RATIO_CANVAS.get(req.ratio or "16:9", (1344, 768)))
            kwargs["height"], kwargs["width"] = h, w
        if self.seed is not None:
            import torch
            kwargs["generator"] = torch.Generator("cuda").manual_seed(self.seed)
        if self.steps is not None:
            kwargs["num_inference_steps"] = self.steps

        if self._pipe is not None and isinstance(self._pipe, tuple):
            # 双卡两段式：先条件编码，再生成（拆分口径见 split_dual_card_kwargs）
            conditioner, rest = self._pipe
            cond_kwargs, rest_kwargs = split_dual_card_kwargs(self.variant, kwargs)
            state = conditioner(**cond_kwargs)
            results = rest(state=state, output_type="pt",
                           output=["videos", "audio", "sampling_rate"], **rest_kwargs)
        else:
            results = self._pipe(output_type="pt",
                                 output=["videos", "audio", "sampling_rate"], **kwargs)
        video = results["videos"]
        audio = results["audio"]
        sr = results.get("sampling_rate", 32000)

        vid_path = workdir / f"h3_{int(time.time())}.mp4"
        if audio is not None:
            self._save_video_with_audio(video, audio, sr, vid_path)
        else:
            self._save_video(video, vid_path)

        meta = ArtifactMeta(adapter=self.name, model="MiniMax-H3-Base",
                            version=self.version,
                            params={"variant": self.variant, "duration": req.duration,
                                    "num_frames": kwargs["num_frames"],
                                    # 可重建：完整提示落盘（对齐"模型可见 ⟺ 日志"）
                                    "prompt": req.text},
                            seed=self.seed, elapsed_s=time.time() - t0)
        return Artifact(kind="video", path=vid_path, meta=meta)

    @staticmethod
    def _save_video(video, path: Path):
        """兼容三种产物类型：torch.Tensor / numpy.ndarray / PIL.Image(或列表)。"""
        import imageio
        import numpy as np
        from PIL import Image

        def _norm(f):
            arr = np.asarray(f)
            if arr.dtype != np.uint8:
                arr = (arr * 255).clip(0, 255).astype("uint8")
            return arr

        if isinstance(video, (list, tuple)):
            frames = [_norm(f) for f in video]
        elif isinstance(video, Image.Image):
            frames = [_norm(video)]
        elif hasattr(video, "detach"):       # torch.Tensor，H3 输出 (B, T, C, H, W)
            if video.dim() == 5:
                video = video[0]             # -> (T, C, H, W)
            if video.dim() == 4:             # (T, C, H, W) -> (T, H, W, C)
                video = video.permute(0, 2, 3, 1)
                frames = [_norm(f.float().cpu().numpy()) for f in video]
            elif video.dim() == 3:           # (C, H, W) 单帧
                frames = [_norm(video.permute(1, 2, 0).float().cpu().numpy())]
            else:
                raise ValueError(f"未知视频张量维度: {tuple(video.shape)}")
        else:                                # numpy
            arr = np.asarray(video)
            if arr.ndim == 4:                # (T, H, W, C)
                frames = [_norm(f) for f in arr]
            else:
                frames = [_norm(arr)]
        imageio.mimsave(str(path), frames, fps=24)

    @staticmethod
    def _save_video_with_audio(video, audio, sr: int, path: Path):
        tmp_v = path.with_suffix(".noaudio.mp4")
        MiniMaxH3Local._save_video(video, tmp_v)
        import soundfile as sf
        import numpy as np
        wav = path.with_suffix(".wav")
        if hasattr(audio, "detach"):          # torch.Tensor
            a = audio.squeeze(0)
            if a.dim() == 2:
                a = a.T
            a = a.float().cpu().numpy()
        else:                                 # numpy
            a = np.asarray(audio).squeeze(0)
            if a.ndim == 2 and a.shape[0] > a.shape[1]:
                a = a.T
        sf.write(str(wav), a, sr)
        import subprocess
        subprocess.run(["ffmpeg", "-y", "-i", str(tmp_v), "-i", str(wav),
                        "-c:v", "copy", "-c:a", "aac", "-shortest", str(path)],
                       capture_output=True, check=True)
        tmp_v.unlink(missing_ok=True)
        wav.unlink(missing_ok=True)


# ==========================================================================
# 官方 API（完整 H3：Context-IR + 2K）
# ==========================================================================
@register("generator.minimax-h3-api")
class MiniMaxH3API:
    name = "generator.minimax-h3-api"
    capabilities = {
        "max_duration_s": 15, "audio": True, "refs": 9,
        "first_last_frame": True, "resolution": "2K", "backend": "api",
        # 单价声明（规划预估口径；单一正源 _MINIMAX_RATES_USD_PER_S，
        # 与运行时 _estimate_cost 同源；实际计费以官方为准）
        "cost_rates_usd_per_s": dict(_MINIMAX_RATES_USD_PER_S),
    }
    param_schema = {
        "api_key": {"type": "secret", "default": "", "help": "MiniMax API key（缺省读环境）"},
        "base_url": {"type": "str", "default": "https://api.minimaxi.com"},
        "resolution": {"type": "str", "default": "768P", "choices": ["768P", "2K"]},
        "duration": {"type": "int", "default": 8, "help": "单次生成时长（秒）"},
        "ratio": {"type": "str", "default": "16:9"},
    }

    def __init__(self, api_key: str = "", base_url: str = "https://api.minimaxi.com",
                 resolution: str = "768P", duration: int = 8, ratio: str = "16:9"):
        # 凭据延迟到第一次生成调用解析（E32 准备：规划期 dry-run 不依赖 key）
        self.api_key = api_key
        self.base_url = base_url
        self.resolution = resolution
        self.duration = duration
        self.ratio = ratio

    def generate(self, req: GenRequest, workdir: Path, **kw) -> Artifact:
        import requests
        workdir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        key = self.api_key or _load_minimax_key()   # 首次 I/O 前解析凭据（fail loud）
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

        content: List[Dict[str, Any]] = [{"type": "text", "text": req.text}]
        if req.first_frame is not None:
            content.append({"type": "image_url",
                            "image_url": {"url": _upload(req.first_frame, headers, self.base_url)},
                            "role": "first_frame"})
        if req.last_frame is not None:
            content.append({"type": "image_url",
                            "image_url": {"url": _upload(req.last_frame, headers, self.base_url)},
                            "role": "last_frame"})
        for r in req.refs:
            content.append({"type": "image_url",
                            "image_url": {"url": _upload(r, headers, self.base_url)},
                            "role": "reference_image"})

        payload: Dict[str, Any] = {
            "model": "MiniMax-H3",
            "content": content,
            "resolution": self.resolution,
            "duration": req.duration or self.duration,
            "ratio": req.ratio or ("adaptive" if req.first_frame or req.refs else self.ratio),
        }
        resp = requests.post(f"{self.base_url}/v2/video_generation", headers=headers,
                             json=payload, timeout=120)  # type: ignore[arg-type]
        resp.raise_for_status()
        task_id = resp.json()["task_id"]

        # 轮询
        while True:
            q = requests.get(f"{self.base_url}/v2/query/video_generation",
                             headers=headers, params={"task_id": task_id}, timeout=60)
            q.raise_for_status()
            task = q.json()["task"]
            status = task["status"]
            if status == "succeeded":
                break
            if status in ("failed", "cancelled"):
                raise RuntimeError(f"H3 任务失败: {task}")
            time.sleep(5)

        url = task["content"]["url"]
        out = workdir / f"h3api_{task_id}.mp4"
        with requests.get(url, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            with open(out, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)

        meta = ArtifactMeta(adapter=self.name, model="MiniMax-H3-API",
                            params={"resolution": self.resolution, "duration": payload["duration"]},
                            elapsed_s=time.time() - t0,
                            cost_usd=_estimate_cost(self.resolution, int(payload["duration"])))
        return Artifact(kind="video", path=out, meta=meta)


def _upload(path: Path, headers: Dict[str, str], base_url: str) -> str:
    import requests
    r = requests.post(f"{base_url}/v1/files/upload?purpose=video_generation",
                      headers=headers, files={"file": open(path, "rb")}, timeout=120)
    r.raise_for_status()
    return r.json()["file"]["url"]


def _load_minimax_key() -> str:
    import os
    k = os.environ.get("MINIMAX_API_KEY")
    if k:
        return k
    settings = Path.home() / ".claude" / "settings.json"
    if settings.exists():
        try:
            cfg = json.loads(settings.read_text(encoding="utf-8"))
            k = cfg.get("env", {}).get("MINIMAX_API_KEY")
            if k:
                return k
        except Exception:
            pass
    raise RuntimeError("未找到 MiniMax API key（MINIMAX_API_KEY 或 ~/.claude/settings.json）")


def _estimate_cost(resolution: str, duration: int) -> float:
    """官方定价近似（USD/秒）。单一价格正源：_MINIMAX_RATES_USD_PER_S。

    同一事实只在一处维护（capabilities 声明目录与运行时计费共用）；
    未知分辨率响亮失败，不猜价。
    """
    rate = _MINIMAX_RATES_USD_PER_S.get(str(resolution).upper())
    if rate is None:
        raise RuntimeError(
            f"未声明分辨率 '{resolution}' 的单价（已知: {sorted(_MINIMAX_RATES_USD_PER_S)}）")
    return round(duration * rate, 4)
