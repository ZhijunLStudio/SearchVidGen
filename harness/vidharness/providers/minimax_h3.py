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
RATIO_CANVAS = {
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "1:1": (768, 768),
    "4:3": (1024, 768),
    "3:4": (768, 1024),
    "21:9": (1568, 672),
}


# ==========================================================================
# 本地 H3-Base（diffusers）
# ==========================================================================
@register("generator.minimax-h3-local")
class MiniMaxH3Local:
    name = "generator.minimax-h3-local"
    capabilities = {
        "max_duration_s": 15, "audio": True, "refs": 9,
        "first_last_frame": True, "resolution": "768p", "backend": "local",
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
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu   # 如 "4,6"
            import torch
            from diffusers import ComponentsManager, MiniMaxH3Blocks
            # 官方双卡方案：text_encoder(~62GB) 一张卡，transformer+VAE(~62GB) 另一张卡，
            # 各自 auto_cpu_offload。单卡 80GB 在首尾帧模式下峰值显存溢出。
            workflow = MiniMaxH3Blocks()
            text_manager = ComponentsManager()
            text_manager.enable_auto_cpu_offload(device="cuda:0")
            conditioner = workflow.sub_blocks.pop("text_encoder").init_pipeline(
                self.model_path, components_manager=text_manager)
            conditioner.load_components(dtype=torch.bfloat16,
                                        pretrained_model_name_or_path=self.model_path)

            manager = ComponentsManager()
            manager.enable_auto_cpu_offload(device="cuda:1")
            rest = workflow.init_pipeline(self.model_path, components_manager=manager)
            rest.load_components(workflow=self.variant, dtype=torch.bfloat16,
                                 pretrained_model_name_or_path=self.model_path)
            self.version = "diffusers-main"
            self._pipe = (conditioner, rest)
        return self._pipe

    def generate(self, req: GenRequest, workdir: Path, **kw) -> Artifact:
        workdir.mkdir(parents=True, exist_ok=True)
        conditioner, rest = self._get_pipe()
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
            from PIL import Image
            kwargs["reference_images"] = [Image.open(r).convert("RGB") for r in req.refs]
        # 纯文本模式必须显式指定画布（无首帧可推断尺寸）
        if req.first_frame is None:
            w, h = req.style.get("canvas", RATIO_CANVAS.get(req.ratio or "16:9", (1344, 768)))
            kwargs["height"], kwargs["width"] = h, w
        if self.seed is not None:
            import torch
            kwargs["generator"] = torch.Generator("cuda").manual_seed(self.seed)
        if self.steps is not None:
            kwargs["num_inference_steps"] = self.steps

        # 两段式调用：先条件编码，再生成
        state = conditioner(prompt=kwargs.pop("prompt"))
        results = rest(state=state, output_type="pt",
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
                                    "num_frames": kwargs["num_frames"]},
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
            import torch
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
        import imageio, torch
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
    }

    def __init__(self, api_key: str = "", base_url: str = "https://api.minimaxi.com",
                 resolution: str = "768P", duration: int = 8, ratio: str = "16:9"):
        self.api_key = api_key or _load_minimax_key()
        self.base_url = base_url
        self.resolution = resolution
        self.duration = duration
        self.ratio = ratio

    def generate(self, req: GenRequest, workdir: Path, **kw) -> Artifact:
        import requests
        workdir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

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

        payload = {
            "model": "MiniMax-H3",
            "content": content,
            "resolution": self.resolution,
            "duration": req.duration or self.duration,
            "ratio": req.ratio or ("adaptive" if req.first_frame or req.refs else self.ratio),
        }
        r = requests.post(f"{self.base_url}/v2/video_generation", headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        task_id = r.json()["task_id"]

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
                            cost_usd=_estimate_cost(self.resolution, payload["duration"]))
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
    # 官方定价大致区间（CN）：768P 约 ¥0.3/秒，2K 约 ¥0.8/秒；按 7.2 汇率近似 USD
    rate = 0.8 if resolution == "2K" else 0.3
    return round(duration * rate / 7.2, 4)
