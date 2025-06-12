# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# MODIFIED FOR BATCH PROCESSING
import argparse
import logging
import os
import sys
import warnings
from pathlib import Path
from PIL import Image
import glob
import torch
import torch.distributed as dist


import wan
from wan.configs import SIZE_CONFIGS, WAN_CONFIGS, MAX_AREA_CONFIGS
from wan.utils.utils import cache_video, str2bool

warnings.filterwarnings('ignore')

def _init_logging(rank):
    """初始化日志记录器"""
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)

def _parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Batch generate videos from images and prompts using Wan-AI.")
    
    # --- 核心输入参数 ---
    parser.add_argument("--project_dir", type=str, required=True, help="Path to the project directory containing 'generated_images' and 'img2vid_prompts.txt'.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated video clips.")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="The path to the Wan-AI model checkpoint directory.")

    # --- 模型和生成参数 ---
    parser.add_argument("--size", type=str, default="832*480", help="Resolution of the generated videos.")
    parser.add_argument("--frame_num", type=int, default=81, help="Number of frames to generate for each video clip.")
    parser.add_argument("--sample_steps", type=int, default=40, help="The sampling steps.")
    parser.add_argument("--sample_guide_scale", type=float, default=6.0, help="Classifier free guidance scale.")
    parser.add_argument("--sample_shift", type=float, default=3.0, help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument("--base_seed", type=int, default=42, help="Base seed for reproducibility.")

    # --- 性能和分布式计算参数 ---
    parser.add_argument("--offload_model", type=str2bool, default=None, help="Offload model to CPU to save GPU memory.")
    parser.add_argument("--ulysses_size", type=int, default=1, help="Ulysses parallelism size.")
    parser.add_argument("--ring_size", type=int, default=1, help="Ring attention parallelism size.")
    parser.add_argument("--t5_fsdp", action="store_true", default=False, help="Use FSDP for T5.")
    parser.add_argument("--t5_cpu", action="store_true", default=False, help="Place T5 model on CPU.")
    parser.add_argument("--dit_fsdp", action="store_true", default=False, help="Use FSDP for DiT.")

    return parser.parse_args()


def main():
    args = _parse_args()
    
    # 1. 初始化分布式环境
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        if rank == 0: logging.info(f"Offload model automatically set to {args.offload_model}.")

    # 2. 准备输入数据 (只在主进程 rank 0 中进行)
    image_paths = []
    prompts = []
    if rank == 0:
        logging.info(f"Loading inputs from project directory: {args.project_dir}")
        
        # 查找所有场景图片
        images_dir = Path(args.project_dir) / "generated_images"
        image_paths = sorted(glob.glob(str(images_dir / "scene*.png")))
        if not image_paths:
            logging.error(f"❌ No images found in {images_dir}")
            sys.exit(1)

        # 读取所有提示词
        prompts_file = Path(args.project_dir) / "img2vid_prompts.txt"
        if not prompts_file.exists():
            logging.error(f"❌ Prompts file not found at {prompts_file}")
            sys.exit(1)
        with open(prompts_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        
        if len(image_paths) != len(prompts):
            logging.warning(f"⚠️ Mismatch: Found {len(image_paths)} images but {len(prompts)} prompts. Will process up to the smaller count.")
            
        num_scenes = min(len(image_paths), len(prompts))
        image_paths = image_paths[:num_scenes]
        prompts = prompts[:num_scenes]
        
        logging.info(f"✅ Found {num_scenes} scenes to process.")
        
        # 确保输出目录存在
        os.makedirs(args.output_dir, exist_ok=True)


    # 3. 加载模型 (只加载一次!)
    if rank == 0: logging.info("Loading Wan-AI I2V model... This may take a while.")
    
    # 任务配置硬编码为 i2v-14B
    task = "i2v-14B" 
    cfg = WAN_CONFIGS[task]
    
    wan_i2v = wan.WanI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=args.t5_cpu,
    )
    if rank == 0: logging.info("✅ Model loaded successfully.")

    # 4. 循环生成所有视频片段
    # 主进程(rank 0)将任务分发给所有进程
    for i in range(len(image_paths)):
        if rank == 0:
            scene_num = i + 1
            image_path = image_paths[i]
            prompt = prompts[i]
            
            logging.info("-" * 50)
            logging.info(f"🎬 Processing Scene {scene_num}/{len(image_paths)}")
            logging.info(f"  Image: {image_path}")
            logging.info(f"  Prompt: {prompt[:100]}...")

            img = Image.open(image_path).convert("RGB")
        
        # 在所有进程中同步数据。主进程广播图像和提示。
        # 这里我们简化处理，假设图像和提示词不需要广播，因为每个rank都会调用generate
        # 但在实际分布式环境中，更好的做法是广播数据。为了简化，我们暂时跳过这步。
        # 注意：在当前脚本中，只有 rank 0 有 image_paths 和 prompts，
        # 为了让所有rank都能执行，我们将在rank 0中执行生成，然后保存。
        # 这是一个简化，如果需要真正的分布式推理，需要广播 image 和 prompt。

        if rank == 0:
            video_tensor = wan_i2v.generate(
                prompt,
                img,
                max_area=MAX_AREA_CONFIGS[args.size],
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver='unipc',
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=args.base_seed + i,  # 为每个场景使用不同的种子以增加多样性
                offload_model=args.offload_model
            )

            # 保存生成的视频片段
            output_filename = f"scene_{i+1:02d}.mp4"
            save_path = os.path.join(args.output_dir, output_filename)
            logging.info(f"  Saving video to: {save_path}")
            
            cache_video(
                tensor=video_tensor[None],
                save_file=save_path,
                fps=cfg.sample_fps,
                normalize=True,
                value_range=(-1, 1)
            )
            logging.info(f"✅ Scene {scene_num} completed.")

    if rank == 0:
        logging.info("-" * 50)
        logging.info("🎉 All video clips have been generated successfully!")


if __name__ == "__main__":
    main()