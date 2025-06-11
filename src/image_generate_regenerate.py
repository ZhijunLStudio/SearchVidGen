import os
from pathlib import Path
from PIL import Image
import torch
from datetime import datetime

# 假设 pipeline 和 apply_group_offloading 已经正确导入
from pipeline import InstantCharacterFluxPipeline
from diffusers.hooks import apply_group_offloading

class ImageGenerator:
    def __init__(self):
        self.ip_adapter_path = '/data/home/lizhijun/llm/flux-hf/InstantCharacter-main/tencent/InstantCharacter/instantcharacter_ip-adapter.bin'
        self.base_model = '/data/home/lizhijun/llm/flux-hf/model/flux-dev'
        self.image_encoder_path = 'google/siglip-so400m-patch14-384'
        self.image_encoder_2_path = 'facebook/dinov2-giant'

        print("Initializing Flux model...")
        self.pipe = InstantCharacterFluxPipeline.from_pretrained(
            self.base_model, torch_dtype=torch.bfloat16
        )
        # offload 优化
        for module in [self.pipe.transformer,
                       self.pipe.text_encoder,
                       self.pipe.text_encoder_2,
                       self.pipe.vae]:
            apply_group_offloading(
                module,
                offload_type="leaf_level",
                offload_device=torch.device("cpu"),
                onload_device=torch.device("cuda"),
                use_stream=True,
            )
        self.pipe.init_adapter(
            image_encoder_path=self.image_encoder_path,
            image_encoder_2_path=self.image_encoder_2_path,
            subject_ipadapter_cfg=dict(
                subject_ip_adapter_path=self.ip_adapter_path,
                nb_token=1024
            ),
        )
        print("Flux model initialized.")

    def generate(self, prompt: str, reference_image: str = None,
                 output_path: Path = None, **kwargs) -> Path:
        if output_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output_path = Path(f"generated_image_{ts}.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return self._generate_with_flux(prompt, reference_image, output_path, **kwargs)

    def _generate_with_flux(self, prompt, reference_image, output_path, **kwargs):
        ref_img = None
        if reference_image and os.path.exists(reference_image):
            try:
                ref_img = Image.open(reference_image).convert('RGB')
            except:
                pass

        steps = kwargs.get("steps", 30)
        guidance = kwargs.get("guidance_scale", 3.8)
        subj_scale = kwargs.get("subject_scale", 0.9)
        seed = kwargs.get("seed", 42)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        gen = torch.Generator(device=device).manual_seed(seed)

        if ref_img:
            img = self.pipe(
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                subject_scale=subj_scale,
                subject_image=ref_img,
                generator=gen,
            ).images[0]
        else:
            img = self.pipe(
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=gen,
            ).images[0]

        img.save(output_path)
        return output_path

# ---------------------------------
#           主程序部分
# ---------------------------------
if __name__ == "__main__":
    # --- 配置区 ---
    input_content_dir = Path("/data/home/lizhijun/opensource/SearchVidGen/generated_video_content/卡皮巴拉")
    reference_image_path = 'img/20.png'
    output_images_subfolder = "generated_images"
    base_seed = 123456

    # ✨ 核心配置：指定要生成的场次列表，以及每个场次生成几张图片
    scene_indices_to_generate = [1]  # 要处理的场次号
    num_images_per_scene = 3                  # 每个场次生成 3 张图

    # ✨ 更新后的 Prompt 覆盖字典 (使用“视觉描述”策略)
    override_prompts = {
        1: "Anime style, high quality, consistent character design. In a cozy home office, a young man leans back thoughtfully in his chair. He is at a minimalist wooden desk where a modern laptop is open. The laptop's illuminated screen displays a dark-themed user interface with colorful, syntax-highlighted text, while its outer lid is a simple matte black. The scene is bathed in warm afternoon light, and a steaming ceramic coffee cup sits on the desk next to a thriving potted plant. The camera captures his thoughtful expression, with the laptop screen clearly visible beside him.",
        2: "Anime style, high quality, consistent character design, a young student sitting in a bright library, surrounded by stacks of programming books. They are focusing intently on a tablet; the tablet's bright screen shows an educational website interface with diagrams, bullet points, and blocks of colored text. The back of the tablet is a clean, white surface. A notebook filled with handwritten notes is beside them. The tablet screen is directly facing the student.",
        7: "Anime style, high quality, consistent character design, a teenager lying on a cozy bed with a laptop. The laptop's glowing screen is playing an online video tutorial, showing a presenter's screen with presentation slides, flowcharts, and colorful text. The lid on the back of the screen has a cute anime sticker on it. The laptop is tilted up toward the teen’s face, surrounded by plush toys and a soft bedside lamp casting a warm glow.",
        10: "Anime style, high quality, consistent character design, a serene outdoor café with a freelancer typing on a laptop. On the laptop's bright, lit-up screen, the user interface of a development application is visible, featuring side panels, menus, and a main window with structured, colorful text. The laptop's outer shell is a sleek, metallic silver. A steaming cup of tea is beside them, and cherry blossom petals drift in the air. The laptop screen faces the freelancer (front view)."
    }

    generation_params = {
        "steps": 30,
        "guidance_scale": 3.8,
        "subject_scale": 0.9,
    }
    # --- 配置结束 ---

    # 1. 读取所有 prompt
    prompts_file = input_content_dir / "img2img_prompts.txt"
    if not prompts_file.exists():
        print(f"Error: {prompts_file} 不存在。")
        exit(1)
    with open(prompts_file, "r", encoding="utf-8") as f:
        all_prompts = [l.strip() for l in f if l.strip()]

    # 2. 准备输出目录
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = input_content_dir / output_images_subfolder / f"rerun_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"🚀 将为场次 {scene_indices_to_generate} 各生成 {num_images_per_scene} 张图片。")
    print(f"   输出目录: {output_dir}")
    
    # 3. 初始化 Flux 生成器
    generator = ImageGenerator()

    # 4. 循环生成
    results = []
    # 外层循环：遍历指定的每个场次
    for idx in scene_indices_to_generate:
        if idx < 1 or idx > len(all_prompts):
            print(f"⚠️ 场次 {idx} 越界，跳过。")
            continue

        prompt = override_prompts.get(idx, all_prompts[idx - 1])
        print(f"\n- - - - - - - - - - - - - - - - - - - - - - -")
        print(f"🔄 开始处理场次 {idx:02d}，将生成 {num_images_per_scene} 张图片...")
        print(f"   Prompt: {prompt}")

        # 内层循环：为当前场次生成指定数量的图片
        for i in range(num_images_per_scene):
            run_num = i + 1
            # 确保每张图都有唯一的 seed 和文件名
            # 添加一个较大的数乘以场次号，避免不同场次的seed过于接近
            seed = base_seed + (idx * 1000) + i 
            out_path = output_dir / f"scene{idx:02d}_run{run_num:02d}.png"
            
            print(f"  -> 正在生成第 {run_num}/{num_images_per_scene} 张... (Seed: {seed})")
            
            try:
                p = generator.generate(
                    prompt=prompt,
                    reference_image=reference_image_path,
                    output_path=out_path,
                    seed=seed,
                    **generation_params
                )
                print(f"  ✅ 保存成功: {p.name}")
                results.append(p)
            except Exception as e:
                print(f"  ❌ 生成失败: {e}")

    print(f"\n🎉 全部完成，总共生成 {len(results)} 张图片，保存在目录：{output_dir}")