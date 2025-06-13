import os
from pathlib import Path
from PIL import Image
import requests
import io
import base64
import torch
from datetime import datetime
import re # Import re for safe folder naming

# Make sure 'pipeline' is discoverable (e.g., in the same directory or in PYTHONPATH)
# Assuming pipeline.py contains InstantCharacterFluxPipeline
from pipeline import InstantCharacterFluxPipeline
from diffusers.hooks import apply_group_offloading


class ImageGenerator:
    def __init__(self, model_type: str = "placeholder"):
        """
        初始化图片生成器
        model_type: 可以是 "placeholder", "stable_diffusion", "flux" 等
        """
        self.model_type = model_type

        # For 'flux' model, initialize it here since it's heavy and shared
        if self.model_type == "flux":
            # Configuration for Flux model
            self.ip_adapter_path = '/data/home/lizhijun/llm/flux-hf/InstantCharacter-main/tencent/InstantCharacter/instantcharacter_ip-adapter.bin'
            self.base_model = '/data/home/lizhijun/llm/flux-hf/model/flux-dev'
            self.image_encoder_path = 'google/siglip-so400m-patch14-384'
            self.image_encoder_2_path = 'facebook/dinov2-giant'

            print("Initializing Flux model...")
            # Check if model is already loaded to prevent re-initialization in case of multiple calls
            if not hasattr(self, 'pipe') or self.pipe is None:
                self.pipe = InstantCharacterFluxPipeline.from_pretrained(self.base_model, torch_dtype=torch.bfloat16)

                # Apply model optimizations
                apply_group_offloading(
                    self.pipe.transformer,
                    offload_type="leaf_level",
                    offload_device=torch.device("cpu"),
                    onload_device=torch.device("cuda"),
                    use_stream=True,
                )
                apply_group_offloading(
                    self.pipe.text_encoder,
                    offload_device=torch.device("cpu"),
                    onload_device=torch.device("cuda"),
                    offload_type="leaf_level",
                    use_stream=True,
                )
                apply_group_offloading(
                    self.pipe.text_encoder_2,
                    offload_device=torch.device("cpu"),
                    onload_device=torch.device("cuda"),
                    offload_type="leaf_level",
                    use_stream=True,
                )
                apply_group_offloading(
                    self.pipe.vae,
                    offload_device=torch.device("cpu"),
                    onload_device=torch.device("cuda"),
                    offload_type="leaf_level",
                    use_stream=True,
                )

                self.pipe.init_adapter(
                    image_encoder_path=self.image_encoder_path,
                    image_encoder_2_path=self.image_encoder_2_path,
                    subject_ipadapter_cfg=dict(subject_ip_adapter_path=self.ip_adapter_path, nb_token=1024),
                )
            print("Flux model initialized.")


    def generate(self, prompt: str, reference_image: str = None,
                output_path: Path = None, **kwargs) -> Path:
        """
        生成图片

        Args:
            prompt: 图片生成提示词
            reference_image: 参考图片路径（可选）
            output_path: 输出路径
            **kwargs: 其他参数（如size, steps等）

        Returns:
            生成的图片路径
        """
        if output_path is None:
            # Generate a default unique name if not provided
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output_path = Path(f"generated_image_{timestamp}.png")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure output directory exists

        if self.model_type == "placeholder":
            # 生成占位图片
            return self._generate_placeholder(prompt, output_path)
        elif self.model_type == "stable_diffusion":
            # 调用Stable Diffusion API (placeholder for actual implementation)
            return self._generate_placeholder(prompt, output_path) # Fallback to placeholder
        elif self.model_type == "flux":
            # 调用Flux模型
            return self._generate_with_flux(prompt, reference_image, output_path, **kwargs)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

    def _generate_placeholder(self, prompt: str, output_path: Path) -> Path:
        """生成占位图片"""
        # 创建一个简单的占位图片
        img = Image.new('RGB', (832, 480), color='lightblue')

        # 添加文字（需要PIL的ImageDraw和ImageFont）
        try:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(img)

            # 尝试使用系统字体
            try:
                # Use a common cross-platform font name if available, or specify a path
                font = ImageFont.truetype("arial.ttf", 20)
            except IOError:
                font = ImageFont.load_default()

            # 绘制提示词
            text = f"Placeholder\n{prompt[:50]}..."
            draw.text((10, 10), text, fill='black', font=font)
        except ImportError:
            # If ImageDraw/ImageFont are not available, just save blank image
            pass

        img.save(output_path)
        return output_path

    def _generate_with_sd(self, prompt: str, reference_image: str,
                         output_path: Path, **kwargs) -> Path:
        """使用Stable Diffusion生成图片"""
        # Placeholder for actual SD API integration
        # Example of how to structure an API call:
        # api_url = "YOUR_SD_API_ENDPOINT"
        # headers = {"Content-Type": "application/json"}
        # payload = {
        #     "prompt": prompt,
        #     "reference_image_base64": base64.b64encode(open(reference_image, "rb").read()).decode('utf-8') if reference_image else None,
        #     **kwargs
        # }
        # response = requests.post(api_url, headers=headers, json=payload)
        # response.raise_for_status()
        # image_bytes = base64.b64decode(response.json()['image_base64'])
        # image = Image.open(io.BytesIO(image_bytes))
        # image.save(output_path)
        # return output_path
        print("Stable Diffusion generation is a placeholder; generating a placeholder image.")
        return self._generate_placeholder(prompt, output_path)


    def _generate_with_flux(self, prompt: str, reference_image: str,
                           output_path: Path, **kwargs) -> Path:
        """使用Flux模型生成图片"""
        if not hasattr(self, 'pipe') or self.pipe is None:
            raise RuntimeError("Flux model not initialized. Please set model_type to 'flux' during ImageGenerator instantiation and ensure initialization completed.")

        if not reference_image or not os.path.exists(reference_image):
            print(f"Warning: Reference image not found at: {reference_image}. Flux model might not generate consistent character without it. Proceeding without reference image consistency (if model supports).")
            # If a reference image is strictly required and not provided, consider raising an error
            # For InstantCharacterFluxPipeline, subject_image is optional for general generation but crucial for character consistency.
            ref_image = None
        else:
            try:
                ref_image = Image.open(reference_image).convert('RGB')
            except Exception as e:
                print(f"Error loading reference image {reference_image}: {e}. Proceeding without reference image consistency.")
                ref_image = None


        # Generation parameters from kwargs, with defaults
        num_inference_steps = kwargs.get("steps", 30)
        guidance_scale = kwargs.get("guidance_scale", 3.8)
        subject_scale = kwargs.get("subject_scale", 0.9)
        seed = kwargs.get("seed", 42) # Default seed if not provided

        # Ensure generator is on CUDA if pipe is
        generator_device = "cuda" if torch.cuda.is_available() else "cpu"
        current_generator = torch.Generator(device=generator_device).manual_seed(seed)


        # Call pipe with subject_image if available, otherwise without
        if ref_image:
            image = self.pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                subject_scale=subject_scale,
                subject_image=ref_image,
                generator=current_generator,
            ).images[0]
        else:
            image = self.pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=current_generator,
            ).images[0]


        image.save(output_path)
        return output_path


# --- Main execution block to generate images from a specified prompt file ---
if __name__ == "__main__":
    # --- Configuration ---
    # Path to the directory containing img2img_prompts.txt (e.g., from LLMClient output)
    # REPLACE THIS WITH THE ACTUAL PATH TO YOUR GENERATED CONTENT FOLDER
    input_content_dir = Path("generated_video_content/春天在哪里_20250614_002954") # Example path
    # If the above path doesn't exist, you'll need to run llm_client.py first
    # Or, manually create a directory and put an img2img_prompts.txt file inside it.

    # Path to your reference image (e.g., the anime boy character)
    reference_image_path = 'img/12.png'

    # Output subfolder name within input_content_dir for generated images
    output_images_subfolder = "generated_images"

    # Fixed seed for consistency (can be varied for each image if needed)
    base_seed = 123456

    # Generation parameters for Flux
    generation_params = {
        "steps": 30,
        "guidance_scale": 3.8,
        "subject_scale": 0.9,
    }
    # --- End Configuration ---


    # --- Load Prompts ---
    prompts_file_path = input_content_dir / "img2img_prompts.txt"
    if not prompts_file_path.exists():
        print(f"Error: img2img_prompts.txt not found at {prompts_file_path}.")
        print("Please ensure you have run llm_client.py to generate content or specify the correct input_content_dir.")
        exit()

    img2img_prompts = []
    with open(prompts_file_path, "r", encoding="utf-8") as f:
        for line in f:
            # Remove any leading/trailing whitespace including newlines
            clean_line = line.strip()
            if clean_line: # Only add non-empty lines
                img2img_prompts.append(clean_line)

    if not img2img_prompts:
        print(f"Warning: No prompts found in {prompts_file_path}. Exiting.")
        exit()

    print(f"Loaded {len(img2img_prompts)} prompts from {prompts_file_path}.")

    # --- Setup Output Directory for Images ---
    # Create the dedicated subfolder for images within the input_content_dir
    output_image_dir = input_content_dir / output_images_subfolder
    output_image_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generated images will be saved to: {output_image_dir}")

    # --- Initialize Image Generator ---
    generator = ImageGenerator(model_type="flux")

    print("\n--- Starting Image Generation ---")
    generated_image_paths = []
    for i, prompt in enumerate(img2img_prompts):
        scene_num = i + 1 # Scenes are 1-indexed
        
        # You might want to extract the "scene name" from a corresponding file
        # or use a default if it's not crucial for filename here.
        # For simplicity, we'll use "sceneXX" in the filename.
        filename = f"scene{scene_num:02d}.png"
        filepath = output_image_dir / filename

        # Generate a unique seed for each image
        current_seed = base_seed + i

        print(f"\nGenerating image for Scene {scene_num}: {prompt[:70]}...") # Show first 70 chars of prompt

        # Generate the image using the ImageGenerator
        generated_path = generator.generate(
            prompt=prompt,
            reference_image=reference_image_path,
            output_path=filepath,
            seed=current_seed,
            **generation_params
        )
        if generated_path:
            generated_image_paths.append(generated_path)
            print(f"✅ Saved: {generated_path}")
        else:
            print(f"❌ Failed to generate image for Scene {scene_num}.")

    print(f"\n--- Image Generation Complete ---")
    print(f"Total images generated: {len(generated_image_paths)}")
    print(f"All generated images are in: {output_image_dir}")

    # You now have all generated image paths in `generated_image_paths`
    # This list can be passed to your VideoGenerator for the next step.