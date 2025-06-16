import os
import random
import torch
from pathlib import Path
from PIL import Image
from datetime import datetime
import logging
from typing import Dict, List, Any, Optional, Tuple

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from pipeline import InstantCharacterFluxPipeline
    from diffusers.hooks import apply_group_offloading
    FLUX_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Flux模型相关依赖导入失败: {e}")
    FLUX_AVAILABLE = False

class ImageGenerator:
    def __init__(self, model_type: str = "flux", use_offload: bool = True):
        """
        初始化图片生成器
        
        Args:
            model_type: 模型类型，目前支持 "flux" 和 "placeholder"
            use_offload: 是否使用模型卸载（节省显存）
        """
        self.model_type = model_type
        self.use_offload = use_offload
        self.pipe = None
        self.is_initialized = False
        
        # 默认配置
        self.ip_adapter_path = '/data/home/lizhijun/llm/flux-hf/InstantCharacter-main/tencent/InstantCharacter/instantcharacter_ip-adapter.bin'
        self.base_model = '/data/home/lizhijun/llm/flux-hf/model/flux-dev'
        self.image_encoder_path = 'google/siglip-so400m-patch14-384'
        self.image_encoder_2_path = 'facebook/dinov2-giant'
        
        logger.info(f"ImageGenerator initialized with model_type='{model_type}', use_offload={use_offload}")

    def initialize_model(self):
        """初始化模型"""
        if self.is_initialized:
            logger.info("模型已初始化，跳过")
            return True
            
        if self.model_type == "flux":
            return self._initialize_flux_model()
        elif self.model_type == "placeholder":
            logger.info("使用占位符模式，无需初始化模型")
            self.is_initialized = True
            return True
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

    def _initialize_flux_model(self):
        """初始化Flux模型"""
        if not FLUX_AVAILABLE:
            logger.error("Flux模型依赖不可用，请检查安装")
            return False
            
        try:
            logger.info("正在初始化Flux模型...")
            
            # 检查模型文件是否存在
            if not os.path.exists(self.base_model):
                logger.error(f"基础模型路径不存在: {self.base_model}")
                return False
                
            if not os.path.exists(self.ip_adapter_path):
                logger.error(f"IP Adapter路径不存在: {self.ip_adapter_path}")
                return False
            
            # 加载模型
            self.pipe = InstantCharacterFluxPipeline.from_pretrained(
                self.base_model, 
                torch_dtype=torch.bfloat16
            )
            
            # 应用offload优化
            if self.use_offload:
                logger.info("应用模型卸载优化...")
                self._apply_offloading()
            else:
                logger.info("将模型加载到CUDA...")
                self.pipe.to("cuda")
            
            # 初始化适配器
            logger.info("初始化IP Adapter...")
            self.pipe.init_adapter(
                image_encoder_path=self.image_encoder_path,
                image_encoder_2_path=self.image_encoder_2_path,
                subject_ipadapter_cfg=dict(
                    subject_ip_adapter_path=self.ip_adapter_path, 
                    nb_token=1024
                ),
            )
            
            self.is_initialized = True
            logger.info("Flux模型初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"Flux模型初始化失败: {e}")
            return False

    def _apply_offloading(self):
        """应用模型卸载优化"""
        try:
            # 为各个组件应用offload
            components = [
                ("transformer", self.pipe.transformer),
                ("text_encoder", self.pipe.text_encoder),
                ("text_encoder_2", self.pipe.text_encoder_2),
                ("vae", self.pipe.vae)
            ]
            
            for name, component in components:
                if component is not None:
                    logger.info(f"应用offload到 {name}")
                    apply_group_offloading(
                        component,
                        offload_type="leaf_level",
                        offload_device=torch.device("cpu"),
                        onload_device=torch.device("cuda"),
                        use_stream=True,
                    )
                    
        except Exception as e:
            logger.warning(f"应用offload优化失败: {e}")
            # 如果offload失败，回退到普通CUDA加载
            logger.info("回退到普通CUDA加载")
            self.pipe.to("cuda")

    def update_config(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"更新配置: {key} = {value}")
        
        # 如果模型已初始化且配置有变化，需要重新初始化
        if self.is_initialized and self.model_type == "flux":
            logger.info("配置已更新，重新初始化模型")
            self.is_initialized = False
            self.pipe = None

    def generate(self, prompt: str, reference_image: str = None, 
                output_path: Path = None, **kwargs) -> Path:
        """
        生成图片
        
        Args:
            prompt: 图片生成提示词
            reference_image: 参考图片路径
            output_path: 输出路径
            **kwargs: 其他参数
            
        Returns:
            生成的图片路径
        """
        # 确保模型已初始化
        if not self.initialize_model():
            raise RuntimeError("模型初始化失败")
        
        # 设置输出路径
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output_path = Path(f"generated_image_{timestamp}.png")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 根据模型类型生成图片
        if self.model_type == "placeholder":
            return self._generate_placeholder(prompt, output_path)
        elif self.model_type == "flux":
            return self._generate_with_flux(prompt, reference_image, output_path, **kwargs)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

    def _generate_placeholder(self, prompt: str, output_path: Path) -> Path:
        """生成占位图片"""
        try:
            img = Image.new('RGB', (832, 480), color='lightblue')
            
            # 添加文字
            try:
                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(img)
                
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except (OSError, IOError):
                    font = ImageFont.load_default()
                
                # 绘制提示词
                text_lines = [
                    "占位图片 / Placeholder Image",
                    f"提示词: {prompt[:30]}...",
                    f"输出: {output_path.name}",
                    f"时间: {datetime.now().strftime('%H:%M:%S')}"
                ]
                
                y_offset = 10
                for line in text_lines:
                    draw.text((10, y_offset), line, fill='black', font=font)
                    y_offset += 30
                    
            except ImportError:
                logger.warning("PIL ImageDraw不可用，生成空白占位图")
            
            img.save(output_path)
            logger.info(f"占位图片已保存: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"生成占位图片失败: {e}")
            raise

    def _generate_with_flux(self, prompt: str, reference_image: str, 
                           output_path: Path, **kwargs) -> Path:
        """使用Flux模型生成图片"""
        if not self.is_initialized:
            raise RuntimeError("Flux模型未初始化")
        
        try:
            # 处理参考图片
            ref_image = None
            if reference_image and os.path.exists(reference_image):
                try:
                    ref_image = Image.open(reference_image).convert('RGB')
                    logger.info(f"成功加载参考图片: {reference_image}")
                except Exception as e:
                    logger.warning(f"加载参考图片失败: {e}")
                    ref_image = None
            else:
                logger.warning(f"参考图片不存在或未提供: {reference_image}")
            
            # 生成参数
            num_inference_steps = kwargs.get("steps", 28)
            guidance_scale = kwargs.get("guidance_scale", 3.5)
            subject_scale = kwargs.get("subject_scale", 0.9)
            seed = kwargs.get("seed", random.randint(1000, 999999))
            
            # 设置随机种子
            device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=device).manual_seed(seed)
            
            logger.info(f"开始生成图片: steps={num_inference_steps}, guidance_scale={guidance_scale}, seed={seed}")
            
            # 生成图片
            if ref_image:
                image = self.pipe(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    subject_scale=subject_scale,
                    subject_image=ref_image,
                    generator=generator,
                ).images[0]
            else:
                image = self.pipe(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                ).images[0]
            
            # 保存图片
            image.save(output_path)
            logger.info(f"图片生成成功: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Flux图片生成失败: {e}")
            raise

    def generate_batch(self, prompts: List[str], reference_image: str = None,
                      output_dir: Path = None, **kwargs) -> List[Path]:
        """
        批量生成图片
        
        Args:
            prompts: 提示词列表
            reference_image: 参考图片路径
            output_dir: 输出目录
            **kwargs: 其他参数
            
        Returns:
            生成的图片路径列表
        """
        if output_dir is None:
            output_dir = Path("generated_images")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        generated_paths = []
        
        for i, prompt in enumerate(prompts):
            try:
                # 生成唯一的种子
                base_seed = kwargs.get("seed", 123456)
                current_seed = base_seed + i
                
                # 设置输出文件名
                filename = f"scene{i+1:02d}.png"
                output_path = output_dir / filename
                
                # 更新参数
                current_kwargs = kwargs.copy()
                current_kwargs["seed"] = current_seed
                
                logger.info(f"生成第 {i+1}/{len(prompts)} 张图片: {filename}")
                
                # 生成图片
                generated_path = self.generate(
                    prompt=prompt,
                    reference_image=reference_image,
                    output_path=output_path,
                    **current_kwargs
                )
                
                if generated_path and generated_path.exists():
                    generated_paths.append(generated_path)
                    logger.info(f"✅ 第 {i+1} 张图片生成成功")
                else:
                    logger.error(f"❌ 第 {i+1} 张图片生成失败")
                    
            except Exception as e:
                logger.error(f"生成第 {i+1} 张图片时出错: {e}")
                continue
        
        logger.info(f"批量生成完成: {len(generated_paths)}/{len(prompts)} 张图片成功")
        return generated_paths

    def cleanup(self):
        """清理资源"""
        if self.pipe is not None:
            logger.info("清理模型资源")
            del self.pipe
            self.pipe = None
            self.is_initialized = False
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("清理GPU缓存")

    def __del__(self):
        """析构函数"""
        self.cleanup()

# 测试代码
if __name__ == "__main__":
    # 测试占位符模式
    generator = ImageGenerator(model_type="placeholder")
    
    test_prompt = "A beautiful anime girl in a spring garden"
    output_path = Path("test_output.png")
    
    try:
        result = generator.generate(test_prompt, output_path=output_path)
        print(f"测试生成成功: {result}")
    except Exception as e:
        print(f"测试失败: {e}")
    
    # 批量测试
    test_prompts = [
        "Anime style, a girl walking in the park",
        "Anime style, a boy reading under a tree",
        "Anime style, children playing in the playground"
    ]
    
    try:
        results = generator.generate_batch(test_prompts, output_dir=Path("test_batch"))
        print(f"批量测试完成: {len(results)} 张图片")
    except Exception as e:
        print(f"批量测试失败: {e}")

