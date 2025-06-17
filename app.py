import gradio as gr
import json
import os
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import subprocess
import sys
import tempfile
import threading
import time
from PIL import Image

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 导入自定义模块
try:
    from src.llm_client import LLMClient
    from src.image_generate_class import ImageGenerator
    from src.vlm_validator import VLMValidator
except ImportError as e:
    print(f"⚠️ 导入模块失败: {e}")
    print("请确保src目录下有相应的模块文件")

class VideoGenerationPipeline:
    def __init__(self):
        self.current_project_dir = None
        self.llm_client = None
        self.image_generator = None
        self.vlm_validator = None
        self.load_prompts_config()
        
        # 设置本地临时目录
        self.temp_dir = Path("./tmp")
        self.temp_dir.mkdir(exist_ok=True)
        
        # 图像生成状态管理
        self.image_generation_status = {}
        self.current_prompts = []
        self.current_images = []
        
    def load_prompts_config(self):
        """加载prompt配置"""
        config_path = Path("config/prompts.json")
        config_path.parent.mkdir(exist_ok=True)
        
        default_config = {
            "system_prompts": {
                "content_generation": "你是一位专业的短视频剧本家和AI绘画/视频提示词工程师。",
                "image_validation": "你是一位专业的图像分析师，请仔细观察图像内容。"
            },
            "validation_prompt": "请分析这张图片，描述图片中的主要内容、场景、人物动作和情感表达。判断是否符合预期的提示词要求。"
        }
        
        if not config_path.exists():
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, ensure_ascii=False, indent=2)
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.prompts_config = json.load(f)

    def generate_content(self, search_query: str, api_key: str, base_url: str, 
                        model_name: str, scene_count: int) -> Tuple[str, str, str, str, str, str]:
        """生成内容的五种结果"""
        try:
            if not all([search_query, api_key, model_name]):
                return "❌ 请填写完整的搜索词、API Key和模型名称", "", "", "", "", ""
            
            # 初始化LLM客户端
            self.llm_client = LLMClient(api_key=api_key, base_url=base_url, model=model_name)
            
            # 创建项目目录
            safe_search_query = "".join(c for c in search_query if c.isalnum() or c in (' ', '_', '-')).strip()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            project_name = f"{safe_search_query}_{timestamp}"
            self.current_project_dir = Path("generated_video_content") / project_name
            self.current_project_dir.mkdir(parents=True, exist_ok=True)
            
            # 修改LLM客户端以支持自定义场景数量
            result = self._generate_custom_content(search_query, scene_count)
            
            if result["success"]:
                data = result["data"]
                self.current_project_dir = Path(data["output_folder"])
                
                # 将列表转换为换行分隔的字符串
                img_prompts_text = "\n".join(data["img2img_prompts"])
                vid_prompts_text = "\n".join(data["img2vid_prompts"])
                narrations_text = "\n".join(data["narrations"])
                
                # 更新当前提示词列表
                self.current_prompts = data["img2img_prompts"]
                self.current_images = [None] * len(self.current_prompts)
                
                print(f"✅ 内容生成成功！")
                print(f"项目目录：{self.current_project_dir}")
                print(f"生成了 {len(self.current_prompts)} 个场景")
                
                return (
                    f"✅ 内容生成成功！\n项目目录：{self.current_project_dir}",
                    img_prompts_text,
                    vid_prompts_text,
                    narrations_text,
                    data["business_points"],
                    data["service_overview"]
                )
            else:
                return f"❌ 生成失败：{result['error']}", "", "", "", "", ""
                
        except Exception as e:
            return f"❌ 错误：{str(e)}", "", "", "", "", ""

    def _generate_custom_content(self, search_query: str, scene_count: int) -> Dict[str, Any]:
        """生成自定义场景数量的内容"""
        try:
            prompt = self._build_custom_prompt(search_query, scene_count)
            
            response = self.llm_client.client.chat.completions.create(
                model=self.llm_client.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4196,
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            parsed_content = self._parse_generated_content(content, scene_count)
            
            # 保存到项目目录
            self._save_content_to_project(parsed_content)
            parsed_content["output_folder"] = str(self.current_project_dir)
            
            return {"success": True, "data": parsed_content}
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _build_custom_prompt(self, search_query: str, scene_count: int) -> str:
        """构建自定义场景数量的prompt"""
        style_prefix = "Anime style, high quality, consistent character design, "
        
        return f"""
你是一位专业的短视频剧本家和AI绘画/视频提示词工程师。
请根据搜索词"{search_query}"，为我生成一个引人入胜的短视频的完整内容方案，共{scene_count}个场景。

## 核心叙事原则：
1. 叙事连贯性：所有{scene_count}个场景必须构成一个完整、连贯的故事
2. 时序一致性：场景必须遵循严格的时间顺序

## 生成内容要求：

### 1. 图片生成提示词（{scene_count}个场景）
- 每个提示词用英文编写，长度在40-60个单词之间
- 必须以"{style_prefix}"开头
- 包含详细的主体、场景、动作、镜头语言和氛围描述

### 2. 视频生成提示词（{scene_count}个场景）
- 每个提示词用英文编写，长度在50-80个单词之间
- 必须以"{style_prefix}"开头
- 必须包含具体的镜头运动描述

### 3. 旁白文本（{scene_count}段）
- 每段不超过15个中文字
- 语言简洁有力，富有感染力

### 4. 视频业务点
- 3-5个核心价值点或故事主旨
- 中文描述

### 5. 服务概述
- 100字以内的中文描述
- 说明视频的整体故事、价值和目标受众

请严格按照以下JSON格式返回：
```json
{{
    "img2img_prompts": [
        "{style_prefix}场景1...",
        "{style_prefix}场景2...",
        ...
    ],
    "img2vid_prompts": [
        "{style_prefix}镜头运动 + 场景1...",
        "{style_prefix}镜头运动 + 场景2...",
        ...
    ],
    "narrations": [
        "旁白1",
        "旁白2",
        ...
    ],
    "business_points": "业务点描述...",
    "service_overview": "服务概述..."
}}```
"""

    def _parse_generated_content(self, content: str, scene_count: int) -> Dict[str, Any]:
        """解析生成的内容"""
        try:
            import re
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = content
                
            data = json.loads(json_str)
            
            # 确保列表长度为指定场景数
            for field in ["img2img_prompts", "img2vid_prompts", "narrations"]:
                if field in data:
                    while len(data[field]) < scene_count:
                        data[field].append(f"Default {field} {len(data[field])+1}")
                    data[field] = data[field][:scene_count]
                else:
                    data[field] = [f"Default {field} {i+1}" for i in range(scene_count)]
            
            return data
            
        except Exception as e:
            # 返回默认数据
            return {
                "img2img_prompts": [f"Anime style scene {i+1}" for i in range(scene_count)],
                "img2vid_prompts": [f"Anime style video scene {i+1}" for i in range(scene_count)],
                "narrations": [f"旁白{i+1}" for i in range(scene_count)],
                "business_points": "默认业务点",
                "service_overview": "默认服务概述"
            }

    def _save_content_to_project(self, content: Dict[str, Any]):
        """保存内容到项目目录"""
        # 保存各种提示词文件
        with open(self.current_project_dir / "img2img_prompts.txt", "w", encoding="utf-8") as f:
            for prompt in content["img2img_prompts"]:
                f.write(f"{prompt}\n")
                
        with open(self.current_project_dir / "img2vid_prompts.txt", "w", encoding="utf-8") as f:
            for prompt in content["img2vid_prompts"]:
                f.write(f"{prompt}\n")
                
        with open(self.current_project_dir / "narrations.txt", "w", encoding="utf-8") as f:
            for narration in content["narrations"]:
                f.write(f"{narration}\n")
                
        with open(self.current_project_dir / "business_points.txt", "w", encoding="utf-8") as f:
            f.write(content["business_points"])
            
        with open(self.current_project_dir / "service_overview.txt", "w", encoding="utf-8") as f:
            f.write(content["service_overview"])
            
        with open(self.current_project_dir / "full_output.json", "w", encoding="utf-8") as f:
            json.dump(content, f, ensure_ascii=False, indent=2)

    def save_edited_content(self, img_prompts: str, vid_prompts: str, 
                           narrations: str, business_points: str, service_overview: str) -> str:
        """保存编辑后的内容"""
        try:
            if not self.current_project_dir:
                return "❌ 没有当前项目，请先生成内容"
            
            # 将字符串转换为列表
            img_list = [line.strip() for line in img_prompts.split('\n') if line.strip()]
            vid_list = [line.strip() for line in vid_prompts.split('\n') if line.strip()]
            nar_list = [line.strip() for line in narrations.split('\n') if line.strip()]
            
            content = {
                "img2img_prompts": img_list,
                "img2vid_prompts": vid_list,
                "narrations": nar_list,
                "business_points": business_points,
                "service_overview": service_overview
            }
            
            # 更新当前提示词
            self.current_prompts = img_list
            self.current_images = [None] * len(self.current_prompts)
            
            self._save_content_to_project(content)
            return f"✅ 内容已保存到：{self.current_project_dir}"
            
        except Exception as e:
            return f"❌ 保存失败：{str(e)}"

    def regenerate_content(self, search_query: str, api_key: str, base_url: str, 
                          model_name: str, scene_count: int, dissatisfaction: str) -> Tuple[str, str, str, str, str, str]:
        """重新生成内容"""
        try:
            if not dissatisfaction.strip():
                return self.generate_content(search_query, api_key, base_url, model_name, scene_count)
            
            # 在原有prompt基础上加入不满意的点
            modified_prompt = self._build_custom_prompt(search_query, scene_count) + f"\n\n请特别注意避免以下问题：{dissatisfaction}"
            
            self.llm_client = LLMClient(api_key=api_key, base_url=base_url, model=model_name)
            
            response = self.llm_client.client.chat.completions.create(
                model=self.llm_client.model,
                messages=[{"role": "user", "content": modified_prompt}],
                max_tokens=4196,
                temperature=0.8  # 稍微增加随机性
            )
            
            content = response.choices[0].message.content
            parsed_content = self._parse_generated_content(content, scene_count)
            
            # 覆盖保存到同一个项目目录
            if self.current_project_dir:
                self._save_content_to_project(parsed_content)
                
                # 更新当前提示词
                self.current_prompts = parsed_content["img2img_prompts"]
                self.current_images = [None] * len(self.current_prompts)
                
                # 将列表转换为换行分隔的字符串
                img_prompts_text = "\n".join(parsed_content["img2img_prompts"])
                vid_prompts_text = "\n".join(parsed_content["img2vid_prompts"])
                narrations_text = "\n".join(parsed_content["narrations"])
                
                return (
                    f"✅ 内容重新生成成功！\n已覆盖保存到：{self.current_project_dir}",
                    img_prompts_text,
                    vid_prompts_text,
                    narrations_text,
                    parsed_content["business_points"],
                    parsed_content["service_overview"]
                )
            else:
                return "❌ 没有当前项目目录", "", "", "", "", ""
                
        except Exception as e:
            return f"❌ 重新生成失败：{str(e)}", "", "", "", "", ""

    def initialize_image_generator(self, ip_adapter_path: str, base_model: str, 
                                  image_encoder_path: str, image_encoder_2_path: str, use_offload: bool):
        """初始化图片生成器"""
        try:
            # 尝试创建ImageGenerator实例
            try:
                self.image_generator = ImageGenerator(model_type="flux", use_offload=use_offload)
            except TypeError:
                try:
                    self.image_generator = ImageGenerator(model_type="flux")
                except:
                    self.image_generator = ImageGenerator()
            
            # 更新配置
            if hasattr(self.image_generator, 'update_config'):
                self.image_generator.update_config(
                    ip_adapter_path=ip_adapter_path,
                    base_model=base_model,
                    image_encoder_path=image_encoder_path,
                    image_encoder_2_path=image_encoder_2_path,
                    use_offload=use_offload
                )
            else:
                # 直接设置属性
                self.image_generator.ip_adapter_path = ip_adapter_path
                self.image_generator.base_model = base_model
                self.image_generator.image_encoder_path = image_encoder_path
                self.image_generator.image_encoder_2_path = image_encoder_2_path
                if hasattr(self.image_generator, 'use_offload'):
                    self.image_generator.use_offload = use_offload
            
            return True, "图片生成器初始化成功"
            
        except Exception as e:
            return False, f"图片生成器初始化失败: {str(e)}"

    def generate_single_image(self, slot_index: int, custom_prompt: str, reference_image, 
                             ip_adapter_path: str, base_model: str, image_encoder_path: str, 
                             image_encoder_2_path: str, use_offload: bool, steps: int, 
                             guidance_scale: float, subject_scale: float) -> Tuple[str, Any]:
        """生成单张图片"""
        try:
            if not self.current_project_dir:
                return "❌ 请先生成内容", None
            
            if reference_image is None:
                return "❌ 请上传角色一致性参考图", None
            
            # 使用自定义提示词或默认提示词
            if slot_index < len(self.current_prompts):
                prompt = custom_prompt.strip() if custom_prompt.strip() else self.current_prompts[slot_index]
            else:
                prompt = custom_prompt.strip() if custom_prompt.strip() else f"Anime style scene {slot_index + 1}"
            
            # 保存参考图到项目目录
            ref_image_path = self.current_project_dir / "reference_image.png"
            reference_image.save(ref_image_path)
            
            # 初始化图片生成器（如果未初始化）
            if not self.image_generator:
                success, message = self.initialize_image_generator(
                    ip_adapter_path, base_model, image_encoder_path, 
                    image_encoder_2_path, use_offload
                )
                if not success:
                    return f"❌ {message}", None
            
            # 创建图片输出目录
            images_dir = self.current_project_dir / "generated_images"
            images_dir.mkdir(exist_ok=True)
            
            # 生成单张图片
            output_path = images_dir / f"image_{slot_index:03d}.png"
            
            # 调用图片生成器
            try:
                generated_path = self.image_generator.generate(
                    prompt=prompt,
                    reference_image=str(ref_image_path),
                    output_path=str(output_path),
                    steps=steps,
                    guidance_scale=guidance_scale,
                    subject_scale=subject_scale,
                    seed=random.randint(1000, 999999)
                )
                    
            except TypeError:
                # 如果参数不对，尝试更简单的调用
                try:
                    generated_path = self.image_generator.generate(
                        prompt, str(ref_image_path), str(output_path)
                    )
                except Exception as e:
                    return f"❌ 图片生成调用失败: {str(e)}", None
            
            if generated_path and Path(generated_path).exists():
                # 读取生成的图片
                img = Image.open(generated_path)
                # 更新图片槽状态
                if slot_index < len(self.current_images):
                    self.current_images[slot_index] = str(generated_path)
                return f"✅ 重新生成第 {slot_index + 1} 张图片成功", img
            else:
                return f"❌ 重新生成第 {slot_index + 1} 张图片失败", None
                
        except Exception as e:
            return f"❌ 生成失败：{str(e)}", None

    def batch_generate_images_direct(self, reference_image, ip_adapter_path: str, base_model: str, 
                                    image_encoder_path: str, image_encoder_2_path: str, use_offload: bool, 
                                    steps: int, guidance_scale: float, subject_scale: float, 
                                    progress=gr.Progress()):
        """直接批量生成图片，返回PIL Image对象"""
        try:
            if not self.current_project_dir:
                return ["❌ 请先生成内容"] + [None] * 10
            
            if not self.current_prompts:
                return ["❌ 没有找到提示词，请先生成内容"] + [None] * 10
            
            if reference_image is None:
                return ["❌ 请上传角色一致性参考图"] + [None] * 10
            
            # 保存参考图
            ref_image_path = self.current_project_dir / "reference_image.png"
            reference_image.save(ref_image_path)
            
            # 初始化生成器
            progress(0, desc="🔧 正在初始化图片生成器...")
            success, message = self.initialize_image_generator(
                ip_adapter_path, base_model, image_encoder_path, 
                image_encoder_2_path, use_offload
            )
            if not success:
                return [f"❌ {message}"] + [None] * 10
            
            # 创建输出目录
            images_dir = self.current_project_dir / "generated_images"
            images_dir.mkdir(exist_ok=True)
            
            # 生成图片
            slot_images = [None] * 10
            generated_count = 0
            total_prompts = len(self.current_prompts)
            
            for i, prompt in enumerate(self.current_prompts):
                progress((i + 1) / total_prompts, desc=f"🎨 正在生成第 {i + 1}/{total_prompts} 张图片...")
                
                output_path = images_dir / f"image_{i:03d}.png"
                
                try:
                    generated_path = self.image_generator.generate(
                        prompt=prompt,
                        reference_image=str(ref_image_path),
                        output_path=str(output_path),
                        steps=steps,
                        guidance_scale=guidance_scale,
                        subject_scale=subject_scale,
                        seed=random.randint(1000, 999999)
                    )
                    
                    if generated_path and Path(generated_path).exists():
                        # 读取图片并转换为PIL Image对象
                        img = Image.open(generated_path)
                        
                        if i < 10:
                            slot_images[i] = img  # 直接传递PIL Image对象
                        
                        generated_count += 1
                        print(f"✅ 第 {i + 1} 张图片生成成功: {generated_path}")
                    else:
                        print(f"❌ 第 {i + 1} 张图片生成失败")
                        
                except Exception as e:
                    print(f"生成第 {i + 1} 张图片时出错: {e}")
                    continue
            
            progress(1.0, desc="🎉 图片生成完成！")
            status = f"🎉 图片生成完成！成功生成 {generated_count}/{total_prompts} 张图片"
            return [status] + slot_images
            
        except Exception as e:
            print(f"批量生成出错: {e}")
            import traceback
            traceback.print_exc()
            return [f"❌ 生成失败：{str(e)}"] + [None] * 10

    def validate_images_with_vlm(self, api_key: str, base_url: str, model_name: str, 
                                progress=gr.Progress()) -> Tuple[str, List[Dict]]:
        """使用视觉语言模型验证图片"""
        try:
            if not self.current_project_dir:
                return "❌ 没有当前项目，请先生成内容和图片", []
            
            images_dir = self.current_project_dir / "generated_images"
            if not images_dir.exists():
                return "❌ 找不到生成的图片，请先生成图片", []
            
            if not all([api_key, model_name]):
                return "❌ 请填写完整的VLM API配置", []
            
            progress(0.1, "正在初始化视觉语言模型...")
            
            # 初始化VLM验证器
            self.vlm_validator = VLMValidator(
                api_key=api_key,
                base_url=base_url,
                model=model_name
            )
            
            progress(0.2, "开始验证图片...")
            
            # 获取图片文件列表
            image_files = sorted(list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")))
            
            validation_results = []
            
            for i, image_path in enumerate(image_files):
                progress(0.2 + (i + 1) / len(image_files) * 0.7, f"正在验证第 {i + 1}/{len(image_files)} 张图片...")
                
                try:
                    # 获取对应的提示词
                    prompt = self.current_prompts[i] if i < len(self.current_prompts) else "Unknown prompt"
                    
                    # 验证单张图片
                    result = self.vlm_validator.validate_single_image(
                        image_path=str(image_path),
                        original_prompt=prompt
                    )
                    
                    if result["success"]:
                        validation_data = {
                            "index": i + 1,
                            "image_path": str(image_path),
                            "original_prompt": prompt,
                            "analysis": result["analysis"],
                            "score": result.get("score", 0),
                            "suggestions": result.get("suggestions", ""),
                            "compliance": result.get("compliance", "unknown"),
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                    else:
                        validation_data = {
                            "index": i + 1,
                            "image_path": str(image_path),
                            "original_prompt": prompt,
                            "analysis": f"验证失败: {result.get('error', '未知错误')}",
                            "score": 0,
                            "suggestions": "请重新生成图片",
                            "compliance": "error",
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                    
                    validation_results.append(validation_data)
                    
                except Exception as e:
                    validation_data = {
                        "index": i + 1,
                        "image_path": str(image_path),
                        "original_prompt": prompt,
                        "analysis": f"验证出错: {str(e)}",
                        "score": 0,
                        "suggestions": "请检查配置重新验证",
                        "compliance": "error",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    validation_results.append(validation_data)
            
            # 保存验证结果
            results_file = self.current_project_dir / "validation_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(validation_results, f, ensure_ascii=False, indent=2)
            
            progress(1.0, "验证完成！")
            
            # 生成状态报告
            successful_count = len([r for r in validation_results if r["compliance"] not in ["error", "unknown"]])
            status_report = f"✅ 图片验证完成！验证了 {len(validation_results)} 张图片，{successful_count} 张成功"
            
            return status_report, validation_results
                
        except Exception as e:
            return f"❌ 验证过程出错: {str(e)}", []

# 创建全局pipeline实例
pipeline = VideoGenerationPipeline()

def create_interface():
    """创建Gradio界面"""
    
    # 设置Gradio临时目录为当前目录下的tmp
    temp_dir = Path("./tmp")
    temp_dir.mkdir(exist_ok=True)
    os.environ["GRADIO_TEMP_DIR"] = str(temp_dir.absolute())
    
    with gr.Blocks(title="🎬 AI视频生成排版工具", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🎬 AI视频生成排版工具
        
        **完整的视频内容生成pipeline：搜索词 → 内容生成 → 图片生成 → 多模态验证**
        """)
        
        # 状态变量
        current_scene_count = gr.State(5)
        
        with gr.Tab("📝 内容生成"):
            gr.Markdown("### 🔧 基础配置")
            
            with gr.Row():
                with gr.Column(scale=2):
                    search_query = gr.Textbox(
                        label="搜索词",
                        placeholder="请输入主题，如：春天在哪里、卡皮巴拉的一天"
                    )
                    
                with gr.Column(scale=1):
                    scene_count = gr.Slider(
                        minimum=3,
                        maximum=10,
                        value=5,
                        step=1,
                        label="场景数量"
                    )
            
            gr.Markdown("### 🤖 语言模型配置")
            
            with gr.Row():
                llm_api_key = gr.Textbox(
                    label="API Key",
                    type="password",
                    placeholder="sk-..."
                )
                
                llm_base_url = gr.Textbox(
                    label="Base URL",
                    placeholder="https://api.openai.com/v1"
                )
                
                llm_model = gr.Textbox(
                    label="模型名称",
                    value="deepseek-v3-250324",
                    placeholder="gpt-4o-mini, deepseek-v3等"
                )
            
            with gr.Row():
                generate_btn = gr.Button("🚀 生成内容", variant="primary", size="lg")
                save_btn = gr.Button("💾 保存编辑", variant="secondary", size="lg")
            
            gr.Markdown("### 📊 生成结果")
            
            with gr.Row():
                generation_status = gr.Textbox(
                    label="生成状态",
                    lines=3,
                    interactive=False,
                    show_copy_button=True
                )
            
            # 五种结果的显示和编辑区域
            with gr.Tab("🖼️ 图片生成提示词"):
                img_prompts = gr.Textbox(
                    label="图片生成提示词（每行一个）",
                    lines=15,
                    placeholder="Anime style, high quality..."
                )
            
            with gr.Tab("🎥 视频生成提示词"):
                vid_prompts = gr.Textbox(
                    label="视频生成提示词（每行一个）",
                    lines=15,
                    placeholder="Anime style, slow push-in shot..."
                )
            
            with gr.Tab("🎙️ 旁白文本"):
                narrations = gr.Textbox(
                    label="旁白文本（每行一个）",
                    lines=10,
                    placeholder="旁白1\n旁白2\n..."
                )
            
            with gr.Tab("💼 业务点"):
                business_points = gr.Textbox(
                    label="视频业务点",
                    lines=5,
                    placeholder="3-5个核心价值点..."
                )
            
            with gr.Tab("📋 服务概述"):
                service_overview = gr.Textbox(
                    label="服务概述",
                    lines=5,
                    placeholder="100字以内的描述..."
                )
            
            # 重新生成区域
            gr.Markdown("### 🔄 重新生成")
            
            with gr.Row():
                dissatisfaction = gr.Textbox(
                    label="不满意的点",
                    placeholder="请描述当前结果的不满意之处，如：故事不够连贯、角色描述不够详细等"
                )
                
                regenerate_btn = gr.Button("🔄 重新生成", variant="secondary")
        
        with gr.Tab("🎨 图片生成"):
            gr.Markdown("### 🖼️ 角色一致性配置")
            
            with gr.Row():
                with gr.Column(scale=1):
                    reference_image = gr.Image(
                        label="角色一致性参考图 - 上传一张角色参考图，确保生成图片的角色一致性",
                        type="pil"
                    )
                    
                with gr.Column(scale=2):
                    gr.Markdown("### ⚙️ 模型配置")
                    
                    ip_adapter_path = gr.Textbox(
                        label="IP Adapter路径",
                        value="/data/home/lizhijun/llm/flux-hf/InstantCharacter-main/tencent/InstantCharacter/instantcharacter_ip-adapter.bin"
                    )
                    
                    base_model = gr.Textbox(
                        label="基础模型路径",
                        value="/data/home/lizhijun/llm/flux-hf/model/flux-dev"
                    )
                    
                    with gr.Row():
                        image_encoder_path = gr.Textbox(
                            label="图像编码器1",
                            value="google/siglip-so400m-patch14-384"
                        )
                        
                        image_encoder_2_path = gr.Textbox(
                            label="图像编码器2", 
                            value="facebook/dinov2-giant"
                        )
            
            gr.Markdown("### 🔧 生成参数")
            
            with gr.Row():
                use_offload = gr.Checkbox(
                    label="启用模型卸载 - 启用后使用CPU卸载节省显存，但速度较慢",
                    value=True
                )
                
                steps = gr.Slider(
                    minimum=10,
                    maximum=50,
                    value=28,
                    step=1,
                    label="推理步数 - 更多步数通常质量更好但速度更慢"
                )
            
            with gr.Row():
                guidance_scale = gr.Slider(
                    minimum=1.0,
                    maximum=10.0,
                    value=3.5,
                    step=0.1,
                    label="引导强度 - 控制生成图片对提示词的遵循程度"
                )
                
                subject_scale = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.9,
                    step=0.1,
                    label="角色强度 - 控制参考图片角色特征的保持程度"
                )
            
            gr.Markdown("### 🚀 图片生成")
            
            with gr.Row():
                batch_generate_btn = gr.Button("🎨 批量生成所有图片", variant="primary", size="lg")
            
            with gr.Row():
                image_generation_status = gr.Textbox(
                    label="图片生成状态",
                    lines=2,
                    interactive=False
                )
            
            # 动态图片生成槽位
            gr.Markdown("### 🖼️ 图片生成槽位")
            
            # 创建图片槽位组件
            image_slot_components = []
            for i in range(10):
                with gr.Row(visible=False) as slot_row:
                    with gr.Column(scale=3):
                        slot_prompt = gr.Textbox(
                            label=f"场景 {i+1} 提示词",
                            lines=2,
                            placeholder="可以修改提示词后重新生成单张图片"
                        )
                    
                    with gr.Column(scale=2):
                        slot_image = gr.Image(
                            label=f"场景 {i+1}",
                            type="pil",
                            height=200
                        )
                    
                    with gr.Column(scale=1):
                        slot_generate_btn = gr.Button(
                            f"🎨 重新生成第{i+1}张",
                            variant="secondary",
                            size="lg"  # 增大按钮
                        )
                        
                        slot_status = gr.Textbox(
                            label="状态",
                            lines=1,
                            interactive=False,
                            value="⏳ 等待生成"  # 默认状态
                        )
                
                image_slot_components.append({
                    "index": i,
                    "row": slot_row,
                    "prompt": slot_prompt,
                    "image": slot_image,
                    "button": slot_generate_btn,
                    "status": slot_status
                })
        
        with gr.Tab("🔍 多模态验证"):
            gr.Markdown("### 🤖 视觉语言模型配置")
            
            with gr.Row():
                vlm_api_key = gr.Textbox(
                    label="VLM API Key",
                    type="password",
                    placeholder="sk-..."
                )
                
                vlm_base_url = gr.Textbox(
                    label="VLM Base URL",
                    placeholder="https://api.openai.com/v1"
                )
                
                vlm_model = gr.Textbox(
                    label="VLM模型名称",
                    value="gpt-4o-mini",
                    placeholder="gpt-4o-mini, claude-3-sonnet等"
                )
            
            with gr.Row():
                validate_btn = gr.Button("🔍 开始验证", variant="primary", size="lg")
            
            validation_status = gr.Textbox(
                label="验证状态",
                lines=3,
                interactive=False
            )
            
            # 验证结果可视化
            gr.Markdown("### 📊 验证结果详情")
            
            validation_results_display = gr.JSON(
                label="详细验证结果",
                visible=True
            )
        
        with gr.Tab("📊 项目管理"):
            gr.Markdown("### 📁 当前项目状态")
            
            project_info = gr.Textbox(
                label="项目信息",
                lines=10,
                interactive=False
            )
            
            with gr.Row():
                refresh_project_btn = gr.Button("🔄 刷新项目信息", variant="secondary")
                export_project_btn = gr.Button("📦 导出项目", variant="primary")
        
        # 事件绑定
        def update_scene_count(count):
            current_scene_count.value = count
            return count
        
        def update_image_slots_visibility(count):
            """根据场景数量显示对应数量的图片槽位"""
            print(f"更新槽位可见性，场景数量: {count}")  # 调试输出
            updates = []
            for i in range(10):  # 只处理10个槽位
                if i < count:
                    updates.append(gr.update(visible=True))
                    print(f"槽位 {i+1}: 可见")  # 调试输出
                else:
                    updates.append(gr.update(visible=False))
                    print(f"槽位 {i+1}: 隐藏")  # 调试输出
            return updates
        
        def update_slots_with_prompts():
            """用生成的提示词更新槽位"""
            if not pipeline.current_prompts:
                return [gr.update() for _ in range(10)]
            
            updates = []
            for i in range(10):
                if i < len(pipeline.current_prompts):
                    updates.append(gr.update(value=pipeline.current_prompts[i]))
                else:
                    updates.append(gr.update())
            return updates
        
        def reset_slot_status():
            """重置所有槽位状态为等待生成"""
            status_updates = []
            for i in range(10):
                if i < len(pipeline.current_prompts):
                    status_updates.append(gr.update(value="⏳ 等待生成"))
                else:
                    status_updates.append(gr.update())
            return status_updates
        
        def update_slot_status_batch_complete():
            """批量生成完成后更新槽位状态"""
            status_updates = []
            for i in range(10):
                if i < len(pipeline.current_prompts):
                    status_updates.append(gr.update(value="✅ 生成完成"))
                else:
                    status_updates.append(gr.update())
            return status_updates
        
        scene_count.change(
            fn=update_scene_count, 
            inputs=[scene_count], 
            outputs=[current_scene_count]
        )
        
        # 生成内容
        generate_btn.click(
            fn=pipeline.generate_content,
            inputs=[search_query, llm_api_key, llm_base_url, llm_model, scene_count],
            outputs=[generation_status, img_prompts, vid_prompts, narrations, business_points, service_overview]
        ).then(
            fn=update_slots_with_prompts,
            outputs=[slot["prompt"] for slot in image_slot_components]
        ).then(
            fn=update_image_slots_visibility,
            inputs=[scene_count],
            outputs=[slot["row"] for slot in image_slot_components]
        ).then(
            fn=reset_slot_status,
            outputs=[slot["status"] for slot in image_slot_components]
        )
        
        # 保存编辑
        save_btn.click(
            fn=pipeline.save_edited_content,
            inputs=[img_prompts, vid_prompts, narrations, business_points, service_overview],
            outputs=[generation_status]
        ).then(
            fn=update_slots_with_prompts,
            outputs=[slot["prompt"] for slot in image_slot_components]
        ).then(
            fn=lambda prompts_text: len([p for p in prompts_text.split('\n') if p.strip()]),
            inputs=[img_prompts],
            outputs=[current_scene_count]
        ).then(
            fn=update_image_slots_visibility,
            inputs=[current_scene_count],
            outputs=[slot["row"] for slot in image_slot_components]
        ).then(
            fn=reset_slot_status,
            outputs=[slot["status"] for slot in image_slot_components]
        )
        
        # 重新生成
        regenerate_btn.click(
            fn=pipeline.regenerate_content,
            inputs=[search_query, llm_api_key, llm_base_url, llm_model, scene_count, dissatisfaction],
            outputs=[generation_status, img_prompts, vid_prompts, narrations, business_points, service_overview]
        ).then(
            fn=update_slots_with_prompts,
            outputs=[slot["prompt"] for slot in image_slot_components]
        ).then(
            fn=update_image_slots_visibility,
            inputs=[scene_count],
            outputs=[slot["row"] for slot in image_slot_components]
        ).then(
            fn=reset_slot_status,
            outputs=[slot["status"] for slot in image_slot_components]
        )
        
        # 批量生成图片
        batch_generate_btn.click(
            fn=pipeline.batch_generate_images_direct,
            inputs=[reference_image, ip_adapter_path, base_model, image_encoder_path, 
                   image_encoder_2_path, use_offload, steps, guidance_scale, subject_scale],
            outputs=[image_generation_status] + [slot["image"] for slot in image_slot_components],
            show_progress=True
        ).then(
            fn=update_slot_status_batch_complete,
            outputs=[slot["status"] for slot in image_slot_components]
        )
        
        # 单张图片生成
        def create_single_generate_function(slot_index):
            """为每个槽位创建单独的生成函数"""
            def single_generate(custom_prompt, ref_img, ip_path, base_mdl, enc_path1, enc_path2, 
                               offload, step_count, guide_scale, subj_scale):
                return pipeline.generate_single_image(
                    slot_index, custom_prompt, ref_img, ip_path, base_mdl, 
                    enc_path1, enc_path2, offload, step_count, guide_scale, subj_scale
                )
            return single_generate
        
        # 为每个槽位绑定单独的生成函数
        for slot in image_slot_components:
            slot_index = slot["index"]
            single_gen_func = create_single_generate_function(slot_index)
            
            slot["button"].click(
                fn=single_gen_func,
                inputs=[
                    slot["prompt"], reference_image, ip_adapter_path, base_model,
                    image_encoder_path, image_encoder_2_path, use_offload, steps,
                    guidance_scale, subject_scale
                ],
                outputs=[slot["status"], slot["image"]]
            )
        
        # 验证图片
        validate_btn.click(
            fn=pipeline.validate_images_with_vlm,
            inputs=[vlm_api_key, vlm_base_url, vlm_model],
            outputs=[validation_status, validation_results_display],
            show_progress=True
        )
        
        # 刷新项目信息
        def refresh_project_info():
            if pipeline.current_project_dir and pipeline.current_project_dir.exists():
                info_lines = [
                    f"📁 项目目录: {pipeline.current_project_dir}",
                    f"📅 创建时间: {datetime.fromtimestamp(pipeline.current_project_dir.stat().st_ctime)}",
                    "",
                    "📋 项目文件:",
                ]
                
                for file_path in sorted(pipeline.current_project_dir.glob("*")):
                    if file_path.is_file():
                        size = file_path.stat().st_size
                        info_lines.append(f"  📄 {file_path.name} ({size} bytes)")
                    elif file_path.is_dir():
                        file_count = len(list(file_path.glob("*")))
                        info_lines.append(f"  📁 {file_path.name}/ ({file_count} files)")
                
                return "\n".join(info_lines)
            else:
                return "❌ 没有当前项目"
        
        refresh_project_btn.click(
            fn=refresh_project_info,
            outputs=[project_info]
        )
        
        # 导出项目
        def export_project():
            if not pipeline.current_project_dir or not pipeline.current_project_dir.exists():
                return "❌ 没有可导出的项目"
            
            try:
                # 创建导出目录
                export_dir = Path("exports")
                export_dir.mkdir(exist_ok=True)
                
                # 创建zip文件
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_filename = export_dir / f"{pipeline.current_project_dir.name}_{timestamp}.zip"
                
                import zipfile
                with zipfile.ZipFile(export_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file_path in pipeline.current_project_dir.rglob('*'):
                        if file_path.is_file():
                            arcname = file_path.relative_to(pipeline.current_project_dir)
                            zipf.write(file_path, arcname)
                
                return f"✅ 项目已导出到: {export_filename}"
                
            except Exception as e:
                return f"❌ 导出失败: {str(e)}"
        
        export_project_btn.click(
            fn=export_project,
            outputs=[generation_status]
        )
    
    return app

if __name__ == "__main__":
    print("🚀 正在启动AI视频生成排版工具...")
    print("📍 端口: 7861")
    
    try:
        # 创建本地临时目录
        temp_dir = Path("./tmp")
        temp_dir.mkdir(exist_ok=True)
        print(f"✅ 临时目录设置: {temp_dir.absolute()}")
        
        app = create_interface()
        print("✅ 界面创建成功")
        
        # 检查端口是否可用
        import socket
        def is_port_available(port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('', port))
                    return True
                except socket.error:
                    return False
        
        port = 7861
        if not is_port_available(port):
            print(f"⚠️  端口 {port} 被占用，尝试其他端口...")
            for new_port in range(7862, 7870):
                if is_port_available(new_port):
                    port = new_port
                    print(f"✅ 使用端口: {port}")
                    break
            else:
                print("❌ 找不到可用端口")
                exit(1)
        
        print(f"🌐 启动服务器: http://192.168.99.119:{port}")
        
        # 使用最简单的启动方式避免权限问题
        app.launch(
            server_name="192.168.99.119",
            server_port=port,
            share=False
        )
        
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n🔧 故障排除建议：")
        print("1. 检查端口7861是否被占用")
        print("2. 确认src目录和相关模块存在")
        print("3. 检查Python依赖是否完整")
        print("4. 检查Gradio版本兼容性")
        print("5. 尝试重新安装Gradio: pip install --upgrade gradio")
