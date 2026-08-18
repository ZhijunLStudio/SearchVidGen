import os
import base64
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from PIL import Image
from io import BytesIO
from openai import OpenAI
import time
import concurrent.futures

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VLMValidator:
    """视觉语言模型验证器"""
    
    def __init__(self, api_key: str, base_url: str = None, model: str = "gpt-4o-mini"):
        """
        初始化VLM验证器
        
        Args:
            api_key: OpenAI API密钥
            base_url: API基础URL
            model: 使用的模型名称
        """
        self.model = model
        
        client_args = {"api_key": api_key}
        if base_url:
            client_args["base_url"] = base_url
            
        self.client = OpenAI(**client_args)
        
        # 加载验证提示词配置
        self.load_validation_prompts()
        
        logger.info(f"VLM验证器初始化完成，模型: {model}")

    def load_validation_prompts(self):
        """加载验证提示词配置"""
        try:
            config_path = Path("config/prompts.json")
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.validation_prompts = config.get("validation_prompts", {})
            else:
                # 默认验证提示词
                self.validation_prompts = {
                    "image_content_analysis": """
请仔细观察这张图片，详细描述以下内容：

1. **主要人物或角色**：
   - 外观特征（发型、服装、体型等）
   - 面部表情和情感状态
   - 姿势和动作

2. **场景环境**：
   - 背景设置和环境氛围
   - 重要的道具或物体
   - 光线和色彩基调

3. **动作和行为**：
   - 角色正在进行的活动
   - 动作的动态感和自然度
   - 与环境的互动

4. **构图和视觉效果**：
   - 整体构图和视角
   - 画面的平衡感和焦点
   - 艺术风格和质量

5. **情感表达和故事感**：
   - 传达的情绪和氛围
   - 是否具有故事性
   - 观众的视觉体验

请用中文详细描述，每个方面都要具体分析。
""",

                    "prompt_consistency_check": """
请将这张图片与以下提示词进行对比分析：

**原始提示词：** {prompt}

请评估以下几个方面：

1. **内容一致性**：
   - 图片内容与提示词描述的匹配程度（1-10分）
   - 哪些关键元素体现得很好？
   - 哪些重要元素缺失或不准确？

2. **角色一致性**：
   - 角色外观是否符合预期？
   - 角色表情和姿态是否合适？
   - 与参考图的一致性如何？

3. **场景还原度**：
   - 背景环境是否符合描述？
   - 氛围和光线效果如何？
   - 道具和细节是否到位？

4. **艺术质量**：
   - 整体画面质量评分（1-10分）
   - 构图和色彩搭配
   - 是否达到专业标准？

5. **改进建议**：
   - 如果重新生成，应该强调哪些关键词？
   - 哪些描述需要更加具体？
   - 是否建议调整生成参数？

请用中文回答，并给出具体的评分和建议。
""",

                    "batch_validation_summary": """
你已经分析了一系列场景图片，请提供整体的验证总结：

1. **整体质量评估**：
   - 所有图片的平均质量水平
   - 最佳和最差的场景分析
   - 整体风格一致性如何？

2. **角色一致性分析**：
   - 角色在不同场景中的一致性
   - 哪些场景的角色表现最好？
   - 需要重点改进的角色方面

3. **故事连贯性**：
   - 场景之间的逻辑连接
   - 时间发展的合理性
   - 情节推进的流畅度

4. **技术问题总结**：
   - 常见的生成问题
   - 需要优化的技术参数
   - 提示词改进方向

5. **优化建议**：
   - 整体项目的改进方案
   - 优先级最高的修改建议
   - 下一步行动计划

请提供专业、详细的分析报告。
"""
                }
        except Exception as e:
            logger.warning(f"加载验证提示词配置失败: {e}")
            # 使用默认配置
            pass

    def encode_image_to_base64(self, image_path: Path) -> str:
        """将图片编码为base64"""
        try:
            with Image.open(image_path) as img:
                # 转换为RGB格式
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 压缩图片以减少API调用成本
                max_size = (1024, 1024)
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # 编码为base64
                buffer = BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                image_bytes = buffer.getvalue()
                
                return base64.b64encode(image_bytes).decode('utf-8')
        except Exception as e:
            logger.error(f"图片编码失败 {image_path}: {e}")
            return None

    def validate_single_image(self, image_path: Path, prompt: str = None, 
                            validation_type: str = "content_analysis") -> Dict[str, Any]:
        """
        验证单张图片
        
        Args:
            image_path: 图片路径
            prompt: 原始生成提示词
            validation_type: 验证类型 ("content_analysis" 或 "prompt_consistency")
            
        Returns:
            验证结果字典
        """
        try:
            # 编码图片
            base64_image = self.encode_image_to_base64(image_path)
            if not base64_image:
                return {
                    "success": False,
                    "error": "图片编码失败",
                    "image_path": str(image_path)
                }
            
            # 选择验证提示词
            if validation_type == "content_analysis":
                system_prompt = self.validation_prompts["image_content_analysis"]
            elif validation_type == "prompt_consistency" and prompt:
                system_prompt = self.validation_prompts["prompt_consistency_check"].format(prompt=prompt)
            else:
                return {
                    "success": False,
                    "error": "无效的验证类型或缺少提示词",
                    "image_path": str(image_path)
                }
            
            # 构建API请求
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]
            
            # 调用API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1500,
                temperature=0.3
            )
            
            analysis = response.choices[0].message.content
            
            return {
                "success": True,
                "image_path": str(image_path),
                "analysis": analysis,
                "validation_type": validation_type,
                "prompt": prompt if prompt else None,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            logger.error(f"验证图片失败 {image_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "image_path": str(image_path)
            }

    def validate_batch_images(self, project_dir: Path, max_workers: int = 3) -> Dict[str, Any]:
        """
        批量验证项目中的所有图片
        
        Args:
            project_dir: 项目目录路径
            max_workers: 并发工作线程数
            
        Returns:
            批量验证结果
        """
        try:
            # 检查项目目录
            images_dir = project_dir / "generated_images"
            if not images_dir.exists():
                return {
                    "success": False,
                    "error": "找不到图片目录",
                    "project_dir": str(project_dir)
                }
            
            # 读取提示词文件
            prompts_file = project_dir / "img2img_prompts.txt"
            prompts = []
            if prompts_file.exists():
                with open(prompts_file, 'r', encoding='utf-8') as f:
                    prompts = [line.strip() for line in f if line.strip()]
            
            # 获取所有图片文件
            image_files = sorted([f for f in images_dir.glob("*.png") if f.is_file()])
            
            if not image_files:
                return {
                    "success": False,
                    "error": "未找到图片文件",
                    "project_dir": str(project_dir)
                }
            
            logger.info(f"开始批量验证 {len(image_files)} 张图片")
            
            # 并发处理图片验证
            validation_results = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有验证任务
                future_to_image = {}
                
                for i, image_path in enumerate(image_files):
                    # 获取对应的提示词
                    prompt = prompts[i] if i < len(prompts) else None
                    
                    # 提交验证任务
                    future = executor.submit(
                        self.validate_single_image,
                        image_path,
                        prompt,
                        "prompt_consistency" if prompt else "content_analysis"
                    )
                    future_to_image[future] = (image_path, prompt)
                
                # 收集结果
                for future in concurrent.futures.as_completed(future_to_image):
                    image_path, prompt = future_to_image[future]
                    try:
                        result = future.result()
                        validation_results.append(result)
                        
                        if result["success"]:
                            logger.info(f"✅ 验证完成: {image_path.name}")
                        else:
                            logger.error(f"❌ 验证失败: {image_path.name} - {result.get('error', 'Unknown error')}")
                            
                    except Exception as e:
                        logger.error(f"验证任务异常 {image_path}: {e}")
                        validation_results.append({
                            "success": False,
                            "error": str(e),
                            "image_path": str(image_path)
                        })
            
            # 生成总结报告
            summary = self._generate_validation_summary(validation_results)
            
            # 保存验证结果
            self._save_validation_results(project_dir, validation_results, summary)
            
            return {
                "success": True,
                "project_dir": str(project_dir),
                "total_images": len(image_files),
                "successful_validations": len([r for r in validation_results if r["success"]]),
                "validation_results": validation_results,
                "summary": summary,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            logger.error(f"批量验证失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "project_dir": str(project_dir)
            }

    def _generate_validation_summary(self, validation_results: List[Dict]) -> str:
        """生成验证总结报告"""
        successful_results = [r for r in validation_results if r["success"]]
        failed_results = [r for r in validation_results if not r["success"]]
        
        summary_lines = [
            "# 🔍 图片验证总结报告",
            f"📊 **验证统计**: 总计 {len(validation_results)} 张图片",
            f"✅ **成功验证**: {len(successful_results)} 张",
            f"❌ **验证失败**: {len(failed_results)} 张",
            "",
        ]
        
        if successful_results:
            summary_lines.extend([
                "## 📋 验证结果详情",
                ""
            ])
            
            for i, result in enumerate(successful_results, 1):
                image_name = Path(result["image_path"]).name
                summary_lines.extend([
                    f"### {i}. {image_name}",
                    f"**验证时间**: {result.get('timestamp', 'Unknown')}",
                    f"**分析结果**:",
                    result["analysis"][:200] + "..." if len(result["analysis"]) > 200 else result["analysis"],
                    ""
                ])
        
        if failed_results:
            summary_lines.extend([
                "## ❌ 验证失败的图片",
                ""
            ])
            
            for result in failed_results:
                image_name = Path(result["image_path"]).name
                summary_lines.append(f"- **{image_name}**: {result.get('error', 'Unknown error')}")
        
        summary_lines.extend([
            "",
            "## 💡 改进建议",
            "",
            "1. 对于验证失败的图片，建议检查图片文件完整性",
            "2. 根据分析结果优化提示词的具体描述",
            "3. 考虑调整生成参数以提高图片质量",
            "4. 重点关注角色一致性和场景连贯性",
            "",
            f"📅 **报告生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        ])
        
        return "\n".join(summary_lines)

    def _save_validation_results(self, project_dir: Path, validation_results: List[Dict], summary: str):
        """保存验证结果到文件"""
        try:
            # 保存详细结果JSON
            results_file = project_dir / "validation_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(validation_results, f, ensure_ascii=False, indent=2)
            
            # 保存总结报告
            summary_file = project_dir / "validation_summary.md"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(summary)
            
            # 保存简化的文本报告
            simple_report_file = project_dir / "validation_report.txt"
            with open(simple_report_file, 'w', encoding='utf-8') as f:
                successful_count = len([r for r in validation_results if r["success"]])
                f.write(f"验证完成: {successful_count}/{len(validation_results)} 张图片验证成功\n\n")
                
                for i, result in enumerate(validation_results, 1):
                    if result["success"]:
                        image_name = Path(result["image_path"]).name
                        f.write(f"{i}. {image_name}: ✅ 验证通过\n")
                        # 提取关键信息
                        analysis = result["analysis"]
                        if "评分" in analysis or "分" in analysis:
                            score_lines = [line for line in analysis.split('\n') if '分' in line][:2]
                            for line in score_lines:
                                f.write(f"   {line.strip()}\n")
                    else:
                        image_name = Path(result["image_path"]).name
                        f.write(f"{i}. {image_name}: ❌ 验证失败 - {result.get('error', 'Unknown')}\n")
                    f.write("\n")
            
            logger.info(f"验证结果已保存到: {project_dir}")
            
        except Exception as e:
            logger.error(f"保存验证结果失败: {e}")

    def get_validation_report(self, project_dir: Path) -> str:
        """获取验证报告文本"""
        try:
            summary_file = project_dir / "validation_summary.md"
            if summary_file.exists():
                with open(summary_file, 'r', encoding='utf-8') as f:
                    return f.read()
            
            # 如果markdown文件不存在，尝试读取简化报告
            simple_report_file = project_dir / "validation_report.txt"
            if simple_report_file.exists():
                with open(simple_report_file, 'r', encoding='utf-8') as f:
                    return f.read()
            
            return "未找到验证报告文件"
            
        except Exception as e:
            logger.error(f"读取验证报告失败: {e}")
            return f"读取验证报告失败: {str(e)}"

# 测试代码
if __name__ == "__main__":
    # 这里需要有效的API密钥进行测试
    test_api_key = "your-api-key-here"
    test_base_url = "https://api.openai.com/v1"
    
    if test_api_key != "your-api-key-here":
        validator = VLMValidator(test_api_key, test_base_url)
        
        # 创建测试图片路径
        test_project_dir = Path("test_project")
        test_project_dir.mkdir(exist_ok=True)
        
        # 测试批量验证
        try:
            result = validator.validate_batch_images(test_project_dir)
            if result["success"]:
                print(f"批量验证成功: {result['successful_validations']}/{result['total_images']}")
                print("\n验证报告:")
                print(validator.get_validation_report(test_project_dir))
            else:
                print(f"批量验证失败: {result['error']}")
        except Exception as e:
            print(f"测试失败: {e}")
    else:
        print("请设置有效的API密钥进行测试")
