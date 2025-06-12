import json
import os
from openai import OpenAI
from typing import Dict, List, Any
import re
from datetime import datetime
from pathlib import Path

class LLMClient:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = "deepseek-v3-250324"):
        """初始化LLM客户端"""
        self.model = model

        # 从环境变量或参数获取配置
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        base_url = base_url or os.getenv("OPENAI_BASE_URL")

        if not api_key:
            raise ValueError("请设置OPENAI_API_KEY环境变量或传入api_key参数")

        client_args = {"api_key": api_key}
        if base_url:
            client_args["base_url"] = base_url

        self.client = OpenAI(**client_args)

    def generate_video_content(self, search_query: str, output_base_dir: str = "output") -> Dict[str, Any]:
        """
        生成视频相关的所有文本内容，并保存到指定文件夹。

        Args:
            search_query (str): 搜索词。
            output_base_dir (str): 基础输出目录，所有生成内容将保存到此目录下的子文件夹中。

        Returns:
            Dict[str, Any]: 包含生成结果的字典，包括成功状态、数据或错误信息。
        """
        try:
            # 构建prompt
            prompt = self._build_content_generation_prompt(search_query)

            # 调用API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4196,
                temperature=0.7
            )

            content = response.choices[0].message.content

            # 解析返回的内容
            parsed_content = self._parse_generated_content(content)

            # --- 保存生成内容到文件 ---
            # 清理搜索词，用于文件夹命名，避免非法字符
            safe_search_query = re.sub(r'[\\/:*?"<>|]', '', search_query)[:50].strip()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_folder_name = f"{safe_search_query}_{timestamp}"

            # 使用Pathlib创建目录
            output_dir = Path(output_base_dir) / output_folder_name
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Generated content will be saved to: {output_dir}")

            # 保存图片生成提示词
            with open(output_dir / "img2img_prompts.txt", "w", encoding="utf-8") as f:
                for p in parsed_content["img2img_prompts"]:
                    f.write(f"{p}\n")
            print("  - img2img_prompts.txt saved.")

            # 保存视频生成提示词
            with open(output_dir / "img2vid_prompts.txt", "w", encoding="utf-8") as f:
                for p in parsed_content["img2vid_prompts"]:
                    f.write(f"{p}\n")
            print("  - img2vid_prompts.txt saved.")

            # 保存旁白文本
            with open(output_dir / "narrations.txt", "w", encoding="utf-8") as f:
                for n in parsed_content["narrations"]:
                    f.write(f"{n}\n")
            print("  - narrations.txt saved.")

            # 保存业务点
            with open(output_dir / "business_points.txt", "w", encoding="utf-8") as f:
                f.write(parsed_content["business_points"])
            print("  - business_points.txt saved.")

            # 保存服务概述
            with open(output_dir / "service_overview.txt", "w", encoding="utf-8") as f:
                f.write(parsed_content["service_overview"])
            print("  - service_overview.txt saved.")

            # 保存完整的JSON输出
            with open(output_dir / "full_output.json", "w", encoding="utf-8") as f:
                json.dump(parsed_content, f, ensure_ascii=False, indent=2)
            print("  - full_output.json saved.")

            # 将生成的文件夹路径添加到返回数据中
            parsed_content["output_folder"] = str(output_dir)

            return {
                "success": True,
                "data": parsed_content
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _build_content_generation_prompt(self, search_query: str) -> str:
        """构建内容生成的prompt - 基于AI视频生成提示词配方优化"""
        style_prefix = "Anime style, high quality, consistent character design, "

        return f"""
请根据搜索词"{search_query}"，生成一个教育类短视频的完整内容方案。请严格按照以下AI视频生成提示词配方要求：

## 提示词配方要求：

**高级公式**: Prompt = 主体（主体描述）+ 场景（场景描述）+ 动作（动作描述）+ 镜头语言 + 氛围 + 风格

**镜头运动公式**: Prompt = 镜头运动描述 + 主体（主体描述）+ 场景（场景描述）+ 动作（动作描述）+ 镜头语言 + 氛围 + 风格

## 生成内容要求：

### 1. **图片生成提示词**（10个场景）
- 每个提示词用英文编写，**长度应在40-60个单词之间**
- **必须以"{style_prefix}"开头，以确保动漫风格和一致性**
- 包含详细的主体描述（外观、服装、表情等）
- 包含丰富的场景描述（环境、背景、前景、光线等）
- 包含具体的动作描述（姿态、表情、互动等）
- 包含镜头语言（拍摄角度、构图、景深等）
- 包含氛围描述（情绪、感觉、气氛等）
- 确保视觉风格一致性和高质量渲染效果

### 2. **视频生成提示词**（10个场景）
- 每个提示词用英文编写，**长度应在50-80个单词之间**
- **必须以"{style_prefix}"开头，以确保动漫风格和一致性**
- **必须包含具体的镜头运动描述**（如：slow push-in shot, tracking shot, dolly movement, pan, tilt, zoom等）
- 基于对应的图片场景，描述5秒内的动态变化
- 包含人物的具体动作变化（面部表情、肢体动作、姿态转换等）
- 包含环境的动态元素（光线变化、物体移动、背景变化等）
- 包含镜头运动的时间控制（每个镜头运动持续时间控制在5秒内）
- 确保动作的自然性和流畅性

### 3. **旁白文本**（10段）
- 每段不超过15个中文字
- 与对应场景内容匹配
- 语言简洁有力，适合语音合成
- 具有教育价值和感染力

### 4. **视频业务点**
- 3-5个核心业务价值点
- 中文描述，突出教育价值

### 5. **服务概述**
- 100字以内的中文描述
- 说明视频的整体价值和目标受众

## 示例参考：

**图片提示词示例**:
"{style_prefix}A dedicated young professional with focused amber eyes and neat business attire, sitting at a modern ergonomic workspace in a bright open-plan office environment. Multiple monitors display colorful data visualizations and code interfaces, while a steaming coffee cup and small succulent plant add personal touches to the organized desk. Soft natural lighting filters through large windows, creating a productive yet comfortable atmosphere with contemporary corporate design and warm professional energy."

**视频提示词示例**:
"{style_prefix}Slow circular dolly shot starting from behind the professional's chair, gradually moving around to capture their concentrated typing motion and facial expressions. The camera reveals changing data on multiple monitors in real-time while steam gently rises from the coffee cup. The professional occasionally pauses to adjust their posture, sip coffee, or review documents with precise hand movements. Background colleagues work quietly as natural lighting subtly shifts across the workspace, maintaining consistent productive office atmosphere throughout the 5-second sequence."

请严格按照以下JSON格式返回：
```json
{{
    "img2img_prompts": [
        "{style_prefix}详细的场景描述...",
        "{style_prefix}详细的场景描述...",
        ...
    ],
    "img2vid_prompts": [
        "{style_prefix}镜头运动描述 + 详细的动态场景描述...",
        "{style_prefix}镜头运动描述 + 详细的动态场景描述...",
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

注意：请确保所有提示词都遵循AI视频生成的最佳实践，包括镜头运动的专业性、视觉细节的丰富性、以及动作描述的精确性。
"""

    def _parse_generated_content(self, content: str) -> Dict[str, Any]:
        """解析生成的内容"""
        try:
            # 尝试提取JSON
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 尝试直接解析
                json_str = content

            data = json.loads(json_str)

            # 验证数据结构
            required_fields = ["img2img_prompts", "img2vid_prompts", "narrations",
                             "business_points", "service_overview"]

            for field in required_fields:
                if field not in data:
                    raise ValueError(f"缺少必需字段: {field}")

            # 确保列表长度为10
            for field in ["img2img_prompts", "img2vid_prompts", "narrations"]:
                if len(data[field]) != 10:
                    print(f"Warning: {field} has {len(data[field])} elements, expected 10. Attempting to adjust.")
                    # Adjust length to 10 if not already
                    while len(data[field]) < 10:
                        data[field].append(f"Default {field.replace('_', ' ')} item {len(data[field])+1}.")
                    data[field] = data[field][:10]
            return data

        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}. Attempting manual parse.")
            return self._manual_parse_content(content)
        except Exception as e:
            raise ValueError(f"内容解析失败: {str(e)}")

    def _manual_parse_content(self, content: str) -> Dict[str, Any]:
        """手动解析内容（备用方案）"""
        result = {
            "img2img_prompts": [],
            "img2vid_prompts": [],
            "narrations": [],
            "business_points": "",
            "service_overview": ""
        }

        lines = content.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if "图片生成提示词" in line or "img2img" in line.lower():
                current_section = "img2img_prompts"
            elif "视频生成提示词" in line or "img2vid" in line.lower():
                current_section = "img2vid_prompts"
            elif "旁白" in line or "narration" in line.lower():
                current_section = "narrations"
            elif "业务点" in line:
                current_section = "business_points"
            elif "服务概述" in line:
                current_section = "service_overview"
            elif current_section and line:
                if current_section in ["img2img_prompts", "img2vid_prompts", "narrations"]:
                    # 移除可能的序号
                    clean_line = re.sub(r'^\d+[\.\)]\s*', '', line)
                    if clean_line and len(result[current_section]) < 10:
                        result[current_section].append(clean_line)
                else:
                    result[current_section] += line + " "

        # Ensure lists have exactly 10 elements by padding or truncating
        for field in ["img2img_prompts", "img2vid_prompts", "narrations"]:
            while len(result[field]) < 10:
                result[field].append(f"Default {field.replace('_', ' ')} item {len(result[field])+1}.")
            result[field] = result[field][:10]

        result["business_points"] = result["business_points"].strip()
        result["service_overview"] = result["service_overview"].strip()

        return result


if __name__ == "__main__":
    client = LLMClient()
    # Using "如何学习Python编程" as an example search query
    search_query_example = "打工人的一天"
    output_result = client.generate_video_content(search_query_example, output_base_dir="generated_video_content")

    if output_result["success"]:
        print("\n--- LLM Content Generation Successful ---")
        print(f"All content saved to: {output_result['data']['output_folder']}")
        print(f"First img2img prompt: {output_result['data']['img2img_prompts'][0]}")
        print(f"First img2vid prompt: {output_result['data']['img2vid_prompts'][0]}")
        print(f"First narration: {output_result['data']['narrations'][0]}")
        print(f"Service Overview: {output_result['data']['service_overview']}")
    else:
        print("\n--- LLM Content Generation Failed ---")
        print(f"Error: {output_result['error']}")

    # Example of another query
    # search_query_example_2 = "健身的好处"
    # output_result_2 = client.generate_video_content(search_query_example_2, output_base_dir="generated_video_content")
    # if output_result_2["success"]:
    #     print(f"\nAnother content set saved to: {output_result_2['data']['output_folder']}")

