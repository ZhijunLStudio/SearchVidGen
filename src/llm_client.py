import json
import os
from openai import OpenAI
from typing import Dict, List, Any
import re
from datetime import datetime
from pathlib import Path # Use pathlib for robust path handling

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
            safe_search_query = re.sub(r'[\\/:*?"<>|]', '', search_query)[:50].strip() # Limit length for path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_folder_name = f"{safe_search_query}_{timestamp}"

            # 使用Pathlib创建目录
            output_dir = Path(output_base_dir) / output_folder_name
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Generated content will be saved to: {output_dir}")

            # 保存图片生成提示词 (No numbering)
            with open(output_dir / "img2img_prompts.txt", "w", encoding="utf-8") as f:
                for p in parsed_content["img2img_prompts"]:
                    f.write(f"{p}\n")
            print("  - img2img_prompts.txt saved.")

            # 保存视频生成提示词 (No numbering)
            with open(output_dir / "img2vid_prompts.txt", "w", encoding="utf-8") as f:
                for p in parsed_content["img2vid_prompts"]:
                    f.write(f"{p}\n")
            print("  - img2vid_prompts.txt saved.")

            # 保存旁白文本 (No numbering)
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
        """构建内容生成的prompt"""
        # Define the consistent style prefix
        style_prefix = "Anime style, high quality, consistent character design, " # Added "consistent character design" for better character consistency

        return f"""
            请根据搜索词"{search_query}"，生成一个教育类短视频的完整内容方案。

            要求生成以下内容：

            1. **图片生成提示词**（10个场景）
            - 每个提示词用英文编写，**长度应在20-40个单词之间**，提供丰富且具体的视觉细节。
            - 描述清晰的视觉场景，包含人物、环境、物品和光线等细节。
            - **所有图片提示词都必须以"{style_prefix}"开头，以确保动漫风格和一致性。**
            - 适合用于图片生成AI，强调高质量、电影感或特定艺术风格。
            - 保持视觉风格一致性。

            2. **视频生成提示词**（10个场景）
            - 每个提示词用英文编写，**长度应在30-60个单词之间**，详细描述动态效果和镜头运动。
            - 基于对应的图片场景，描述动态效果，包含人物动作、物体变化、镜头推拉摇移等动态元素。
            - **所有视频提示词都必须以"{style_prefix}"开头，以确保动漫风格和一致性。**
            - 时长约5秒的动作描述。

            3. **旁白文本**（10段）
            - 每段不超过15个中文字。
            - 与对应场景内容匹配。
            - 语言简洁有力。
            - 适合语音合成。

            4. **视频业务点**
            - 3-5个核心业务价值点。
            - 中文描述。
            - 突出教育价值。

            5. **服务概述**
            - 100字以内的中文描述。
            - 说明视频的整体价值和目标受众。

            请严格按照以下JSON格式返回：
            ```json
            {{
                "img2img_prompts": [
                    "{style_prefix}A cozy home office with a modern laptop displaying intricate Python code, bathed in warm afternoon natural light, a ceramic coffee cup steaming gently on a minimalist wooden desk next to a thriving potted plant.",
                    "{style_prefix}A young student sitting in a bright library, surrounded by stacks of programming books, focusing intently on a Python tutorial displayed on a tablet, with a notebook filled with handwritten notes beside them.",
                    ...
                ],
                "img2vid_prompts": [
                    "{style_prefix}A smooth time-lapse sequence showing a person entering a home office, turning on a laptop, hands beginning to type rapidly on the keyboard, and Python code scrolling quickly on the screen, followed by a gentle pan to the steaming coffee cup.",
                    "{style_prefix}A slow zoom into the student's face in the library, showing their concentration as they highlight a section in the Python tutorial, then a quick cut to their hand writing notes in the notebook.",
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
                    data[field] = data[field][:10] # Truncate if more than 10
            return data

        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}. Attempting manual parse.")
            # If JSON parsing fails, try manual parsing
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
                    if clean_line and len(result[current_section]) < 10: # Only add if not already 10
                        result[current_section].append(clean_line)
                else:
                    # For business points and service overview, just append line
                    result[current_section] += line + " "

        # Ensure lists have exactly 10 elements by padding or truncating
        for field in ["img2img_prompts", "img2vid_prompts", "narrations"]:
            while len(result[field]) < 10:
                result[field].append(f"Default {field.replace('_', ' ')} item {len(result[field])+1}.")
            result[field] = result[field][:10] # Truncate if more than 10

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
        # You can now access the parsed data directly from output_result['data']
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