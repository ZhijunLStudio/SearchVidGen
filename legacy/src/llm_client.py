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
                max_tokens=4196, # 保持较大的token以容纳完整的10个场景
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
        """
        构建内容生成的prompt - 优化版，强调叙事和时序连贯性
        """
        style_prefix = "Anime style, high quality, consistent character design, "

        return f"""
你是一位专业的短视频剧本家和AI绘画/视频提示词工程师。
请根据搜索词"{search_query}"，为我生成一个引人入胜的短视频的完整内容方案。

## 核心叙事原则（必须严格遵守）：

1.  **叙事连贯性**: 所有10个场景必须构成一个完整、连贯的故事或逻辑流程。后一个场景必须是前一个场景的自然发展和延续。
2.  **时序一致性**: 场景必须遵循严格的、不可逆转的时间顺序。例如，如果故事跨越一天，场景必须从早到晚（如：清晨 -> 中午 -> 黄昏 -> 夜晚）或从夜到日演进，**绝不允许出现“白天-黑夜-白天”这样的时间跳跃**。这种时间流逝必须在光照、环境、角色状态和行为中得到明确体现。

## 提示词配方要求：

**高级公式**: Prompt = 主体（主体描述）+ 场景（场景描述）+ 动作（动作描述）+ 镜头语言 + 氛围 + 风格
**镜头运动公式**: Prompt = 镜头运动描述 + 主体（主体描述）+ 场景（场景描述）+ 动作（动作描述）+ 镜头语言 + 氛围 + 风格

## 生成内容要求：

### 1. **图片生成提示词**（10个场景）
- **严格遵循上述核心叙事原则**，确保场景按时间顺序推进。
- 每个提示词用英文编写，**长度在40-60个单词之间**。
- **必须以"{style_prefix}"开头**，确保风格统一。
- 包含详细的主体、场景、动作、镜头语言和氛围描述。
- 体现出时间的变化，例如通过光线（如：early morning golden hour, harsh midday sun, warm sunset glow, soft moonlight）来表现。

### 2. **视频生成提示词**（10个场景）
- **严格遵循上述核心叙事原则**，展示与图片场景对应的、连贯的动态发展。
- 每个提示词用英文编写，**长度在50-80个单词之间**。
- **必须以"{style_prefix}"开头**。
- **必须包含具体的镜头运动描述**（如：slow push-in shot, tracking shot, dolly movement, pan, tilt, zoom）。
- 描述5秒内的动态变化，包括人物动作、表情和环境元素的细微变化，以延续故事。

### 3. **旁白文本**（10段）
- 每段不超过15个中文字。
- 与对应场景内容和情绪紧密匹配，共同推动故事发展。
- 语言简洁有力，富有感染力。

### 4. **视频业务点**
- 3-5个核心价值点或故事主旨。
- 中文描述，突出视频的核心信息或情感。

### 5. **服务概述**
- 100字以内的中文描述。
- 说明视频的整体故事、价值和目标受众。

请严格按照以下JSON格式返回：
```json
{{
    "img2img_prompts": [
        "{style_prefix}场景1：故事的开端...",
        "{style_prefix}场景2：故事的发展...",
        ...
    ],
    "img2vid_prompts": [
        "{style_prefix}镜头运动 + 场景1的动态延续...",
        "{style_prefix}镜头运动 + 场景2的动态延续...",
        ...
    ],
    "narrations": [
        "旁白1",
        "旁白2",
        ...
    ],
    "business_points": "业务点或故事主旨描述...",
    "service_overview": "服务概述..."
}}```

注意：请确保你的回答完美地融合了叙事结构和AI生成技术，创造出一个有逻辑、有情感、有时序的完整故事。
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
                    clean_line = re.sub(r'^\d+[\.\)]\s*', '', line).strip('",')
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
    # 确保您已设置环境变量 OPENAI_API_KEY 和 OPENAI_BASE_URL
    # 或者在初始化时传入参数: client = LLMClient(api_key="...", base_url="...")
    client = LLMClient()
    
    # 使用您的故事类查询作为示例
    search_query_example = "春天在哪里"
    # search_query_example = "卡皮巴拉的一天" # 也可以用这个
    
    output_result = client.generate_video_content(search_query_example, output_base_dir="generated_video_content")

    if output_result["success"]:
        print("\n--- LLM Content Generation Successful ---")
        print(f"All content saved to: {output_result['data']['output_folder']}")
        print("\n--- Sample Generated Content ---")
        print(f"First img2img prompt: {output_result['data']['img2img_prompts'][0]}")
        print(f"Last img2img prompt: {output_result['data']['img2img_prompts'][-1]}")
        print(f"First narration: {output_result['data']['narrations'][0]}")
        print(f"Last narration: {output_result['data']['narrations'][-1]}")
        print(f"Service Overview: {output_result['data']['service_overview']}")
    else:
        print("\n--- LLM Content Generation Failed ---")
        print(f"Error: {output_result['error']}")