import json
import argparse
import os
import time
import re
from openai import OpenAI
from typing import Dict, List, Any
import uuid

class KnowledgeVideoGenerator:
    def __init__(self, api_key: str, base_url: str = None, model: str = "gpt-4"):
        """初始化知识视频生成器客户端
        
        Args:
            api_key: OpenAI API密钥
            base_url: API基础URL (可选，用于代理)
            model: 使用的OpenAI模型
        """
        self.model = model
        self.prompts_data = None
        
        # 设置OpenAI客户端
        client_args = {"api_key": api_key}
        if base_url:
            client_args["base_url"] = base_url
            
        self.client = OpenAI(**client_args)
        
    def load_prompts(self, prompts_file: str) -> Dict:
        """从JSON文件加载prompt配置
        
        Args:
            prompts_file: JSON配置文件路径
        
        Returns:
            加载的prompt配置字典
        """
        with open(prompts_file, 'r', encoding='utf-8') as f:
            self.prompts_data = json.load(f)
            
        print(f"成功加载 {len(self.prompts_data)} 个prompt模板")
        return self.prompts_data
        
    def call_openai_api(self, 
                       prompt: str, 
                       max_tokens: int = 2048, 
                       temperature: float = 0.7,
                       top_p: float = 0.9) -> str:
        """调用OpenAI API
        
        Args:
            prompt: 输入的prompt文本
            max_tokens: 生成的最大token数
            temperature: 采样温度
            top_p: 采样的概率阈值
            
        Returns:
            生成的文本
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            
            # 提取生成的文本
            generated_text = response.choices[0].message.content
            return generated_text
            
        except Exception as e:
            print(f"OpenAI API调用失败: {e}")
            return ""
    
    def extract_json_from_text(self, text: str) -> str:
        """从文本中提取JSON数据
        
        Args:
            text: 可能包含JSON的文本
            
        Returns:
            提取出的JSON字符串
        """
        # 尝试几种常见的JSON提取模式
        
        # 模式1: 提取被```json和```包围的内容
        json_pattern1 = r"```json\s*([\s\S]*?)\s*```"
        matches = re.search(json_pattern1, text)
        if matches:
            return matches.group(1).strip()
        
        # 模式2: 提取被```包围的内容
        json_pattern2 = r"```\s*([\s\S]*?)\s*```"
        matches = re.search(json_pattern2, text)
        if matches:
            return matches.group(1).strip()
        
        # 模式3: 寻找[{开头和}]结尾的内容
        json_pattern3 = r"(\[\s*\{[\s\S]*\}\s*\])"
        matches = re.search(json_pattern3, text)
        if matches:
            return matches.group(1).strip()
        
        # 模式4: 寻找{开头和}结尾的内容
        json_pattern4 = r"(\{\s*[\s\S]*\}\s*)"
        matches = re.search(json_pattern4, text)
        if matches:
            return matches.group(1).strip()
            
        # 如果上述都失败，返回原文本
        return text
    
    def parse_json_safely(self, text: str) -> Any:
        """安全地解析JSON文本
        
        Args:
            text: JSON文本
            
        Returns:
            解析后的Python对象
        """
        try:
            # 尝试从文本中提取JSON
            json_str = self.extract_json_from_text(text)
            
            # 尝试解析JSON
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            print("问题文本:", text[:200] + "..." if len(text) > 200 else text)
            return None
    
    def format_prompt(self, prompt_id: str, input_values: Dict[str, Any]) -> str:
        """根据模板和输入值格式化prompt
        
        Args:
            prompt_id: 要使用的prompt模板ID
            input_values: 模板中变量的值
            
        Returns:
            格式化后的prompt
        """
        if not self.prompts_data:
            raise ValueError("请先使用load_prompts加载prompt配置")
            
        if prompt_id not in self.prompts_data:
            raise ValueError(f"未找到ID为 {prompt_id} 的prompt模板")
            
        prompt_template = self.prompts_data[prompt_id]["template"]
        
        # 替换模板中的变量
        for key, value in input_values.items():
            placeholder = f"{{{key}}}"
            prompt_template = prompt_template.replace(placeholder, str(value))
            
        return prompt_template
    
    def process_task(self, task_type: str, input_data: Dict[str, Any], 
                    max_tokens: int = 2048, temperature: float = 0.7) -> Dict[str, Any]:
        """处理特定类型的任务
        
        Args:
            task_type: 任务类型，对应prompts.json中的键
            input_data: 任务输入数据
            max_tokens: 生成的最大token数
            temperature: 采样温度
            
        Returns:
            包含生成结果的字典
        """
        prompt = self.format_prompt(task_type, input_data)
        
        print(f"\n执行任务: {task_type}")
        print(f"输入数据: {json.dumps(input_data, ensure_ascii=False)[:200]}...")
        
        start_time = time.time()
        result = self.call_openai_api(prompt, max_tokens, temperature)
        end_time = time.time()
        
        print(f"任务完成，耗时: {end_time - start_time:.2f}秒")
        
        return {
            "task_type": task_type,
            "input": input_data,
            "output": result,
            "time_taken": end_time - start_time
        }
        
    def generate_video_description_and_script(self, search_query: str, temperature: float = 0.7) -> Dict[str, str]:
        """根据搜索查询生成整段式视频描述和口播台词
        
        Args:
            search_query: 用户搜索查询
            temperature: 生成温度
            
        Returns:
            包含视频描述和口播台词的字典
        """
        print(f"\n开始为搜索查询 '{search_query}' 生成视频描述和口播台词...")
        
        # 1. 搜索意图分析
        intent_result = self.process_task(
            "search_intent_analysis",
            {"search_query": search_query},
            max_tokens=1500,
            temperature=0.7
        )
        
        # 2. 知识内容研究与拓展
        knowledge_result = self.process_task(
            "knowledge_research",
            {
                "search_query": search_query,
                "intent_analysis": intent_result["output"]
            },
            max_tokens=2500,
            temperature=0.7
        )
        
        # 3. 生成视频描述和口播台词
        desc_result = self.process_task(
            "video_description_and_script",
            {
                "search_query": search_query,
                "intent_analysis": intent_result["output"],
                "knowledge_content": knowledge_result["output"]
            },
            max_tokens=3000,
            temperature=temperature
        )
        
        # 提取视频描述和口播台词
        result = desc_result["output"]
        video_description_match = re.search(r"【视频描述】\s*([\s\S]*?)(?=\n【口播台词】|\Z)", result)
        script_match = re.search(r"【口播台词】\s*([\s\S]*)", result)
        
        video_description = video_description_match.group(1).strip() if video_description_match else ""
        script = script_match.group(1).strip() if script_match else ""
        
        return {
            "search_query": search_query,
            "intent_analysis": intent_result,
            "knowledge_content": knowledge_result,
            "description_result": desc_result,
            "video_description": video_description,
            "script": script
        }

    
    def auto_generate_styles(self, search_query: str, num_styles: int = 3) -> Dict[str, Any]:
        """自动设计并生成多种不同风格的视频
        
        Args:
            search_query: 搜索词
            num_styles: 要生成的风格数量
            
        Returns:
            包含多个风格设计的字典
        """
        results = {"search_query": search_query}
        
        # 1. 搜索意图分析 - 只需要做一次
        intent_result = self.process_task(
            "search_intent_analysis",
            {"search_query": search_query},
            max_tokens=1500,
            temperature=0.7
        )
        results["intent_analysis"] = intent_result
        
        # 2. 知识内容研究与拓展 - 只需要做一次
        knowledge_result = self.process_task(
            "knowledge_research",
            {
                "search_query": search_query,
                "intent_analysis": intent_result["output"]
            },
            max_tokens=2500,
            temperature=0.7
        )
        results["knowledge_content"] = knowledge_result
        
        # 3. 自动设计多种视频风格
        style_design_result = self.process_task(
            "auto_style_design",
            {
                "search_query": search_query,
                "intent_analysis": intent_result["output"],
                "knowledge_content": knowledge_result["output"],
                "num_styles": num_styles
            },
            max_tokens=2000,
            temperature=0.8
        )
        results["auto_style_design"] = style_design_result
        
        # 尝试解析JSON风格定义
        try:
            # 使用安全的JSON解析
            style_designs = self.parse_json_safely(style_design_result["output"])
            
            if not style_designs:
                # 解析失败，生成默认风格
                raise ValueError("无法解析风格定义")
                
            if not isinstance(style_designs, list):
                # 如果不是列表，尝试提取styles字段
                if isinstance(style_designs, dict) and "styles" in style_designs:
                    style_designs = style_designs["styles"]
                else:
                    raise ValueError("无法解析风格定义，预期格式为列表或包含styles字段的字典")
            
            # 确保我们有足够的风格
            actual_styles = min(len(style_designs), num_styles)
            
            # 4. 为每种风格生成不同的内容
            results["style_variations"] = []
            
            for i in range(actual_styles):
                style = style_designs[i]
                style_name = style.get("name", f"风格 #{i+1}")
                style_description = style.get("description", "")
                
                print(f"\n生成 '{style_name}' 风格的视频方案...")
                
                # 给每个风格设置略微不同的温度，增加多样性
                temp_variation = 0.6 + (i * 0.1) % 0.3
                
                # 为该风格生成脚本
                script_result = self.process_task(
                    "video_script_generation",
                    {
                        "search_query": search_query,
                        "knowledge_content": knowledge_result["output"],
                        "style_name": style_name,
                        "style_description": style_description
                    },
                    max_tokens=3000,
                    temperature=temp_variation
                )
                
                # 为该风格生成分镜头
                storyboard_result = self.process_task(
                    "storyboard_design",
                    {
                        "video_script": script_result["output"],
                        "style_name": style_name,
                        "style_description": style_description
                    },
                    max_tokens=3000,
                    temperature=temp_variation
                )
                
                # 为该风格生成视觉和声音设计
                visual_audio_result = self.process_task(
                    "visual_audio_design",
                    {
                        "search_query": search_query,
                        "video_script": script_result["output"],
                        "style_name": style_name,
                        "style_description": style_description
                    },
                    max_tokens=1500,
                    temperature=temp_variation
                )
                
                # 为该风格生成视频合成指令
                synthesis_result = self.process_task(
                    "video_synthesis_instructions",
                    {
                        "search_query": search_query,
                        "storyboard": storyboard_result["output"],
                        "visual_audio_design": visual_audio_result["output"],
                        "style_name": style_name,
                        "style_description": style_description
                    },
                    max_tokens=2000,
                    temperature=temp_variation
                )
                
                # 将该风格的所有结果添加到结果列表
                results["style_variations"].append({
                    "style_info": {
                        "name": style_name,
                        "description": style_description
                    },
                    "video_script": script_result,
                    "storyboard": storyboard_result,
                    "visual_audio_design": visual_audio_result,
                    "synthesis_instructions": synthesis_result
                })
                
        except Exception as e:
            print(f"处理风格设计时出错: {e}")
            print("将生成预定义的多种风格视频")
            
            # 如果风格解析失败，使用预定义的风格列表
            predefined_styles = [
                {
                    "name": "传统文化解析",
                    "description": "以清晰、权威的方式讲解春联的历史文化背景，强调知识性和教育价值"
                },
                {
                    "name": "生活场景体验",
                    "description": "通过生活化的场景和温暖的氛围，展示春联在现代生活中的应用和意义"
                },
                {
                    "name": "视觉艺术探索",
                    "description": "关注春联的艺术性和视觉美感，通过精美的书法和设计展示其审美价值"
                }
            ]
            
            # 确保样式数量不超过要求
            num_predefined = min(len(predefined_styles), num_styles)
            
            results["style_variations"] = []
            for i in range(num_predefined):
                style = predefined_styles[i]
                style_name = style["name"]
                style_description = style["description"]
                
                print(f"\n生成 '{style_name}' 风格的视频方案...")
                
                # 给每个风格设置略微不同的温度
                temp_variation = 0.6 + (i * 0.1) % 0.3
                
                # 为该风格生成内容
                script_result = self.process_task(
                    "video_script_generation",
                    {
                        "search_query": search_query,
                        "knowledge_content": knowledge_result["output"],
                        "style_name": style_name,
                        "style_description": style_description
                    },
                    max_tokens=3000,
                    temperature=temp_variation
                )
                
                storyboard_result = self.process_task(
                    "storyboard_design",
                    {
                        "video_script": script_result["output"],
                        "style_name": style_name,
                        "style_description": style_description
                    },
                    max_tokens=3000,
                    temperature=temp_variation
                )
                
                visual_audio_result = self.process_task(
                    "visual_audio_design",
                    {
                        "search_query": search_query,
                        "video_script": script_result["output"],
                        "style_name": style_name,
                        "style_description": style_description
                    },
                    max_tokens=1500,
                    temperature=temp_variation
                )
                
                synthesis_result = self.process_task(
                    "video_synthesis_instructions",
                    {
                        "search_query": search_query,
                        "storyboard": storyboard_result["output"],
                        "visual_audio_design": visual_audio_result["output"],
                        "style_name": style_name,
                        "style_description": style_description
                    },
                    max_tokens=2000,
                    temperature=temp_variation
                )
                
                results["style_variations"].append({
                    "style_info": {
                        "name": style_name,
                        "description": style_description
                    },
                    "video_script": script_result,
                    "storyboard": storyboard_result,
                    "visual_audio_design": visual_audio_result,
                    "synthesis_instructions": synthesis_result
                })
        
        return results

def main():
    parser = argparse.ArgumentParser(description="知识视频生成系统 - OpenAI版")
    parser.add_argument("--api_key", required=True, help="OpenAI API密钥")
    parser.add_argument("--base_url", default=None, help="API基础URL (可选，用于代理)")
    parser.add_argument("--model", default="gpt-4", help="OpenAI模型名称")
    parser.add_argument("--prompts", default="config/knowledge_prompts.json", help="Prompt配置文件路径")
    parser.add_argument("--search", required=True, help="搜索查询词")
    parser.add_argument("--output", default="knowledge_video_results.json", help="结果输出文件")
    parser.add_argument("--styles", type=int, default=3, help="要生成的风格数量")
    parser.add_argument("--single_desc", action="store_true", help="生成整段式视频描述和口播台词，而非多风格视频")
    
    args = parser.parse_args()
    
    generator = KnowledgeVideoGenerator(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model
    )
    generator.load_prompts(args.prompts)
    
    # 创建输出文件夹
    output_dir = os.path.splitext(args.output)[0]
    os.makedirs(output_dir, exist_ok=True)
    
    if args.single_desc:
        # 使用新功能生成单一整段式视频描述和口播台词
        results = generator.generate_video_description_and_script(args.search)
        
        # 保存结果 JSON
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 保存人类易读格式
        readable_output = f"{output_dir}/video_description_and_script.txt"
        with open(readable_output, "w", encoding="utf-8") as f:
            f.write(f"# 搜索查询：{args.search}\n\n")
            f.write("## 视频描述\n\n")
            f.write(results["video_description"])
            f.write("\n\n## 口播台词\n\n")
            f.write(results["script"])
        
        print(f"\n处理完成! 视频描述和口播台词已保存至: {readable_output}")
    
    else:
        # 原有的多风格视频生成逻辑
        results = generator.auto_generate_styles(args.search, args.styles)
        
        # 保存结果
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        print(f"\n处理完成! 结果已保存至: {args.output}")
        
        # 将每个风格的合成指令单独保存为文件，方便查看
        for i, style_variation in enumerate(results["style_variations"]):
            style_name = style_variation["style_info"]["name"].replace(" ", "_")
            style_file = f"{output_dir}/{style_name}_instructions.txt"
            
            with open(style_file, "w", encoding="utf-8") as f:
                f.write(f"# {style_variation['style_info']['name']} - 视频合成指令\n\n")
                f.write(f"风格描述: {style_variation['style_info']['description']}\n\n")
                f.write(style_variation["synthesis_instructions"]["output"])
                
            print(f"风格 '{style_variation['style_info']['name']}' 的指令已保存至: {style_file}")


if __name__ == "__main__":
    main()
