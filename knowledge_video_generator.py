import json
import argparse
import os
import time
import re
import datetime
from openai import OpenAI
from typing import Dict, List, Any

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
            print("问题文本:", text[:200] + "…" if len(text) > 200 else text)
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
        
        # 打印输入数据（有限长度）
        input_str = json.dumps(input_data, ensure_ascii=False)
        if len(input_str) > 200:
            print(f"输入数据: {input_str[:200]}…")
        else:
            print(f"输入数据: {input_str}")
        
        start_time = time.time()
        result = self.call_openai_api(prompt, max_tokens, temperature)
        end_time = time.time()
        
        print(f"任务完成，耗时: {end_time - start_time:.2f}秒")
        
        # 打印输出数据（有限长度）
        if result:
            output_preview = result[:200] + "…" if len(result) > 200 else result
            print(f"输出摘要: {output_preview}")
        
        return {
            "task_type": task_type,
            "input": input_data,
            "output": result,
            "time_taken": end_time - start_time
        }
    
    def design_video_styles(self, search_query: str, intent_analysis: str, knowledge_content: str, num_styles: int = 3) -> List[Dict]:
        """设计多种视频风格
        
        Args:
            search_query: 搜索查询
            intent_analysis: 意图分析结果
            knowledge_content: 知识内容
            num_styles: 要设计的风格数量
            
        Returns:
            风格列表，每个风格是一个字典
        """
        style_design_result = self.process_task(
            "auto_style_design",
            {
                "search_query": search_query,
                "intent_analysis": intent_analysis,
                "knowledge_content": knowledge_content,
                "num_styles": num_styles
            },
            max_tokens=2000,
            temperature=0.8
        )
        
        # 尝试解析JSON风格定义
        try:
            style_designs = self.parse_json_safely(style_design_result["output"])
            
            if not style_designs:
                raise ValueError("无法解析风格定义")
                
            if not isinstance(style_designs, list):
                # 如果不是列表，尝试提取styles字段
                if isinstance(style_designs, dict) and "styles" in style_designs:
                    style_designs = style_designs["styles"]
                else:
                    raise ValueError("无法解析风格定义，预期格式为列表或包含styles字段的字典")
            
            # 确保每个风格都有name和description字段
            for style in style_designs:
                if "name" not in style:
                    style["name"] = "未命名风格"
                if "description" not in style:
                    style["description"] = "无描述"
            
            return style_designs
            
        except Exception as e:
            print(f"处理风格设计时出错: {e}")
            print("将使用基本风格")
            
            # 创建一个基本风格
            return [{
                "name": "标准知识讲解风格",
                "description": "以清晰、直观的方式讲解知识点，适合大多数学习者",
                "target_audience": "一般受众",
                "visual_characteristics": "清晰简洁的画面",
                "narrative_approach": "直观讲解"
            }]
    
    def generate_video_description_and_script(self, search_query: str, style: Dict = None, temperature: float = 0.7) -> Dict[str, Any]:
        """根据搜索查询生成整段式视频描述和口播台词
        
        Args:
            search_query: 用户搜索查询
            style: 视频风格信息 (可选)
            temperature: 生成温度
            
        Returns:
            包含视频描述和口播台词的字典
        """
        if style:
            print(f"\n开始为搜索查询 '{search_query}' 生成 '{style['name']}' 风格的视频描述和口播台词...")
        else:
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
        
        # 如果没有提供风格，则自动设计一个
        if not style:
            style_design_result = self.process_task(
                "optimal_style_design",
                {
                    "search_query": search_query,
                    "intent_analysis": intent_result["output"],
                    "knowledge_content": knowledge_result["output"]
                },
                max_tokens=1500,
                temperature=0.7
            )
            
            try:
                style = self.parse_json_safely(style_design_result["output"])
                if not style or not isinstance(style, dict) or "name" not in style:
                    style = {
                        "name": "自动设计风格",
                        "description": style_design_result["output"]
                    }
            except Exception:
                style = {
                    "name": "自动设计风格",
                    "description": style_design_result["output"]
                }
                
            print(f"选择的视频风格: {style['name']}")
        
        # 3. 生成视频描述和口播台词
        desc_result = self.process_task(
            "styled_video_description_and_script",
            {
                "search_query": search_query,
                "intent_analysis": intent_result["output"],
                "knowledge_content": knowledge_result["output"],
                "style_name": style.get("name", "默认风格"),
                "style_description": style.get("description", "")
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
            "style": style,
            "video_description": video_description,
            "script": script,
            "intent_analysis": intent_result,
            "knowledge_content": knowledge_result,
            "description_result": desc_result
        }
    
    def generate_multiple_styles(self, search_query: str, user_styles: List[str] = None, num_styles: int = 3, temperature: float = 0.7) -> Dict[str, Any]:
        """生成多种风格的视频描述和口播台词
        
        Args:
            search_query: 搜索查询
            user_styles: 用户指定的风格列表
            num_styles: 要生成的风格数量
            temperature: 生成温度
            
        Returns:
            包含多种风格视频描述的字典
        """
        print(f"\n开始为搜索查询 '{search_query}' 生成多种风格的视频描述和口播台词...")
        
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
        
        styles = []
        
        # 如果用户提供了风格列表，则使用这些风格
        if user_styles and len(user_styles) > 0:
            for style_name in user_styles:
                styles.append({
                    "name": style_name.strip(),
                    "description": f"用户指定的 {style_name.strip()} 风格"
                })
        else:
            # 否则，自动设计风格
            styles = self.design_video_styles(
                search_query,
                intent_result["output"],
                knowledge_result["output"],
                num_styles
            )
        
        print(f"将生成 {len(styles)} 种不同风格的视频:")
        for i, style in enumerate(styles):
            print(f"{i+1}. {style['name']}")
        
        results = []
        
        # 3. 为每种风格生成视频描述和口播台词
        for i, style in enumerate(styles):
            print(f"\n生成风格 {i+1}/{len(styles)}: {style['name']}")
            
            # 为不同风格设置略微不同的温度，增加多样性
            temp_variation = temperature + (i * 0.05) % 0.2
            
            desc_result = self.process_task(
                "styled_video_description_and_script",
                {
                    "search_query": search_query,
                    "intent_analysis": intent_result["output"],
                    "knowledge_content": knowledge_result["output"],
                    "style_name": style["name"],
                    "style_description": style.get("description", "")
                },
                max_tokens=3000,
                temperature=temp_variation
            )
            
            # 提取视频描述和口播台词
            result_text = desc_result["output"]
            video_description_match = re.search(r"【视频描述】\s*([\s\S]*?)(?=\n【口播台词】|\Z)", result_text)
            script_match = re.search(r"【口播台词】\s*([\s\S]*)", result_text)
            
            video_description = video_description_match.group(1).strip() if video_description_match else ""
            script = script_match.group(1).strip() if script_match else ""
            
            # 添加到结果列表
            results.append({
                "style": style,
                "video_description": video_description,
                "script": script
            })
        
        return {
            "search_query": search_query,
            "intent_analysis": intent_result,
            "knowledge_content": knowledge_result,
            "styles": results
        }

def main():
    parser = argparse.ArgumentParser(description="知识视频描述和台词生成系统")
    parser.add_argument("--api_key", required=True, help="OpenAI API密钥")
    parser.add_argument("--base_url", default=None, help="API基础URL (可选，用于代理)")
    parser.add_argument("--model", default="gpt-4", help="OpenAI模型名称")
    parser.add_argument("--prompts", default="config/video_prompts.json", help="Prompt配置文件路径")
    parser.add_argument("--search", required=True, help="搜索查询词")
    parser.add_argument("--output_dir", default="results", help="结果输出目录")
    parser.add_argument("--temperature", type=float, default=0.7, help="生成温度")
    parser.add_argument("--styles", default=None, help="要生成的视频风格，多个风格用逗号分隔")
    parser.add_argument("--num_styles", type=int, default=3, help="若不指定具体风格，自动生成的风格数量")
    
    args = parser.parse_args()
    
    # 创建生成器实例
    generator = KnowledgeVideoGenerator(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model
    )
    
    # 加载prompts
    generator.load_prompts(args.prompts)
    
    # 解析用户指定的风格
    user_styles = None
    if args.styles:
        user_styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    
    # 生成结果
    if user_styles or args.num_styles > 1:
        # 生成多风格视频
        results = generator.generate_multiple_styles(
            args.search,
            user_styles,
            args.num_styles,
            args.temperature
        )
    else:
        # 生成单一风格视频
        single_result = generator.generate_video_description_and_script(
            args.search, 
            temperature=args.temperature
        )
        
        # 封装为与多风格相同的格式
        results = {
            "search_query": args.search,
            "intent_analysis": single_result["intent_analysis"],
            "knowledge_content": single_result["knowledge_content"],
            "styles": [{
                "style": single_result["style"],
                "video_description": single_result["video_description"],
                "script": single_result["script"]
            }]
        }
    
    # 创建以搜索词+时间命名的子文件夹
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # 处理搜索词为有效的文件夹名
    safe_search_query = re.sub(r'[\\/:*?"<>|]', '_', args.search)
    result_dir = os.path.join(args.output_dir, f"{safe_search_query}_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    
    # 保存详细结果为JSON
    json_path = os.path.join(result_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 保存简化的JSON结果（仅包含风格、描述和台词）
    simplified_results = {
        "search_query": args.search,
        "styles": []
    }
    
    for item in results["styles"]:
        simplified_results["styles"].append({
            "style_name": item["style"]["name"],
            "style_description": item["style"].get("description", ""),
            "video_description": item["video_description"],
            "script": item["script"]
        })
    
    simplified_json_path = os.path.join(result_dir, "simplified_results.json")
    with open(simplified_json_path, "w", encoding="utf-8") as f:
        json.dump(simplified_results, f, ensure_ascii=False, indent=2)
    
    # 保存每种风格的视频描述和台词为易读的文本文件
    for i, style_result in enumerate(results["styles"]):
        style_name = style_result["style"]["name"]
        safe_style_name = re.sub(r'[\\/:*?"<>|]', '_', style_name)
        txt_path = os.path.join(result_dir, f"{safe_style_name}_description_and_script.txt")
        
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"# 搜索查询：{args.search}\n\n")
            f.write(f"## 视频风格：{style_name}\n\n")
            
            if "description" in style_result["style"]:
                f.write(f"{style_result['style']['description']}\n\n")
            
            f.write("## 视频描述\n\n")
            f.write(style_result["video_description"])
            f.write("\n\n## 口播台词\n\n")
            f.write(style_result["script"])
    
    print(f"\n处理完成! 结果已保存到目录: {result_dir}")
    print(f"- 详细JSON结果: {json_path}")
    print(f"- 简化JSON结果: {simplified_json_path}")
    print(f"- 各风格视频描述和台词保存为独立文件")

if __name__ == "__main__":
    main()
