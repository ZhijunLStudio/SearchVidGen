import os
import json
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
from openai import OpenAI

# 加载配置
def load_config():
    with open("config/llm_config.json", "r", encoding="utf-8") as f:
        return json.load(f)

# 加载提示词模板
def load_prompt(prompt_name="default"):
    with open("config/prompt.json", "r", encoding="utf-8") as f:
        return json.load(f).get(prompt_name, "")

# 获取落地页文本内容
def fetch_landing_page_content(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text(separator='\n', strip=True)
        return text[:2000]  # 限定最大长度
    except Exception as e:
        return f"[无法获取页面内容: {e}]"

# 初始化 OpenAI 客户端
def get_openai_client(api_key, base_url):
    return OpenAI(api_key=api_key, base_url=base_url)

# 使用 OpenAI SDK 调用模型生成内容
def call_llm(client, model, prompt_text):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[大模型调用失败: {e}]"

# 搜索并处理客户信息
def search_customer_info(excel_path, search_index, prompt_name="ad_zh"):
    try:
        search_terms_df = pd.read_excel(excel_path, sheet_name='搜索词')
        customer_info_df = pd.read_excel(excel_path, sheet_name='客户信息')

        if search_index <= 0 or search_index > len(search_terms_df):
            print(f"索引超出范围，搜索词表有 {len(search_terms_df)} 行")
            return

        search_term = search_terms_df.iloc[search_index - 1, 0]
        print(f"\n搜索词: {search_term}")

        config = load_config()
        prompt_template = load_prompt(prompt_name)
        client = get_openai_client(config["api_key"], config["api_url"])

        found = False

        for idx, row in customer_info_df.iterrows():
            row_strs = [str(value).lower() for value in row.values if pd.notna(value)]
            if any(str(search_term).lower() in cell for cell in row_strs):
                found = True
                # 结构化客户信息
                content_lines = [f"搜索词：{search_term}\n", "客户信息："]
                for col in row.index:
                    if col.strip().lower() != "客户名称":
                        content_lines.append(f"{col}: {row[col]}")

                # 获取落地页内容
                if '落地页' in row and pd.notna(row['落地页']):
                    page_text = fetch_landing_page_content(row['落地页'])
                    content_lines.append("\n落地页内容预览：\n" + page_text[:1000])
                else:
                    content_lines.append("\n落地页内容预览：暂无")

                # 构建给模型的输入
                full_text_for_model = "\n".join(content_lines)
                llm_input = f"{prompt_template}\n\n{full_text_for_model}\n\n请基于以上信息生成视频广告的描述和台词："
                ad_content = call_llm(client, config["model"], llm_input)

                # 添加原始内容到输出
                content_lines.append("\n生成的广告内容：\n" + ad_content)
                output_text = "\n".join(content_lines)

                # 创建以搜索词命名的文件夹
                safe_term = "".join(c if c.isalnum() else "_" for c in str(search_term))
                folder_path = f"output_text/{safe_term}"
                os.makedirs(folder_path, exist_ok=True)
                
                # 保存文件，命名格式: 搜索词_行号_时间戳.txt
                timestamp = time.strftime("%Y%m%d%H%M%S")
                filename = f"{folder_path}/{safe_term}_行{idx+1}_{timestamp}.txt"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(output_text)
                
                # 尝试解析并保存JSON文件
                try:
                    # 尝试找出JSON内容部分
                    json_content = ad_content.strip()
                    # 如果模型输出包含了额外的前导文本，尝试提取JSON部分
                    import re
                    json_match = re.search(r'\{.*\}', json_content, re.DOTALL)
                    if json_match:
                        json_content = json_match.group(0)
                    
                    # 解析并格式化JSON
                    content_json = json.loads(json_content)
                    json_filename = f"{folder_path}/{safe_term}_行{idx+1}_{timestamp}.json"
                    with open(json_filename, "w", encoding="utf-8") as f:
                        json.dump(content_json, f, ensure_ascii=False, indent=2)
                    print(f"✅ 匹配项已保存至 {filename} 和 {json_filename}")
                except Exception as json_err:
                    print(f"⚠️ JSON解析失败，仅保存了文本文件: {json_err}")
                    print(f"✅ 匹配项已保存至 {filename}")

        if not found:
            print(f"未找到匹配客户：{search_term}")

    except Exception as e:
        print(f"发生错误: {e}")


# 主程序入口
if __name__ == "__main__":
    excel_file = "origin_data/data.xlsx"
    os.makedirs("output_text", exist_ok=True)

    while True:
        try:
            search_index = int(input("\n请输入要搜索的行号 (输入0退出): "))
            if search_index == 0:
                print("程序已退出")
                break

            prompt_name = input("请输入使用的提示词模板名（中文模板:ad_zh, 英文模板:ad_en, 默认:ad_zh: ").strip() or "default"
            search_customer_info(excel_file, search_index, prompt_name=prompt_name)

        except ValueError:
            print("请输入有效的数字")
