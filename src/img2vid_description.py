#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_img2vid_prompts.py

功能：
  - 遍历指定文件夹下 generated_images/ 子目录中的所有图片（按文件名排序）
  - 针对每张图片，仅根据图像内容，调用 OpenAI o4-mini 模型
    生成一条自然流畅的英文描述，突出主体、场景与动作（不要使用标签或“Subject:”“Scene:”等固定格式，也不要提及时间/时长）
  - 实时将每条 prompt 追加写入 img2vid_video_prompts.txt，并立即 flush，避免中途丢失
  - 显示进度条

用法：
  pip install openai pillow tqdm
  python generate_img2vid_prompts.py \
    --api_key sk-... \
    --base_url https://api.openai-proxy.org/v1 \
    --folder /path/to/your/folder
"""

import os
import argparse
import base64
from io import BytesIO
from PIL import Image
from openai import OpenAI
from tqdm import tqdm

def encode_image_to_datauri(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def build_instruction() -> str:
    return (
        "You are shown an image. Based only on what you see, imagine what happens next. "
        "Write one concise English sentence or two that naturally describe the main focus, "
        "the setting around it, and the action or movement taking place. "
        "Do not use labels, bullet points, or mention time or duration."
    )

def main():
    parser = argparse.ArgumentParser(
        description="Generate natural-language video prompts from images"
    )
    parser.add_argument("--api_key", "-k", required=True, help="OpenAI API key")
    parser.add_argument("--base_url", "-u", required=True, help="OpenAI base URL")
    parser.add_argument(
        "--folder", "-f", required=True,
        help="输入/输出目录，内含 generated_images/ 子目录"
    )
    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    images_dir = os.path.join(args.folder, "generated_images")
    output_txt = os.path.join(args.folder, "img2vid_video_prompts.txt")

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"找不到图片目录: {images_dir}")

    img_files = sorted([
        fn for fn in os.listdir(images_dir)
        if fn.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    if not img_files:
        print("未找到任何图片，退出。")
        return

    instruction = build_instruction()

    # 先清空（或新建）目标文件
    with open(output_txt, "w", encoding="utf-8") as fo:
        pass

    # 逐张生成并写入
    with open(output_txt, "a", encoding="utf-8") as fo:
        for img_name in tqdm(img_files, desc="Processing images"):
            img_path = os.path.join(images_dir, img_name)
            try:
                img = Image.open(img_path).convert("RGB")
                datauri = encode_image_to_datauri(img)
            except Exception as e:
                print(f"[ERROR] 无法打开图片 {img_name}: {e}")
                fo.write("\n")
                fo.flush()
                continue

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image_url", "image_url": {"url": datauri, "detail": "high"}}
                    ]
                }
            ]
            try:
                resp = client.chat.completions.create(
                    model="o4-mini",
                    messages=messages
                )
                text = resp.choices[0].message.content.strip().replace("\n", " ")
            except Exception as e:
                print(f"[WARN] 模型调用失败 {img_name}: {e}")
                text = ""

            # 实时写入并 flush
            fo.write(text + "\n")
            fo.flush()

    print(f"完成，共处理 {len(img_files)} 张图片，结果已写入 {output_txt}")

if __name__ == "__main__":
    main()
