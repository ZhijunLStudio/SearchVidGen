#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_img2vid_prompts.py

功能：
  - 遍历指定文件夹下 generated_images/ 子目录中的所有图片（按文件名排序）
  - 针对每张图片，仅根据图像内容，调用 OpenAI o4-mini 模型
    生成一条自然流畅的英文描述，突出主体、场景与动作（不要使用标签或“Subject:”“Scene:”等固定格式，也不要提及时间/时长）
  - 如果调用失败，最多重试3次，每次失败后指数退避
  - 支持并发处理，默认4个 workers，可通过 --workers 调整
  - 实时将每条 prompt 追加写入 img2vid_video_prompts.txt，并立即 flush，避免中途丢失
  - 显示进度条

用法：
  pip install openai pillow tqdm
  python generate_img2vid_prompts.py \
    --api_key sk-... \
    --base_url https://api.openai-proxy.org/v1 \
    --folder /path/to/your/folder \
    [--workers 4]
"""

import os
import argparse
import base64
import time
from io import BytesIO
from PIL import Image
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def encode_image_to_datauri(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def build_instruction() -> str:
    return (
        "You are shown a single image. Based solely on its visual details, "
        "write one cohesive English paragraph of at least 50 words describing "
        "exactly what actions the people in the scene are performing or are about to perform. "
        "Focus on their movements, gestures, facial expressions, interactions with objects or each other, "
        "and how the environment responds (e.g. wind, light, surfaces). "
        "Do NOT use bullet points, labels (like “Subject:” or “Scene:”), or mention time spans or durations."
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
    parser.add_argument(
        "--workers", "-w", type=int, default=4,
        help="并发 workers 数量，默认 4"
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

    def process_image(img_name: str) -> str:
        """
        打开图片、编码、调用 API，最多重试 3 次，返回生成的文字（失败时返回空字符串）
        """
        img_path = os.path.join(images_dir, img_name)
        try:
            img = Image.open(img_path).convert("RGB")
            datauri = encode_image_to_datauri(img)
        except Exception as e:
            print(f"[ERROR] 无法打开图片 {img_name}: {e}")
            return ""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": datauri, "detail": "high"}}
                ]
            }
        ]

        text = ""
        for attempt in range(1, 4):
            try:
                resp = client.chat.completions.create(
                    model="o4-mini",
                    messages=messages
                )
                text = resp.choices[0].message.content.strip().replace("\n", " ")
                break
            except Exception as e:
                # 429/其他错误统一重试
                wait = 2 ** attempt
                print(f"[WARN] 调用失败 {img_name} (尝试第 {attempt} 次): {e}，{wait}s 后重试")
                time.sleep(wait)
        else:
            print(f"[ERROR] {img_name} 超过最大重试次数，跳过。")

        return text

    # 并发调用并保持原有顺序写入
    with ThreadPoolExecutor(max_workers=args.workers) as executor, \
         open(output_txt, "a", encoding="utf-8") as fo:

        # executor.map 会保持结果顺序
        for text in tqdm(executor.map(process_image, img_files),
                         total=len(img_files),
                         desc="Processing images"):
            fo.write(text + "\n")
            fo.flush()

    print(f"完成，共处理 {len(img_files)} 张图片，结果已写入 {output_txt}")

if __name__ == "__main__":
    main()
