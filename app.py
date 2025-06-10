import gradio as gr
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple
import tempfile
import shutil
from pathlib import Path

# 导入其他模块
from src.llm_client import LLMClient
from src.image_generator import ImageGenerator
from src.video_generator import VideoGenerator
from src.audio_generator import AudioGenerator
from src.video_processor import VideoProcessor

class IntentVideoApp:
    def __init__(self):
        self.llm_client = LLMClient()
        self.image_generator = ImageGenerator()
        self.video_generator = VideoGenerator()
        self.audio_generator = AudioGenerator()
        self.video_processor = VideoProcessor()
        
        # 创建工作目录
        self.work_dir = Path("workspace")
        self.work_dir.mkdir(exist_ok=True)
        
        # 当前会话数据
        self.current_session = {
            "search_query": "",
            "prompts": {},
            "images": [],
            "videos": [],
            "audios": [],
            "final_video": None
        }
    
    def generate_prompts(self, search_query: str) -> Tuple[str, str, str, str, str]:
        """生成所有文本内容"""
        if not search_query:
            return "请输入搜索词", "", "", "", ""
        
        self.current_session["search_query"] = search_query
        
        # 调用LLM生成所有内容
        result = self.llm_client.generate_video_content(search_query)
        
        if result["success"]:
            prompts = result["data"]
            self.current_session["prompts"] = prompts
            
            # 格式化输出
            img2img_prompts = "\n".join([f"{i+1}. {p}" for i, p in enumerate(prompts["img2img_prompts"])])
            img2vid_prompts = "\n".join([f"{i+1}. {p}" for i, p in enumerate(prompts["img2vid_prompts"])])
            narrations = "\n".join([f"{i+1}. {n}" for i, n in enumerate(prompts["narrations"])])
            
            return (
                img2img_prompts,
                img2vid_prompts,
                narrations,
                prompts["business_points"],
                prompts["service_overview"]
            )
        else:
            error_msg = f"生成失败: {result.get('error', '未知错误')}"
            return error_msg, error_msg, error_msg, error_msg, error_msg
    
    def update_prompts(self, img2img_text: str, img2vid_text: str, narrations_text: str,
                      business_points: str, service_overview: str) -> str:
        """更新提示词"""
        try:
            # 解析文本为列表
            img2img_prompts = [line.split('. ', 1)[1].strip() for line in img2img_text.strip().split('\n') if '. ' in line]
            img2vid_prompts = [line.split('. ', 1)[1].strip() for line in img2vid_text.strip().split('\n') if '. ' in line]
            narrations = [line.split('. ', 1)[1].strip() for line in narrations_text.strip().split('\n') if '. ' in line]
            
            # 更新会话数据
            self.current_session["prompts"].update({
                "img2img_prompts": img2img_prompts,
                "img2vid_prompts": img2vid_prompts,
                "narrations": narrations,
                "business_points": business_points,
                "service_overview": service_overview
            })
            
            return "✅ 提示词已更新"
        except Exception as e:
            return f"❌ 更新失败: {str(e)}"
    
    def generate_images(self, reference_image=None, progress=gr.Progress()) -> List[str]:
        """生成图片"""
        if not self.current_session["prompts"].get("img2img_prompts"):
            return []
        
        progress(0, desc="开始生成图片...")
        
        # 创建图片保存目录
        image_dir = self.work_dir / f"images_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        image_dir.mkdir(exist_ok=True)
        
        prompts = self.current_session["prompts"]["img2img_prompts"]
        images = []
        
        for i, prompt in enumerate(prompts):
            progress((i+1)/len(prompts), desc=f"生成第 {i+1}/{len(prompts)} 张图片")
            
            # 生成图片
            image_path = self.image_generator.generate(
                prompt=prompt,
                reference_image=reference_image,
                output_path=image_dir / f"scene_{i+1:02d}.png"
            )
            
            if image_path:
                images.append(str(image_path))
        
        self.current_session["images"] = images
        return images
    
    def generate_single_video(self, scene_idx: int, custom_prompt: str = None) -> str:
        """生成或重新生成单个视频片段"""
        if scene_idx < 0 or scene_idx >= len(self.current_session["images"]):
            return None
        
        # 使用自定义prompt或默认prompt
        prompt = custom_prompt or self.current_session["prompts"]["img2vid_prompts"][scene_idx]
        image_path = self.current_session["images"][scene_idx]
        
        # 创建视频保存目录
        video_dir = self.work_dir / "videos"
        video_dir.mkdir(exist_ok=True)
        
        video_path = self.video_generator.generate(
            image_path=image_path,
            prompt=prompt,
            output_path=video_dir / f"scene_{scene_idx+1:02d}.mp4"
        )
        
        if video_path:
            # 更新视频列表
            if len(self.current_session["videos"]) <= scene_idx:
                self.current_session["videos"].extend([None] * (scene_idx + 1 - len(self.current_session["videos"])))
            self.current_session["videos"][scene_idx] = str(video_path)
        
        return str(video_path) if video_path else None
    
    def generate_all_videos(self, progress=gr.Progress()) -> List[str]:
        """批量生成所有视频"""
        if not self.current_session["images"]:
            return []
        
        progress(0, desc="开始生成视频...")
        
        videos = []
        for i in range(len(self.current_session["images"])):
            progress((i+1)/len(self.current_session["images"]), desc=f"生成第 {i+1}/{len(self.current_session['images'])} 个视频")
            
            video_path = self.generate_single_video(i)
            if video_path:
                videos.append(video_path)
        
        return videos
    
    def generate_audios(self, progress=gr.Progress()) -> List[str]:
        """生成音频"""
        if not self.current_session["prompts"].get("narrations"):
            return []
        
        progress(0, desc="开始生成音频...")
        
        # 创建音频保存目录
        audio_dir = self.work_dir / "audios"
        audio_dir.mkdir(exist_ok=True)
        
        narrations = self.current_session["prompts"]["narrations"]
        audios = []
        
        for i, narration in enumerate(narrations):
            progress((i+1)/len(narrations), desc=f"生成第 {i+1}/{len(narrations)} 个音频")
            
            audio_path = self.audio_generator.generate(
                text=narration,
                output_path=audio_dir / f"narration_{i+1:02d}.wav"
            )
            
            if audio_path:
                audios.append(str(audio_path))
        
        self.current_session["audios"] = audios
        return audios
    
    def merge_final_video(self, progress=gr.Progress()) -> str:
        """合并最终视频"""
        if not self.current_session["videos"] or not self.current_session["audios"]:
            return None
        
        progress(0, desc="开始合并视频...")
        
        # 创建输出目录
        output_dir = self.work_dir / "final"
        output_dir.mkdir(exist_ok=True)
        
        # 合并视频
        final_path = self.video_processor.merge_videos_with_audio(
            video_paths=self.current_session["videos"],
            audio_paths=self.current_session["audios"],
            output_path=output_dir / f"final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        )
        
        if final_path:
            self.current_session["final_video"] = str(final_path)
        
        return str(final_path) if final_path else None

def create_interface():
    """创建Gradio界面"""
    app = IntentVideoApp()
    
    with gr.Blocks(title="意图视频生成器", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎬 意图视频生成器
        
        通过搜索词自动生成包含图片、视频、音频和旁白的完整视频内容。
        """)
        
        with gr.Tab("📝 内容生成"):
            with gr.Row():
                search_input = gr.Textbox(
                    label="搜索词",
                    placeholder="输入搜索词，例如：如何学习Python编程",
                    scale=3
                )
                generate_btn = gr.Button("生成内容", variant="primary", scale=1)
            
            with gr.Row():
                with gr.Column():
                    img2img_prompts = gr.Textbox(
                        label="图片生成提示词（10个片段）",
                        lines=10,
                        interactive=True
                    )
                
                with gr.Column():
                    img2vid_prompts = gr.Textbox(
                        label="视频生成提示词（10个片段）",
                        lines=10,
                        interactive=True
                    )
            
            with gr.Row():
                with gr.Column():
                    narrations = gr.Textbox(
                        label="旁白文本（每段不超过15字）",
                        lines=10,
                        interactive=True
                    )
                
                with gr.Column():
                    business_points = gr.Textbox(
                        label="视频业务点",
                        lines=4,
                        interactive=True
                    )
                    service_overview = gr.Textbox(
                        label="服务概述",
                        lines=4,
                        interactive=True
                    )
            
            update_btn = gr.Button("更新提示词", variant="secondary")
            update_status = gr.Textbox(label="状态", interactive=False)
        
        with gr.Tab("🖼️ 图片生成"):
            with gr.Row():
                with gr.Column(scale=1):
                    ref_image = gr.Image(
                        label="参考图片（可选）",
                        type="filepath"
                    )
                    generate_images_btn = gr.Button("生成所有图片", variant="primary")
                
                with gr.Column(scale=3):
                    image_gallery = gr.Gallery(
                        label="生成的图片",
                        columns=5,
                        rows=2,
                        height="auto"
                    )
        
        with gr.Tab("🎥 视频生成"):
            with gr.Row():
                generate_videos_btn = gr.Button("批量生成视频", variant="primary")
            
            with gr.Row():
                video_gallery = gr.Gallery(
                    label="生成的视频片段",
                    columns=5,
                    rows=2,
                    height="auto"
                )
            
            gr.Markdown("### 单个视频重新生成")
            with gr.Row():
                scene_selector = gr.Number(
                    label="场景编号",
                    minimum=1,
                    maximum=10,
                    step=1,
                    value=1
                )
                custom_video_prompt = gr.Textbox(
                    label="自定义视频提示词（可选）",
                    placeholder="留空使用默认提示词"
                )
                regenerate_video_btn = gr.Button("重新生成", variant="secondary")
            
            single_video_output = gr.Video(label="重新生成的视频")
        
        with gr.Tab("🎵 音频生成"):
            generate_audios_btn = gr.Button("生成所有音频", variant="primary")
            audio_outputs = gr.File(
                label="生成的音频文件",
                file_count="multiple"
            )
        
        with gr.Tab("🎬 最终合成"):
            merge_btn = gr.Button("合并生成最终视频", variant="primary", size="lg")
            final_video = gr.Video(label="最终视频")
            
            with gr.Row():
                business_display = gr.Textbox(label="视频业务点", interactive=False)
                service_display = gr.Textbox(label="服务概述", interactive=False)
        
        # 事件绑定
        generate_btn.click(
            fn=app.generate_prompts,
            inputs=[search_input],
            outputs=[img2img_prompts, img2vid_prompts, narrations, business_points, service_overview]
        )
        
        update_btn.click(
            fn=app.update_prompts,
            inputs=[img2img_prompts, img2vid_prompts, narrations, business_points, service_overview],
            outputs=[update_status]
        )
        
        generate_images_btn.click(
            fn=app.generate_images,
            inputs=[ref_image],
            outputs=[image_gallery]
        )
        
        generate_videos_btn.click(
            fn=app.generate_all_videos,
            outputs=[video_gallery]
        )
        
        regenerate_video_btn.click(
            fn=lambda idx, prompt: app.generate_single_video(int(idx)-1, prompt),
            inputs=[scene_selector, custom_video_prompt],
            outputs=[single_video_output]
        )
        
        generate_audios_btn.click(
            fn=app.generate_audios,
            outputs=[audio_outputs]
        )
        
        merge_btn.click(
            fn=app.merge_final_video,
            outputs=[final_video]
        ).then(
            fn=lambda: (app.current_session["prompts"].get("business_points", ""),
                       app.current_session["prompts"].get("service_overview", "")),
            outputs=[business_display, service_display]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )
