import os
import subprocess
from pathlib import Path
from PIL import Image
import numpy as np
import cv2

class VideoGenerator:
    def __init__(self, model_type: str = "placeholder"):
        """
        初始化视频生成器
        model_type: 可以是 "placeholder", "wan2.1", "cogvideox" 等
        """
        self.model_type = model_type
        
        # 模型配置
        self.model_configs = {
            "wan2.1": {
                "model_path": "/data/home/lizhijun/llm/flux-hf/model/Wan-AI/Wan2.1-I2V-14B-480P",
                "size": "832*480",
                "sample_shift": 8,
                "sample_guide_scale": 6,
                "ring_size": 4
            }
        }
    
    def generate(self, image_path: str, prompt: str, output_path: Path = None, **kwargs) -> Path:
        """
        生成视频
        
        Args:
            image_path: 输入图片路径
            prompt: 视频生成提示词
            output_path: 输出路径
            **kwargs: 其他参数
            
        Returns:
            生成的视频路径
        """
        if output_path is None:
            output_path = Path(f"generated_video_{hash(prompt)}.mp4")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.model_type == "placeholder":
            return self._generate_placeholder(image_path, prompt, output_path)
        elif self.model_type == "wan2.1":
            return self._generate_with_wan(image_path, prompt, output_path, **kwargs)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def _generate_placeholder(self, image_path: str, prompt: str, output_path: Path) -> Path:
        """生成占位视频"""
        # 读取图片
        if os.path.exists(image_path):
            img = cv2.imread(str(image_path))
        else:
            # 创建占位图片
            img = np.ones((480, 832, 3), dtype=np.uint8) * 128
        
        # 创建一个简单的视频（5秒，30fps）
        fps = 30
        duration = 5
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (img.shape[1], img.shape[0]))
        
        # 添加一些简单的动画效果
        for i in range(fps * duration):
            frame = img.copy()
            
            # 添加移动的文字或效果
            progress = i / (fps * duration)
            x = int(progress * img.shape[1])
            
            # 绘制进度条
            cv2.rectangle(frame, (0, img.shape[0]-20), (x, img.shape[0]), (0, 255, 0), -1)
            
            # 添加文字
            text = f"Frame {i+1}/{fps*duration}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        cv2.destroyAllWindows()
        
        return output_path
    
    def _generate_with_wan(self, image_path: str, prompt: str, output_path: Path, **kwargs) -> Path:
        """使用Wan2.1模型生成视频"""
        config = self.model_configs["wan2.1"]
        
        # 构建命令
        cmd = [
            "torchrun",
            "--nproc_per_node=4",
            "generate.py",
            "--task", "i2v-14B",
            "--size", config["size"],
            "--ckpt_dir", config["model_path"],
            "--image", str(image_path),
            "--prompt", prompt,
            "--save_file", str(output_path),
            "--base_seed", str(kwargs.get("seed", 42)),
            "--dit_fsdp",
            "--t5_fsdp",
            "--ring_size", str(config["ring_size"]),
            "--sample_shift", str(config["sample_shift"]),
            "--sample_guide_scale", str(config["sample_guide_scale"])
        ]
        
        try:
            # 执行命令
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if output_path.exists():
                return output_path
            else:
                print(f"视频生成失败: {result.stderr}")
                return None
                
        except subprocess.CalledProcessError as e:
            print(f"视频生成出错: {e}")
            print(f"错误输出: {e.stderr}")
            # 返回占位视频
            return self._generate_placeholder(image_path, prompt, output_path)
        except FileNotFoundError:
            print("未找到generate.py或相关依赖，使用占位视频")
            return self._generate_placeholder(image_path, prompt, output_path)
    
    def batch_generate(self, image_prompt_pairs: list, output_dir: Path, **kwargs) -> list:
        """批量生成视频"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        for i, (image_path, prompt) in enumerate(image_prompt_pairs):
            output_path = output_dir / f"video_{i+1:02d}.mp4"
            video_path = self.generate(image_path, prompt, output_path, **kwargs)
            if video_path:
                results.append(str(video_path))
        
        return results

# 用于测试
if __name__ == "__main__":
    generator = VideoGenerator(model_type="placeholder")
    video_path = generator.generate(
        image_path="test.jpg",
        prompt="A person walking in the park"
    )
    print(f"Generated video: {video_path}")
