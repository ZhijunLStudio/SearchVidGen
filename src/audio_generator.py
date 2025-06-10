import os
import numpy as np
import soundfile as sf
import torch
from pathlib import Path
# 确保你已经安装了 kokoro 库：pip install kokoro>=0.8.1 "misaki[zh]>=0.8.1"
from kokoro import KModel, KPipeline
import re # Import re for cleaning up loaded text if needed

class KokoroAudioGenerator:
    def __init__(self, model_path: str = 'models/Kokoro-82M-v1.1-zh'):
        """
        初始化 Kokoro TTS 音频生成器
        model_path: Kokoro 模型文件的本地路径
        """
        self.model_path = Path(model_path)
        self.sample_rate = 24000 # Kokoro 模型的采样率通常是 24000 Hz
        self.voice = 'zm_009'    # 默认中文女声，可以根据需要更改为 'zm_010' (男声)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' # 优先使用 GPU

        self._init_kokoro_model()

    def _init_kokoro_model(self):
        """初始化 Kokoro TTS 模型"""
        try:
            # 模型文件路径
            model_file = self.model_path / "kokoro-v1_1-zh.pth"
            config_file = self.model_path / "config.json"

            # 检查模型文件是否存在
            if not model_file.exists() or not config_file.exists():
                raise FileNotFoundError(
                    f"Kokoro 模型文件未找到。请确保 '{model_file.name}' 和 '{config_file.name}' "
                    f"存在于路径: {self.model_path}"
                )

            # 只有当模型未加载时才进行加载
            if not hasattr(self, 'model') or self.model is None:
                print("Loading Kokoro model...")
                self.model = KModel(
                    repo_id='hexgrad/Kokoro-82M-v1.1-zh',
                    model=str(model_file), # KModel 需要字符串路径
                    config=str(config_file)
                ).to(self.device).eval()

                self.zh_pipeline = KPipeline(
                    lang_code='z',
                    repo_id='hexgrad/Kokoro-82M-v1.1-zh', # 这里也保持一致
                    model=self.model
                )
                print(f"Kokoro 模型加载成功，使用设备: {self.device}")
            else:
                print("Kokoro 模型已加载。")

        except ImportError:
            print("错误: 'kokoro' 库未安装。请运行: pip install kokoro>=0.8.1 \"misaki[zh]>=0.8.1\"")
            self.model = None
            self.zh_pipeline = None
        except FileNotFoundError as e:
            print(f"错误: Kokoro 模型初始化失败 - {e}")
            self.model = None
            self.zh_pipeline = None
        except Exception as e:
            print(f"错误: 初始化 Kokoro 模型时发生未知错误 - {e}")
            self.model = None
            self.zh_pipeline = None

    def generate(self, text: str, output_path: Path = None, **kwargs) -> Path | None:
        """
        使用 Kokoro TTS 生成音频
        Args:
            text: 要转换的文本
            output_path: 生成音频的保存路径。如果未指定，将生成一个默认文件名。
            **kwargs: 其他参数，例如 'voice' 可以改变发音人。
        Returns:
            生成的音频文件路径，如果生成失败则返回 None。
        """
        if self.model is None or self.zh_pipeline is None:
            print("错误: Kokoro 模型未成功加载，无法生成音频。")
            return None

        if output_path is None:
            # 默认文件名，基于文本内容的哈希值和时间戳
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            output_path = Path(f"generated_kokoro_audio_{timestamp}.wav")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True) # 确保输出目录存在

        try:
            current_voice = kwargs.get('voice', self.voice)
            generator = self.zh_pipeline(text, voice=current_voice)
            wav_data = next(generator).audio

            sf.write(str(output_path), wav_data, self.sample_rate)
            print(f"音频已生成并保存到: {output_path}")
            return output_path
        except Exception as e:
            print(f"错误: 使用 Kokoro 生成音频失败 - {e}")
            return None

    def batch_generate(self, texts: list[str], output_dir: Path, **kwargs) -> list[Path]:
        """
        批量生成音频
        Args:
            texts: 文本列表，每个元素对应一个要生成的音频。
            output_dir: 批量生成音频的输出目录。
            **kwargs: 传递给 generate 方法的额外参数。
        Returns:
            生成的音频文件路径列表。
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for i, text in enumerate(texts):
            # 为每个文本生成一个带序号的文件名
            output_path = output_dir / f"narration_{i+1:02d}.wav" # Use narration_01.wav, narration_02.wav etc.
            print(f"Generating audio for narration {i+1}: '{text}'")
            audio_path = self.generate(text, output_path, **kwargs)
            if audio_path:
                results.append(audio_path)
            else:
                print(f"Failed to generate audio for narration {i+1}.")
        return results

# --- Main execution block for loading narrations and generating audio ---
if __name__ == "__main__":
    # --- Configuration ---
    # !!! IMPORTANT: Set this to the actual path of your LLMClient output folder !!!
    # This folder should contain your 'narrations.txt' file.
    input_content_dir = Path("generated_video_content/如何学习Python编程_20250611_010357") # EXAMPLE PATH

    # Output subfolder name within input_content_dir for generated audios
    output_audio_subfolder = "audios"

    # Path to your Kokoro model files (adjust if different)
    kokoro_model_base_path = '/data/home/lizhijun/opensource/kokoro-pipeline/models/Kokoro-82M-v1.1-zh'
    # --- End Configuration ---


    # --- Initialize Audio Generator ---
    print("\n--- Initializing Kokoro Audio Generator ---")
    kokoro_gen = KokoroAudioGenerator(model_path=kokoro_model_base_path)

    # --- Load Narrations ---
    narrations_file_path = input_content_dir / "narrations.txt"
    if not narrations_file_path.exists():
        print(f"Error: narrations.txt not found at {narrations_file_path}.")
        print("Please ensure you have run llm_client.py to generate content or specify the correct input_content_dir.")
        exit()

    narrations = []
    with open(narrations_file_path, "r", encoding="utf-8") as f:
        for line in f:
            clean_line = line.strip() # Remove any leading/trailing whitespace
            if clean_line:
                narrations.append(clean_line)

    if not narrations:
        print(f"Warning: No narrations found in {narrations_file_path}. Exiting.")
        exit()

    print(f"Loaded {len(narrations)} narrations from {narrations_file_path}.")

    # --- Setup Output Directory for Audios ---
    output_audio_dir = input_content_dir / output_audio_subfolder
    output_audio_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generated audios will be saved to: {output_audio_dir}")

    # --- Batch Generate Audios ---
    print("\n--- Starting Audio Generation ---")
    generated_audio_paths = kokoro_gen.batch_generate(narrations, output_audio_dir)

    if generated_audio_paths:
        print(f"\n--- Audio Generation Complete ---")
        print(f"Total audios generated: {len(generated_audio_paths)}")
        print(f"All generated audios are in: {output_audio_dir}")
        for p in generated_audio_paths:
            print(f"- {p}")
    else:
        print("\n--- Audio Generation Failed ---")
        print("No audio files were generated.")

    # You now have all generated audio paths in `generated_audio_paths`
    # This list can be passed to your VideoGenerator along with image paths.