# import os
# import subprocess
# import tempfile
# from pathlib import Path
# import json

# class VideoProcessor:
#     def __init__(self):
#         """初始化视频处理器"""
#         self.check_ffmpeg()
    
#     def check_ffmpeg(self):
#         """检查FFmpeg是否可用"""
#         try:
#             subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
#             self.ffmpeg_available = True
#         except:
#             print("警告: FFmpeg未安装或不可用")
#             self.ffmpeg_available = False
    
#     def get_duration(self, media_path: str) -> float:
#         """获取媒体文件时长"""
#         if not self.ffmpeg_available:
#             return 0.0
        
#         cmd = [
#             'ffprobe', '-v', 'quiet',
#             '-show_entries', 'format=duration',
#             '-of', 'json',
#             str(media_path)
#         ]
        
#         try:
#             result = subprocess.run(cmd, capture_output=True, text=True, check=True)
#             data = json.loads(result.stdout)
#             return float(data['format']['duration'])
#         except:
#             return 0.0
    
#     def extend_video(self, video_path: str, target_duration: float, 
#                     output_path: str = None) -> str:
#         """延长视频到目标时长"""
#         if not self.ffmpeg_available:
#             return video_path
        
#         current_duration = self.get_duration(video_path)
#         if current_duration >= target_duration:
#             return video_path
        
#         if output_path is None:
#             output_path = f"extended_{Path(video_path).name}"
        
#         extension_duration = target_duration - current_duration
        
#         try:
#             # 使用最后一帧延长
#             cmd = [
#                 'ffmpeg', '-y', '-i', str(video_path),
#                 '-vf', f'tpad=stop_mode=clone:stop_duration={extension_duration}',
#                 '-c:a', 'copy',
#                 str(output_path)
#             ]
            
#             subprocess.run(cmd, capture_output=True, check=True)
#             return output_path
            
#         except subprocess.CalledProcessError:
#             return video_path
    
#     def create_slow_zoom_extension(self, video_path: str, extension_duration: float,
#                                   output_path: str) -> bool:
#         """创建缓慢放大的延长效果"""
#         if not self.ffmpeg_available:
#             return False
        
#         video_duration = self.get_duration(video_path)
#         if video_duration == 0:
#             return False
        
#         # 使用最后2秒作为素材
#         source_duration = min(2.0, video_duration)
#         source_start = max(0, video_duration - source_duration)
        
#         temp_source = f"temp_source_{os.path.basename(output_path)}"
#         temp_extension = f"temp_ext_{os.path.basename(output_path)}"
        
#         try:
#             # 1. 提取最后的片段
#             cmd_extract = [
#                 'ffmpeg', '-y',
#                 '-ss', str(source_start),
#                 '-i', str(video_path),
#                 '-t', str(source_duration),
#                 '-c:v', 'libx264',
#                 '-c:a', 'aac',
#                 temp_source
#             ]
#             subprocess.run(cmd_extract, capture_output=True, check=True)
            
#             # 2. 创建缓慢放大效果（从100%到110%）
#             zoom_filter = (
#                 f"scale='iw*(1+0.1*t/{extension_duration})':'ih*(1+0.1*t/{extension_duration})',"
#                 f"crop=in_w:in_h:(in_w-out_w)/2:(in_h-out_h)/2,"
#                 f"fade=t=out:st={extension_duration-1.5}:d=1.5"
#             )
            
#             # 计算需要循环的次数
#             loop_count = int(extension_duration / source_duration) + 1
            
#             cmd_animate = [
#                 'ffmpeg', '-y',
#                 '-stream_loop', str(loop_count),
#                 '-i', temp_source,
#                 '-vf', zoom_filter,
#                 '-t', str(extension_duration),
#                 '-c:v', 'libx264',
#                 '-pix_fmt', 'yuv420p',
#                 '-an',  # 移除音频
#                 temp_extension
#             ]
#             subprocess.run(cmd_animate, capture_output=True, check=True)
            
#             # 3. 拼接原视频和延长部分
#             with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
#                 f.write(f"file '{os.path.abspath(video_path)}'\n")
#                 f.write(f"file '{os.path.abspath(temp_extension)}'\n")
#                 list_file = f.name
            
#             cmd_concat = [
#                 'ffmpeg', '-y',
#                 '-f', 'concat',
#                 '-safe', '0',
#                 '-i', list_file,
#                 '-c', 'copy',
#                 str(output_path)
#             ]
#             subprocess.run(cmd_concat, capture_output=True, check=True)
            
#             os.unlink(list_file)
#             return True
            
#         except subprocess.CalledProcessError:
#             return False
#         finally:
#             # 清理临时文件
#             for temp_file in [temp_source, temp_extension]:
#                 if os.path.exists(temp_file):
#                     os.remove(temp_file)
    
#     def merge_video_with_audio(self, video_path: str, audio_path: str,
#                               output_path: str, extend_video: bool = True) -> str:
#         """合并单个视频和音频"""
#         if not self.ffmpeg_available:
#             return video_path
        
#         video_duration = self.get_duration(video_path)
#         audio_duration = self.get_duration(audio_path)
        
#         # 如果需要延长视频以匹配音频
#         if extend_video and audio_duration > video_duration:
#             extension_duration = audio_duration - video_duration + 0.5  # 额外0.5秒缓冲
            
#             temp_extended = f"temp_extended_{os.path.basename(video_path)}"
            
#             # 尝试创建动画延长效果
#             if not self.create_slow_zoom_extension(video_path, extension_duration, temp_extended):
#                 # 如果失败，使用简单延长
#                 temp_extended = self.extend_video(video_path, audio_duration + 0.5, temp_extended)
            
#             video_path = temp_extended
        
#         try:
#             # 合并视频和音频
#             cmd = [
#                 'ffmpeg', '-y',
#                 '-i', str(video_path),
#                 '-i', str(audio_path),
#                 '-c:v', 'copy',
#                 '-c:a', 'aac',
#                 '-map', '0:v:0',
#                 '-map', '1:a:0',
#                 '-shortest',
#                 str(output_path)
#             ]
            
#             subprocess.run(cmd, capture_output=True, check=True)
            
#             # 清理临时文件
#             if 'temp_extended' in locals() and os.path.exists(temp_extended):
#                 os.remove(temp_extended)
            
#             return output_path
            
#         except subprocess.CalledProcessError as e:
#             print(f"合并视频音频失败: {e}")
#             return video_path
    
#     def merge_videos_with_audio(self, video_paths: list, audio_paths: list,
#                                output_path: Path, extension_duration: float = 2.5) -> Path:
#         """合并多个视频片段和对应的音频"""
#         if not self.ffmpeg_available:
#             return None
        
#         output_path = Path(output_path)
#         output_path.parent.mkdir(parents=True, exist_ok=True)
        
#         # 创建临时目录
#         temp_dir = Path(tempfile.mkdtemp())
        
#         try:
#             # 1. 为每个片段创建完整的视频（视频+音频）
#             segments = []
            
#             for i, (video_path, audio_path) in enumerate(zip(video_paths, audio_paths)):
#                 if not video_path or not audio_path:
#                     continue
                
#                 segment_path = temp_dir / f"segment_{i+1:02d}.mp4"
                
#                 # 合并单个片段的视频和音频
#                 merged_path = self.merge_video_with_audio(
#                     video_path, audio_path, 
#                     str(segment_path), 
#                     extend_video=True
#                 )
                
#                 if merged_path and os.path.exists(merged_path):
#                     segments.append(merged_path)
            
#             if not segments:
#                 print("没有成功的片段")
#                 return None
            
#             # 2. 拼接所有片段
#             with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
#                 for segment in segments:
#                     f.write(f"file '{os.path.abspath(segment)}'\n")
#                 list_file = f.name
            
#             cmd_concat = [
#                 'ffmpeg', '-y',
#                 '-f', 'concat',
#                 '-safe', '0',
#                 '-i', list_file,
#                 '-c', 'copy',
#                 str(output_path)
#             ]
            
#             subprocess.run(cmd_concat, capture_output=True, check=True)
            
#             os.unlink(list_file)
            
#             # 验证输出
#             if output_path.exists():
#                 duration = self.get_duration(str(output_path))
#                 print(f"最终视频时长: {duration:.2f}秒")
#                 return output_path
            
#             return None
            
#         except Exception as e:
#             print(f"视频合并失败: {e}")
#             return None
#         finally:
#             # 清理临时文件
#             import shutil
#             if temp_dir.exists():
#                 shutil.rmtree(temp_dir)
    
#     def add_subtitles(self, video_path: str, subtitles: list, output_path: str) -> str:
#         """为视频添加字幕"""
#         if not self.ffmpeg_available:
#             return video_path
        
#         # 创建SRT字幕文件
#         srt_path = Path(video_path).with_suffix('.srt')
        
#         with open(srt_path, 'w', encoding='utf-8') as f:
#             for i, (start_time, end_time, text) in enumerate(subtitles):
#                 f.write(f"{i+1}\n")
#                 f.write(f"{self._format_time(start_time)} --> {self._format_time(end_time)}\n")
#                 f.write(f"{text}\n\n")
        
#         try:
#             # 添加字幕
#             cmd = [
#                 'ffmpeg', '-y',
#                 '-i', str(video_path),
#                 '-vf', f"subtitles={srt_path}:force_style='Fontsize=24,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2'",
#                 '-c:a', 'copy',
#                 str(output_path)
#             ]
            
#             subprocess.run(cmd, capture_output=True, check=True)
            
#             # 清理字幕文件
#             os.unlink(srt_path)
            
#             return output_path
            
#         except subprocess.CalledProcessError:
#             if os.path.exists(srt_path):
#                 os.unlink(srt_path)
#             return video_path
    
#     def _format_time(self, seconds: float) -> str:
#         """格式化时间为SRT格式"""
#         hours = int(seconds // 3600)
#         minutes = int((seconds % 3600) // 60)
#         secs = seconds % 60
#         return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')

# # 用于测试
# if __name__ == "__main__":
#     processor = VideoProcessor()
    
#     # 测试获取时长
#     # duration = processor.get_duration("test.mp4")
#     # print(f"Duration: {duration}")
    
#     # 测试合并
#     # result = processor.merge_video_with_audio("video.mp4", "audio.wav", "output.mp4")
#     # print(f"Merged: {result}")



import os
import re
import subprocess
import tempfile
from pathlib import Path

def get_duration(path):
    """通用：获取音视频时长（秒）"""
    cmd = [
        'ffprobe', '-v', 'quiet',
        '-show_entries', 'format=duration',
        '-of', 'csv=p=0', str(path)
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(r.stdout.strip())
    except:
        return None

import tempfile
import os
import subprocess

import tempfile, os, subprocess

import subprocess, os

import subprocess, os

def create_slow_zoom_extension(video_path, extension_duration, output_path):
    """
    用 tpad 复制最后一帧来延长视频，并用 mpeg4 编码，
    以规避 x264 在当前环境下不可用的问题。
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"tpad=stop_mode=clone:stop_duration={extension_duration}",
        "-an",                   # 丢弃音轨，后面再单独拼音频
        "-c:v", "mpeg4",         # 换成你的环境能用的编码器
        "-qscale:v", "3",        # 可选：控制 mpeg4 质量，范围 1–31，数字越小质量越好
        str(output_path)
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        # 如果连 mpeg4 都不行，再尝试 libopenh264
        err = e.stderr.decode()
        print("⛔ 延长动画（tpad+mpeg4）失败：", err.strip().splitlines()[0])
        print("→ 再试一次：tpad+libopenh264…")
        cmd[6] = "libopenh264"  # 将 -c:v mpeg4 改为 -c:v libopenh264
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e2:
            print("⛔ 延长动画（tpad+libopenh264）也失败：", e2.stderr.decode().strip().splitlines()[0])
            return False




def create_video_segment(video_path, audio_path, ext_dur, output_path):
    """对一对 (video, audio) 做延长 + 对齐音频，输出到 output_path。"""
    vd = get_duration(video_path)
    ad = get_duration(audio_path)
    if vd is None or ad is None:
        print("⛔ 时长获取失败", video_path, audio_path)
        return False

    extended_vd = vd + ext_dur
    print(f"→ 视频 {vd:.2f}s + 延长 {ext_dur:.2f}s = {extended_vd:.2f}s，音频 {ad:.2f}s")

    # 1) 先尝试缓放延长
    tmp_ext_vid = tempfile.mktemp(suffix='.mp4')
    ok = create_slow_zoom_extension(video_path, ext_dur, tmp_ext_vid)
    if not ok:
        # 失败则退回到简单帧克隆延长
        subprocess.run([
            'ffmpeg', '-y',
            '-i', video_path,
            '-vf', f"tpad=stop_mode=clone:stop_duration={ext_dur}",
            '-an',
            tmp_ext_vid
        ], check=True, capture_output=True)

    # 2) 如果音频更长，则在视频末尾拼黑屏
    final_vid = tmp_ext_vid
    if ad > extended_vd:
        black_dur = ad - extended_vd
        tmp_black = tempfile.mktemp(suffix='.mp4')
        tmp_full  = tempfile.mktemp(suffix='.mp4')
        subprocess.run([
            'ffmpeg', '-y',
            '-f', 'lavfi',
            '-i', f"color=c=black:s=1920x1080:d={black_dur}:r=30",
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            tmp_black
        ], check=True, capture_output=True)
        with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as ff:
            ff.write(f"file '{os.path.abspath(tmp_ext_vid)}'\n")
            ff.write(f"file '{os.path.abspath(tmp_black)}'\n")
            lst = ff.name
        subprocess.run([
            'ffmpeg', '-y',
            '-f', 'concat', '-safe', '0',
            '-i', lst,
            '-c', 'copy',
            tmp_full
        ], check=True, capture_output=True)
        os.unlink(lst)
        os.remove(tmp_black)
        final_vid = tmp_full

    # 3) 合并音视频：只有音频比视频长时才加 -shortest
    cmd = [
        'ffmpeg', '-y',
        '-i', final_vid,
        '-i', audio_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-map', '0:v:0',
        '-map', '1:a:0'
    ]
    if ad > extended_vd:
        cmd.append('-shortest')
    cmd.append(output_path)
    subprocess.run(cmd, check=True, capture_output=True)

    # 清理临时
    for f in (tmp_ext_vid, final_vid):
        if os.path.exists(f) and f != video_path:
            os.remove(f)
    return True


def concat_segments(segments, output_path):
    """concat 多段落"""
    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
        for seg in segments:
            f.write(f"file '{os.path.abspath(seg)}'\n")
        lst = f.name
    try:
        subprocess.run([
            'ffmpeg', '-y',
            '-f', 'concat', '-safe', '0',
            '-i', lst,
            '-c', 'copy',
            output_path
        ], check=True, capture_output=True)
        return True
    finally:
        os.unlink(lst)

def natural_sort_key(name):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'(\d+)', name)]

def main(base_dir, extension_duration, output_name):
    vd_dir = os.path.join(base_dir, 'videos')
    ad_dir = os.path.join(base_dir, 'audios')
    tmp_dir = os.path.join(base_dir, 'temp_segments')
    final_out = os.path.join(base_dir, output_name)

    os.makedirs(tmp_dir, exist_ok=True)
    print(f"工作目录: {base_dir}")
    print(f"视频子目录: {vd_dir}")
    print(f"音频子目录: {ad_dir}")
    print(f"延长时长: {extension_duration}s")
    print("==========")

    vids = [f for f in os.listdir(vd_dir)
            if f.lower().endswith(('.mp4','.mov','.mkv','.avi'))]
    auds = [f for f in os.listdir(ad_dir)
            if f.lower().endswith(('.mp3','.wav','.aac','.m4a'))]
    vids.sort(key=natural_sort_key)
    auds.sort(key=natural_sort_key)
    print("发现视频:", vids)
    print("发现音频:", auds)

    segments = []
    for i, v in enumerate(vids):
        if i >= len(auds):
            break
        src_v = os.path.join(vd_dir, v)
        src_a = os.path.join(ad_dir, auds[i])
        out_seg = os.path.join(tmp_dir, f"segment_{i+1:02d}.mp4")
        print(f"\n▶ 处理第 {i+1} 段: {v} + {auds[i]}")
        if create_video_segment(src_v, src_a, extension_duration, out_seg):
            d = get_duration(out_seg) or 0
            print(f"  ✓ 完成, 时长 {d:.2f}s")
            segments.append(out_seg)
        else:
            print("  ✗ 失败")

    if not segments:
        print("没有生成任何段落，退出")
        return

    print("\n▶ 开始拼接所有段落")
    if concat_segments(segments, final_out):
        tot = get_duration(final_out) or 0
        print(f"✓ 最终视频生成: {final_out}，时长 {tot:.2f}s")
    else:
        print("✗ 拼接失败")

    print("临时文件夹:", tmp_dir)

if __name__ == '__main__':
    # 在这里硬编码你的参数：
    base_dir = r"generated_video_content/打工人的一天_20250612_220505"      # <-- 改成自己的目录
    extension_duration = 2.5                  # <-- 每段视频要延长的秒数
    output_name = "final_output.mp4"          # <-- 最终输出文件名

    # 检查 ffmpeg 是否可用
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except:
        print("✗ 未检测到 ffmpeg，请先安装")
        exit(1)

    main(base_dir, extension_duration, output_name)
