/**
 * FFmpeg 操作：探测/抽帧/拼接。
 * ffmpeg 缺失在第一次使用即响亮失败并给安装指引。
 * @module @zhijunlstudio/dsh-video-tool
 */

import { execFileSync } from 'node:child_process'
import { existsSync, mkdirSync, unlinkSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'

export interface MediaInfo {
  durationS: number
  fps: number
  width: number
  height: number
  hasAudio: boolean
  codec: string
}

export function findBinary(name: string, configured?: string): string {
  const candidate = configured ?? name
  try {
    execFileSync(candidate, ['-version'], { stdio: 'ignore' })
    return candidate
  } catch {
    throw new Error(
      '找不到 ' + name + ' 可执行文件（尝试过: ' + JSON.stringify(candidate) + '）。'
      + '请安装 ffmpeg 并在工具配置里设置 ffmpegPath，或放入 PATH。',
    )
  }
}

function probeJson(ffprobe: string, path: string): Record<string, unknown> {
  const out = execFileSync(ffprobe, [
    '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', path,
  ], { encoding: 'utf-8' })
  const data = JSON.parse(out) as unknown
  if (typeof data !== 'object' || data === null) {
    throw new Error('ffprobe 输出不是 JSON 对象: ' + path)
  }
  return data as Record<string, unknown>
}

/** 探测视频媒体信息（video_verify 与编排共用）。 */
export function probeVideo(path: string, ffmpegPath?: string): MediaInfo {
  if (!existsSync(path)) {
    throw new Error('媒体文件不存在: ' + path)
  }
  const ffprobe = findBinary('ffprobe', ffmpegPath !== undefined ? ffmpegPath.replace(/ffmpeg$/, 'ffprobe') : undefined)
  const data = probeJson(ffprobe, path)
  const streams = data.streams
  if (!Array.isArray(streams) || streams.length === 0) {
    throw new Error('ffprobe 未发现流: ' + path)
  }
  const videoStream = streams.find(s => (s as Record<string, unknown>).codec_type === 'video') as Record<string, unknown> | undefined
  if (videoStream === undefined) {
    throw new Error('没有视频流: ' + path)
  }
  const audioStream = streams.find(s => (s as Record<string, unknown>).codec_type === 'audio')
  const fpsRaw = typeof videoStream.r_frame_rate === 'string' ? videoStream.r_frame_rate : '0/1'
  const [num, den] = fpsRaw.split('/').map(part => Number.parseFloat(part))
  const fps = Number.isFinite(num) && Number.isFinite(den) && den !== 0 ? num / den : 0
  const durationS = typeof data.format === 'object' && data.format !== null
    ? Number.parseFloat(String((data.format as Record<string, unknown>).duration ?? '0'))
    : 0
  return {
    durationS: Number.isFinite(durationS) ? durationS : 0,
    fps: Number.isFinite(fps) ? Math.round(fps * 100) / 100 : 0,
    width: Number(videoStream.width ?? 0),
    height: Number(videoStream.height ?? 0),
    hasAudio: audioStream !== undefined,
    codec: String(videoStream.codec_name ?? ''),
  }
}

/** 抽一帧 JPG 到 outDir，返回帧文件路径。 */
export function extractFrame(video: string, atSeconds: number, outDir: string, name: string, ffmpegPath?: string): string {
  const ffmpeg = findBinary('ffmpeg', ffmpegPath)
  mkdirSync(outDir, { recursive: true })
  const out = join(outDir, name + '.jpg')
  execFileSync(ffmpeg, [
    '-y', '-ss', atSeconds.toFixed(3), '-i', video, '-frames:v', '1', '-q:v', '3', out,
  ], { stdio: 'ignore' })
  if (!existsSync(out)) {
    throw new Error('抽帧失败（ffmpeg 无输出）: ' + video + ' @' + atSeconds + 's')
  }
  return out
}

/** concat demuxer 拼接多个 mp4 到 outPath（-c copy，保留音轨）。 */
export function concatVideos(videos: string[], outPath: string, ffmpegPath?: string): void {
  const ffmpeg = findBinary('ffmpeg', ffmpegPath)
  if (videos.length === 0) {
    throw new Error('concatVideos: 没有输入片段')
  }
  const listPath = join(outPath + '.list.txt')
  writeFileSync(listPath, videos.map(video => "file '" + video.replace(/'/g, "'\\''") + "'").join('\n') + '\n', 'utf-8')
  try {
    execFileSync(ffmpeg, [
      '-y', '-f', 'concat', '-safe', '0', '-i', listPath, '-c', 'copy', outPath,
    ], { stdio: 'ignore' })
  } finally {
    try {
      unlinkSync(listPath)
    } catch {
      // 临时文件清理失败不阻塞
    }
  }
  if (!existsSync(outPath)) {
    throw new Error('FFmpeg 拼接失败，未产出 ' + outPath)
  }
}
