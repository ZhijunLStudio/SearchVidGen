/**
 * 媒体抽帧：ffmpeg 采样 → base64 data URL（裁判的多模态输入）。
 *
 * ffmpeg 缺失在第一次使用即响亮失败并给安装指引（vidharness E12 同口径：
 * 环境缺口不静默，也不假装看过媒体）。
 * @module @zhijunlstudio/dsh-video-provider
 */

import { execFileSync } from 'node:child_process'
import { existsSync, mkdirSync, readFileSync } from 'node:fs'
import { extname, join } from 'node:path'

const VIDEO_EXTS = new Set(['.mp4', '.mov', '.mkv', '.avi', '.webm'])

export interface SampledFrame {
  source: string
  atSeconds: number
  dataUrl: string
  mimeType: string
}

function findBinary(name: string, configured?: string): string {
  const candidate = configured ?? name
  try {
    execFileSync(candidate, ['-version'], { stdio: 'ignore' })
    return candidate
  } catch {
    throw new Error(
      '找不到 ' + name + ' 可执行文件（尝试过: ' + JSON.stringify(candidate) + '）。'
      + '请在提供者配置里设置 ffmpegPath，或安装 ffmpeg 并放入 PATH。',
    )
  }
}

function probeDuration(ffprobe: string, media: string): number {
  const out = execFileSync(ffprobe, [
    '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', media,
  ], { encoding: 'utf-8' })
  const duration = Number.parseFloat(out.trim())
  if (!Number.isFinite(duration) || duration <= 0) {
    throw new Error('ffprobe 无法读取时长: ' + media)
  }
  return duration
}

function frameDataUrl(path: string, atSeconds: number, ffmpeg: string, source: string, outDir: string, index: number): SampledFrame {
  const out = join(outDir, 'frame_' + String(index).padStart(2, '0') + '.jpg')
  execFileSync(ffmpeg, [
    '-y', '-ss', atSeconds.toFixed(3), '-i', source, '-frames:v', '1', '-q:v', '3', out,
  ], { stdio: 'ignore' })
  if (!existsSync(out)) {
    throw new Error('抽帧失败（ffmpeg 无输出）: ' + source + ' @' + atSeconds + 's')
  }
  const buffer = readFileSync(out)
  return {
    source,
    atSeconds,
    dataUrl: 'data:image/jpeg;base64,' + buffer.toString('base64'),
    mimeType: 'image/jpeg',
  }
}

/**
 * 对媒体列表采样若干帧。
 * @param media - 视频/图像绝对路径；图像直接整张转 data URL。
 * @param count - 每段视频采样帧数。
 * @param workdir - 帧产物目录。
 * @param ffmpegPath - 可选自定义 ffmpeg 路径。
 * @param signal - 取消信号。
 */
export async function sampleFrames(
  media: string[],
  count: number,
  workdir: string,
  ffmpegPath?: string,
  signal?: AbortSignal,
): Promise<SampledFrame[]> {
  const ffmpeg = findBinary('ffmpeg', ffmpegPath)
  const ffprobe = findBinary('ffprobe', ffmpegPath !== undefined ? ffmpegPath.replace(/ffmpeg$/, 'ffprobe') : undefined)
  mkdirSync(workdir, { recursive: true })
  const frames: SampledFrame[] = []
  let index = 0
  for (const source of media) {
    signal?.throwIfAborted()
    const ext = extname(source).toLowerCase()
    if (!VIDEO_EXTS.has(ext)) {
      // 图像直接作为单帧
      const buffer = readFileSync(source)
      frames.push({
        source,
        atSeconds: 0,
        dataUrl: 'data:image/jpeg;base64,' + buffer.toString('base64'),
        mimeType: 'image/jpeg',
      })
      index++
      continue
    }
    const duration = probeDuration(ffprobe, source)
    const effective = Math.max(1, Math.min(count, Math.floor(duration * 2)))
    for (let i = 0; i < effective; i++) {
      signal?.throwIfAborted()
      const at = effective === 1 ? 0 : (duration * i) / (effective - 1)
      frames.push(frameDataUrl(source, at, ffmpeg, source, workdir, index))
      index++
    }
  }
  return frames
}
