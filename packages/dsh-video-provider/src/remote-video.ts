/**
 * remote 视频生成器：HTTP 协议端点。
 *
 * 两种协议（配置 protocol 字段选择）：
 * - openai：POST {base}/videos/generations → {id}，GET {base}/videos/{id}
 *   轮询至 completed → 下载视频 URL；
 * - minimax：POST {base}/v2/video_generation（content 数组 + 图片角色）→
 *   {task_id}，GET {base}/v2/query/video_generation 轮询至 succeeded →
 *   下载 content.url（参考官方文档与 vidharness generator.minimax-h3-api）。
 *
 * 未知协议参数经 defaultParams 原样透传——端点支持什么就配什么。
 * @module @zhijunlstudio/dsh-video-provider
 */

import { mkdirSync } from 'node:fs'
import { join } from 'node:path'
import type { GeneratorCallOptions, GeneratorProvider, GeneratedVideo, GeneratorRequest } from '@zhijunlstudio/dsh-video'
import type { Context } from '@deepseek-ai/cordis'
import type { RemoteVideoProviderConfig } from './config.ts'
import { resolveCredential } from './credentials.ts'
import { downloadToFile, getJson, postJson } from './http.ts'

const DEFAULT_POLL_MS = 5_000
const DEFAULT_TIMEOUT_MS = 30 * 60_000

/** 轮询等待（可中止）。 */
async function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  await new Promise<void>((resolve, reject) => {
    const timer = setTimeout(resolve, ms)
    signal?.addEventListener('abort', () => {
      clearTimeout(timer)
      reject(signal.reason ?? new Error('已取消'))
    }, { once: true })
  })
}

function asRecord(value: unknown): Record<string, unknown> {
  if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
    return value as Record<string, unknown>
  }
  throw new Error('协议响应必须是 JSON 对象，得到: ' + JSON.stringify(value).slice(0, 200))
}

/** 从多种形状里取视频下载 URL（openai: video.url / url；minimax: task.content.url）。 */
function pickVideoUrl(data: Record<string, unknown>): string {
  const video = data.video
  if (typeof video === 'object' && video !== null && !Array.isArray(video)) {
    const url = (video as Record<string, unknown>).url
    if (typeof url === 'string' && url.length > 0) return url
  }
  const content = data.content
  if (typeof content === 'object' && content !== null && !Array.isArray(content)) {
    const url = (content as Record<string, unknown>).url
    if (typeof url === 'string' && url.length > 0) return url
  }
  const url = data.url
  if (typeof url === 'string' && url.length > 0) return url
  throw new Error('协议响应里找不到视频下载 URL: ' + JSON.stringify(data).slice(0, 300))
}

/** 远程生成器提供者。 */
export class RemoteVideoProvider implements GeneratorProvider {
  readonly id: string
  readonly label?: string
  readonly capabilities

  constructor(
    private readonly config: RemoteVideoProviderConfig,
    private readonly ctx: Context,
  ) {
    this.id = config.id
    this.label = config.label
    this.capabilities = {
      backend: 'remote' as const,
      ...config.capabilities,
    }
  }

  async generate(request: GeneratorRequest, options: GeneratorCallOptions): Promise<GeneratedVideo> {
    const key = await resolveCredential(this.ctx, this.config.credential ?? 'MINIMAX_API_KEY')
    const authHeaders = { Authorization: 'Bearer ' + key }
    mkdirSync(options.workdir, { recursive: true })
    const target = join(options.workdir, 'out.mp4')
    const timeoutMs = this.config.timeoutMs ?? DEFAULT_TIMEOUT_MS
    const pollMs = this.config.pollIntervalMs ?? DEFAULT_POLL_MS
    const started = Date.now()
    const url = this.config.protocol === 'minimax'
      ? await this.runMinimax(request, authHeaders, options, timeoutMs, pollMs)
      : await this.runOpenai(request, authHeaders, options, timeoutMs, pollMs)
    options.signal?.throwIfAborted()
    await downloadToFile(url, target, {}, { signal: options.signal, timeoutMs: 300_000 })
    return {
      path: target,
      durationS: request.duration,
      seed: request.seed,
      model: this.config.model,
      backend: 'remote',
      params: { protocol: this.config.protocol, ...this.config.defaultParams },
      elapsedMs: Date.now() - started,
    }
  }

  private async runOpenai(
    request: GeneratorRequest,
    authHeaders: Record<string, string>,
    options: GeneratorCallOptions,
    timeoutMs: number,
    pollMs: number,
  ): Promise<string> {
    const base = this.config.baseUrl.replace(/\/+$/, '')
    const body: Record<string, unknown> = {
      model: this.config.model,
      prompt: request.text,
      ...this.config.defaultParams,
    }
    if (request.duration !== undefined) body.duration = request.duration
    if (request.seed !== undefined) body.seed = request.seed
    if (request.resolution !== undefined) body.size = resolutionToSize(request.resolution)
    if ((request.refs?.length ?? 0) > 0) body.images = request.refs
    options.onProgress?.('remote(' + this.id + '): 提交任务 ' + base + '/videos/generations')
    const created = asRecord(await postJson(base + '/videos/generations', body, authHeaders, { signal: options.signal, timeoutMs }))
    const id = created.id
    if (typeof id !== 'string' || id.length === 0) {
      throw new Error('openai 协议响应缺少任务 id: ' + JSON.stringify(created).slice(0, 200))
    }
    // 立即完成（部分实现同步返回 video url）
    const immediate = tryPickUrl(created)
    if (immediate !== undefined) return immediate
    for (let attempt = 0; ; attempt++) {
      options.signal?.throwIfAborted()
      await sleep(pollMs, options.signal)
      const status = asRecord(await getJson(base + '/videos/' + encodeURIComponent(id), authHeaders, { signal: options.signal, timeoutMs }))
      const state = status.status
      if (state === 'completed' || state === 'succeeded') return pickVideoUrl(status)
      if (state === 'failed' || state === 'cancelled') {
        throw new Error('openai 视频任务失败: ' + JSON.stringify(status).slice(0, 400))
      }
      options.onProgress?.('remote(' + this.id + '): 轮询 #' + (attempt + 1) + ' 状态 ' + String(state))
    }
  }

  private async runMinimax(
    request: GeneratorRequest,
    authHeaders: Record<string, string>,
    options: GeneratorCallOptions,
    timeoutMs: number,
    pollMs: number,
  ): Promise<string> {
    const base = this.config.baseUrl.replace(/\/+$/, '')
    const content: Record<string, unknown>[] = [{ type: 'text', text: request.text }]
    const upload = async (path: string, role: string): Promise<string> => {
      // MiniMax 上传协议：multipart 到 /v1/files/upload?purpose=video_generation
      const form = new FormData()
      const fs = await import('node:fs')
      form.append('file', new Blob([fs.readFileSync(path)], { type: 'image/jpeg' }), 'image.jpg')
      const response = await fetch(base + '/v1/files/upload?purpose=video_generation', {
        method: 'POST', headers: authHeaders, body: form, signal: options.signal,
      })
      const text = await response.text()
      if (!response.ok) throw new Error('minimax 图片上传失败(' + response.status + '): ' + text.slice(0, 300))
      const data = asRecord(JSON.parse(text))
      const file = data.file
      if (typeof file !== 'object' || file === null) throw new Error('minimax 上传响应缺少 file: ' + text.slice(0, 300))
      const url = (file as Record<string, unknown>).url
      if (typeof url !== 'string') throw new Error('minimax 上传响应缺少 file.url: ' + text.slice(0, 300))
      return url
    }
    if (request.firstFrame !== undefined) {
      content.push({ type: 'image_url', image_url: { url: await upload(request.firstFrame, 'first_frame') }, role: 'first_frame' })
    }
    if (request.lastFrame !== undefined) {
      content.push({ type: 'image_url', image_url: { url: await upload(request.lastFrame, 'last_frame') }, role: 'last_frame' })
    }
    for (const ref of request.refs ?? []) {
      content.push({ type: 'image_url', image_url: { url: await upload(ref, 'reference_image') }, role: 'reference_image' })
    }
    const body: Record<string, unknown> = {
      model: this.config.model,
      content,
      resolution: request.resolution ?? this.config.defaultParams?.resolution ?? '768P',
      duration: request.duration ?? this.config.defaultParams?.duration ?? 8,
      ratio: request.ratio ?? ((request.firstFrame ?? request.refs?.length) ? 'adaptive' : '16:9'),
      ...this.config.defaultParams,
    }
    options.onProgress?.('remote(' + this.id + '): 提交任务 ' + base + '/v2/video_generation')
    const submitted = asRecord(await postJson(base + '/v2/video_generation', body, authHeaders, { signal: options.signal, timeoutMs }))
    const taskId = submitted.task_id
    if (typeof taskId !== 'string' || taskId.length === 0) {
      throw new Error('minimax 协议响应缺少 task_id: ' + JSON.stringify(submitted).slice(0, 200))
    }
    for (let attempt = 0; ; attempt++) {
      options.signal?.throwIfAborted()
      await sleep(pollMs, options.signal)
      const qs = base + '/v2/query/video_generation?task_id=' + encodeURIComponent(taskId)
      const data = asRecord(await getJson(qs, authHeaders, { signal: options.signal, timeoutMs }))
      const task = data.task
      if (typeof task !== 'object' || task === null) {
        throw new Error('minimax 查询响应缺少 task: ' + JSON.stringify(data).slice(0, 300))
      }
      const state = (task as Record<string, unknown>).status
      if (state === 'succeeded') return pickVideoUrl(task as Record<string, unknown>)
      if (state === 'failed' || state === 'cancelled') {
        throw new Error('minimax 视频任务失败: ' + JSON.stringify(task).slice(0, 400))
      }
      options.onProgress?.('remote(' + this.id + '): 轮询 #' + (attempt + 1) + ' 状态 ' + String(state))
    }
  }
}

function tryPickUrl(data: Record<string, unknown>): string | undefined {
  try {
    return pickVideoUrl(data)
  } catch {
    return undefined
  }
}

/** 分辨率标签 → openai size 字符串（常见映射；未知原样透传）。 */
export function resolutionToSize(resolution: string): string {
  const sizes: Record<string, string> = {
    '480p': '854x480',
    '768p': '1280x768',
    '1080p': '1920x1080',
    '2K': '2048x1152',
    '16:9': '1280x720',
    '1:1': '1024x1024',
    '9:16': '720x1280',
  }
  return sizes[resolution] ?? resolution
}
