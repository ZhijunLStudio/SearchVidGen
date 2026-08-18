/**
 * local 视频生成器：本机引擎（vh gen-single JSON 子进程契约）。
 *
 * 契约（与 harness/vidharness/cli.py cmd_gen_single 对齐）：
 * - 输入：spec.json（text/refs/duration/ratio/seed/generator/out）；
 * - 输出：stdout 单行 JSON {video:{path,...}, judge, costUsd, elapsedS, runDir}；
 * - 失败：非零退出 + stderr 可读原因；超时/取消由本提供者管理。
 *
 * 这是与外部进程的 wire 边界：解析结果做严格校验（防御性校验在
 * 此处是正当的——子进程输出不可信）。
 * @module @zhijunlstudio/dsh-video-provider
 */

import { existsSync, mkdirSync, writeFileSync } from 'node:fs'
import { spawn } from 'node:child_process'
import { join } from 'node:path'
import type { GeneratorCallOptions, GeneratorProvider, GeneratedVideo, GeneratorRequest } from '@zhijunlstudio/dsh-video'
import type { LocalVideoProviderConfig } from './config.ts'

const DEFAULT_TIMEOUT_MS = 45 * 60_000
const KILL_GRACE_MS = 10_000

/** 本地引擎生成器提供者。 */
export class LocalVideoProvider implements GeneratorProvider {
  readonly id: string
  readonly label?: string
  readonly capabilities

  constructor(private readonly config: LocalVideoProviderConfig) {
    this.id = config.id
    this.label = config.label
    this.capabilities = {
      backend: 'local' as const,
      ...config.capabilities,
    }
  }

  async generate(request: GeneratorRequest, options: GeneratorCallOptions): Promise<GeneratedVideo> {
    mkdirSync(options.workdir, { recursive: true })
    const spec = {
      text: request.text,
      refs: request.refs ?? [],
      duration: request.duration,
      ratio: request.ratio,
      seed: request.seed,
      generator: {
        adapter: this.config.adapter ?? 'generator.minimax-h3-local',
        params: this.config.params ?? {},
      },
      gpu_price_usd_per_hour: this.config.gpuPriceUsdPerHour,
      ffmpeg_dir: this.config.ffmpegDir,
      out: options.workdir,
    }
    const specPath = join(options.workdir, 'gen-single.spec.json')
    writeFileSync(specPath, JSON.stringify(spec, null, 2), 'utf-8')

    const argv = this.config.pythonPath !== undefined
      ? [this.config.pythonPath, '-m', 'vidharness.cli', 'gen-single', specPath]
      : [this.config.engineCmd ?? 'vh', 'gen-single', specPath]
    options.onProgress?.('local(' + this.id + '): 启动引擎 ' + argv.join(' '))

    const started = Date.now()
    const { stdout, stderr, code } = await this.runEngine(argv, options)
    if (code !== 0) {
      throw new Error('本地引擎 gen-single 失败(exit ' + String(code) + '): ' + stderr.slice(-2000))
    }
    options.signal?.throwIfAborted()
    const result = validateLocalResult(JSON.parse(stdout) as unknown)
    if (!existsSync(result.video.path)) {
      throw new Error('本地引擎返回的视频文件不存在: ' + result.video.path)
    }
    return {
      path: result.video.path,
      durationS: result.video.durationS,
      seed: result.video.seed,
      model: result.video.model,
      backend: 'local',
      params: { adapter: result.video.model, engine: this.config.engineCmd ?? 'vh' },
      costUsd: result.costUsd,
      elapsedMs: Date.now() - started,
    }
  }

  private async runEngine(
    argv: string[],
    options: GeneratorCallOptions,
  ): Promise<{ stdout: string; stderr: string; code: number | null }> {
    return new Promise((resolve, reject) => {
      const child = spawn(argv[0], argv.slice(1), {
        cwd: this.config.cwd,
        env: process.env,
        stdio: ['ignore', 'pipe', 'pipe'],
      })
      let stdout = ''
      let stderr = ''
      let settled = false
      const timeoutMs = this.config.timeoutMs ?? DEFAULT_TIMEOUT_MS
      const timer = setTimeout(() => finish('timeout'), timeoutMs)

      function finish(kind: 'timeout' | 'abort' | 'exit'): void {
        if (settled) return
        settled = true
        clearTimeout(timer)
        if (kind === 'timeout') {
          child.kill('SIGTERM')
          setTimeout(() => child.kill('SIGKILL'), KILL_GRACE_MS).unref()
        }
      }

      const onAbort = (): void => {
        finish('abort')
        child.kill('SIGTERM')
        setTimeout(() => child.kill('SIGKILL'), KILL_GRACE_MS).unref()
      }
      options.signal?.addEventListener('abort', onAbort, { once: true })

      child.stdout.on('data', (chunk: Buffer) => {
        stdout += chunk.toString('utf-8')
      })
      child.stderr.on('data', (chunk: Buffer) => {
        const text = chunk.toString('utf-8')
        stderr += text
        for (const line of text.split(/\r?\n/)) {
          if (line.trim().length > 0) options.onProgress?.('local(' + this.id + '): ' + line.trim())
        }
      })
      child.on('error', (error) => {
        settled = true
        clearTimeout(timer)
        options.signal?.removeEventListener('abort', onAbort)
        reject(new Error('无法启动本地引擎 ' + argv[0] + ': ' + error.message
          + '（请确认 vh 在 PATH，或配置 pythonPath 指向 vidharness 所在环境的 python）'))
      })
      child.on('close', (code) => {
        if (settled) {
          options.signal?.removeEventListener('abort', onAbort)
          if (options.signal?.aborted) {
            reject(options.signal.reason ?? new Error('已取消'))
          } else {
            reject(new Error('本地引擎超时被终止(' + timeoutMs + 'ms)；stderr 尾部: ' + stderr.slice(-1000)))
          }
          return
        }
        settled = true
        clearTimeout(timer)
        options.signal?.removeEventListener('abort', onAbort)
        resolve({ stdout: stdout.trim(), stderr, code })
      })
    })
  }
}

interface LocalResult {
  video: {
    path: string
    durationS?: number
    seed?: number
    model: string
  }
  costUsd?: number
  elapsedS?: number
  judge?: unknown
  runDir?: string
}

/** 子进程输出校验（wire 边界）：形状不对响亮失败，绝不吞字段。 */
export function validateLocalResult(raw: unknown): LocalResult {
  // 必须是 function 声明：TS 控制流收窄只认声明/throw 的 never 调用
  function fail(message: string): never {
    throw new Error('本地引擎输出不符合 JSON 契约: ' + message + '；原始输出: ' + JSON.stringify(raw).slice(0, 500))
  }
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) fail('顶层必须是 JSON 对象')
  const data = raw as Record<string, unknown>
  const video = data.video
  if (typeof video !== 'object' || video === null || Array.isArray(video)) fail('缺少 video 对象')
  const v = video as Record<string, unknown>
  const path = v.path
  const model = v.model
  if (typeof path !== 'string' || path.length === 0) fail('video.path 必须是非空字符串')
  if (typeof model !== 'string' || model.length === 0) fail('video.model 必须是非空字符串')
  if (!path.startsWith('/')) fail('video.path 必须是绝对路径')
  const result: LocalResult = { video: { path, model } }
  const durationS = v.durationS
  if (durationS !== undefined && durationS !== null) {
    if (typeof durationS !== 'number' || !Number.isFinite(durationS)) fail('video.durationS 必须是有限数')
    result.video.durationS = durationS
  }
  const seed = v.seed
  if (seed !== undefined && seed !== null) {
    if (typeof seed !== 'number' || !Number.isInteger(seed)) fail('video.seed 必须是整数')
    result.video.seed = seed
  }
  const costUsd = data.costUsd
  if (costUsd !== undefined && costUsd !== null) {
    if (typeof costUsd !== 'number' || !Number.isFinite(costUsd)) fail('costUsd 必须是有限数')
    result.costUsd = costUsd
  }
  const elapsedS = data.elapsedS
  if (elapsedS !== undefined && elapsedS !== null) {
    if (typeof elapsedS !== 'number' || !Number.isFinite(elapsedS)) fail('elapsedS 必须是有限数')
    result.elapsedS = elapsedS
  }
  if (data.judge !== undefined) result.judge = data.judge
  if (typeof data.runDir === 'string') result.runDir = data.runDir
  return result
}