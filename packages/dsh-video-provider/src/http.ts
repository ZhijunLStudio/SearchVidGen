/**
 * 极小 HTTP 帮手：fetch + 超时 + AbortSignal 合并。
 * @module @zhijunlstudio/dsh-video-provider
 */

import { createWriteStream } from 'node:fs'
import { Readable } from 'node:stream'

export interface HttpOptions {
  timeoutMs?: number
  signal?: AbortSignal
}

/** 将外部 signal 与超时合并为一个 abort signal。 */
export function withTimeout(signal: AbortSignal | undefined, timeoutMs: number): { signal: AbortSignal; clear: () => void } {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(new Error('请求超时（' + timeoutMs + 'ms）')), timeoutMs)
  if (signal !== undefined) {
    if (signal.aborted) {
      clearTimeout(timer)
      controller.abort(signal.reason)
    } else {
      signal.addEventListener('abort', () => controller.abort(signal.reason), { once: true })
    }
  }
  return { signal: controller.signal, clear: () => clearTimeout(timer) }
}

/** JSON POST；非 2xx 抛错并带响应体摘要。 */
export async function postJson(url: string, body: unknown, headers: Record<string, string>, options: HttpOptions = {}): Promise<unknown> {
  const { signal, clear } = withTimeout(options.signal, options.timeoutMs ?? 120_000)
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...headers },
      body: JSON.stringify(body),
      signal,
    })
    const text = await response.text()
    if (!response.ok) {
      throw new Error('POST ' + url + ' 失败(' + response.status + '): ' + text.slice(0, 400))
    }
    return text.length > 0 ? JSON.parse(text) : undefined
  } finally {
    clear()
  }
}

/** JSON GET；非 2xx 抛错并带响应体摘要。 */
export async function getJson(url: string, headers: Record<string, string>, options: HttpOptions = {}): Promise<unknown> {
  const { signal, clear } = withTimeout(options.signal, options.timeoutMs ?? 60_000)
  try {
    const response = await fetch(url, { method: 'GET', headers, signal })
    const text = await response.text()
    if (!response.ok) {
      throw new Error('GET ' + url + ' 失败(' + response.status + '): ' + text.slice(0, 400))
    }
    return text.length > 0 ? JSON.parse(text) : undefined
  } finally {
    clear()
  }
}

/** 下载二进制到本地文件（流式）。 */
export async function downloadToFile(url: string, target: string, headers: Record<string, string>, options: HttpOptions = {}): Promise<void> {
  const { signal, clear } = withTimeout(options.signal, options.timeoutMs ?? 300_000)
  try {
    const response = await fetch(url, { method: 'GET', headers, signal })
    if (!response.ok) {
      throw new Error('下载失败(' + response.status + '): ' + url)
    }
    const writeStream = createWriteStream(target)
    const webStream = response.body
    if (webStream === null) {
      throw new Error('下载响应无 body: ' + url)
    }
    const nodeStream = Readable.fromWeb(webStream as never)
    await new Promise<void>((resolve, reject) => {
      nodeStream.pipe(writeStream)
      writeStream.on('finish', () => resolve())
      writeStream.on('error', reject)
      nodeStream.on('error', reject)
    })
  } finally {
    clear()
  }
}