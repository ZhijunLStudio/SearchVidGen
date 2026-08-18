/**
 * 裁判提供者：OpenAI 兼容 chat completions 端点。
 *
 * 一行配置覆盖三类裁判：本地 vLLM VLM（视频/图像模态）、DeepSeek API
 * （文本模态）、任何 OpenAI 兼容多模态端点。媒体输入 = ffmpeg 抽帧 →
 * data URL 图像部件；文本模态不发媒体。
 *
 * 协议纪律：
 * - 输出解析优先 JSON，正则兜底；都失败则抛可操作的错误
 *   （vidharness E21：不可解析的反馈不是空信号，必须可见）；
 * - 只返回原始 {scores, feedback, raw}，加权/阈值结算归消费者
 *   （computeVerdict 在 @zhijunlstudio/dsh-video）。
 * @module @zhijunlstudio/dsh-video-provider
 */

import { join } from 'node:path'
import type { Context } from '@deepseek-ai/cordis'
import type {
  JudgeCallOptions, JudgeCriterion, JudgeProvider, JudgeRequest, JudgeScores,
} from '@zhijunlstudio/dsh-video'
import type { RemoteJudgeProviderConfig } from './config.ts'
import { resolveCredential } from './credentials.ts'
import { postJson } from './http.ts'
import { sampleFrames } from './frames.ts'

const DEFAULT_FRAME_SAMPLES = 4
const DEFAULT_TIMEOUT_MS = 5 * 60_000

function buildPrompt(criteria: readonly JudgeCriterion[]): string {
  const lines = criteria.map(c => '  - ' + c.name + ': ' + c.question)
  return [
    '你是视频/图像质量裁判。根据给出的媒体，按以下维度打分（0-10，可一位小数）：',
    ...lines,
    '',
    '严格只输出 JSON（不要任何其他文字）：',
    '{"<维度名>": <0-10 分数>, "feedback": "<一句话，说明优点与问题>"}',
    '所有维度都必须给出分数。',
  ].join('\n')
}

/** 从裁判文本解析原始评分（JSON 优先，正则兜底，失败抛可操作错误）。 */
export function parseJudgeScores(text: string, criteria: readonly JudgeCriterion[]): { scores: Record<string, number>; feedback: string } {
  const trimmed = text.trim()
  // 1) JSON：支持 {"维度": 分, "feedback": ...} 与 {"scores": {...}, "feedback": ...}
  const brace = trimmed.match(/\{[\s\S]*\}/)
  if (brace !== null) {
    try {
      const data = JSON.parse(brace[0]) as unknown
      if (typeof data === 'object' && data !== null && !Array.isArray(data)) {
        const record = data as Record<string, unknown>
        const pool = (typeof record.scores === 'object' && record.scores !== null && !Array.isArray(record.scores))
          ? record.scores as Record<string, unknown>
          : record
        const scores: Record<string, number> = {}
        for (const criterion of criteria) {
          const value = pool[criterion.name] ?? pool[criterion.name.toLowerCase()]
          if (typeof value === 'number' && Number.isFinite(value)) scores[criterion.name] = value
          else if (typeof value === 'string' && value.trim() !== '') {
            const parsed = Number.parseFloat(value)
            if (Number.isFinite(parsed)) scores[criterion.name] = parsed
          }
        }
        const feedback = typeof record.feedback === 'string' ? record.feedback : ''
        if (Object.keys(scores).length === criteria.length) return { scores, feedback }
      }
    } catch {
      // 落到正则兜底
    }
  }
  // 2) 正则兜底："维度名[:：] 分数"
  const scores: Record<string, number> = {}
  for (const criterion of criteria) {
    const escaped = criterion.name.replace(/[.*+?^$|{}()\[\]\\]/g, '\\$&')
    const re = new RegExp(escaped + '\\s*[:：]?\\s*(\\d+(?:\\.\\d+)?)')
    const match = re.exec(trimmed)
    if (match !== null) {
      const parsed = Number.parseFloat(match[1])
      if (Number.isFinite(parsed)) scores[criterion.name] = parsed
    }
  }
  if (Object.keys(scores).length === criteria.length) return { scores, feedback: trimmed }
  // 3) 可操作失败（不吞、不伪造）
  const missing = criteria.filter(c => scores[c.name] === undefined).map(c => c.name)
  throw new Error(
    '裁判输出不可解析（缺维度: ' + missing.join(', ') + '）。'
    + '请重试并严格只输出 JSON：{"<维度名>": <0-10 分数>, "feedback": "<一句话>"}。'
    + '原文前 300 字: ' + trimmed.slice(0, 300),
  )
}

/** 远程裁判提供者。 */
export class RemoteJudgeProvider implements JudgeProvider {
  readonly id: string
  readonly label?: string
  readonly modalities: readonly ('video' | 'image' | 'text')[]

  constructor(
    private readonly config: RemoteJudgeProviderConfig,
    private readonly ctx: Context,
  ) {
    this.id = config.id
    this.label = config.label
    this.modalities = config.modalities
  }

  async judge(request: JudgeRequest, options: JudgeCallOptions): Promise<JudgeScores> {
    const key = await resolveCredential(this.ctx, this.config.credential ?? 'DEEPSEEK_API_KEY')
    const base = this.config.baseUrl.replace(/\/+$/, '')
    const frames = request.modality === 'text'
      ? []
      : await sampleFrames(
        request.media,
        this.config.frameSamples ?? DEFAULT_FRAME_SAMPLES,
        join(options.workdir, 'frames'),
        this.config.ffmpegPath,
        options.signal,
      )
    const content: unknown[] = [{ type: 'text', text: buildPrompt(request.criteria) }]
    for (const frame of frames) {
      content.push({ type: 'image_url', image_url: { url: frame.dataUrl } })
    }
    const body = {
      model: this.config.model,
      messages: [{ role: 'user', content }],
      max_tokens: this.config.maxTokens ?? 4096,
      temperature: this.config.temperature ?? 0,
    }
    options.onProgress?.('judge(' + this.id + '): 调用 ' + base + '/chat/completions')
    const data = await postJson(base + '/chat/completions', body, {
      Authorization: 'Bearer ' + key,
    }, { signal: options.signal, timeoutMs: this.config.timeoutMs ?? DEFAULT_TIMEOUT_MS })
    if (typeof data !== 'object' || data === null || Array.isArray(data)) {
      throw new Error('裁判端点响应不是 JSON 对象: ' + JSON.stringify(data).slice(0, 200))
    }
    const choices = (data as Record<string, unknown>).choices
    if (!Array.isArray(choices) || choices.length === 0) {
      throw new Error('裁判端点响应缺少 choices: ' + JSON.stringify(data).slice(0, 300))
    }
    const first = choices[0] as Record<string, unknown>
    const message = first.message
    if (typeof message !== 'object' || message === null) {
      throw new Error('裁判端点响应缺少 message: ' + JSON.stringify(first).slice(0, 300))
    }
    const text = (message as Record<string, unknown>).content
    if (typeof text !== 'string' || text.length === 0) {
      throw new Error('裁判端点响应缺少文本 content: ' + JSON.stringify(message).slice(0, 300))
    }
    const parsed = parseJudgeScores(text, request.criteria)
    return { scores: parsed.scores, feedback: parsed.feedback, raw: { text } }
  }
}
