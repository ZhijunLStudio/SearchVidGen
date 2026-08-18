/**
 * 剧本规划（消费者拥有提示契约与输出契约）。
 *
 * 提示契约移植自 vidharness seams/script.py：换提供者不换面向模型的语言；
 * 输出契约：优先 fenced JSON，其次首尾花括号，失败抛可诊断错误。
 * 单段兜底（query 即视频指令）必须可见：报告里记 scriptError，不静默降级。
 * @module @zhijunlstudio/dsh-video-tool
 */

import type { Context } from '@deepseek-ai/cordis'
import { BlockAssembler, createUserMessage } from '@deepseek-ai/dsh-llm'
import type { ScriptConfig } from './config.ts'

export interface SegmentPlan {
  video_prompt: string
  narration?: string
  duration?: number
}

export interface ScriptResult {
  segments: SegmentPlan[]
  source: 'llm' | 'single'
  scriptError?: string
  raw?: string
}

export function buildScriptPrompt(query: string, segments: number, brief?: string, style?: string): string {
  const parts = [
    '你是一位专业的短视频导演兼视频生成提示词工程师。',
    '目标：根据用户意图规划一部 ' + segments + ' 个镜头的短视频。',
    '用户意图：' + query,
  ]
  if (brief !== undefined && brief.trim() !== '') {
    parts.push('补充要求：' + brief.trim())
  }
  if (style !== undefined && style.trim() !== '') {
    parts.push('视觉风格（每个镜头都要遵守）：' + style.trim())
  }
  parts.push(
    '每个镜头的 video_prompt 用英文写（40-80 词），包含主体、动作、镜头运动与环境氛围，'
    + '适合视频生成模型直接执行；narration 是可选中文旁白（≤20 字）；duration 为 3-15 的整数秒。',
    '严格只输出 JSON（不要其他文字）：',
    '{"segments": [{"video_prompt": "...", "narration": "...", "duration": 8}]}',
  )
  return parts.join('\n')
}

/** 解析剧本 JSON：fenced → 首尾花括号；失败抛可诊断错误（不吞）。 */
export function parseScriptJson(content: string, minSegments = 1): SegmentPlan[] {
  const trimmed = content.trim()
  let raw = trimmed
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/)
  if (fenced !== null) raw = fenced[1]
  let data: unknown
  try {
    data = JSON.parse(raw)
  } catch {
    const brace = raw.match(/\{[\s\S]*\}/)
    if (brace === null) {
      throw new Error('剧本输出无法解析（无 JSON）：原文前 300 字: ' + trimmed.slice(0, 300))
    }
    try {
      data = JSON.parse(brace[0])
    } catch (error) {
      throw new Error('剧本 JSON 解析失败: ' + String(error) + '；原文前 300 字: ' + trimmed.slice(0, 300))
    }
  }
  if (typeof data !== 'object' || data === null || Array.isArray(data)) {
    throw new Error('剧本 JSON 顶层必须是对象: ' + trimmed.slice(0, 300))
  }
  const list = (data as Record<string, unknown>).segments
  if (!Array.isArray(list) || list.length < minSegments) {
    throw new Error('剧本 JSON 缺少 segments 数组（至少 ' + minSegments + ' 条）: ' + trimmed.slice(0, 300))
  }
  const segments: SegmentPlan[] = []
  for (const item of list) {
    if (typeof item !== 'object' || item === null) {
      throw new Error('剧本 segment 必须是对象: ' + JSON.stringify(item).slice(0, 200))
    }
    const record = item as Record<string, unknown>
    if (typeof record.video_prompt !== 'string' || record.video_prompt.trim().length === 0) {
      throw new Error('剧本 segment 缺少非空 video_prompt: ' + JSON.stringify(item).slice(0, 200))
    }
    const plan: SegmentPlan = { video_prompt: record.video_prompt.trim() }
    if (typeof record.narration === 'string' && record.narration.trim().length > 0) {
      plan.narration = record.narration.trim().slice(0, 200)
    }
    if (typeof record.duration === 'number' && Number.isFinite(record.duration)) {
      plan.duration = Math.min(15, Math.max(3, Math.round(record.duration)))
    }
    segments.push(plan)
  }
  return segments
}

/** 用宿主 LLM 生成分镜计划；segments<=1 或 LLM 失败时走单段兜底（错误可见）。 */
export async function generateScript(
  ctx: Context,
  config: ScriptConfig,
  query: string,
  segments: number,
  brief?: string,
  style?: string,
  signal?: AbortSignal,
): Promise<ScriptResult> {
  if (segments <= 1) {
    return { segments: [{ video_prompt: query }], source: 'single' }
  }
  const provider = config.provider ?? 'deepseek-official'
  const model = config.model ?? 'deepseek-v4-flash'
  const prompt = buildScriptPrompt(query, segments, brief, style)
  try {
    const assembler = new BlockAssembler()
    for await (const chunk of ctx.llm.stream({
      provider,
      model,
      messages: [createUserMessage({ content: [{ type: 'text', text: prompt }], source: { kind: 'user' } })],
      system: '你只输出 JSON，不输出任何解释。',
      temperature: config.temperature ?? 0.9,
      maxTokens: config.maxTokens ?? 8192,
      signal,
    })) {
      signal?.throwIfAborted()
      assembler.push(chunk)
    }
    if (assembler.finish !== undefined && assembler.finish.kind !== 'stop') {
      throw new Error('剧本 LLM 调用未正常结束: ' + JSON.stringify(assembler.finish))
    }
    const text = assembler.blocks()
      .filter(block => block.type === 'text')
      .map(block => block.text)
      .join('\n')
      .trim()
    if (text.length === 0) {
      throw new Error('剧本 LLM 没有输出文本')
    }
    return { segments: parseScriptJson(text, 1), source: 'llm', raw: text.slice(0, 4000) }
  } catch (error) {
    // 剧本失败不静默：单段兜底 + 报告里记录原因（模型可见⟺日志）
    return {
      segments: [{ video_prompt: query }],
      source: 'single',
      scriptError: (error as Error).message ?? String(error),
    }
  }
}
