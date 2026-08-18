/**
 * video_generate_search 的编排闭环（Consumer）：
 * 剧本 → 逐段生成+评测重试 → 跨段检查 → FFmpeg 总装 → 报告。
 *
 * 职责边界（vidharness E1-E46 教训的直接移植）：
 * - 加权/阈值结算归本消费者（computeVerdict），提供者只给原始评分；
 * - 裁判不可用 ≠ 静默通过：报告 judged=false + 每段 verdict=null 可见；
 * - 跨段抽帧失败记 error 记录，不假装衔接检查做过；
 * - 无提供者/无裁判/无 ffmpeg 均响亮失败并给配置指引。
 * @module @zhijunlstudio/dsh-video-tool
 */

import { mkdirSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import type { Context } from '@deepseek-ai/cordis'
import {
  computeVerdict,
  type GeneratorProvider, type GeneratedVideo, type JudgeProvider, type JudgeCriterion, type Verdict,
} from '@zhijunlstudio/dsh-video'
import {
  DEFAULT_CROSS_CRITERIA, DEFAULT_SEGMENT_CRITERIA, defaultOutputRoot, type Config,
} from './config.ts'
import { concatVideos, extractFrame, probeVideo } from './ffmpeg.ts'
import { generateScript, type SegmentPlan } from './script.ts'

export interface SearchRequest {
  query: string
  brief?: string
  provider?: string
  judge?: string
  segments?: number
  duration?: number
  ratio?: string
  resolution?: string
  style?: string
  refs?: string[]
  firstFrame?: string
  lastFrame?: string
}

export interface SegmentReport {
  index: number
  prompt: string
  video: GeneratedVideo
  verdict: Verdict | null
  attempts: number
  judgeId?: string
}

export interface SearchReport {
  runId: string
  query: string
  script: { segments: number; source: 'llm' | 'single'; scriptError?: string }
  segments: SegmentReport[]
  cross: { pair: [number, number]; verdict?: Verdict; error?: string }[]
  final: { path: string; durationS?: number }
  totalCostUsd: number
  judged: boolean
  generator: string
  judge?: string
  workdir: string
}

export interface OrchestratorOptions {
  workdir: string
  signal?: AbortSignal
  onProgress?: (line: string) => void
}

function resolveJudgeForVideo(ctx: Context, preferredId: string | undefined): JudgeProvider | undefined {
  if (preferredId !== undefined) return ctx.video.getJudge(preferredId)
  try {
    return ctx.video.routeJudge({ media: [], modality: 'video', criteria: [] })
  } catch {
    return undefined
  }
}

/** 逐段生成 + 评测重试闭环。 */
async function generateSegmentWithJudge(
  generator: GeneratorProvider,
  judge: JudgeProvider | undefined,
  plan: SegmentPlan,
  criteria: readonly JudgeCriterion[],
  config: Config,
  index: number,
  options: OrchestratorOptions,
): Promise<{ video: GeneratedVideo; verdict: Verdict | null; attempts: number }> {
  const maxAttempts = (config.maxRetries ?? 2) + 1
  const injectFeedback = config.injectFeedback ?? true
  const feedbackPrefix = config.feedbackPrefix ?? '请修正以下问题后重新生成：'
  const segDir = join(options.workdir, 'segments', 'seg' + String(index).padStart(2, '0'))
  let text = plan.video_prompt
  let lastVideo: GeneratedVideo | undefined
  let lastVerdict: Verdict | null = null
  let attempts = 0

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    attempts = attempt
    options.signal?.throwIfAborted()
    options.onProgress?.('[' + index + '/' + '] 第 ' + attempt + ' 次生成（' + generator.id + '）')
    mkdirSync(segDir, { recursive: true })
    const request = {
      text,
      duration: plan.duration,
    }
    lastVideo = await generator.generate(request, {
      workdir: segDir,
      signal: options.signal,
      onProgress: options.onProgress,
    })
    if (judge === undefined || criteria.length === 0) {
      return { video: lastVideo, verdict: null, attempts }
    }
    options.onProgress?.('[' + index + '] 评测（' + judge.id + '）')
    const scores = await judge.judge(
      { media: [lastVideo.path], modality: 'video', criteria: [...criteria] },
      { workdir: join(segDir, 'judge'), signal: options.signal, onProgress: options.onProgress },
    )
    lastVerdict = computeVerdict(scores, criteria)
    options.onProgress?.('[' + index + '] 评分 ' + lastVerdict.weightedScore + '/10，'
      + (lastVerdict.passed ? '通过' : '未通过: ' + lastVerdict.failed.join(', ')))
    if (lastVerdict.passed || attempt >= maxAttempts) {
      return { video: lastVideo, verdict: lastVerdict, attempts }
    }
    if (injectFeedback && scores.feedback.trim().length > 0) {
      text = plan.video_prompt + '\n' + feedbackPrefix + scores.feedback.trim()
    }
  }
  // 上面循环必然 return；此处仅满足类型系统
  return { video: lastVideo as GeneratedVideo, verdict: lastVerdict, attempts }
}

/** 完整编排：剧本 → 逐段 → 跨段 → 总装 → 报告落盘并返回。 */
export async function runSearchGeneration(
  ctx: Context,
  config: Config,
  request: SearchRequest,
  runId: string,
  options: OrchestratorOptions,
): Promise<SearchReport> {
  const outputRoot = config.outputRoot ?? defaultOutputRoot()
  const workdir = options.workdir
  mkdirSync(workdir, { recursive: true })
  options.onProgress?.('工作目录: ' + workdir)

  // 0) 生成器/裁判解析（fail loud：没有提供者给指引，不假装能生成）
  let generator: GeneratorProvider
  try {
    generator = ctx.video.routeGenerator({
      text: request.query,
      duration: request.duration,
      ratio: request.ratio,
    }, { preferredId: request.provider, backend: config.preferBackend })
  } catch (error) {
    const ids = ctx.video.generatorIds()
    throw new Error(
      '无法解析视频生成提供者: ' + (error as Error).message
      + '。已注册生成器: ' + (ids.length > 0 ? ids.join(', ') : '（无）')
      + '。请在 @zhijunlstudio/dsh-video-provider 的配置里添加 generators 行。',
    )
  }
  const judge = request.judge !== undefined ? ctx.video.getJudge(request.judge) : resolveJudgeForVideo(ctx, undefined)
  if (request.judge !== undefined && judge !== undefined && !judge.modalities.includes('video')) {
    throw new Error('裁判 ' + request.judge + ' 不支持 video 模态（其模态: ' + judge.modalities.join(', ') + '）')
  }
  if (judge === undefined) {
    options.onProgress?.('⚠ 没有可用视频裁判：将跳过评测（judged=false，成片未经验证）')
  }

  // 1) 剧本
  const segmentsCount = Math.min(8, Math.max(1, Math.round(request.segments ?? 3)))
  const script = await generateScript(
    ctx, config.script ?? {}, request.query, segmentsCount, request.brief, request.style, options.signal)
  options.onProgress?.('剧本: ' + script.segments.length + ' 段（来源 ' + script.source
    + (script.scriptError !== undefined ? '，降级原因: ' + script.scriptError : '') + '）')

  // 2) 逐段生成 + 评测闭环
  const criteria = config.criteria ?? DEFAULT_SEGMENT_CRITERIA
  const segmentReports: SegmentReport[] = []
  for (let i = 0; i < script.segments.length; i++) {
    const plan = script.segments[i]
    const { video, verdict, attempts } = await generateSegmentWithJudge(
      generator, judge, plan, criteria, config, i + 1, options)
    segmentReports.push({
      index: i + 1,
      prompt: plan.video_prompt,
      video,
      verdict,
      attempts,
      judgeId: judge?.id,
    })
  }

  // 3) 跨段检查（≥2 段且有视频裁判；抽帧失败记 error 可见）
  const cross: SearchReport['cross'] = []
  if (script.segments.length >= 2 && judge !== undefined) {
    const crossCriteria = config.crossCriteria ?? DEFAULT_CROSS_CRITERIA
    const framesDir = join(workdir, 'frames')
    for (let i = 1; i < segmentReports.length; i++) {
      options.signal?.throwIfAborted()
      try {
        const prev = segmentReports[i - 1].video
        const curr = segmentReports[i].video
        const prevLast = extractFrame(prev.path, Math.max(0, (probeVideo(prev.path, config.ffmpegPath).durationS || 1) * 0.9), framesDir, 'seg' + String(i).padStart(2, '0') + '_last', config.ffmpegPath)
        const currFirst = extractFrame(curr.path, 0.05, framesDir, 'seg' + String(i + 1).padStart(2, '0') + '_first', config.ffmpegPath)
        options.onProgress?.('跨段检查 [' + i + ' → ' + (i + 1) + ']')
        const scores = await judge.judge(
          { media: [prevLast, currFirst], modality: 'image', criteria: [...crossCriteria] },
          { workdir: join(framesDir, 'judge'), signal: options.signal, onProgress: options.onProgress },
        )
        cross.push({ pair: [i, i + 1], verdict: computeVerdict(scores, crossCriteria) })
      } catch (error) {
        cross.push({ pair: [i, i + 1], error: (error as Error).message ?? String(error) })
        options.onProgress?.('⚠ 跨段检查失败（已记录）: ' + String((error as Error).message ?? error))
      }
    }
  }

  // 4) 总装
  options.onProgress?.('FFmpeg 总装 ' + segmentReports.length + ' 段')
  const finalDir = join(workdir, 'final')
  mkdirSync(finalDir, { recursive: true })
  const finalPath = join(finalDir, 'final_video.mp4')
  concatVideos(segmentReports.map(s => s.video.path), finalPath, config.ffmpegPath)
  let finalDuration: number | undefined
  try {
    finalDuration = probeVideo(finalPath, config.ffmpegPath).durationS
  } catch {
    // 时长探测失败不阻塞交付；video_verify 可单独复检
  }

  // 5) 报告
  const report: SearchReport = {
    runId,
    query: request.query,
    script: {
      segments: script.segments.length,
      source: script.source,
      scriptError: script.scriptError,
    },
    segments: segmentReports,
    cross,
    final: { path: finalPath, durationS: finalDuration },
    totalCostUsd: Math.round(segmentReports.reduce((sum, s) => sum + (s.video.costUsd ?? 0), 0) * 10000) / 10000,
    judged: judge !== undefined,
    generator: generator.id,
    judge: judge?.id,
    workdir,
  }
  writeFileSync(join(workdir, 'report.json'), JSON.stringify(report, null, 2), 'utf-8')
  return report
}

/** 供 video_generate 复用的单段直出（无剧本、无评测）。 */
export async function runSingleGeneration(
  ctx: Context,
  config: Config,
  request: SearchRequest,
  options: OrchestratorOptions,
): Promise<{ video: GeneratedVideo; generator: GeneratorProvider; providerId: string }> {
  const generator = ctx.video.routeGenerator({
    text: request.query,
    duration: request.duration,
    ratio: request.ratio,
  }, { preferredId: request.provider, backend: config.preferBackend })
  const segDir = join(options.workdir, 'single')
  mkdirSync(segDir, { recursive: true })
  const video = await generator.generate(
    {
      text: request.query,
      duration: request.duration,
      ratio: request.ratio,
      resolution: request.resolution,
      refs: request.refs,
      firstFrame: request.firstFrame,
      lastFrame: request.lastFrame,
    },
    { workdir: segDir, signal: options.signal, onProgress: options.onProgress },
  )
  return { video, generator, providerId: generator.id }
}
