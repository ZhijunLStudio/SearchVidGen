/**
 * 视频生成模式的 5 个模型面工具。
 * @module @zhijunlstudio/dsh-video-tool
 */

import { mkdirSync } from 'node:fs'
import { extname, join } from 'node:path'
import type { Context } from '@deepseek-ai/cordis'
import { defineTool } from '@deepseek-ai/dsh-tools'
import type { ContentBlock } from '@deepseek-ai/dsh-llm'
import { computeVerdict, type JudgeCriterion, type JudgeModality } from '@zhijunlstudio/dsh-video'
import {
  DEFAULT_CROSS_CRITERIA, DEFAULT_SEGMENT_CRITERIA, defaultOutputRoot, type Config,
} from './config.ts'
import { probeVideo } from './ffmpeg.ts'
import { startVideoJob } from './jobs.ts'
import { runSearchGeneration, runSingleGeneration, type SearchRequest } from './orchestrator.ts'

function text(value: string): ContentBlock[] {
  return [{ type: 'text', text: value }]
}

function newRunId(): string {
  const stamp = new Date()
  const pad = (n: number): string => String(n).padStart(2, '0')
  const random = Math.random().toString(36).slice(2, 6)
  return 'video-' + stamp.getFullYear() + pad(stamp.getMonth() + 1) + pad(stamp.getDate())
    + '-' + pad(stamp.getHours()) + pad(stamp.getMinutes()) + pad(stamp.getSeconds()) + '-' + random
}

const MEDIA_VIDEO_EXTS = new Set(['.mp4', '.mov', '.mkv', '.avi', '.webm'])

function inferModality(media: string[]): JudgeModality {
  return media.some(path => MEDIA_VIDEO_EXTS.has(extname(path).toLowerCase())) ? 'video' : 'image'
}

function judgeCriteria(args: { criteria?: JudgeCriterion[]; preset?: string }, config: Config): JudgeCriterion[] {
  if (args.criteria !== undefined && args.criteria.length > 0) return args.criteria
  if (args.preset === 'cross') return config.crossCriteria ?? DEFAULT_CROSS_CRITERIA
  return config.criteria ?? DEFAULT_SEGMENT_CRITERIA
}

/** 注册全部工具（fiber 卸载自动反注册）。 */
export function registerVideoTools(ctx: Context, config: Config): void {
  const outputRoot = config.outputRoot ?? defaultOutputRoot()
  const runInBackgroundDefault = config.runInBackground ?? true

  ctx.tools.register(defineTool({
    name: 'video_adapters',
    description: '列出已注册的视频生成提供者与裁判的声明目录（capabilities/参数/模态）。生成视频前可用它了解当前可用的模型端点与本地引擎。',
    parameters: {},
    output: {
      schema: {
        type: 'object',
        properties: {
          generators: { type: 'json', description: '生成器目录（id/capabilities/paramCatalog）' },
          judges: { type: 'json', description: '裁判目录（id/modalities）' },
        },
        additionalProperties: true,
      },
      render: (_args, value) => {
        const catalog = value as { generators: unknown[]; judges: unknown[] }
        return text([
          '视频生成提供者（' + catalog.generators.length + '）:',
          ...(catalog.generators.length === 0
            ? ['（无）——请在 @zhijunlstudio/dsh-video-provider 配置里添加 generators 行（remote: URL/key/model；local: 本机 vh 引擎）。']
            : catalog.generators.map(g => JSON.stringify(g))),
          '裁判（' + catalog.judges.length + '）:',
          ...(catalog.judges.length === 0
            ? ['（无）——请在配置里添加 judges 行（OpenAI 兼容端点：URL/key/model）。']
            : catalog.judges.map(j => JSON.stringify(j))),
        ].join('\n'))
      },
    },
    execute: async () => JSON.parse(JSON.stringify(ctx.video.catalog())),
  }))

  ctx.tools.register(defineTool({
    name: 'video_generate',
    description: '把一句视频指令直接生成一段视频（单段直出，无剧本无评测）。默认作为后台任务执行（返回 jobId，用 job_* 工具查看进度/结果）；run_in_background: false 时同步等待。生成模型由 provider 参数指定或按请求能力自动路由（video_adapters 查看可用项）。',
    parameters: {
      text: { type: 'string', required: true, description: '视频指令（英文提示词效果最佳，一两句话即可）' },
      provider: { type: 'string', description: '指定生成器 id（省略则自动路由；video_adapters 查看可用 id）' },
      duration: { type: 'number', description: '时长秒数（3-15）' },
      ratio: { type: 'string', description: '宽高比：16:9 / 1:1 / 9:16' },
      resolution: { type: 'string', description: '分辨率标签（如 768p；须在提供者声明内）' },
      seed: { type: 'integer', description: '随机种子（可复现）' },
      audio: { type: 'boolean', description: '要求原生音轨' },
      refs: { type: 'array', items: { type: 'string' }, description: '参考图绝对路径（角色/风格锚点）' },
      firstFrame: { type: 'string', description: '首帧条件图绝对路径' },
      lastFrame: { type: 'string', description: '尾帧条件图绝对路径' },
      run_in_background: { type: 'boolean', description: '默认 true：作为后台 job 执行' },
    },
    output: {
      schema: {
        oneOf: [
          {
            type: 'object',
            properties: {
              kind: { type: 'string', const: 'background' },
              jobId: { type: 'string' },
            },
            additionalProperties: false,
          },
          {
            type: 'object',
            properties: {
              kind: { type: 'string', const: 'done' },
              video: { type: 'json' },
              provider: { type: 'string' },
            },
            additionalProperties: false,
          },
        ],
      },
      render: (_args, value) => {
        const result = value as { kind: string; jobId?: string; video?: { path?: string } }
        if (result.kind === 'background') {
          return text('已启动后台视频生成任务 ' + String(result.jobId) + '，用 job_output/job_list 查看进度与结果。')
        }
        return text('视频已生成: ' + String(result.video?.path ?? ''))
      },
    },
    execute: async (args, exec) => {
      const request: SearchRequest = {
        query: args.text,
        provider: args.provider,
        duration: args.duration,
        ratio: args.ratio,
        resolution: args.resolution,
        refs: args.refs,
        firstFrame: args.firstFrame,
        lastFrame: args.lastFrame,
      }
      const background = args.run_in_background ?? runInBackgroundDefault
      const workdir = join(outputRoot, newRunId(), 'generate')
      mkdirSync(workdir, { recursive: true })
      const doRun = async (signal: AbortSignal, onProgress: (line: string) => void) => {
        const { video, generator } = await runSingleGeneration(
          ctx, config, request, { workdir, signal, onProgress })
        return JSON.parse(JSON.stringify({ kind: 'done' as const, video, provider: generator.id }))
      }
      if (background) {
        const started = startVideoJob(ctx, exec.agent, 'video_generate: ' + args.text.slice(0, 80), async handles => {
          const result = await doRun(handles.abort.signal, line => handles.progress(line))
          return { status: 'completed' as const, detail: '视频: ' + result.video.path }
        })
        return { kind: 'background' as const, jobId: started.jobId }
      }
      return await doRun(exec.signal, () => {})
    },
  }))

  ctx.tools.register(defineTool({
    name: 'video_generate_search',
    description: '从搜索词或一两句话出发生成完整短视频：LLM 分镜剧本 → 逐段生成+评测闭环（失败反馈重试）→ 跨段一致性检查 → FFmpeg 拼接成片，产出成片 mp4 与报告（含评分/成本）。默认后台执行；成片路径与报告在任务完成后可用 video_verify 复检。',
    parameters: {
      query: { type: 'string', required: true, description: '搜索词或一两句话的意图描述（如"雨夜，一只小猫在旧书店橱窗前躲雨"）' },
      brief: { type: 'string', description: '补充要求（风格/受众/时长）' },
      provider: { type: 'string', description: '指定生成器 id（省略自动路由）' },
      judge: { type: 'string', description: '指定裁判 id（省略自动选视频模态裁判；无裁判则跳过评测并标记 judged=false）' },
      segments: { type: 'integer', description: '分镜段数（1-8，默认 3；1 段时跳过 LLM 剧本）' },
      duration: { type: 'number', description: '每段时长秒数（3-15）' },
      ratio: { type: 'string', description: '宽高比：16:9 / 1:1 / 9:16' },
      style: { type: 'string', description: '统一视觉风格要求（写入每个镜头提示词）' },
      run_in_background: { type: 'boolean', description: '默认 true：作为后台 job 执行' },
    },
    output: {
      schema: {
        oneOf: [
          {
            type: 'object',
            properties: {
              kind: { type: 'string', const: 'background' },
              jobId: { type: 'string' },
            },
            additionalProperties: false,
          },
          {
            type: 'object',
            properties: {
              kind: { type: 'string', const: 'done' },
              report: { type: 'json', description: '完整报告（分段/评分/成本/成片路径）' },
            },
            additionalProperties: false,
          },
        ],
      },
      render: (_args, value) => {
        const result = value as { kind: string; jobId?: string; report?: { final?: { path?: string }; judged?: boolean } }
        if (result.kind === 'background') {
          return text('已启动后台视频生成任务 ' + String(result.jobId) + '，用 job_output/job_list 查看进度；完成后我会收到结果通知。')
        }
        const report = result.report
        return text('成片: ' + String(report?.final?.path ?? '')
          + (report?.judged === false ? '（未评测：没有可用视频裁判）' : ''))
      },
    },
    execute: async (args, exec) => {
      const request: SearchRequest = {
        query: args.query,
        brief: args.brief,
        provider: args.provider,
        judge: args.judge,
        segments: args.segments,
        duration: args.duration,
        ratio: args.ratio,
        style: args.style,
      }
      const background = args.run_in_background ?? runInBackgroundDefault
      const runId = newRunId()
      const workdir = join(outputRoot, runId, 'search')
      mkdirSync(workdir, { recursive: true })
      const doRun = (signal: AbortSignal, onProgress: (line: string) => void) =>
        runSearchGeneration(ctx, config, request, runId, { workdir, signal, onProgress })
      if (background) {
        const started = startVideoJob(ctx, exec.agent, 'video_generate_search: ' + args.query.slice(0, 80), async handles => {
          const report = await doRun(handles.abort.signal, line => handles.progress(line))
          return {
            status: 'completed' as const,
            detail: '成片: ' + report.final.path
              + (report.judged ? '，加权分 ' + report.segments.reduce((s, x) => s + (x.verdict?.weightedScore ?? 0), 0) : '，未评测'),
          }
        })
        return { kind: 'background' as const, jobId: started.jobId }
      }
      const report = await doRun(exec.signal, () => {})
      return JSON.parse(JSON.stringify({ kind: 'done' as const, report }))
    },
  }))

  ctx.tools.register(defineTool({
    name: 'video_judge',
    description: '用配置的裁判评测已有视频/图像：按维度打分（0-10）并给出通过/未通过结算与修正反馈。media 传媒体绝对路径（视频/图片）；criteria 省略时用默认逐段维度（preset: "segment"），跨段两帧检查用 preset: "cross"。text 模态（纯文本裁判）没有媒体输入——把待评文本直接写进各维度的 question。',
    parameters: {
      media: { type: 'array', items: { type: 'string' }, required: true, description: '媒体文件绝对路径（mp4 或图片）' },
      criteria: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            name: { type: 'string', required: true },
            question: { type: 'string', required: true },
            weight: { type: 'number' },
            minScore: { type: 'number' },
          },
          additionalProperties: true,
        },
        description: '自定义评测维度（省略用 preset）',
      },
      preset: { type: 'string', enum: ['segment', 'cross'], description: '维度预设：segment=逐段质量；cross=跨段一致/叙事推进' },
      judge: { type: 'string', description: '指定裁判 id（省略自动按模态路由）' },
      modality: { type: 'string', enum: ['video', 'image', 'text'], description: '评测模态（省略按文件后缀推断）' },
    },
    output: {
      schema: {
        type: 'object',
        properties: {
          judgeId: { type: 'string' },
          scores: { type: 'json' },
          verdict: {
            type: 'object',
            properties: {
              passed: { type: 'boolean' },
              weightedScore: { type: 'number' },
              maxScore: { type: 'number' },
              failed: { type: 'array', items: { type: 'string' } },
              details: { type: 'json' },
            },
            additionalProperties: false,
          },
          feedback: { type: 'string' },
        },
        additionalProperties: false,
      },
      render: (_args, value) => {
        const result = value as { verdict: { passed: boolean; weightedScore: number; failed: string[] }; judgeId: string }
        return text('裁判 ' + result.judgeId + '：加权分 ' + result.verdict.weightedScore + '/10，'
          + (result.verdict.passed ? '通过' : '未通过: ' + result.verdict.failed.join(', ')))
      },
    },
    execute: async (args, exec) => {
      const criteria = judgeCriteria(args, config)
      const modality = args.modality ?? inferModality(args.media)
      const judge = ctx.video.routeJudge({ media: args.media, modality, criteria }, args.judge)
      const scores = await judge.judge(
        { media: args.media, modality, criteria },
        { workdir: join(outputRoot, 'judge-' + Date.now().toString(36)), signal: exec.signal },
      )
      const verdict = computeVerdict(scores, criteria)
      return JSON.parse(JSON.stringify({ judgeId: judge.id, scores: scores.scores, verdict, feedback: scores.feedback }))
    },
  }))

  ctx.tools.register(defineTool({
    name: 'video_verify',
    description: '检验视频产物的媒体事实（ffprobe）：时长/帧率/分辨率/宽高比/音频轨道/文件存在。expect 只写要检查的项；任何一项不满足 ok=false 并逐项说明。适合生成完成后复检成片。',
    parameters: {
      path: { type: 'string', required: true, description: '视频绝对路径' },
      expect: {
        type: 'object',
        properties: {
          minDurationS: { type: 'number' },
          maxDurationS: { type: 'number' },
          minFps: { type: 'number' },
          maxFps: { type: 'number' },
          aspect: { type: 'string', description: '宽高比标签：16:9 / 1:1 / 9:16' },
          audio: { type: 'boolean', description: '要求/禁止原生音轨' },
        },
        additionalProperties: true,
        description: '期望约束（只写要检查的项）',
      },
    },
    output: {
      schema: {
        type: 'object',
        properties: {
          ok: { type: 'boolean' },
          media: { type: 'json' },
          checks: { type: 'json' },
        },
        additionalProperties: true,
      },
      render: (_args, value) => {
        const result = value as { ok: boolean; checks: { name: string; pass: boolean; detail: string }[] }
        const lines = result.checks.map(check => (check.pass ? '✓' : '✗') + ' ' + check.name + ': ' + check.detail)
        return text(['视频检验 ' + (result.ok ? '全部通过' : '存在未通过项') + ':', ...lines].join('\n'))
      },
    },
    execute: async (args) => {
      const media = probeVideo(args.path, config.ffmpegPath)
      const expect = args.expect ?? {}
      const checks: { name: string; pass: boolean; detail: string }[] = []
      const push = (name: string, pass: boolean, detail: string): void => {
        checks.push({ name, pass, detail })
      }
      push('文件存在', true, args.path)
      if (expect.minDurationS !== undefined) {
        push('最小时长', media.durationS >= expect.minDurationS, media.durationS + 's ≥ ' + expect.minDurationS + 's')
      }
      if (expect.maxDurationS !== undefined) {
        push('最大时长', media.durationS <= expect.maxDurationS, media.durationS + 's ≤ ' + expect.maxDurationS + 's')
      }
      if (expect.minFps !== undefined) {
        push('最小帧率', media.fps >= expect.minFps, media.fps + 'fps ≥ ' + expect.minFps + 'fps')
      }
      if (expect.maxFps !== undefined) {
        push('最大帧率', media.fps <= expect.maxFps, media.fps + 'fps ≤ ' + expect.maxFps + 'fps')
      }
      if (expect.aspect !== undefined) {
        const ratios: Record<string, number> = { '16:9': 16 / 9, '1:1': 1, '9:16': 9 / 16, '4:3': 4 / 3 }
        const expected = ratios[expect.aspect]
        if (expected === undefined) {
          push('宽高比', false, '未知宽高比标签 ' + expect.aspect)
        } else {
          const actual = media.height > 0 ? media.width / media.height : 0
          const pass = Math.abs(actual - expected) / expected < 0.05
          push('宽高比 ' + expect.aspect, pass, media.width + 'x' + media.height + '（' + Math.round(actual * 100) / 100 + '）')
        }
      }
      if (expect.audio !== undefined) {
        push(expect.audio ? '有音轨' : '无音轨', media.hasAudio === expect.audio, media.hasAudio ? '检测到音轨' : '未检测到音轨')
      }
      return JSON.parse(JSON.stringify({ ok: checks.every(check => check.pass), media, checks }))
    },
  }))
}
