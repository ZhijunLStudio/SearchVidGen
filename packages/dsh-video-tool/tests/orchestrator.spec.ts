import { chmodSync, existsSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { beforeEach, describe, expect, it } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import { LlmAdapter, LlmRuntime, type GenerateOptions, type StreamChunk } from '@deepseek-ai/dsh-llm'
import { VideoRegistry } from '@zhijunlstudio/dsh-video'
import type { GeneratorProvider, JudgeProvider } from '@zhijunlstudio/dsh-video'
import { parseScriptJson } from '../src/script.ts'
import { runSearchGeneration, runSingleGeneration } from '../src/orchestrator.ts'
import type { Config } from '../src/config.ts'

// ---------- fake ffmpeg/ffprobe ----------
function fakeBinaries(dir: string): { ffmpeg: string; ffprobe: string } {
  const ffmpeg = join(dir, 'ffmpeg')
  const ffprobe = join(dir, 'ffprobe')
  writeFileSync(ffmpeg, [
    '#!/bin/sh',
    '# fake ffmpeg: -version ok; 其他调用把最后一个参数当作输出，写占位内容',
    'if [ "$1" = "-version" ]; then exit 0; fi',
    'out=""',
    'for a in "$@"; do out="$a"; done',
    'echo "fake-ffmpeg-output" > "$out"',
    'exit 0',
  ].join('\n'))
  const probeJson = '{"streams":[{"codec_type":"video","codec_name":"h264","width":1280,"height":720,"r_frame_rate":"24/1"},{"codec_type":"audio","codec_name":"aac"}],"format":{"duration":"8.0"}}'
  writeFileSync(ffprobe, [
    '#!/bin/sh',
    'if [ "$1" = "-version" ]; then exit 0; fi',
    "echo '" + probeJson + "'",
    'exit 0',  ].join('\n'))
  chmodSync(ffmpeg, 0o755)
  chmodSync(ffprobe, 0o755)
  return { ffmpeg, ffprobe }
}

// ---------- 测试上下文 ----------
interface TestWorld {
  ctx: Context
  registry: VideoRegistry
  config: Config
  dir: string
  generator: GeneratorProvider
  judge: JudgeProvider
}

function makeWorld(): TestWorld {
  const dir = mkdtempSync(join(tmpdir(), 'dsh-video-tool-'))
  const bins = fakeBinaries(dir)
  const ctx = new Context()
  const registry = new VideoRegistry(ctx)
  void new LlmRuntime(ctx)

  const generator: GeneratorProvider = {
    id: 'mock-gen',
    capabilities: {
      backend: 'remote', minDurationS: 3, maxDurationS: 15, audio: true,
      maxRefs: 2, firstLastFrame: false, resolutions: ['768p'],
    },
    async generate(_request, options) {
      const out = join(options.workdir, 'out.mp4')
      writeFileSync(out, 'fake-segment')
      return { path: out, model: 'mock', backend: 'remote', params: {}, durationS: 8 }
    },
  }
  registry.registerGenerator(generator)

  let calls = 0
  const judge: JudgeProvider = {
    id: 'mock-judge',
    modalities: ['video', 'image', 'text'],
    async judge(request) {
      calls++
      const pass = calls >= 2
      const scores: Record<string, number> = {}
      for (const criterion of request.criteria) {
        scores[criterion.name] = pass ? 8 : 3
      }
      return { scores, feedback: pass ? '很好' : '主体与指令不符' }
    },
  }
  registry.registerJudge(judge)

  const config: Config = { ffmpegPath: bins.ffmpeg, maxRetries: 2, injectFeedback: true }
  return { ctx, registry, config, dir, generator, judge }
}

describe('parseScriptJson（输出契约）', () => {
  it('解析 fenced JSON', () => {
    const plans = parseScriptJson('\n```json\n{"segments":[{"video_prompt":"a cat","narration":"小猫","duration":8}]}\n```\n')
    expect(plans).toHaveLength(1)
    expect(plans[0].video_prompt).toBe('a cat')
    expect(plans[0].narration).toBe('小猫')
    expect(plans[0].duration).toBe(8)
  })
  it('解析裸 JSON 与首尾花括号', () => {
    expect(parseScriptJson('{"segments":[{"video_prompt":"x"}]}')).toHaveLength(1)
    expect(parseScriptJson('前言 {"segments":[{"video_prompt":"y"}]} 后记')).toHaveLength(1)
  })
  it('失败抛可诊断错误', () => {
    expect(() => parseScriptJson('完全不是 JSON')).toThrow(/无法解析/)
    expect(() => parseScriptJson('{"segments":[]}')).toThrow(/缺少 segments/)
    expect(() => parseScriptJson('{"segments":[{}]}')).toThrow(/video_prompt/)
  })
  it('duration 收敛到 3-15 整数', () => {
    const plans = parseScriptJson('{"segments":[{"video_prompt":"x","duration":99}]}')
    expect(plans[0].duration).toBe(15)
  })
})

describe('runSearchGeneration（编排闭环）', () => {
  let world: TestWorld
  beforeEach(() => {
    world = makeWorld()
  })

  it('单段：生成→评测→总装→报告 judged=true', async () => {
    const { ctx, config, dir } = world
    const report = await runSearchGeneration(ctx, config, { query: '一只猫', segments: 1 }, 'run-1', {
      workdir: join(dir, 'work'),
      onProgress: () => {},
    })
    expect(report.script.source).toBe('single')
    expect(report.judged).toBe(true)
    expect(report.segments).toHaveLength(1)
    expect(report.segments[0].verdict?.passed).toBe(true) // 第一次评分 3 分失败，重试后 8 分通过
    expect(report.segments[0].attempts).toBe(2)
    expect(existsSync(report.final.path)).toBe(true)
    expect(existsSync(join(dir, 'work', 'report.json'))).toBe(true)
  })

  it('多段走 LLM 剧本（假适配器）+ 跨段检查', async () => {
    const { ctx, config, dir } = world
    const scriptJson = '{"segments":[{"video_prompt":"a cat walks","narration":"猫在走","duration":8},{"video_prompt":"a cat sleeps","duration":8}]}'
    class FakeScriptAdapter extends LlmAdapter {
      async *stream(_options: GenerateOptions): AsyncIterable<StreamChunk> {
        yield { type: 'block-start', index: 0, blockType: 'text' }
        yield { type: 'text-delta', index: 0, text: scriptJson }
        yield { type: 'block-end', index: 0, block: { type: 'text', text: scriptJson } }
        yield { type: 'finish', reason: { kind: 'stop' } }
      }
    }
    ctx.llm.registerAdapter(['deepseek-official'], new FakeScriptAdapter())
    const report = await runSearchGeneration(ctx, config, { query: '一只猫的一天', segments: 2 }, 'run-2', {
      workdir: join(dir, 'work2'),
      onProgress: () => {},
    })
    expect(report.script.source).toBe('llm')
    expect(report.script.segments).toBe(2)
    expect(report.segments).toHaveLength(2)
    expect(report.cross).toHaveLength(1)
    expect(report.cross[0].verdict?.passed).toBe(true) // 跨段裁判调用次数已 ≥2，8 分通过
    expect(report.final.durationS).toBe(8)
  })

  it('没有生成器时响亮失败并给配置指引', async () => {
    const { ctx, config, dir } = world
    const bare = new Context()
    void new VideoRegistry(bare)
    await expect(runSearchGeneration(bare, config, { query: 'x' }, 'run-3', {
      workdir: join(dir, 'work3'),
    })).rejects.toThrow(/无法解析视频生成提供者/)
  })

  it('LLM 剧本失败时可见降级（scriptError 记录，不静默）', async () => {
    const { ctx, config, dir } = world
    class GarbageAdapter extends LlmAdapter {
      async *stream(_options: GenerateOptions): AsyncIterable<StreamChunk> {
        yield { type: 'block-start', index: 0, blockType: 'text' }
        yield { type: 'text-delta', index: 0, text: '无法生成 JSON 的胡言乱语' }
        yield { type: 'block-end', index: 0, block: { type: 'text', text: '无法生成 JSON 的胡言乱语' } }
        yield { type: 'finish', reason: { kind: 'stop' } }
      }
    }
    ctx.llm.registerAdapter(['deepseek-official'], new GarbageAdapter())
    const report = await runSearchGeneration(ctx, config, { query: 'x', segments: 3 }, 'run-4', {
      workdir: join(dir, 'work4'),
    })
    expect(report.script.source).toBe('single')
    expect(report.script.scriptError).toBeTruthy()
  })
})

describe('runSingleGeneration（单段直出）', () => {
  it('路由并生成，返回提供者 id', async () => {
    const world = makeWorld()
    const result = await runSingleGeneration(world.ctx, world.config, { query: '一只猫' }, {
      workdir: join(world.dir, 'single'),
    })
    expect(result.providerId).toBe('mock-gen')
    expect(existsSync(result.video.path)).toBe(true)
  })
})
