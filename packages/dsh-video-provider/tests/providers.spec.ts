import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http'
import { chmodSync, existsSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import type { AddressInfo } from 'node:net'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import { RemoteVideoProvider } from '../src/remote-video.ts'
import { LocalVideoProvider, validateLocalResult } from '../src/local-video.ts'
import { parseJudgeScores, RemoteJudgeProvider } from '../src/remote-judge.ts'
import type { RemoteVideoProviderConfig } from '../src/config.ts'

const KEY = 'test-key-123'
let envBackup: NodeJS.ProcessEnv

beforeAll(() => {
  envBackup = process.env
  process.env = { ...process.env, MINIMAX_API_KEY: KEY, DEEPSEEK_API_KEY: KEY }
})
afterAll(() => {
  process.env = envBackup
})

const FAKE_MP4 = Buffer.from('fake-mp4-bytes')

function listen(handler: (req: IncomingMessage, res: ServerResponse) => void): Promise<Server> {
  return new Promise(resolve => {
    const server = createServer(handler)
    server.listen(0, '127.0.0.1', () => resolve(server))
  })
}

function baseUrl(server: Server): string {
  const port = (server.address() as AddressInfo).port
  return 'http://127.0.0.1:' + port
}

function workdir(): string {
  return mkdtempSync(join(tmpdir(), 'dsh-video-provider-'))
}

function remoteConfig(server: Server, overrides: Partial<RemoteVideoProviderConfig> = {}): RemoteVideoProviderConfig {
  return {
    id: 'mock-remote',
    kind: 'remote',
    protocol: 'openai',
    baseUrl: baseUrl(server),
    model: 'mock-video-model',
    capabilities: {
      minDurationS: 3, maxDurationS: 15, audio: true, maxRefs: 2,
      firstLastFrame: false, resolutions: ['768p'],
    },
    pollIntervalMs: 20,
    ...overrides,
  }
}

describe('RemoteVideoProvider / openai 协议', () => {
  it('提交 → 轮询 → 下载 mp4 到 workdir', async () => {
    const requests: string[] = []
    const server = await listen((req, res) => {
      requests.push(req.method + ' ' + (req.url ?? ''))
      if (req.url === '/videos/generations') {
        res.setHeader('Content-Type', 'application/json')
        res.end(JSON.stringify({ id: 'vid-1' }))
      } else if (req.url === '/videos/vid-1') {
        res.setHeader('Content-Type', 'application/json')
        res.end(JSON.stringify({ id: 'vid-1', status: 'completed', video: { url: baseUrl(server) + '/vid.mp4' } }))
      } else if (req.url === '/vid.mp4') {
        res.end(FAKE_MP4)
      } else {
        res.statusCode = 404
        res.end()
      }
    })
    try {
      const provider = new RemoteVideoProvider(remoteConfig(server), new Context())
      const result = await provider.generate({ text: '一只猫' }, { workdir: workdir() })
      expect(existsSync(result.path)).toBe(true)
      expect(readFileSync(result.path)).toEqual(FAKE_MP4)
      expect(result.model).toBe('mock-video-model')
      expect(result.backend).toBe('remote')
      expect(requests).toContain('POST /videos/generations')
    } finally {
      server.close()
    }
  })

  it('任务失败时响亮报错', async () => {
    const server = await listen((_req, res) => {
      res.setHeader('Content-Type', 'application/json')
      if ((res.req?.url ?? '').startsWith('/videos/generations')) {
        res.end(JSON.stringify({ id: 'vid-fail' }))
      } else {
        res.end(JSON.stringify({ id: 'vid-fail', status: 'failed' }))
      }
    })
    try {
      const provider = new RemoteVideoProvider(remoteConfig(server), new Context())
      await expect(provider.generate({ text: 'x' }, { workdir: workdir() })).rejects.toThrow(/任务失败/)
    } finally {
      server.close()
    }
  })
})

describe('RemoteVideoProvider / minimax 协议', () => {
  it('提交 v2/video_generation → 查询 → 下载 content.url', async () => {
    const server = await listen((req, res) => {
      res.setHeader('Content-Type', 'application/json')
      if ((req.url ?? '').startsWith('/v2/video_generation')) {
        res.end(JSON.stringify({ task_id: 'task-1' }))
      } else if ((req.url ?? '').startsWith('/v2/query/video_generation')) {
        res.end(JSON.stringify({ task: { status: 'succeeded', content: { url: baseUrl(server) + '/out.mp4' } } }))
      } else if ((req.url ?? '').startsWith('/out.mp4')) {
        res.end(FAKE_MP4)
      } else {
        res.statusCode = 404
        res.end()
      }
    })
    try {
      const provider = new RemoteVideoProvider(remoteConfig(server, { protocol: 'minimax', baseUrl: baseUrl(server) }), new Context())
      const result = await provider.generate({ text: '一只猫', duration: 8 }, { workdir: workdir() })
      expect(readFileSync(result.path)).toEqual(FAKE_MP4)
    } finally {
      server.close()
    }
  })
})

describe('LocalVideoProvider / vh 子进程契约', () => {
  it('解析 stdout 单行 JSON 并校验产物存在', async () => {
    const dir = workdir()
    const out = join(dir, 'out.mp4')
    writeFileSync(out, 'x')
    const script = join(dir, 'fake-vh.js')
    writeFileSync(script, [
      '#!/usr/bin/env node',
      'const fs = require("fs")',
      'const args = process.argv.slice(2)',
      'const specIdx = args.indexOf("gen-single")',
      'const specPath = args[specIdx + 1]',
      'const spec = JSON.parse(fs.readFileSync(specPath, "utf-8"))',
      'fs.writeFileSync("' + dir + '/captured-spec.json", JSON.stringify(spec))',
      'console.error("progress line 1")',
      'console.error("progress line 2")',
      'console.log(JSON.stringify({',
      '  video: { path: "' + out + '", model: "generator.mock", durationS: 8, seed: 42 },',
      '  judge: null, costUsd: 0.5, elapsedS: 12.3, runDir: "' + dir + '"',
      '}))',
    ].join('\n'))
    chmodSync(script, 0o755)
    const provider = new LocalVideoProvider({
      id: 'mock-local', kind: 'local', engineCmd: script,
      capabilities: { minDurationS: 3, maxDurationS: 15, audio: true, maxRefs: 0, firstLastFrame: false, resolutions: ['768p'] },
    })
    const progress: string[] = []
    const result = await provider.generate(
      { text: '一只猫', seed: 42, duration: 8 },
      { workdir: dir, onProgress: line => progress.push(line) },
    )
    expect(result.path).toBe(out)
    expect(result.seed).toBe(42)
    expect(result.costUsd).toBe(0.5)
    expect(progress.some(line => line.includes('progress line 1'))).toBe(true)
    const captured = JSON.parse(readFileSync(join(dir, 'captured-spec.json'), 'utf-8'))
    expect(captured.text).toBe('一只猫')
    expect(captured.generator.adapter).toBe('generator.minimax-h3-local')
  })

  it('非零退出报错并带 stderr 尾部', async () => {
    const dir = workdir()
    const script = join(dir, 'fake-vh-fail.js')
    writeFileSync(script, '#!/usr/bin/env node\nconsole.error("boom: adapter 不存在")\nprocess.exit(3)\n')
    chmodSync(script, 0o755)
    const provider = new LocalVideoProvider({
      id: 'mock-local-fail', kind: 'local', engineCmd: script,
      capabilities: { minDurationS: 3, maxDurationS: 15, audio: true, maxRefs: 0, firstLastFrame: false, resolutions: ['768p'] },
    })
    await expect(provider.generate({ text: 'x' }, { workdir: dir })).rejects.toThrow(/boom/)
  })
})

describe('validateLocalResult（wire 边界防御）', () => {
  it('拒绝形状不合法的输出', () => {
    expect(() => validateLocalResult(null)).toThrow(/JSON 契约/)
    expect(() => validateLocalResult({ video: {} })).toThrow(/video.path/)
    expect(() => validateLocalResult({ video: { path: 'rel.mp4', model: 'm' } })).toThrow(/绝对路径/)
    expect(() => validateLocalResult({ video: { path: '/a.mp4', model: 'm' }, costUsd: 'oops' })).toThrow(/costUsd/)
  })
  it('接受合法输出', () => {
    const result = validateLocalResult({ video: { path: '/a.mp4', model: 'm', durationS: 8 }, costUsd: 0.5 })
    expect(result.video.path).toBe('/a.mp4')
    expect(result.costUsd).toBe(0.5)
  })
})

describe('parseJudgeScores', () => {
  const criteria = [{ name: '与指令一致性', question: '一致吗' }, { name: '画面质量', question: '质量好吗' }]
  it('JSON 直接解析', () => {
    const parsed = parseJudgeScores('{"与指令一致性": 8.5, "画面质量": 9, "feedback": "不错"}', criteria)
    expect(parsed.scores['与指令一致性']).toBe(8.5)
    expect(parsed.feedback).toBe('不错')
  })
  it('支持 {scores: {...}} 包裹形态', () => {
    const parsed = parseJudgeScores('{"scores": {"与指令一致性": 7, "画面质量": 6.5}, "feedback": "ok"}', criteria)
    expect(parsed.scores['画面质量']).toBe(6.5)
  })
  it('正则兜底：维度名后跟分数', () => {
    const parsed = parseJudgeScores('评估如下：\n与指令一致性: 8\n画面质量：7.5\n整体不错', criteria)
    expect(parsed.scores['与指令一致性']).toBe(8)
    expect(parsed.scores['画面质量']).toBe(7.5)
  })
  it('缺维度抛可操作错误（不吞不伪造）', () => {
    expect(() => parseJudgeScores('只有 与指令一致性: 8', criteria)).toThrow(/缺维度: 画面质量/)
  })
})

describe('RemoteJudgeProvider', () => {
  it('文本模态调用 chat/completions 并解析评分', async () => {
    const server = await listen((req, res) => {
      let body = ''
      req.on('data', chunk => { body += chunk })
      req.on('end', () => {
        expect(JSON.parse(body).model).toBe('judge-mock')
        res.setHeader('Content-Type', 'application/json')
        res.end(JSON.stringify({ choices: [{ message: { content: '{"与指令一致性": 8, "feedback": "好"}' } }] }))
      })
    })
    try {
      const provider = new RemoteJudgeProvider({
        id: 'judge-mock', baseUrl: baseUrl(server), model: 'judge-mock', modalities: ['text'],
      }, new Context())
      const result = await provider.judge(
        { media: [], modality: 'text', criteria: [{ name: '与指令一致性', question: '一致吗？' }] },
        { workdir: workdir() },
      )
      expect(result.scores['与指令一致性']).toBe(8)
    } finally {
      server.close()
    }
  })
})
