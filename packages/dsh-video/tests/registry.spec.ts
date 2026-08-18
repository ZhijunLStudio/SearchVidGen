import { Context } from '@deepseek-ai/cordis'
import { beforeEach, describe, expect, it } from 'vitest'
import { VideoRegistry } from '../src/registry.ts'
import type { GeneratorCapabilities, GeneratorProvider, JudgeProvider } from '../src/types.ts'

const baseCapabilities = (): GeneratorCapabilities => ({
  backend: 'remote',
  minDurationS: 3,
  maxDurationS: 15,
  audio: true,
  maxRefs: 9,
  firstLastFrame: true,
  resolutions: ['768p'],
})

function makeGenerator(id: string, overrides: Partial<GeneratorProvider> = {}): GeneratorProvider {
  return {
    id,
    capabilities: baseCapabilities(),
    async generate() {
      return { path: '/tmp/out.mp4', model: 'test', backend: 'remote', params: {} }
    },
    ...overrides,
  }
}

function makeJudge(id: string, modalities: JudgeProvider['modalities'] = ['video', 'image', 'text']): JudgeProvider {
  return {
    id,
    modalities,
    async judge() {
      return { scores: {}, feedback: '' }
    },
  }
}

let ctx: Context
let registry: VideoRegistry

beforeEach(() => {
  ctx = new Context()
  registry = new VideoRegistry(ctx)
})

describe('generator registration', () => {
  it('registers and disposes', () => {
    const dispose = registry.registerGenerator(makeGenerator('h3'))
    expect(registry.generatorIds()).toEqual(['h3'])
    dispose()
    expect(registry.generatorIds()).toEqual([])
  })

  it('rejects duplicate ids at load time', () => {
    registry.registerGenerator(makeGenerator('h3'))
    expect(() => registry.registerGenerator(makeGenerator('h3'))).toThrow(/重复注册/)
  })

  it('rejects malformed ids', () => {
    expect(() => registry.registerGenerator(makeGenerator('bad id'))).toThrow(/id 不合法/)
    expect(() => registry.registerGenerator(makeGenerator('../escape'))).toThrow(/id 不合法/)
  })

  it('rejects malformed capabilities at registration', () => {
    const bad = (capabilities: Partial<GeneratorCapabilities>) =>
      registry.registerGenerator(makeGenerator('x', { capabilities: { ...baseCapabilities(), ...capabilities } }))
    expect(() => bad({ maxRefs: -1 })).toThrow(/maxRefs/)
    expect(() => bad({ resolutions: [] })).toThrow(/resolutions/)
    expect(() => bad({ backend: 'cloud' as never })).toThrow(/backend/)
    expect(() => bad({ minDurationS: 20 })).toThrow(/不能大于/)
  })
})

describe('generator routing', () => {
  it('routes by request constraints and lists per-candidate reasons when nothing matches', () => {
    const shortOnly = makeGenerator('short', { capabilities: { ...baseCapabilities(), maxDurationS: 5 } })
    const noRefs = makeGenerator('noref', { capabilities: { ...baseCapabilities(), maxRefs: 0 } })
    registry.registerGenerator(shortOnly)
    registry.registerGenerator(noRefs)
    expect(() => registry.routeGenerator({ text: 'x', duration: 10, refs: ['/a.jpg'] }))
      .toThrow(/short: 时长 10s 不在 \[3, 5\]/)
    expect(() => registry.routeGenerator({ text: 'x', refs: Array.from({ length: 10 }, (_, i) => '/r' + i + '.jpg') }))
      .toThrow(/noref: 参考图 10 张超过上限 0/)
  })

  it('fails loud when the request needs unsupported first/last frame conditioning', () => {
    registry.registerGenerator(makeGenerator('t2v', { capabilities: { ...baseCapabilities(), firstLastFrame: false } }))
    expect(() => registry.routeGenerator({ text: 'x', firstFrame: '/f.jpg' }))
      .toThrow(/firstLastFrame/)
  })

  it('fails loud on resolution and audio mismatches', () => {
    registry.registerGenerator(makeGenerator('a', { capabilities: { ...baseCapabilities(), resolutions: ['480p'], audio: false } }))
    expect(() => registry.routeGenerator({ text: 'x', resolution: '1080p' })).toThrow(/分辨率 1080p/)
    expect(() => registry.routeGenerator({ text: 'x', audio: true })).toThrow(/audio/)
  })

  it('explicit preferred id bypasses routing; unknown id lists available ids', () => {
    const first = makeGenerator('first', { capabilities: { ...baseCapabilities(), maxDurationS: 5 } })
    registry.registerGenerator(first)
    expect(registry.routeGenerator({ text: 'x', duration: 12 }, { preferredId: 'first' })).toBe(first)
    expect(() => registry.routeGenerator({ text: 'x' }, { preferredId: 'nope' })).toThrow(/可用: first/)
  })

  it('backend preference filters candidates', () => {
    const remote = makeGenerator('remote', { capabilities: { ...baseCapabilities(), backend: 'remote' } })
    const local = makeGenerator('local', { capabilities: { ...baseCapabilities(), backend: 'local' } })
    registry.registerGenerator(remote)
    registry.registerGenerator(local)
    expect(registry.routeGenerator({ text: 'x' }, { backend: 'local' })).toBe(local)
    expect(() => registry.routeGenerator({ text: 'x' }, { backend: 'local', preferredId: 'remote' })).not.toThrow()
  })

  it('picks the earliest registered among multiple matches (deterministic priority)', () => {
    const first = makeGenerator('first')
    registry.registerGenerator(first)
    registry.registerGenerator(makeGenerator('second'))
    expect(registry.routeGenerator({ text: 'x' })).toBe(first)
  })
})

describe('judge registration and modality guard', () => {
  it('registers, lists, and disposes judges', () => {
    const dispose = registry.registerJudge(makeJudge('vlm'))
    expect(registry.judgeIds()).toEqual(['vlm'])
    dispose()
    expect(registry.judgeIds()).toEqual([])
  })

  it('rejects duplicate ids and empty modalities', () => {
    registry.registerJudge(makeJudge('vlm'))
    expect(() => registry.registerJudge(makeJudge('vlm'))).toThrow(/重复注册/)
    expect(() => registry.registerJudge(makeJudge('empty', []))).toThrow(/至少一种模态/)
  })

  it('routes by modality and fails visibly when the media needs a judge no one offers', () => {
    registry.registerJudge(makeJudge('text-only', ['text']))
    expect(registry.routeJudge({ media: [], modality: 'text', criteria: [] })).not.toBeUndefined()
    expect(() => registry.routeJudge({ media: ['/v.mp4'], modality: 'video', criteria: [] }))
      .toThrow(/没有裁判支持模态 video；已注册: text-only\(text\)/)
  })

  it('preferred judge lacking the modality fails at first call', () => {
    registry.registerJudge(makeJudge('text-only', ['text']))
    registry.registerJudge(makeJudge('vlm', ['video', 'image']))
    expect(() => registry.routeJudge({ media: ['/v.mp4'], modality: 'video', criteria: [] }, 'text-only'))
      .toThrow(/不支持模态 video/)
    expect(() => registry.routeJudge({ media: ['/v.mp4'], modality: 'video', criteria: [] }, 'nope'))
      .toThrow(/未知裁判/)
  })
})

describe('catalog', () => {
  it('exposes declared capabilities, params, and modalities', () => {
    registry.registerGenerator(makeGenerator('h3', {
      label: 'MiniMax H3',
      paramCatalog: { steps: { type: 'integer', help: '去噪步数', default: 30 } },
    }))
    registry.registerJudge(makeJudge('vlm', ['video', 'image']))
    const catalog = registry.catalog()
    expect(catalog.generators).toHaveLength(1)
    expect(catalog.generators[0]).toMatchObject({
      id: 'h3', label: 'MiniMax H3',
      paramCatalog: { steps: { type: 'integer', help: '去噪步数', default: 30 } },
    })
    expect(catalog.generators[0].capabilities.maxDurationS).toBe(15)
    expect(catalog.judges[0].modalities).toEqual(['video', 'image'])
  })
})