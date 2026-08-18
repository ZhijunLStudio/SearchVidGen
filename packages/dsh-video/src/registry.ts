/**
 * ctx.video 注册表：生成器/裁判的注册、目录、按能力路由（fail-loud）。
 *
 * 路由语义：
 * - 显式 provider id 优先；未知 id 响亮失败并列出可用 id；
 * - 自动路由按请求约束过滤（时长/参考数/首尾帧/分辨率/音频）；
 * - 无匹配时错误信息逐候选列出未满足的约束（E3 教训：不静默降级）；
 * - 多个匹配取最早注册者（注册顺序 = 路由优先级），确定性。
 * @module @zhijunlstudio/dsh-video
 */

import { Context, Service } from '@deepseek-ai/cordis'
import type {
  GeneratorCapabilities, GeneratorProvider, GeneratorRequest, JudgeModality,
  JudgeProvider, JudgeRequest, ParamCatalog, RegisteredGenerator, RegisteredJudge,
  VideoCatalog,
} from './types.ts'

declare module '@deepseek-ai/cordis' {
  interface Context {
    video: VideoRegistry
  }
}

/** 提供者 id 规则：配置里用户可见，须为路径安全标识。 */
export const PROVIDER_ID_RE = /^[a-z0-9][a-z0-9.-]*$/

/** capabilities 注册期校验：类型与取值范围在此锁定，路由不再嗅探。 */
export function validateCapabilities(capabilities: GeneratorCapabilities): void {
  const fail = (message: string): never => {
    throw new TypeError('capabilities 校验失败: ' + message)
  }
  if (capabilities.backend !== 'remote' && capabilities.backend !== 'local') {
    fail('backend 必须是 remote | local，得到 ' + JSON.stringify(capabilities.backend))
  }
  for (const field of ['minDurationS', 'maxDurationS'] as const) {
    const value = capabilities[field]
    if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
      fail(field + ' 必须是正有限数，得到 ' + JSON.stringify(value))
    }
  }
  if (capabilities.minDurationS > capabilities.maxDurationS) {
    fail('minDurationS(' + capabilities.minDurationS + ') 不能大于 maxDurationS(' + capabilities.maxDurationS + ')')
  }
  if (typeof capabilities.audio !== 'boolean') fail('audio 必须是布尔值')
  if (typeof capabilities.maxRefs !== 'number' || !Number.isInteger(capabilities.maxRefs) || capabilities.maxRefs < 0) {
    fail('maxRefs 必须是非负整数，得到 ' + JSON.stringify(capabilities.maxRefs))
  }
  if (typeof capabilities.firstLastFrame !== 'boolean') fail('firstLastFrame 必须是布尔值')
  if (!Array.isArray(capabilities.resolutions) || capabilities.resolutions.length === 0
    || capabilities.resolutions.some(item => typeof item !== 'string' || item.length === 0)) {
    fail('resolutions 必须是非空字符串数组')
  }
}

/** 生成器注册表：进程内单实例（Cordis 服务），提供者经 register* 挂载。 */
export class VideoRegistry extends Service {
  private readonly generators = new Map<string, RegisteredGenerator>()
  private readonly judges = new Map<string, RegisteredJudge>()
  private order = 0

  constructor(ctx: Context) {
    super(ctx, 'video')
  }

  /**
   * 注册一个生成器提供者。
   * @returns 注销函数（fiber 卸载时自动反转，无需手动调用）。
   */
  registerGenerator(provider: GeneratorProvider): () => void {
    if (!PROVIDER_ID_RE.test(provider.id)) {
      throw new TypeError('生成器 id 不合法（须匹配 ' + PROVIDER_ID_RE.source + '）: ' + JSON.stringify(provider.id))
    }
    if (this.generators.has(provider.id)) {
      throw new Error('生成器 id 重复注册: ' + provider.id)
    }
    validateCapabilities(provider.capabilities)
    if (provider.paramCatalog !== undefined) validateParamCatalog(provider.paramCatalog)
    const record: RegisteredGenerator = { provider, order: this.order++ }
    this.generators.set(provider.id, record)
    return () => {
      this.generators.delete(provider.id)
    }
  }

  generatorIds(): string[] {
    return [...this.generators.keys()]
  }

  /** 取一个生成器（未知 id 响亮失败，列出可用 id）。 */
  getGenerator(id: string): GeneratorProvider {
    const record = this.generators.get(id)
    if (record === undefined) {
      throw new Error('未知生成器 ' + JSON.stringify(id) + '；可用: ' + this.generatorIds().join(', ') || '（无）')
    }
    return record.provider
  }

  /**
   * 按请求约束路由生成器。
   * @param request - 生成请求（约束来源）。
   * @param preferredId - 显式指定则跳过自动路由（未知 id 响亮失败）。
   * @param backend - 后端偏好过滤（可选）。
   * @returns 命中的生成器。
   */
  routeGenerator(request: GeneratorRequest, options?: { preferredId?: string; backend?: 'remote' | 'local' }): GeneratorProvider {
    if (options?.preferredId !== undefined) {
      return this.getGenerator(options.preferredId)
    }
    const requested = request
    const candidates = [...this.generators.values()].sort((a, b) => a.order - b.order)
    const preferredBackend = options?.backend
    const mismatches: string[] = []
    for (const record of candidates) {
      const capabilities = record.provider.capabilities
      if (preferredBackend !== undefined && capabilities.backend !== preferredBackend) {
        mismatches.push(record.provider.id + ': 后端 ' + capabilities.backend + ' != 期望 ' + preferredBackend)
        continue
      }
      const reasons: string[] = []
      if (requested.duration !== undefined
        && (requested.duration < capabilities.minDurationS || requested.duration > capabilities.maxDurationS)) {
        reasons.push('时长 ' + requested.duration + 's 不在 [' + capabilities.minDurationS + ', ' + capabilities.maxDurationS + ']')
      }
      if (requested.refs !== undefined && requested.refs.length > capabilities.maxRefs) {
        reasons.push('参考图 ' + requested.refs.length + ' 张超过上限 ' + capabilities.maxRefs)
      }
      if ((requested.firstFrame !== undefined || requested.lastFrame !== undefined) && !capabilities.firstLastFrame) {
        reasons.push('需要首/尾帧条件但提供者不支持 firstLastFrame')
      }
      if (requested.resolution !== undefined && !capabilities.resolutions.includes(requested.resolution)) {
        reasons.push('分辨率 ' + requested.resolution + ' 不在 [' + capabilities.resolutions.join(', ') + ']')
      }
      if (requested.audio === true && !capabilities.audio) {
        reasons.push('需要原生音频但提供者不支持 audio')
      }
      if (reasons.length === 0) return record.provider
      mismatches.push(record.provider.id + ': ' + reasons.join('; '))
    }
    throw new Error('没有生成器满足本次请求。逐候选原因: ' + (mismatches.length > 0 ? mismatches.join(' | ') : '（无已注册生成器）'))
  }

  /** 注册一个裁判提供者。 */
  registerJudge(provider: JudgeProvider): () => void {
    if (!PROVIDER_ID_RE.test(provider.id)) {
      throw new TypeError('裁判 id 不合法（须匹配 ' + PROVIDER_ID_RE.source + '）: ' + JSON.stringify(provider.id))
    }
    if (this.judges.has(provider.id)) {
      throw new Error('裁判 id 重复注册: ' + provider.id)
    }
    if (provider.modalities.length === 0) {
      throw new TypeError('裁判 ' + provider.id + ' 必须声明至少一种模态')
    }
    for (const modality of provider.modalities) {
      if (modality !== 'image' && modality !== 'video' && modality !== 'text') {
        throw new TypeError('裁判 ' + provider.id + ' 声明了未知模态: ' + JSON.stringify(modality))
      }
    }
    const record: RegisteredJudge = { provider, order: this.order++ }
    this.judges.set(provider.id, record)
    return () => {
      this.judges.delete(provider.id)
    }
  }

  judgeIds(): string[] {
    return [...this.judges.keys()]
  }

  getJudge(id: string): JudgeProvider {
    const record = this.judges.get(id)
    if (record === undefined) {
      throw new Error('未知裁判 ' + JSON.stringify(id) + '；可用: ' + this.judgeIds().join(', ') || '（无）')
    }
    return record.provider
  }

  /**
   * 按模态路由裁判（模态守卫：媒体评测配到 text-only 裁判在第一次调用即响亮失败）。
   */
  routeJudge(request: JudgeRequest, preferredId?: string): JudgeProvider {
    if (preferredId !== undefined) {
      const provider = this.getJudge(preferredId)
      if (!provider.modalities.includes(request.modality)) {
        throw new Error('裁判 ' + preferredId + ' 不支持模态 ' + request.modality + '；其模态: [' + provider.modalities.join(', ') + ']')
      }
      return provider
    }
    const candidates = [...this.judges.values()].sort((a, b) => a.order - b.order)
    for (const record of candidates) {
      if (record.provider.modalities.includes(request.modality)) return record.provider
    }
    const withModalities = candidates.map(record => record.provider.id + '(' + record.provider.modalities.join('/') + ')')
    throw new Error('没有裁判支持模态 ' + request.modality + '；已注册: ' + (withModalities.join(', ') || '（无）'))
  }

  /** 全量目录（video_adapters 工具的声明面）。 */
  catalog(): VideoCatalog {
    const generators = [...this.generators.values()].sort((a, b) => a.order - b.order).map(record => ({
      id: record.provider.id,
      label: record.provider.label,
      capabilities: record.provider.capabilities,
      paramCatalog: record.provider.paramCatalog ?? {},
    }))
    const judges = [...this.judges.values()].sort((a, b) => a.order - b.order).map(record => ({
      id: record.provider.id,
      label: record.provider.label,
      modalities: record.provider.modalities,
    }))
    return { generators, judges }
  }
}

function validateParamCatalog(catalog: ParamCatalog): void {
  for (const [name, decl] of Object.entries(catalog)) {
    if (typeof decl.help !== 'string' || decl.help.length === 0) {
      throw new TypeError('参数声明 ' + name + ' 必须提供 help')
    }
  }
}
