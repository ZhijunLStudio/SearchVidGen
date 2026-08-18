/**
 * 视频生成能力缝的协议类型。
 *
 * 设计纪律（移植自 vidharness E1-E46 的经验）：
 * - capabilities 是声明（注册期校验），路由只依据声明，不嗅探字符串；
 * - 评测结算（权重/阈值）归消费者，提供者只返回原始评分；
 * - 显式 provider id 优先于自动路由；无匹配时响亮失败并列出逐候选原因，
 *   绝不静默降级（fallback 链是配置层的决定）。
 * @module @zhijunlstudio/dsh-video
 */

/** 提供者后端：remote = 远程 HTTP API；local = 本地引擎（子进程）。 */
export type VideoBackend = 'remote' | 'local'

/** 生成器能力声明：注册期校验，路由/参数降级/成本预估的依据。 */
export interface GeneratorCapabilities {
  /** 后端类型。 */
  backend: VideoBackend
  /** 单次生成时长上下界（秒）。 */
  minDurationS: number
  maxDurationS: number
  /** 是否产出原生音轨。 */
  audio: boolean
  /** 最多可接收的参考图数量。 */
  maxRefs: number
  /** 是否支持首/尾帧条件（跨段衔接）。 */
  firstLastFrame: boolean
  /** 支持的分辨率标签（如 768p / 1080p）。 */
  resolutions: string[]
}

/** 一次生成请求：文本指令 + 可选多模态上下文。 */
export interface GeneratorRequest {
  /** 视频指令文本（模型侧提示词）。 */
  text: string
  /** 参考图绝对路径（角色/风格锚点）。 */
  refs?: string[]
  /** 首帧条件图（跨段衔接）。 */
  firstFrame?: string
  /** 尾帧条件图（跨段衔接）。 */
  lastFrame?: string
  /** 期望时长（秒）；缺省由提供者默认。 */
  duration?: number
  /** 期望宽高比，如 '16:9' | '1:1' | '9:16'。 */
  ratio?: string
  /** 期望分辨率标签（须在 capabilities.resolutions 内）。 */
  resolution?: string
  /** 期望带原生音轨。 */
  audio?: boolean
  /** 逐请求随机种子。 */
  seed?: number
  /** 模型私有参数（provider 协议参数，如 steps/guidance）。 */
  params?: Record<string, unknown>
}

/** 生成调用上下文。 */
export interface GeneratorCallOptions {
  /** 产物落盘目录（已创建）。 */
  workdir: string
  /** 取消信号：中止在途调用。 */
  signal?: AbortSignal
  /** 进度行回调（进入 job 流式输出）。 */
  onProgress?: (line: string) => void
}

/** 一段已生成的视频产物。 */
export interface GeneratedVideo {
  /** 绝对路径。 */
  path: string
  /** 实际时长（秒，可选）。 */
  durationS?: number
  /** 实际种子（可选）。 */
  seed?: number
  /** 模型标识（配置中的 model 名）。 */
  model: string
  /** 后端类型。 */
  backend: VideoBackend
  /** 本次调用生效的参数（可复现依据）。 */
  params: Record<string, unknown>
  /** 估算成本（USD，可选）。 */
  costUsd?: number
  /** 耗时（毫秒，可选）。 */
  elapsedMs?: number
}

/** 提供者参数目录条目（video_adapters 工具的声明面）。 */
export interface ParamDecl {
  type: 'string' | 'integer' | 'number' | 'boolean' | 'enum' | 'path' | 'gpu-list'
  required?: boolean
  default?: unknown
  choices?: readonly unknown[]
  help: string
}

export type ParamCatalog = Record<string, ParamDecl>

/** 生成器提供者：实现 generate 即可，注册进 ctx.video。 */
export interface GeneratorProvider {
  /** 唯一 id（配置里用户可见；格式 ^[a-z0-9][a-z0-9.-]*$）。 */
  id: string
  /** 展示名（可选）。 */
  label?: string
  capabilities: GeneratorCapabilities
  /** 可选参数声明（覆盖能力之外的可调参数）。 */
  paramCatalog?: ParamCatalog
  generate(request: GeneratorRequest, options: GeneratorCallOptions): Promise<GeneratedVideo>
}

/** 注册后的生成器记录（含注册顺序，路由优先级 = 注册顺序）。 */
export interface RegisteredGenerator {
  provider: GeneratorProvider
  order: number
}

/** 评测模态。 */
export type JudgeModality = 'image' | 'video' | 'text'

/** 一条评测维度（阈值/权重在结算时生效，见 computeVerdict）。 */
export interface JudgeCriterion {
  name: string
  question: string
  /** 权重（默认 1.0）。 */
  weight?: number
  /** 最低通过分（0-10，默认 6）。 */
  minScore?: number
}

/** 一次评测请求。 */
export interface JudgeRequest {
  /** 媒体文件绝对路径（text 模态时可为空数组）。 */
  media: string[]
  /** 请求模态：与媒体对应。 */
  modality: JudgeModality
  criteria: JudgeCriterion[]
}

/** 评测调用上下文。 */
export interface JudgeCallOptions {
  workdir: string
  signal?: AbortSignal
  onProgress?: (line: string) => void
}

/** 裁判原始结果：只含原始数据，不计算加权分/阈值判定。 */
export interface JudgeScores {
  /** 维度名 -> 0-10 分。 */
  scores: Record<string, number>
  /** 自然语言反馈（失败时注入下一次生成的修正指令）。 */
  feedback: string
  /** 提供者原始输出（可审计，可选）。 */
  raw?: unknown
}

/** 裁判提供者。 */
export interface JudgeProvider {
  id: string
  label?: string
  /** 该裁判能评测的模态；媒体请求配到 text-only 裁判会在第一次调用即响亮失败。 */
  modalities: readonly JudgeModality[]
  judge(request: JudgeRequest, options: JudgeCallOptions): Promise<JudgeScores>
}

/** 注册后的裁判记录。 */
export interface RegisteredJudge {
  provider: JudgeProvider
  order: number
}

/** 提供者目录（video_adapters 工具的返回形状）。 */
export interface VideoCatalog {
  generators: {
    id: string
    label?: string
    capabilities: GeneratorCapabilities
    paramCatalog: ParamCatalog
  }[]
  judges: {
    id: string
    label?: string
    modalities: readonly JudgeModality[]
  }[]
}
