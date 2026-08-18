/**
 * 配置面：一行配置接入一个模型端点/本地引擎/裁判。
 *
 * 鲁棒性设计（用户要求"指定 URL/key/model，不是叠一堆包"）：
 * - 新模型 = 新增一条配置行，零代码改动；
 * - remote 的 defaultParams 原样透传协议参数（steps/guidance 等），
 *   端点支持什么就配什么，不硬编码厂商字段；
 * - 凭据走 dsh credentials seam（credential 引用名），不落明文；
 * - 配置错误在装配期（schemastery 校验）响亮失败。
 * @module @zhijunlstudio/dsh-video-provider
 */

import type { GeneratorCapabilities, JudgeModality } from '@zhijunlstudio/dsh-video'
import z from 'schemastery'

/** remote 生成器：HTTP 协议端点（OpenAI 兼容 / MiniMax 官方）。 */
export interface RemoteVideoProviderConfig {
  id: string
  kind: 'remote'
  label?: string
  /** 'openai' = OpenAI 视频 API 风格；'minimax' = MiniMax /v2/video_generation 风格。 */
  protocol: 'openai' | 'minimax'
  /** 端点根地址（不含路径）。 */
  baseUrl: string
  /** 模型名（原样传入请求）。 */
  model: string
  /** credentials seam 里的引用名（如 MINIMAX_API_KEY）；缺省读同名环境变量。 */
  credential?: string
  capabilities: {
    minDurationS: number
    maxDurationS: number
    audio: boolean
    maxRefs: number
    firstLastFrame: boolean
    resolutions: string[]
  }
  /** 协议参数透传（原样并入请求体，如 steps/guidance）。 */
  defaultParams?: Record<string, unknown>
  /** 提交/轮询/下载单请求超时（毫秒）。 */
  timeoutMs?: number
  /** 轮询间隔（毫秒）。 */
  pollIntervalMs?: number
}

/** local 生成器：本机引擎（vh gen-single JSON 子进程契约）。 */
export interface LocalVideoProviderConfig {
  id: string
  kind: 'local'
  label?: string
  /** vh 命令名（缺省 'vh'，需在 PATH）。 */
  engineCmd?: string
  /** Python 解释器绝对路径（给了就用 "<python> -m vidharness.cli"，不用 PATH 找 vh）。 */
  pythonPath?: string
  /** 子进程工作目录（vidharness 包可导入的目录；缺省继承宿主 cwd）。 */
  cwd?: string
  /** vidharness 生成器适配器名。 */
  adapter?: string
  /** 适配器参数（model_path/gpu/steps/... 原样透传给 vh）。 */
  params?: Record<string, unknown>
  capabilities: {
    minDurationS: number
    maxDurationS: number
    audio: boolean
    maxRefs: number
    firstLastFrame: boolean
    resolutions: string[]
  }
  /** 整段超时（毫秒，缺省 45 分钟）。 */
  timeoutMs?: number
  /** 成本口径：GPU 单价 USD/卡时（估算用）。 */
  gpuPriceUsdPerHour?: number
  /** ffmpeg/ffprobe 所在目录（prepend 进引擎 PATH；模型环境常不含 ffmpeg）。 */
  ffmpegDir?: string
}

/** 裁判：OpenAI 兼容 chat completions 端点（本地 vLLM / DeepSeek API / 任何 VLM 服务）。 */
export interface RemoteJudgeProviderConfig {
  id: string
  label?: string
  baseUrl: string
  model: string
  /** credentials seam 引用名或环境变量名。 */
  credential?: string
  /** 该裁判能评测的模态；文本裁判只填 ['text']（媒体评测会响亮拒绝）。 */
  modalities: JudgeModality[]
  /** 视频抽帧数（video 模态；缺省 4）。 */
  frameSamples?: number
  maxTokens?: number
  temperature?: number
  timeoutMs?: number
  /** ffmpeg 可执行路径（缺省 PATH 找 'ffmpeg'）。 */
  ffmpegPath?: string
}

export interface Config {
  generators?: (RemoteVideoProviderConfig | LocalVideoProviderConfig)[]
  judges?: RemoteJudgeProviderConfig[]
}

const capabilitySchema = z.object({
  minDurationS: z.number().min(0.01),
  maxDurationS: z.number().min(0.01),
  audio: z.boolean(),
  maxRefs: z.number().min(0).step(1),
  firstLastFrame: z.boolean(),
  resolutions: z.array(z.string().min(1)),
})

export const Config: z<Config> = z.object({
  generators: z.array(z.union([
    z.object({
      id: z.string().pattern(/^[a-z0-9][a-z0-9.-]*$/),
      kind: z.const('remote'),
      label: z.string().required(false),
      protocol: z.union([z.const('openai'), z.const('minimax')]),
      baseUrl: z.string().min(1),
      model: z.string().min(1),
      credential: z.string().min(1).required(false),
      capabilities: capabilitySchema,
      defaultParams: z.dict(z.any()).required(false),
      timeoutMs: z.number().min(1).required(false),
      pollIntervalMs: z.number().min(1).required(false),
    }),
    z.object({
      id: z.string().pattern(/^[a-z0-9][a-z0-9.-]*$/),
      kind: z.const('local'),
      label: z.string().required(false),
      engineCmd: z.string().min(1).required(false),
      pythonPath: z.string().min(1).required(false),
      cwd: z.string().min(1).required(false),
      adapter: z.string().min(1).required(false),
      params: z.dict(z.any()).required(false),
      capabilities: capabilitySchema,
      timeoutMs: z.number().min(1).required(false),
      gpuPriceUsdPerHour: z.number().min(0.0001).required(false),
      ffmpegDir: z.string().min(1).required(false),
    }),
  ])).default([]),
  judges: z.array(z.object({
    id: z.string().pattern(/^[a-z0-9][a-z0-9.-]*$/),
    label: z.string().required(false),
    baseUrl: z.string().min(1),
    model: z.string().min(1),
    credential: z.string().min(1).required(false),
    modalities: z.array(z.union([z.const('video'), z.const('image'), z.const('text')])).default(['text']),
    frameSamples: z.number().min(1).step(1).required(false),
    maxTokens: z.number().min(1).step(1).required(false),
    temperature: z.number().min(0).required(false),
    timeoutMs: z.number().min(1).required(false),
    ffmpegPath: z.string().min(1).required(false),
  })).default([]),
}) as z<Config>