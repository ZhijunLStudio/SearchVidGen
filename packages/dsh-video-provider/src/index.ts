/**
 * @zhijunlstudio/dsh-video-provider —— 配置驱动提供者（Service Provider 角色）。
 *
 * 装配即注册：apply 把 config.generators / config.judges 逐条注册进
 * ctx.video（注册失败在装配期响亮失败）。新模型 = 新配置行。
 * @module @zhijunlstudio/dsh-video-provider
 */

import type { Context } from '@deepseek-ai/cordis'
import type { Config, LocalVideoProviderConfig, RemoteJudgeProviderConfig, RemoteVideoProviderConfig } from './config.ts'
import { RemoteVideoProvider } from './remote-video.ts'
import { LocalVideoProvider } from './local-video.ts'
import { RemoteJudgeProvider } from './remote-judge.ts'

export const name = 'dsh-video-provider'

export const inject = ['video', 'credentials']

export { Config } from './config.ts'
export type { Config as ConfigType, LocalVideoProviderConfig, RemoteJudgeProviderConfig, RemoteVideoProviderConfig } from './config.ts'
export { RemoteVideoProvider } from './remote-video.ts'
export { LocalVideoProvider, validateLocalResult } from './local-video.ts'
export { RemoteJudgeProvider, parseJudgeScores } from './remote-judge.ts'
export { sampleFrames } from './frames.ts'
export { resolveCredential } from './credentials.ts'

export function apply(ctx: Context, config: Config): void {
  for (const generator of config.generators ?? []) {
    ctx.effect(() => {
      const provider = generator.kind === 'remote'
        ? new RemoteVideoProvider(generator as RemoteVideoProviderConfig, ctx)
        : new LocalVideoProvider(generator as LocalVideoProviderConfig)
      return ctx.video.registerGenerator(provider)
    })
  }
  for (const judge of config.judges ?? []) {
    ctx.effect(() => {
      const provider = new RemoteJudgeProvider(judge as RemoteJudgeProviderConfig, ctx)
      return ctx.video.registerJudge(provider)
    })
  }
}
