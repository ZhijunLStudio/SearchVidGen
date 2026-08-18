/**
 * @zhijunlstudio/dsh-video-tool —— 视频生成模式的模型面工具（Consumer 角色）。
 *
 * 本行通常由「视频生成模式」preset（presets/video/agent.cordis.yml）挂到
 * agent 平面（只有该模式的会话带这 5 个工具）；也可全局装配。
 * 依赖 host 平面的 ctx.video（@zhijunlstudio/dsh-video）与提供者
 * （@zhijunlstudio/dsh-video-provider）。
 * @module @zhijunlstudio/dsh-video-tool
 */

import type { Context } from '@deepseek-ai/cordis'
import type { Config } from './config.ts'
import { registerVideoTools } from './tools.ts'

export const name = 'dsh-video-tool'

export const inject = ['tools', 'video', 'jobs', 'llm']

export { Config } from './config.ts'
export type { Config as ConfigType } from './config.ts'
export { registerVideoTools } from './tools.ts'
export { runSearchGeneration, runSingleGeneration } from './orchestrator.ts'
export type { SearchReport, SearchRequest, SegmentReport } from './orchestrator.ts'
export { generateScript, parseScriptJson, buildScriptPrompt } from './script.ts'
export type { SegmentPlan, ScriptResult } from './script.ts'
export { probeVideo, extractFrame, concatVideos } from './ffmpeg.ts'
export { startVideoJob, LineBuffer } from './jobs.ts'

export function apply(ctx: Context, config: Config): void {
  registerVideoTools(ctx, config)
}
