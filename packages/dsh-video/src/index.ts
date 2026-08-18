/**
 * @zhijunlstudio/dsh-video —— 视频生成能力缝（Service Definition）。
 *
 * 三角色：
 * - Service Definition：本包（协议类型 + ctx.video 注册表 + 结算函数）；
 * - Service Provider：@zhijunlstudio/dsh-video-provider（配置驱动，见其 README）；
 * - Consumer：@zhijunlstudio/dsh-video-tool（模型面工具 + 编排闭环）。
 * @module @zhijunlstudio/dsh-video
 */

import type { Context } from '@deepseek-ai/cordis'
import { VideoRegistry } from './registry.ts'

export * from './types.ts'
export { computeVerdict } from './verdict.ts'
export type { Verdict, VerdictDetail } from './verdict.ts'
export { VideoRegistry, validateCapabilities, PROVIDER_ID_RE } from './registry.ts'

export const name = 'dsh-video'

/** 宿主平面装配入口：注册 ctx.video（一个 context 只允许一个实例）。 */
export function apply(ctx: Context): void {
  void new VideoRegistry(ctx)
}
