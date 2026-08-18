/**
 * 凭据解析：credentials seam 优先，环境变量兜底；缺凭据响亮失败并给指引。
 * @module @zhijunlstudio/dsh-video-provider
 */

import type { Context } from '@deepseek-ai/cordis'
import type { CredentialRef } from '@deepseek-ai/dsh-credentials'

/**
 * 按引用名解析凭据（seam → 环境变量）。
 * @param ctx - 宿主上下文（inject: credentials）。
 * @param name - 引用名（如 MINIMAX_API_KEY / DEEPSEEK_API_KEY）。
 * @returns 非空密钥；两个来源都没有时抛错并说明配置位置。
 */
export async function resolveCredential(ctx: Context, name: string): Promise<string> {
  try {
    const resolved = await ctx.credentials.resolve(name as CredentialRef)
    if (resolved !== undefined && resolved.value.length > 0) return resolved.value
  } catch (error) {
    // seam 失败不吞：下面统一走环境变量兜底再 fail loud
    void error
  }
  const env = process.env[name]
  if (env !== undefined && env.length > 0) return env
  throw new Error(
    '未找到凭据 ' + JSON.stringify(name) + '。请二选一配置：'
    + '(1) dsh 凭据文件 ~/.dsh/.credentials.yaml 增加 ' + name + '；'
    + '(2) 环境变量 ' + name + '。密钥不会出现在错误信息中。',
  )
}
