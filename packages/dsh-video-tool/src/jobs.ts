/**
 * 长任务后台化：video 工具族的 ctx.jobs 集成。
 * 后台分支返回 {kind:'background', jobId}；job 完成后运行时自动通知
 * 拥有者 agent（新 turn 携带产物摘要）。
 * @module @zhijunlstudio/dsh-video-tool
 */

import type { Agent } from '@deepseek-ai/dsh-agent'
import type { JobHooks, JobOutcome } from '@deepseek-ai/dsh-jobs'
import type { Context } from '@deepseek-ai/cordis'

declare module '@deepseek-ai/dsh-jobs' {
  interface JobKindMap {
    video: 'video'
  }
}

/** 进度行缓冲（readOutput 消费游标）。 */
export class LineBuffer {
  private lines: string[] = []
  private cursor = 0

  push(line: string): void {
    if (line.trim().length === 0) return
    if (this.lines.length > 500) {
      // 防无界增长：丢弃最旧的一半并保留游标语义
      this.lines = this.lines.slice(250)
      this.cursor = Math.max(0, this.cursor - 250)
    }
    this.lines.push(line)
  }

  drain(): string {
    const pending = this.lines.slice(this.cursor)
    this.cursor = this.lines.length
    return pending.join('\n')
  }
}

export interface VideoJobHandles extends JobHooks {
  abort: AbortController
  /** 进度行写入 job 流式输出（readOutput 消费）。 */
  progress(line: string): void
}

/**
 * 把一段异步工作作为后台 video job 启动。
 * @returns {kind:'background', jobId}（对齐 dsh-tool-bash 的后台分支形状）。
 */
export function startVideoJob(
  ctx: Context,
  owner: Agent | undefined,
  label: string,
  run: (handles: VideoJobHandles) => Promise<JobOutcome>,
): { kind: 'background'; jobId: string } {
  const jobId = ctx.jobs.start({
    kind: 'video',
    label,
    owner,
    outputLimitBytes: 65536,
    run: () => {
      const abort = new AbortController()
      const buffer = new LineBuffer()
      let cancelReason: string | undefined
      let settled: Promise<JobOutcome> | undefined
      const hooks: VideoJobHandles = {
        abort,
        progress(line) {
          buffer.push(line)
        },
        cancel(reason) {
          cancelReason = reason
          abort.abort(new Error(reason ?? '已取消'))
        },
        readOutput() {
          return buffer.drain()
        },
        get done() {
          settled ??= run(hooks).then(outcome => ({
            ...outcome,
            output: outcome.output ?? buffer.drain(),
          })).catch((error: unknown) => ({
            status: 'failed' as const,
            detail: (error as Error).message ?? String(error),
            output: buffer.drain(),
          }))
          return settled
        },
      }
      void hooks.done
      return hooks
    },
  })
  return { kind: 'background', jobId: String(jobId) }
}
