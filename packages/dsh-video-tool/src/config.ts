/**
 * 工具包配置（视频生成模式的工程面）。
 * @module @zhijunlstudio/dsh-video-tool
 */

import z from 'schemastery'
import { homedir } from 'node:os'
import { join } from 'node:path'
import type { JudgeCriterion } from '@zhijunlstudio/dsh-video'

export interface ScriptConfig {
  provider?: string
  model?: string
  temperature?: number
  maxTokens?: number
}

export interface Config {
  /** 产物根目录（缺省 ~/.dsh/video-gen/<runId>/...）。 */
  outputRoot?: string
  /** 长任务缺省走后台 job（工具参数 run_in_background 可覆盖）。 */
  runInBackground?: boolean
  /** 剧本 LLM 路由（缺省 deepseek-official / deepseek-v4-flash）。 */
  script?: ScriptConfig
  /** 评测失败重试上限（生成次数 = maxRetries + 1）。 */
  maxRetries?: number
  /** 失败反馈注入下一次生成。 */
  injectFeedback?: boolean
  feedbackPrefix?: string
  /** 逐段评测默认维度（video_judge 的 preset: 'segment'）。 */
  criteria?: JudgeCriterion[]
  /** 跨段评测默认维度（preset: 'cross'）。 */
  crossCriteria?: JudgeCriterion[]
  /** 自动路由时的后端偏好。 */
  preferBackend?: 'remote' | 'local'
  /** ffmpeg 路径（缺省 PATH）。 */
  ffmpegPath?: string
}

export function defaultOutputRoot(): string {
  return join(homedir(), '.dsh', 'video-gen')
}

export const DEFAULT_SEGMENT_CRITERIA: JudgeCriterion[] = [
  { name: '与指令一致性', question: '视频是否准确呈现生成指令描述的主体、动作、镜头与环境？', weight: 1, minScore: 6 },
  { name: '画面质量', question: '是否存在明显缺陷（肢体畸形、闪变、主体崩坏、水印杂质）？无缺陷给高分', weight: 1, minScore: 6 },
  { name: '音效场景相符', question: '画面内容是否与指令描述的音效场景相符？', weight: 0.8, minScore: 5 },
]

export const DEFAULT_CROSS_CRITERIA: JudgeCriterion[] = [
  { name: '跨段一致性', question: '上一段结尾与下一段开头：人物外貌服装是否一致、场景是否自然衔接？', weight: 1, minScore: 6 },
  { name: '叙事推进', question: '这两帧是否为合理推进的新镜头（机位/景别有变化），而不是同一画面的复制（冻结帧）？完全相同的画面应给低分', weight: 1, minScore: 5 },
]

const criterionSchema = z.object({
  name: z.string().min(1),
  question: z.string().min(1),
  weight: z.number().min(0).required(false),
  minScore: z.number().min(0).max(10).required(false),
})

export const Config: z<Config> = z.object({
  outputRoot: z.string().min(1).required(false),
  runInBackground: z.boolean().required(false),
  script: z.object({
    provider: z.string().min(1).required(false),
    model: z.string().min(1).required(false),
    temperature: z.number().min(0).max(2).required(false),
    maxTokens: z.number().min(1).step(1).required(false),
  }).required(false),
  maxRetries: z.number().min(0).max(5).step(1).required(false),
  injectFeedback: z.boolean().required(false),
  feedbackPrefix: z.string().required(false),
  criteria: z.array(criterionSchema).required(false),
  crossCriteria: z.array(criterionSchema).required(false),
  preferBackend: z.union([z.const('remote'), z.const('local')]).required(false),
  ffmpegPath: z.string().min(1).required(false),
}) as z<Config>
