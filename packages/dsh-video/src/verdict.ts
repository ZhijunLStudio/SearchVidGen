/**
 * 评测结算（消费者拥有）：加权分、逐维度阈值判定。
 *
 * 纪律（vidharness Bug#1 / E9 教训）：权重与 minScore 只在消费者侧生效，
 * 提供者返回的 JudgeScores 只是原始数据，绝不替消费者算总分。
 * 维度缺失或分数不可解析时响亮失败——诚实的"未评测"优于伪造的分数。
 * @module @zhijunlstudio/dsh-video
 */

import type { JudgeCriterion, JudgeScores } from './types.ts'

/** 一条维度的结算结果。 */
export interface VerdictDetail {
  name: string
  score: number
  weight: number
  minScore: number
  pass: boolean
}

/** 一次评测的完整结算。 */
export interface Verdict {
  /** 所有维度都通过。 */
  passed: boolean
  /** 加权平均分（0-10）。 */
  weightedScore: number
  /** 加权满分（10）。 */
  maxScore: number
  /** 未通过的维度名。 */
  failed: string[]
  details: VerdictDetail[]
}

const SCALE = 10

/** 校验单条评分：必须是 0-SCALE 内的有限数。 */
function normalizeScore(name: string, value: unknown): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > SCALE) {
    throw new TypeError('评测维度 ' + JSON.stringify(name) + ' 的分数必须是 0-' + SCALE + ' 内的数值，得到 ' + JSON.stringify(value))
  }
  return value
}

/**
 * 依据原始评分与维度声明计算结算。
 *
 * @param scores - 裁判返回的原始评分（含 feedback）。
 * @param criteria - 维度声明（weight/minScore 在此生效）。
 * @returns 结算；任何维度缺失分数即抛错（宁可响亮失败也不伪造通过）。
 */
export function computeVerdict(scores: JudgeScores, criteria: readonly JudgeCriterion[]): Verdict {
  if (criteria.length === 0) {
    throw new TypeError('computeVerdict: 至少需要一条评测维度')
  }
  const details: VerdictDetail[] = []
  for (const criterion of criteria) {
    const raw = scores.scores[criterion.name]
    if (raw === undefined) {
      throw new TypeError('评测维度 ' + JSON.stringify(criterion.name) + ' 在裁判输出中缺失分数；输出为 ' + JSON.stringify(scores.scores))
    }
    const score = normalizeScore(criterion.name, raw)
    const weight = criterion.weight === undefined ? 1 : criterion.weight
    if (typeof weight !== 'number' || !Number.isFinite(weight) || weight <= 0) {
      throw new TypeError('评测维度 ' + JSON.stringify(criterion.name) + ' 的 weight 必须是正数，得到 ' + JSON.stringify(weight))
    }
    const minScore = criterion.minScore === undefined ? 6 : criterion.minScore
    if (typeof minScore !== 'number' || !Number.isFinite(minScore) || minScore < 0 || minScore > SCALE) {
      throw new TypeError('评测维度 ' + JSON.stringify(criterion.name) + ' 的 minScore 必须是 0-' + SCALE + ' 内的数值，得到 ' + JSON.stringify(minScore))
    }
    details.push({ name: criterion.name, score, weight, minScore, pass: score >= minScore })
  }
  const totalWeight = details.reduce((sum, detail) => sum + detail.weight, 0)
  const weightedScore = details.reduce((sum, detail) => sum + detail.score * detail.weight, 0) / totalWeight
  return {
    passed: details.every(detail => detail.pass),
    weightedScore: Math.round(weightedScore * 100) / 100,
    maxScore: SCALE,
    failed: details.filter(detail => !detail.pass).map(detail => detail.name),
    details,
  }
}
