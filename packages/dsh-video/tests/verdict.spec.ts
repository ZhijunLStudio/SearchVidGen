import { describe, expect, it } from 'vitest'
import { computeVerdict } from '../src/verdict.ts'
import type { JudgeCriterion } from '../src/types.ts'

const criterion = (name: string, overrides: Partial<JudgeCriterion> = {}): JudgeCriterion => ({
  name, question: name + ' 问题', ...overrides,
})

describe('computeVerdict', () => {
  it('computes weighted score and passes when every dimension clears its threshold', () => {
    const verdict = computeVerdict(
      { scores: { a: 8, b: 9 }, feedback: '' },
      [criterion('a', { weight: 1, minScore: 6 }), criterion('b', { weight: 2, minScore: 7 })],
    )
    expect(verdict.passed).toBe(true)
    expect(verdict.weightedScore).toBeCloseTo((8 + 2 * 9) / 3, 2)
    expect(verdict.failed).toEqual([])
    expect(verdict.details).toHaveLength(2)
  })

  it('fails only the dimensions below their own minScore', () => {
    const verdict = computeVerdict(
      { scores: { a: 8, b: 4 }, feedback: '' },
      [criterion('a'), criterion('b', { minScore: 6 })],
    )
    expect(verdict.passed).toBe(false)
    expect(verdict.failed).toEqual(['b'])
  })

  it('defaults weight to 1 and minScore to 6', () => {
    const verdict = computeVerdict(
      { scores: { a: 6 }, feedback: '' },
      [criterion('a')],
    )
    expect(verdict.passed).toBe(true)
    expect(verdict.details[0]).toMatchObject({ weight: 1, minScore: 6 })
  })

  it('throws when a criterion has no score instead of faking a pass', () => {
    expect(() => computeVerdict(
      { scores: { a: 8 }, feedback: '' },
      [criterion('a'), criterion('b')],
    )).toThrow(/缺失分数/)
  })

  it('throws on out-of-range or non-finite scores', () => {
    expect(() => computeVerdict(
      { scores: { a: Number.NaN }, feedback: '' },
      [criterion('a')],
    )).toThrow(/0-10/)
    expect(() => computeVerdict(
      { scores: { a: 11 }, feedback: '' },
      [criterion('a')],
    )).toThrow(/0-10/)
    expect(() => computeVerdict(
      { scores: { a: '9' as unknown as number }, feedback: '' },
      [criterion('a')],
    )).toThrow(/0-10/)
  })

  it('throws on invalid weight or minScore declarations', () => {
    expect(() => computeVerdict(
      { scores: { a: 8 }, feedback: '' },
      [criterion('a', { weight: 0 })],
    )).toThrow(/weight/)
    expect(() => computeVerdict(
      { scores: { a: 8 }, feedback: '' },
      [criterion('a', { minScore: -1 })],
    )).toThrow(/minScore/)
  })

  it('rejects empty criteria', () => {
    expect(() => computeVerdict({ scores: {}, feedback: '' }, [])).toThrow(/至少需要一条/)
  })
})