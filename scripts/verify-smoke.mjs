/**
 * 冒烟产物复检：解析 smoke/local.log 里的 SMOKE_RESULT，
 * 用 video_verify 同款 probeVideo 对成片做事实检验。
 * 用法: node scripts/verify-smoke.mjs
 */
import { readFileSync } from 'node:fs'
import { probeVideo } from '../packages/dsh-video-tool/lib/index.mjs'

const log = readFileSync('/data/lizhijun/work/SearchVidGen/smoke/local.log', 'utf-8')
const line = log.split('\n').find(l => l.startsWith('SMOKE_RESULT '))
if (!line) {
  console.log('NO_RESULT_YET')
  process.exit(2)
}
const result = JSON.parse(line.slice('SMOKE_RESULT '.length))
const media = probeVideo(result.path, '/data/lizhijun/anaconda3/envs/torch/bin/ffmpeg')
const checks = [
  ['文件存在', true],
  ['时长≥4.5s', media.durationS >= 4.5],
  ['帧率 24 附近', Math.abs(media.fps - 24) < 1],
  ['16:9 画布', Math.abs(media.width / media.height - 16 / 9) / (16 / 9) < 0.05],
  ['有原生音轨', media.hasAudio],
]
console.log('SMOKE_VERIFY ' + JSON.stringify({
  path: result.path,
  media,
  checks: checks.map(([name, pass]) => ({ name, pass })),
  ok: checks.every(([, pass]) => pass),
  provider: { model: result.model, backend: result.backend, costUsd: result.costUsd, elapsedMs: result.elapsedMs },
}))
process.exit(checks.every(([, pass]) => pass) ? 0 : 1)
