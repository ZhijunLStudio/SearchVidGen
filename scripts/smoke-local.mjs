/**
 * 真实 GPU 冒烟：TS LocalVideoProvider → vh gen-single → H3 ref2va int8（单卡）→ mp4。
 * 用法: node scripts/smoke-local.mjs
 */
import { LocalVideoProvider } from '../packages/dsh-video-provider/lib/index.mjs'

const provider = new LocalVideoProvider({
  id: 'h3-local-smoke',
  kind: 'local',
  label: '冒烟: H3 ref2va int8 单卡',
  pythonPath: '/data/lizhijun/anaconda3/envs/h3int8/bin/python',
  cwd: '/data/lizhijun/work/SearchVidGen/harness',
  adapter: 'generator.minimax-h3-local',
  params: {
    model_path: '/data-ssd/lizhijun/models/MiniMax-H3',
    gpu: '2',
    variant: 'ref2va',
    steps: 30,
  },
  capabilities: { minDurationS: 5, maxDurationS: 15, audio: true, maxRefs: 9, firstLastFrame: false, resolutions: ['768p'] },
  timeoutMs: 45 * 60 * 1000,
  gpuPriceUsdPerHour: 1.2,
  ffmpegDir: '/data/lizhijun/anaconda3/envs/torch/bin',
})

const started = Date.now()
try {
  const result = await provider.generate(
    {
      text: '一只白色小猫在雨夜的旧书店橱窗前躲雨，霓虹灯倒影在湿润的地面上，镜头缓慢推进',
      refs: ['/data/lizhijun/work/SearchVidGen/harness/assets/anchor_cat.jpg'],
      duration: 5,
      ratio: '16:9',
      seed: 42,
    },
    {
      workdir: '/data/lizhijun/work/SearchVidGen/smoke/local',
      onProgress: line => console.error('[progress]', line),
    },
  )
  console.log('SMOKE_RESULT ' + JSON.stringify({ ...result, elapsedMs: Date.now() - started }))
} catch (error) {
  console.error('SMOKE_FAILED', error && error.message ? error.message : String(error))
  process.exit(1)
}
