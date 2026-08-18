/**
 * 真实裁判冒烟：deepseek-text（OpenAI 兼容 chat completions → DeepSeek API）。
 * 验证配置行的裁判链路：URL/key/model 解析 + JSON 评分解析。
 * 用法: node scripts/smoke-judge.mjs
 */
import { RemoteJudgeProvider } from '../packages/dsh-video-provider/lib/index.mjs'

// 凭据走环境变量兜底（与生产 resolveCredential 的 seam→env 逻辑一致）
const stubCtx = { credentials: { resolve: async () => undefined } }

const provider = new RemoteJudgeProvider({
  id: 'deepseek-text-smoke',
  baseUrl: 'https://api.deepseek.com',
  model: 'deepseek-chat',
  credential: 'DEEPSEEK_API_KEY',
  modalities: ['text'],
}, stubCtx)

try {
  const result = await provider.judge(
    {
      media: [],
      modality: 'text',
      criteria: [
        { name: '与指令一致性', question: '这段文案是否清晰描述了视频内容？' },
      ],
    },
    { workdir: '/data/lizhijun/work/SearchVidGen/smoke/judge' },
  )
  console.log('JUDGE_RESULT ' + JSON.stringify(result))
} catch (error) {
  console.error('JUDGE_FAILED', error && error.message ? error.message : String(error))
  process.exit(1)
}
