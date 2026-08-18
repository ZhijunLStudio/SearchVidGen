# SearchVidGen：DeepSeek Harness 的视频生成模式

SearchVidGen v3 把「输入一个搜索词或一两句话 → 产出带评测闭环的成片视频」
做成 **DeepSeek Harness（dsh）的原生插件家族 + agent preset（模式）**——
不再自建 harness，而是长在 dsh 上：一切皆插件（Cordis），能力缝 = 
服务定义 + 提供者 + 消费者，模式 = agent preset。

> 历史：v1（Gradio 七步流水线，`legacy/`）与 v2（Python 版 harness，
> `harness/`，现降级为执行引擎 + 评测/基准层）见各自目录。

## 架构

```
packages/                          # dsh 插件家族（pnpm monorepo）
├── dsh-video/                     # 能力缝 SD：ctx.video 注册表 + 按能力路由 + 结算协议
├── dsh-video-provider/            # 配置驱动提供者（一行配置一个模型，不装新包）
│   ├── remote  生成器             #   openai / minimax 协议，URL+key+model 接入任意端点
│   ├── local   生成器             #   vh gen-single 子进程（本机 GPU 引擎）
│   └── judge   裁判               #   OpenAI 兼容 chat completions（VLM/文本），视频抽帧
└── dsh-video-tool/                # 消费者：5 个模型面工具 + 编排闭环 + ctx.jobs
presets/video/                     # 「视频生成模式」agent preset（GUI 预设选择器可选）
harness/                           # Python 引擎（vidharness）+ 实验/基准/leaderboard
legacy/                            # v1 Gradio 流水线（归档）
```

**能力缝三角色**：Service Definition（dsh-video 的协议与注册表）/
Service Provider（dsh-video-provider，配置即状态）/
Consumer（dsh-video-tool 的工具与编排）——与 dsh 官方 capability seam 同构。

**模型选择鲁棒，不叠包**：新模型 = 在 provider 配置里加一行
（`baseUrl + model + credential`，未知协议参数经 `defaultParams` 透传），
零代码改动；没配任何提供者时工具响亮失败并给出配置指引。

## 用户旅程（视频生成模式）

GUI 新建会话 → 预设选「视频生成模式」→ 输入
`雨夜，一只小猫在旧书店橱窗前躲雨` → agent 调用 `video_generate_search`
（后台 job）→ LLM 分镜 → 逐段生成 + 评测重试闭环 → 跨段一致性检查 →
FFmpeg 总装 → 成片 mp4 + 报告（评分/成本）→ `video_verify` 复检后交付。

## 模型面工具

| 工具 | 作用 |
|---|---|
| `video_adapters` | 已注册生成器/裁判的声明目录（capabilities/参数/模态） |
| `video_generate` | 一句指令 → 单段视频（后台 job） |
| `video_generate_search` | 搜索词/一两句话 → 剧本→逐段生成+评测闭环→跨段检查→总装→报告 |
| `video_judge` | 已有媒体按维度打分 + 通过/未通过结算 + 修正反馈 |
| `video_verify` | ffprobe 事实检验（时长/帧率/分辨率/宽高比/音轨） |

## 安装（任何人）

```bash
# 1. 三个包装进 web profile（dsh ≥ 0.1.0-rc.6）
dsh plugin --profile web add <dsh-video 包路径>
dsh plugin --profile web add <dsh-video-provider 包路径>
dsh plugin --profile web add <dsh-video-tool 包路径>

# 2. 配置提供者（一行一个模型；写进 ~/.dsh/cordis.patch.yml 的 video-provider 行）
#    remote 示例（无需 GPU、无需 Python）：
#      generators:
#        - {id: h3-api, kind: remote, protocol: minimax,
#           baseUrl: https://api.minimaxi.com, model: MiniMax-H3,
#           credential: MINIMAX_API_KEY,
#           capabilities: {minDurationS: 3, maxDurationS: 15, audio: true,
#                          maxRefs: 9, firstLastFrame: true, resolutions: [768P, 2K]}}
#      judges:
#        - {id: vlm, baseUrl: http://127.0.0.1:8030/v1, model: judge-qwen3.5-27b,
#           credential: DEEPSEEK_API_KEY, modalities: [video, image]}
#    （凭据放 ~/.dsh/.credentials.yaml 或同名环境变量）

# 3. 同步「视频生成模式」预设到 ~/.dsh/.agent-presets/video
node scripts/sync-preset.mjs

# 4. GUI（127.0.0.1:3080）新建会话 → 预设选「视频生成模式」
```

本机 GPU 引擎（local 提供者）另需：vidharness 环境（`harness/`，pip install -e .），
H3 权重路径；模型环境不含 ffmpeg 时在 local 配置行加 `ffmpegDir`（指向
ffmpeg 所在目录，经 spec.ffmpeg_dir 注入引擎 PATH）。remote 提供者与裁判只需要
网络与凭据。

## 检验（测试与验证分层）

1. **TS 单测/集成**：`pnpm vet`（typecheck + vitest + build）——mock HTTP 协议、
   mock vh 子进程、能力路由 fail-loud、评测结算归消费者、预设组合；
2. **Python 契约测试**：`harness/` 内 `python -m pytest tests/`（170 用例）
   ——含 `vh gen-single` 的 stdout JSON 契约（mock 生成器 + 真实 ffmpeg）；
3. **真实冒烟**：`node scripts/smoke-local.mjs`（TS provider → vh → H3 真实 GPU）；
4. **评测/基准/证据**：vidharness 的 `vh doctor / regress / leaderboard / bench`
   继续作为实验与证据层（experiments/ 事件溯源 + E 系列结论）。

## 开发

```bash
export PATH=$HOME/.nvm/versions/node/v22.22.2/bin:$PATH   # node >= 22.19
corepack pnpm install
corepack pnpm vet            # 三包 typecheck + test + build
```

目录内的 `docs/` 保留 v1/v2 时期的调研与证据文档；
harness 的决策记忆在 `harness/.agents/notes/`。
