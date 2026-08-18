# SearchVidGen v3 架构：DeepSeek Harness 插件家族

2026-08-18。本文记录 v1（Gradio 流水线）→ v2（Python 自建 harness）→ v3
（dsh 插件 + 视频生成模式）的方向修正与关键决策。

## 为什么不再自建 harness（v2 → v3）

v2 的 `harness/vidharness` 是 deepseek-harness 哲学的 Python 缩放模仿品：
能力缝/提供者/消费者、事件溯源、bench/leaderboard 都对齐了，但它
平行于 dsh 存在——用户要装两套东西，且永远追不上官方扩展点。
正确做法是把能力做成 **dsh 插件**、把产品形态做成 **agent preset（模式）**，
Python 侧降级为执行引擎与实验/评测层。

## 关键决策

1. **配置驱动的提供者，不叠包**：新模型 = 配置里加一行（URL/key/model +
   协议透传），而不是每模型一个包。remote（openai/minimax）/local（vh 引擎）/
   judge（OpenAI 兼容）三个实现覆盖全部接入形态。
2. **模式 = agent preset**：`presets/video` 同步进 `~/.dsh/.agent-presets/video`，
   GUI 预设选择器可选；工具行挂 agent 平面（其他会话不带视频工具 schema，
   省 token），persona 定义工作流（先 video_adapters → video_generate_search →
   job 跟踪 → video_verify 复检）。
3. **评测结算归消费者**：provide 只返回原始评分；computeVerdict（dsh-video）
   统一结算权重/阈值（移植 vidharness Bug#1 教训）。
4. **能力路由 fail-loud**：注册期校验 capabilities；请求约束逐候选列出未满足
   原因；显式 provider id 优先；模态守卫（媒体评测配到 text-only 裁判第一次
   调用即失败）。
5. **长任务走 ctx.jobs**：video_generate/video_generate_search 默认后台 job，
   完成自动通知 agent；进度走 job 流式输出。
6. **wire 边界严格校验**：TS ↔ vh 子进程的 stdout JSON 契约两侧都有契约测试
   （TS: packages/dsh-video-provider/tests；Python: harness/tests/test_gen_single_contract.py）。

## 子进程契约（vh gen-single --json）

输入 spec.json：text/refs/duration/ratio/seed/generator{adapter,params}/
judge{adapter,params,criteria}/retry/ffmpeg_dir/out；输出 stdout 单行 JSON：
video{path,...}/judge/costUsd/elapsedS/runDir；失败非零退出 + stderr 可读原因。
进度日志全部进 stderr（stdout 只留契约行）。

真实冒烟发现并修复的环境缺口：模型环境（h3int8）不含 ffmpeg（在 torch 环境
bin 下），首次冒烟在去噪完成后保存视频失败。契约因此增加 `ffmpeg_dir`：
调用方把 ffmpeg 所在目录 prepend 进引擎 PATH（TS 配置 `ffmpegDir` 字段），
缺失时仍在最早点响亮失败——环境缺口不静默（E12 同口径）。

## 检验分层

- TS：`pnpm vet`（43 用例）——mock HTTP 服务器/mock vh 脚本/路由/结算/预设；
- Python：`pytest`（170 用例）——注册表/评测/事件溯源/gen-single 契约；
- 真实冒烟：scripts/smoke-local.mjs（TS→vh→H3 GPU 实链路）；
- 证据层：vh doctor/regress/leaderboard/bench（保留 v2 资产）。
