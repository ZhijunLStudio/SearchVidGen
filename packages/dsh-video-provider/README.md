# @zhijunlstudio/dsh-video-provider

配置驱动的提供者（Service Provider 角色）。一行配置接入一个模型，不装新包：

- **remote 生成器**：HTTP 协议端点。`protocol: openai`（/videos/generations 风格）
  或 `protocol: minimax`（/v2/video_generation 官方 API）；`baseUrl + model + credential`
  即可接入任何兼容端点，未知协议参数经 `defaultParams` 透传。
- **local 生成器**：本机引擎（`vh gen-single --json` 子进程契约，
  SearchVidGen/harness 的 vidharness）。`pythonPath + cwd + adapter + params`。
- **judge 裁判**：OpenAI 兼容 chat completions（本地 vLLM / DeepSeek API / 任何 VLM），
  视频评测自动 ffmpeg 抽帧；`modalities` 声明模态，媒体评测配到 text-only 裁判响亮失败。

凭据走 dsh credentials seam（`credential` 引用名）或同名环境变量；缺凭据在第一次调用时
响亮失败并给配置指引。配置错误在装配期（schemastery）拒绝。
