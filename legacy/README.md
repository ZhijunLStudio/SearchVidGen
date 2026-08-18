# legacy：v1 归档（Gradio 七步流水线，2026-08-14 前）

本目录冻结保存 SearchVidGen v1 的完整代码，仅供历史参考，不再维护：

- `app.py` / `config/` / `source/` —— Gradio 应用、prompt 配置与示例素材
- `src/` —— 七步流水线脚本（llm_client → image_generator →
  img2vid_description → video_generator.sh → audio_generator →
  video_processor）与 Wan2.1/InstantCharacter 相关代码
- `README.v1.md` / `README.en-US.v1.md` —— v1 时期的项目说明

现行架构（v3，dsh 插件家族）见 [../README.md](../README.md) 与
[../docs/architecture-v3.md](../docs/architecture-v3.md)。
