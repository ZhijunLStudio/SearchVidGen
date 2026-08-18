# @zhijunlstudio/dsh-video-tool

视频生成模式的模型面工具集（Consumer 角色）：

- `video_adapters`：已注册生成器/裁判的声明目录
- `video_generate`：一句指令 → 单段视频（后台 job）
- `video_generate_search`：搜索词/一两句话 → LLM 分镜 → 逐段生成+评测重试闭环 →
  跨段一致性检查 → FFmpeg 总装 → 成片+报告（后台 job）
- `video_judge`：已有媒体打分 + 通过/未通过结算 + 修正反馈
- `video_verify`：ffprobe 事实检验（时长/帧率/分辨率/宽高比/音轨）

本包默认不全局挂载：由「视频生成模式」preset（SearchVidGen/presets/video）挂到 agent 平面。
