# dsh 插件化：视频生成模式取代自建 harness 的产品层（2026-08-18）

## 决策

SearchVidGen 的产品面不再自建 harness：改为 DeepSeek Harness 插件家族
（@zhijunlstudio/dsh-video / -provider / -tool）+ agent preset「视频生成模式」。
vidharness 降级为两件事：

1. **执行引擎**：新增 `vh gen-single --json` 子进程契约（stdout 单行 JSON，
   进度走 stderr），被 TS local 提供者驱动；
2. **实验与评测层**：bench/leaderboard/doctor/regress/记忆机制保留不变。

## 关键点

- 提供者配置驱动（一行一个模型），不叠包；能力路由/评测结算/模态守卫的
  教训（E3/E11/E15/Bug#1）全部移植进 TS 包并测试。
- SegmentDirector 的 judge 配置改为可选（gen-single 无裁判规格时不装配），
  single 任务计划支持 context.duration——核心故事路径零改动（170 测试全绿）。
- 真实冒烟链路：TS LocalVideoProvider → vh gen-single → H3 ref2va int8 单卡
  （GPU 2，h3int8 环境）→ mp4。
- 冒烟暴露的环境缺口：h3int8 环境无 ffmpeg（在 torch 环境 bin）→ 首次冒烟
  去噪完成后保存失败。契约增加 spec.ffmpeg_dir（TS 配置 ffmpegDir），
  gen-single 把该目录 prepend 进 PATH；缺失仍 fail loud。契约测试
  test_ffmpeg_dir_prepends_to_path 锁定。

## 否决

- 否决「每模型一个 npm 包」：安装面爆炸，新模型要发版；配置行即可表达
  URL/key/model + 协议参数透传。
- 否决「TS 重写 Python 编排」：SegmentDirector 已过 E16 真实全链路验证，
  TS 只做薄编排（video_generate_search 单段直通 vh，多段剧本在 TS 侧用
  ctx.llm + 评测闭环，保持一致语义）。
