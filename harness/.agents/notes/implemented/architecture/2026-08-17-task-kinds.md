# Agent Note: 任务种类路由——消除"一切皆故事"偏见（E46，通用化 R1）

Status: implemented

## Problem

用户直击要害："一直都在以故事的形式做视频？这不是通用化的思想。"
自查确认故事偏见嵌在四层：
①配置层**强制** `pipeline.script`（一切任务必须带 LLM 剧本缝）；
②提供者系统人格是"导演/分镜计划"；
③流水线硬编码 剧本→跨段叙事评测→旁白总装；
④所有任务 YAML 都是 story 形态。核心机制（注册表/缝/bench/事件
溯源）本身是通用的——偏见在编排层，不在底座。

## Decision

1. **`task.kind` 路由**：`story`（原有 LLM 分镜故事）/ `single`
   （query 即视频指令 → 单条生成+评测 → 成片）/ `shots`
   （task.clips 多条指令 → 各自生成+评测 → 拼接）。
   非 story 任务：无 LLM 剧本、无跨段叙事评测、无旁白、无
   script_optimize/script_judge/audio_verify 阶段——用户指令直接
   成为生成输入。
2. **配置语义**：kind 缺省 story（向后兼容）；story 仍强制 script
   缝；shots 强制 clips（每条含 video_prompt，duration/ratio 可选）；
   clips 只对 shots 有意义（fail loud）；script 缝对非 story 任务
   变为可选。
3. **缝保持不变**：GenRequest/评测/事件溯源/不变量全部复用——
   通用性来自编排层的任务形态，不靠再造底座（对齐"能力缝完整
   不残缺"：single/shots 只是 SegmentDirector 的另一种消费方式）。
4. **标题暂只 story**：非 story 无 LLM 剧本适配器（cli 守卫跳过）；
   独立标题适配器留作后续。
5. 测试 +3：kind 校验规则矩阵、single 路由（无 script/cross 阶段、
   不变量通过）、shots 路由（每条 clip 生成 + 无旁白总装）。

## Alternatives considered

- **再造一个"通用流水线"类**：否决。生成+评测+成片三段在三种形态
  下完全同构，只有 story 多了 LLM 剧本与叙事评测——路由是唯一
  需要的差异面。
- **把旁白/叙事拆成可选插件**：未来方向（如 shots 需要旁白时）。
  本轮先立 task.kind 骨架，插件化留待按需触发。

## Consequences

- 通用化的第一步落地：非故事任务从配置到流水线全链路可行
  （tasks/shot_single.yaml、tasks/shots.yaml）。
- 真实 GPU 验收**尝试**：GPU 4,6 被共享机器租户抢占（加载中途
  49.6GB 被占 → OOM fail-loud，run 事件完整可续跑）——共享机
  争用现实再次实证；run 保留 experiments/shot_single/ 待续跑。
- 后续路线（不预防性建设）：①shots 旁白/TTS 可选；②独立标题
  适配器（非 story 任务）；③kind 扩展（如 img2vid 单图任务）在
  真实需求出现时按同模式加。
- 测试 166（+3）；kind 进 manifest（set_meta），跨期可审计。
