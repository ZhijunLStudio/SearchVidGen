# Agent Note: 阶段级裁判路由（judge.stages）

Status: implemented

## Problem

E16 暴露：剧本评测（script_judge）与媒体评测共用同一个裁判实例，而主裁判是
本地 vLLM VLM——smoke 首跑时 vLLM 尚未就绪，剧本评测被跳过（容错降级有效
但该维度无评价）。文本评测本不需要视觉模型：judge.deepseek-text（DeepSeek
官方 API）更便宜、不占 GPU、不依赖本地服务。需要一个"按阶段选裁判"的
显式机制（对齐 DSH 的 provider-routed 模式在阶段级的落地）。

## Decision

任务配置 `judge.stages` 支持**阶段级裁判覆盖**：

```yaml
judge:
  adapter: judge.openai-compat        # 默认（媒体评测）
  params: {...}
  stages:
    script_judge: {adapter: judge.deepseek-text, params: {model: deepseek-chat}}
```

- 允许的阶段键（白名单）：`script_judge` / `script_optimize` /
  `segment_judge` / `cross_judge`（config schema 校验，未知键 fail loud）；
- SegmentDirector 构建 `judges` 映射，未覆盖的阶段回退主裁判；
- 所有裁判消费点（stage_script / optimizer / stage_segments /
  cross_consistency）按映射取实例。

真实冒烟：script_judge 走 judge.deepseek-text，两次尝试（第 1 次解析失败
空评分 → 反馈重试 → 第 2 次 5.0/7.0），judge 产物 adapter 记录正确。

## Alternatives considered

- **每个阶段独立 judge 配置键（judge_script/judge_segment/...）**：否决。
  配置重复且失去"默认主裁判"语义；stages 覆盖与回退更贴合"显式 > 隐式"。
- **按媒体自动推断裁判**：否决。隐式路由不可解释；显式声明阶段 → 裁判
  才可审计（本笔记的 stages 就是显式声明）。
- **直接改 story.yaml 主裁判为 deepseek-text**：否决。段级/跨段评测需要
  VLM；只能按阶段分流。

## Consequences

- story_smoke 的剧本评测不再依赖 VLM 服务就绪（E16 待办①关闭）；
  未来 story.yaml 可同样启用。
- 新阶段（未来新增评测阶段）必须在 JUDGE_STAGE_KEYS 登记才能路由——
  路由白名单是配置协议的一部分。
- 模态守卫仍在消费点兜底：即使误配（如 segment_judge 覆盖为 text-only），
  第一次媒体评测即响亮失败。
