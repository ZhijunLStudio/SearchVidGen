# Agent Note: judge 缝的孪生适配器与模态守卫

Status: implemented

## Problem

对齐 deepseek-harness 的孪生适配器原则（twin-llm-adapters：**从一开始交付两个
真实不同实现的 provider，任何无法同时表达的词汇都是 Service Definition 缺陷**），
judge 缝此前只有一个实现（judge.openai-compat，本地 vLLM VLM）。单实现掩盖
两个词汇缺口：①`modalities` 声明在协议里但**不可强制**——一个 text-only 裁判
被误配到媒体评测时，会静默丢弃媒体假装评过分；②抽帧失败时 `_extract_frame`
返回 None，None 直接传给裁判同样会静默产生垃圾评分。

## Decision

1. **judge 缝第二实现 `judge.deepseek-text`**：DeepSeek 官方 API 文本裁判
   （modalities=["text"]）。剧本评审等 media=[] 场景用它：不占本地 GPU、
   不依赖 vLLM 服务存活、单价更低。真实 API 冒烟通过（评分 7.0 / passed /
   反馈可用，与 vLLM 裁判同协议、同结算路径）。
2. **模态守卫**（孪生适配器暴露的词汇缺口的落地点）：`run_judge` 按媒体
   后缀推断所需模态，超出裁判声明的 modalities 即抛错（"不要静默假装看过"）；
   同时过滤 media 中的 None（抽帧失败的 None 不再传给裁判）。
3. **抽帧失败 fail-visible**：cross_consistency 对首/尾帧缺失的记录
   `{segment_pair, error: "抽帧失败"}` 错误记录，而不是让裁判空评。

modalities 仍留在协议属性（不进 capabilities schema）：它是协议词汇不是
能力键，且守卫放在消费点（run_judge）比注册点更早拦截误配。

## Alternatives considered

- **modalities 并入 capabilities schema 校验**：否决。能力键是"声明-校验"的
  路由词汇，modalities 是协议属性；消费点守卫在"裁判被用于什么媒体"处
  判定，比注册点校验更精确（同一裁判可被合法地用于文本与媒体两处）。
- **text-only 裁判静默忽略媒体**：否决。这是单实现时代的隐藏语义，
  正是本笔记要消灭的"静默假装看过"。
- **把文本裁判做成独立 seam**：否决。协议、结算、解析完全同构，
  只是模态子集——拆 seam 是预防性拆分（capability-seams 笔记原则）。

## Consequences

- 任务配置可用 `judge: {adapter: judge.deepseek-text}` 承接 script_judge，
  vLLM 只服务图像/视频评测；两种裁判经同一 run_judge 结算、同一 judge
  产物目录归档。
- 新裁判提供者必须诚实声明 modalities；误配在第一次媒体评测时响亮失败。
- 真实 API 依赖 DEEPSEEK_API_KEY（本机已配置）；无 key 时实例化即失败
  （与 script 提供者同口径）。
