# Agent Note: 能力缝三层角色映射（seams/providers/consumers）

Status: implemented

## Problem

一个可替换能力涉及三个以不同速率、因不同原因变化的关注点：**约定**（这个能力是什么）、
**实现**（它如何运行）、**消费方**（编排代码面向什么编程）。把三者捆在一起会耦合变化
速率：换一个生成后端时，编排代码也被搅动，尽管协议从未改变。

本决策移植 [DeepSeek Harness 的能力缝 Agent Note](
/data/lizhijun/work/Harness/deepseek-harness/.agents/notes/implemented/architecture/2026-06-13-capability-seams.zh.md)
到 Python 规模。

## Decision

目录结构按**角色**划分，核心只留注册表与实验管理：

- **Service Definition**（`seams/`）：`Protocol` + 数据结构，只依赖协议所需词汇。
  生成缝 = `MediaGenerator` + `GenRequest` + `Artifact`；评测缝 = `Judge` +
  `JudgeCriteria` + `RetryPolicy`。SD 不含任何厂商字段。
- **Service Provider**（`providers/`）：按同一 SD 实现的适配器，`@register("seam.name")`
  加载即注册。新模型 = 新文件 + 一行注册，核心零改动。
- **Consumer**（`consumers/`）：编排与闭环（judge_loop / segment_director /
  script_optimizer / assemble / audio_verify / fallback），只面向 SD 编程，
  从不 import provider 特有类型（例外：provider 自身实现解析逻辑）。

一个 **seam** 是三层角色的组合，单一角色不是 seam。当前 seam 清单：
`generator`（2 个提供者：本地 H3-Base / 官方 H3 API + fallback 合成器）、
`script`（DeepSeek V4 Flash）、`judge`（OpenAI 兼容 VLM）、`transcribe`（SenseVoice）。

**不预防性拆分**：如果一个能力只有一种可设想的提供者和一个消费者，就保持单模块
直到第二个出现（DSH 同款规则）。转写缝目前只有一个提供者，未拆。

## Alternatives considered

- **单文件大流水线**：否决。换模型/换评测器都会牵动整条流水线，无法做对比实验
  （对比实验要求"只换一个变量"）。
- **每能力一个 pip 包**：否决。本仓库规模（约 2000 行）下拆包只增加样板，
  DSH 的拆包理由（独立发布/版本管理）暂不成立；目录角色边界已足够承载变化速率。
- **插件式动态发现（entry points）**：否决。显式注册 + 显式加载（
  `load_builtin_adapters()`）在单仓库下更可审计；动态发现留给多仓库部署时再议。

## Consequences

- 新模型适配器只需读 `seams/generator.py` 的 Protocol 注释即可实现；
  换评测器不触碰任何消费者。
- `seams/__init__.py` 是 SD 的公共出口；消费者与提供者都从这里 import，
  不 import 彼此的模块路径（`consumers.judge_loop.parse_scores` 是唯一例外，
  因解析逻辑同时是提供者侧能力与消费者侧工具，见 judge 结算归属笔记）。
