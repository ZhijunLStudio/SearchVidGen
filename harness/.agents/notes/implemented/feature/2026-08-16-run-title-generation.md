# Agent Note: run 标题自动生成（session title，E40）

Status: implemented

## Problem

run 目录只有 run_id 哈希，leaderboard/对比报告里扫一眼分不清哪个 run
是什么内容。DSH 范式里 session 有 human-readable 标题，vidharness 缺这层；
用户翻 22 个历史 run 全靠 query 列 + 目录名对号入座。

## Decision

1. **`_generate_run_title`（cli.py，session 层）**：用 script 提供者从
   query 提炼 ≤12 字标题。选 script seam 是 E33 先例的复用——归纳是
   自由变换任务，裁判的评分纪律（feedback=pass）与变换冲突。
   两轮尝试：首轮宽松提示 → 未产出 title 字段则次轮收紧提示
   （"上次没有输出 title 字段…"）。产物落 artifacts/title/ +
   `manifest.title` 经事件流（模型可见 ⟺ 日志，重放可恢复）。
2. **失败静默**：标题是 UX 增强，不是证据完整性的一部分——API 挂了
   或缺 key 只降级不拖垮 run（与 lazy credential 同原则）。
3. **`SegmentDirector.run(before_finalize=...)` 钩子**：标题必须在
   finalize 之前挂入。finalize 落盘 manifest + 断言不变量，在其之后
   追加事件会造成"finalized 快照"与最终总额不一致、事件终态混乱。
   钩子由 cli 传入——director 保持库函数职责，标题策略属 session 层。
4. **消费侧单一正源**：`report.collect` 加 title 字段（聚合唯一正源）；
   leaderboard MD 加"标题"列（`|` 转义防表格破构，JSON 基线保留原文）；
   对比 HTML 加列（html.escape）；run 详情页概览加 title 行。

## Alternatives considered

- **director.run 内部自动生成**：否决。director 是库函数；标题是否
  生成、用什么提供者是 session 层策略，钩子注入保持职责分离。
- **finalize 之后 set_meta + 手动 _flush**：否决。finalized 事件之后
  追加 manifest.set/artifact.saved 使事件流终态与总额口径混乱，
  且依赖私有 API。
- **裁判适配器做标题**：否决。同 E33 理由——系统纪律压倒变换任务。

## Consequences

- 每个新 run 多一次 script API 调用（实测 $0.000017/次）；失败静默。
- 老 run 无 title → 表格显示 "-"，新旧兼容（leaderboard 有无标题列
  自适应）。
- E40 证据：真实 DeepSeek API 生成"雨夜暖面"（4 字 ≤12），
  manifest/事件流/leaderboard MD/详情页全链路一致，doctor 干净；
  单测 5 个（生成持久化/重试恢复/失败静默/钩子时序/表格转义）。
- 后续可做：多语言 query 标题策略、title 进 bench 格摘要。
