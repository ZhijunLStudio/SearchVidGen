# Agent Note: 经验记忆提升规则是死代码（修复 + 版本化）

Status: implemented

## Problem

经验记忆文档宣称"同一规范化 complaint 出现 >= promote_threshold 次 → 提升为
经验"，但 `add()` 只累加 `count`、从不置 `promoted`，而 `experience_lines()`
要求 `count>=threshold AND promoted`——于是裁判反馈**永远无法自动提升**。
实证：真实 _memory.jsonl 有 8 条条目、3 条经验行，全部来自 `add_experience()`
（CLI 手动沉淀），重复出现的裁判反馈一条都没被提升过。这是与 E11 同类的
"文档承诺与代码行为脱节"bug。

## Decision

1. **提升发生在 add() 内**：count 到达阈值即置 `promoted=True` + `promoted_at`；
   `experience_lines()` 简化为 `kind=="experience" or promoted`。
2. **记录格式版本化**（对齐 DSH 的 session log version 机制）：
   `MEMORY_FORMAT_VERSION = 1`，每条记录带 `v`；旧文件（无 v 字段）按 v0
   兼容读取、下次 flush 统一升级；未知版本行跳过并记入 `load_warnings`。
3. **损坏必须可观测**：坏行/缺必需字段行不再静默跳过，而是记入
   `load_warnings`（内存是辅助数据不整体拒绝加载，但损坏要可见）。
4. **sources 上限**：每条经验的来源回溯只留最近 5 个（防无限增长）。

## Alternatives considered

- **保留 promoted 双条件，只在 experience_lines 里放宽**：否决。双条件没有
  语义来源（提升就是 count 的事）；删掉冗余状态比维护两个状态一致更简单。
- **坏行直接抛异常拒绝加载**：否决。记忆是辅助数据，坏一行不该让整条
  流水线起不来；但"跳过且不可见"也不行——load_warnings 是折中，
  doctor 类工具未来可消费。
- **sources 全量保留**：否决。长期运行下 sources 线性增长而价值递减。

## Consequences

- 经验记忆从此真正自学习：裁判重复反馈会自动进入生成提示的"经验教训"区
  （跨任务、跨 run、跨 bench 格累积）。
- 版本升级路径：格式演进 bump MEMORY_FORMAT_VERSION；读取侧对旧版本要么
  迁移要么明确跳过并告警。
- 回归测试锁定：threshold=2 时第二次 add 提升、threshold=1 时首次即提升。
  现状更新（8-16 晚）：`recent_feedback` 接口因无生产调用方被移除
  （死代码清理），提升语义不变。
