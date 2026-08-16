# Agent Note: 优化器段数来自任务配置而非 manifest 幽灵字段

Status: implemented

## Problem

`ScriptOptimizer` 生成剧本模板时读 `exp.manifest.get("segments", 4)`——
但 manifest 从未写入过 `segments` 字段。结果：无论任务 YAML 配置 `segments: N`
是多少，自主优化路径永远按默认 4 段生成（2026-08-16 修复，Bug#2）。
`--segments` CLI 覆盖同样被吞。

## Decision

`ScriptOptimizer` 构造函数显式接收 `segments` 参数，由 SegmentDirector 从
任务配置传入（`self.cfg.get("segments", 4)`）；模板组装只消费实例字段。
修正后任务配置 → 优化器 → LLM 模板的链条完整可重建。

## Alternatives considered

- **在 manifest 里补写 segments 字段**：否决。manifest 是运行产物（记录），
  配置的正源是任务 YAML + CLI 覆盖；把配置塞进产物再回读会制造两个正源，
  续跑时二者还可能不一致（配置快照笔记的原则）。
- **优化器直接读 cfg**：否决。优化器是独立消费者，不应持有任务配置的引用；
  显式参数保持其可单测（回归测试正是用参数注入验证的）。

## Consequences

- 回归测试锁定：optimize 发出的 template["segments"] 必须等于构造参数。
- 同类 bug 的通用修复原则：**配置正源是任务 YAML + CLI 覆盖；
  manifest 只是运行记录，任何"从 manifest 读回配置"的模式都视为异味**。
