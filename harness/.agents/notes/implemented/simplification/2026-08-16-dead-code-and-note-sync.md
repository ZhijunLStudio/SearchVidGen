# Agent Note: 死代码清理与笔记一致性审计（维护轮）

Status: implemented

## Problem

两个维护缺口在发布后显现：①`parse_judge_output`（评测结算笔记里"为兼容
保留"的封装）与 `recent_feedback`（记忆笔记里"供局部重试"的接口）都已
没有生产调用方——保留它们违反 simplification 纪律（不增加能力的前提下
移除代码）；②DSH 规则要求 implemented 笔记与现实同步，但事件溯源笔记
缺 stage.* 事件、结算/记忆笔记仍描述已删接口。

## Decision

1. **移除死代码**：`parse_judge_output`（调用方只剩测试，测试改用
   parse_scores+finalize_verdict 两段式）；`recent_feedback`（无生产
   调用方）。两个接口的决策笔记以"现状更新"段落同步，而非改写历史。
2. **笔记一致性审计**：事件溯源笔记补 stage.started/finished；抽查
   笔记引用的文件路径与脚本清单（RUNBOOK 四个 scripts 全部存在）；
   引用路径多为简写（core/xxx.py → vidharness/core/xxx.py），无真实漂移。

## Alternatives considered

- **保留"可能有外部脚本用"的兼容封装**：否决。单仓库无外部消费者；
  死了就是死了，保留是维护税（AGENTS 级原则：删除而非注释）。
- **recent_feedback 接线进重试上下文**：否决。判裁反馈注入（E22）已
  覆盖重试信号；跨 run 上下文是经验记忆的 experience_lines 职责——
  两个机制重叠时留更简单者。

## Consequences

- 公开接口面缩小：解析两段式（parse_scores/finalize_verdict）是唯一
  正解路径；记忆接口 = add/add_experience/experience_lines。
- 后续每轮加入"死代码 grep + 笔记抽查"维护项（v0.2.1 起）。
