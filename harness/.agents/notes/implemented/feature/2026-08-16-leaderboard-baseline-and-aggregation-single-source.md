# Agent Note: leaderboard 基线与"聚合唯一正源"

Status: implemented

## Problem

两个缺口同时暴露：①"公开 leaderboard"是路线图终点，但没有基线机制——指标
只能靠人工从 report 里抄，无法追踪回归；②评测聚合逻辑开始重复——report.collect
汇总全局均分，compare_chains 又自己重写一遍 stage 级聚合（两个口径会漂移）。
对齐 DSH 的"事件投影消费"（投影是唯一读取面，消费方不再各自解析原始文件）。

## Decision

1. **collect() 是聚合的唯一正源**：新增 `stage_scores`（分 stage 维度均分）、
   `stage_passed`（分 stage 通过数）、`models`（生成器 adapter:model）、
   `chain_mode`（manifest 记录）四个字段；全局 scores/passed_rate 保持兼容。
   compare_chains 重构为消费 collect()，删除自带的 eval 聚合代码；
   衔接模式正源链：manifest.chain_mode（新口径）→ config.yaml 快照 → "?"。
2. **leaderboard 基线导出**（`vh leaderboard <task>`）：从 collect() 取数，
   产出 `<publish>/<task>.json`（机器可读基线，可入库 git 追踪、回归可 diff）
   与 `<task>.md`（人读表格）；与上次基线对比给出增量 diff
   （new_runs / removed_runs）。默认 publish 目录 `leaderboards/`
   在 harness 仓库内、随代码入库——首次基线已提交。
3. **seam 一致性元测试**：每个内置提供者必须具备其 seam 的协议成员
   （name、核心方法、能力声明 dict；generator 还必须声明 backend 成本口径）。
   协议成员缺失 = 加载前拦截（比运行到调用点才炸更早）。
4. **`vh doctor --all`**：全量体检一个 experiments 目录下所有 run。

## Alternatives considered

- **leaderboard 由 CI/网站生成**：否决。本仓库规模先本地基线 + git 追踪即可；
  网站是发布形态，不是数据正源。
- **compare_chains 保留独立聚合**：否决。两个聚合口径必然漂移（本次重构就
  发现旧脚本按记录加权均分、collect 按维度均分的语义差异）；单一正源之后，
  语义演进只改一处。
- **基线放 experiments/ 内**：否决。experiments/ 被 gitignore（批量产物），
  基线需要入库追踪才有回归价值；独立 leaderboards/ 目录是可评审的小文件。

## Consequences

- 新增聚合维度（如新的评测 stage）只需改 collect()，所有消费方自动获得；
  消费方禁止再直接 glob eval/*.json 聚合。
- leaderboards/*.json 每次跑完实验后手动 `vh leaderboard <task>` 更新并提交；
  后续轮次可加 git hook 提醒。
- 元测试锁定内置提供者契约；新提供者必须过 seam 一致性检查（含 backend 声明）。
