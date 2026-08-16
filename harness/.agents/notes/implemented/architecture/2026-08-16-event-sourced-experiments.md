# Agent Note: 实验事件溯源（events.jsonl 权威，manifest 是投影）

Status: implemented

## Problem

实验状态只有 `manifest.json` 一个全量重写文件：进程在任何 `_flush()` 之间崩溃，
已完成的产物/评测/成本记录就会丢失或与文件系统不一致；没有任何追加式日志可供
恢复与审计。对齐 deepseek-harness 的 event-sourced sessions 原则（
"模型可见 ⟺ 日志"，会话日志是权威、可从日志重建一切），需要给实验引入
追加式事件流。

## Decision

`experiments/<task>/<run_id>/events.jsonl` 是**权威记录**（append-only，每条事件
一行 JSON，先落事件再更新投影）；`manifest.json` 降级为**投影**（保持原格式，
供 report/compare 等脚本快速读取）。

事件类型（v=1）：`run.created` / `query.bound` / `config.snapshotted`（含 sha256）/ `artifact.saved` / `eval.saved` / `retry` / `manifest.set` / `finalized`；
另有时序事件 `stage.started` / `stage.finished`（阶段生命周期笔记，不参与投影）。

- **崩溃恢复**：打开实验时若事件流完整（首个事件是 run.created），用重放结果
  重建 manifest——投影丢失/损坏不影响恢复；
- **权威性判定**：事件流不完整（2026-08-16 前的旧 run）时 manifest 保持权威，
  新 run 从 run.created 起事件流即权威；`events_complete` 标记此状态；
- **可重建性闭环**：裁判原始输出从 eval/ 迁至 artifacts/judge/（经事件流归档，
  每次裁判调用可重建；eval/ 目录只保留评测明细记录列表）；
- 重试计数、director 元信息（chain_mode/generator_capabilities）一律经
  `record_retry()` / `set_meta()` 走事件流。

## Alternatives considered

- **SQLite 单文件库**：否决。JSONL 追加可读可 grep、零依赖、坏行可跳过；
  本规模（每 run 数十到数百事件）不需要事务与索引。
- **只加事件、manifest 仍为权威**：否决。两个权威必然漂移；重放一致性检查
  （运行时不变量笔记）需要明确谁是正源。
- **事件含完整 config/产物内容（内联）**：否决。config.yaml 与产物文件本身
  已落盘，事件只记哈希与路径；内联会使事件流膨胀且引入双写。

## Consequences

- 新增实验状态变更必须经 `_emit()` 事件，否则重放会丢状态（约定写入 seams
  注释与 RUNBOOK）。
- 旧 run 没有事件流：doctor 提示"无事件流"，manifest 仍可读（report 兼容）。
- 事件格式变更需 bump EVENT_VERSION（对齐 DSH 的 session log version 机制）。
