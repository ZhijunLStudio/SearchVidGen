# Agent Note: 单 run 详情页（vh report --run）

Status: implemented

## Problem

对比报告（report.html）只有聚合行，无法回答"这个 run 里到底发生了什么"：
评测明细、产物清单、事件流、配置快照散落在 run 目录的多个文件里，查看只能
靠手工翻文件。对齐 DSH 的"投影消费"思路，需要每个 run 一个自包含的详情页。

## Decision

`render_run_html(run_dir, out)` 生成单 run 详情页（写入 run 目录内
report.html），`vh report <task> --run <run_id>` 触发。页面五段：

1. **概览**：run_id/query/bench_cell/chain_mode/能力声明/时间/成本/重试；
2. **配置快照**：config.yaml 全文（旧 run 显示"无快照"提示）；
3. **产物表**：每个 stage 产物的适配器/模型/耗时/成本/seed；
4. **评测明细**：eval/*.json 的完整记录（可读 JSON）；
5. **事件流**：末尾 20 条事件摘要 + 总数。

页面只读实验目录（manifest/eval/events/config），不引入新数据源——
详情页是投影的又一种消费形态。

## Alternatives considered

- **在聚合报告里内嵌详情**：否决。聚合表会爆炸；详情按需生成更符合
  "报告 = 投影消费"的分层。
- **把详情页做成 report.html 的链接目标**：已做（聚合表"产物目录"列指向
  run 目录），详情页文件名固定为 report.html，浏览器打开 run 目录即得。
- **引入前端框架**：否决。纯静态 HTML 零依赖，与现有报告一致。

## Consequences

- 人工复盘一个 run 的完整证据链 = 打开一个文件；judge 原始输出仍在
  artifacts/judge/（详情页不重复渲染大 JSON，只列产物表）。
- 详情页依赖 events.jsonl/eval/*.json 的可读性——这两者的格式演变
  （见事件溯源笔记）需保持向后兼容。
