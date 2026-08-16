# Agent Note: 评测结算归属消费者（weight/min_score 是任务配置的拥有物）

Status: implemented

## Problem

Judge seam 原协议把 `weight`/`min_score`/`aliases` 留在消费者侧（YAML），却只把
`{name: question}` 传给提供者；提供者解析时用默认权重重建维度，导致任务 YAML 里
声明的权重（如"旁白自然 weight 1.2"）与阈值（如"音效场景相符 min_score 5"）被静默
替换成默认值（weight=1.0 / min_score=6.0）——加权总分与通过判定都与配置不符
（2026-08-16 修复，Bug#1）。

## Decision

**评测结算（加权/阈值判定）是评测策略，归属任务配置，由消费者统一执行；提供者
只负责模型 I/O 与原始评分解析。** 新协议：

- `judge.judge(media, criteria, workdir)` 的 `criteria` 接收 `criteria_to_spec()`
  产出的完整规格 dict（name → {question, weight, min_score, aliases}），
  兼容旧协议裸字符串问题；
- 提供者 payload 只含原始数据：`{"scores": {维度名: 分数}, "feedback": str}`；
- 消费者经 `run_judge()` 统一结算：`finalize_verdict(scores, feedback, criteria)`
  加权、按 min_score 判定、缺失维度计 0 且判未通过（解析失败不得静默通过）；
- 现状更新（8-16 晚）：`parse_judge_output` 兼容封装已随死代码清理移除——
  解析只保留 parse_scores + finalize_verdict 两段。

对齐 DeepSeek Harness 的"显式 > 隐式"：缺省与策略应用是拥有方的显式
`resolve(request): Spec` 步骤（这里是 `run_judge`），不是提供者 `run()` 里的
隐藏 `?? default`。

## Alternatives considered

- **提供者侧保留结算，把完整规格传进去**：否决。结算逻辑会随每个提供者复制，
  且评测策略（重试阈值、权重）与模型 I/O 是两种变化速率——改阈值不应触碰
  模型适配器。
- **judge 只返回文本，解析全在消费者**：否决。解析依赖模型输出格式
  （别名兜底、JSON 位置），与提供者绑定；但解析结果（原始分）与结算分离已足够。
- **缺失维度按均值补**：否决。宁严勿松：缺失维度 = 该维度没评上，判未通过
  触发反馈重试，比补假数据更接近真实质量。

## Consequences

- `seams/judge.py` 新增 `criteria_to_spec` / `spec_to_criteria`；所有消费者
  （judge_loop / stage_script / script_optimizer / cross_consistency）经
  `run_judge` 结算，策略只在一处生效。
- 外部脚本（collect_evidence）的裸字符串 criteria 仍兼容。
- 回归测试锁定：weight=1.2/min_score=5.5 的组合必须产生加权分与正确判定。
