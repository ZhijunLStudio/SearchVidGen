# Agent Note: 评分解析失败的可操作反馈（E21 暴露）

Status: implemented

## Problem

E21 的 script_judge 连续两次空评分：裁判输出不可解析时 feedback 为空（或
只是无法直接使用的散文），重试循环没有有效信号注入，第二次尝试同样失败——
评测闭环的"失败反馈重试"在解析层断链。

## Decision

两个裁判提供者（vllm_judge / judge.deepseek-text）在 `parse_scores` 得到
空评分时，feedback 统一替换为 `unparseable_feedback(raw)`：

- 明确的结构化指令（"严格只输出 JSON：{维度: 分数, feedback: ...}"）；
- 原文前 200 字作为上下文（模型能看到自己错在哪）。

反馈经 RetryPolicy 注入下一次生成/评测，第二次尝试有真实信号可依。
scores 非空时行为完全不变（正常路径零影响）。

## Alternatives considered

- **解析失败直接判 0 分不再重试**：否决。偶发格式错误不该吞噬重试机会；
  现有缺失维度判未通过的语义保持不变，只是反馈从空变可用。
- **在 finalize_verdict 里兜底**：否决。它拿不到原文；反馈生成需要 raw，
  只能放在提供者解析点（raw 的所有者）。

## Consequences

- 回归测试锁定：garbage 输出 → feedback 含"评分解析失败"与指令；
  正常 JSON 输出 → feedback 不受影响。
- 该反馈会进入经验记忆（script_judge 的反馈入 memory 的既有逻辑），
  解析失败教训也能被记忆并注入未来提示。
