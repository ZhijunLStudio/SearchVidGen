# Agent Note: 语义聚类合并经验记忆（vh memory-consolidate，E33）

Status: implemented

## Problem

E32 记录：~20 条"叙事缺乏起承转合"语义近重复因文本不同而永不提升，
语义聚类标记为未来项。本轮兑现——且中间发现两个必须绕开的坑：
①裁判提供者的系统纪律强制 feedback="pass"，无法做自由变换任务；
②第一版 consolidate 会把无标签条目丢弃（数据丢失）。

## Decision

1. **用 script 提供者做归纳**（而非裁判）：归纳是自由变换任务，不是评分；
   script.deepseek-v4-flash 输出 `{"label": "<短语>"}` 经 parse_script_json
   自动解析。裁判的评分纪律（feedback=pass）与变换任务冲突——
   能力选择按任务形状，不是按名字。
2. `ExperienceMemory.consolidate(canonicalize)`（纯逻辑、无 LLM 依赖）：
   未提升条目按规范短语归并（count 累加/sources 截断）→ 达标提升；
   **无标签条目原样保留**（第一版丢弃 bug 已修 + 回归测试）。
3. `vh memory-consolidate`（纯 API）：真实运行 23 条 → 4 组归并、
   **1 条自动提升（"叙事缺乏起承转合"）**、16 条无标签保留——
   E14 的提升机制首次在真实数据上由语义聚类驱动产出经验。

## Alternatives considered

- **裁判 + 强提示硬掰**：否决。实测两次尝试（含重试与"禁止写 pass"）
  仍返回 pass——系统提示纪律压倒用户问题。
- **嵌入向量聚类**：否决。无嵌入基础设施；LLM 归纳用现有 seam、
  零新依赖，且结果可读可审计。

## Consequences

- 自学习闭环完整：反馈 → 清洗（E32）→ 语义聚类（E33）→ 提升 →
  注入提示（E28）——每一环都有真实数据证据。
- 归纳质量依赖 script 提供者：温度 0、JSON 模式；无标签条目保留待
  下次合并（幂等重跑安全）。
