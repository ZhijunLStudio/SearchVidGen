# Agent Note: 剧本优化闭环量化工具与裁判口径差异发现

Status: implemented

## Problem

剧本自主优化（ScriptOptimizer）的"多轮进化"一直只有机制没有量化：E23 证明
反馈重试单次生效，但 optimize 整体开/关到底值多少分、多少成本，没有任何
数字。同时 scripts/ 缺少纯 API 的证据工具（不占 GPU 即可验证剧本层）。

## Decision

`scripts/compare_script_optimize.py`：控制变量对比 optimize 开/关——
同一 query、同一裁判（judge.deepseek-text）、同一评测维度，各 N=3 试；
off = 单次生成，on = 2轮×2候选；成本从 exp.manifest.total_cost_usd
统一结算（剧本+裁判调用都入产物计费）。

实测（E24，query=雨夜小猫）：
- 均分 2.65 → **5.46**（+2.81）；通过率 1/3 → 2/3
- 成本 off $0.0003/试 → on $0.0009/试（约 3×，2轮×2候选 ≈ 4 次生成）
- 但 5.46 仍低于 6.0 阈值、距 target 8.0 远——**当前优化预算不足**，
  候选数/轮次/温度策略是下一步调参对象

## Alternatives considered

- **用 vh bench 跑（含 GPU 段生成）**：否决。剧本层对比不需要生成视频；
  纯 API 工具把反馈周期从 30 分钟压到 1 分钟。
- **只测单次优化（非重复试验）**：否决。LLM 方差大，N=3 才有粗粒度
  可信度；试验数做成参数。

## Consequences

- **裁判口径差异（E24 警报，E25 部分修正）**：E24 观察到的 vLLM ≈9-10 vs
  deepseek-text 2.65-5.46 差异，经校准机制定位为**大半是解析失败伪影**
  （缺失维度计 0 + 非 JSON 输出），而非纯尺度差。JSON 模式（
  response_format=json_object）把 deepseek-text 解析成功率从 40% 提到
  80%；校准实测（n=4/维）维度级偏移：叙事完整 +0.5 / 旁白自然 +0.5 /
  **可生成性 -1.5**（deepseek-text 更宽松）。混用裁判的 leaderboard
  数据仍需标注裁判来源（calibration/ 提供偏移参考），但差异量级已从
  "不可比" 收窄到"可标注修正"。
- 优化闭环的 ROI 有了基线：每 +1 分剧本的成本 ≈ $0.0003（当前预算下），
  未来调参可直接对比此基线。
