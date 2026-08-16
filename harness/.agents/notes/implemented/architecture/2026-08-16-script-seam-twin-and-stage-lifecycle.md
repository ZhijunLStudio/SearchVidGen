# Agent Note: script 缝孪生化、提示契约上移 SD、阶段生命周期事件

Status: implemented

## Problem

三个独立缺口：①script 缝只有一个提供者（deepseek-v4-flash 绑定官方 API），
孪生适配器原则要求第二实现以暴露协议缺口；②剧本提示与 JSON 解析逻辑埋在
deepseek_script 提供者内——"面向模型的语言"本是 seam 的约定，却随提供者私有化，
换提供者就要复制一份；③事件流只有产物/评测/重试事件，没有阶段边界，
无法回答"这个 run 进行到哪一步、各阶段耗时多少"。

## Decision

1. **script 缝第二实现 `script.openai-compat`**：任意 OpenAI 兼容端点
   （本地 vLLM、第三方 API），base_url/model/api_key 显式声明。
   计费口径：未配置单价时 `billing="unpriced"`（cost=0，不编造单价），
   配置后按 token 计费——对齐"提供者声明成本"原则。
2. **提示/解析契约上移 Service Definition**（seams/script.py）：
   `build_script_prompt(query, template)`（协议骨架+目标+经验注入）与
   `parse_script_json(content)` 归 seam 所有；两个提供者共用，
   deepseek_script 重构后行为不变（回归测试锁定）。
3. **阶段生命周期事件**：director.run 的每个阶段发
   `stage.started`/`stage.finished`（finally 保证结束事件）；不变量新增
   **配对检查**（已 finalize 的 run，每个 started 必须有 finished）；
   单 run 详情页新增阶段时间线。

## Alternatives considered

- **提示契约留在各提供者各自实现**：否决。这正是孪生适配器要防的漂移——
  契约属于 SD；两个实现若各自维护提示，评测口径就无法跨提供者比较。
- **计费未知时按 DeepSeek 价估**：否决。编造单价污染成本对比；
  unpriced 是诚实的"无口径"，bench 预估与 finalize 都明确跳过。
- **阶段事件写入 manifest（而非仅事件流）**：否决。manifest 是投影；
  时间线是时序数据，事件流才是正源（详情页从事件流重建时间线）。

## Consequences

- 新 script 提供者 = 实现 generate + 声明 param_schema，提示/解析零复制。
- story_smoke.yaml 落地为最小真实冒烟任务（2 段×20 步），新基建的
  首次真实 GPU 端到端验证跑在它上面。
- 不变量配对检查使"中断的 run 被误认为完整"不可能发生。
