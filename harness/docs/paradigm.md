# 范式对照：deepseek-harness → VidHarness

本文件是 VidHarness 与 [deepseek-harness](
/data/lizhijun/work/Harness/deepseek-harness) 范式对齐的**可审查映射**：
每条 DSH 原则 → 本仓库落地（文件/机制）→ 验证证据（E 系列实验 / 测试）。
决策的"为什么"在 `.agents/notes/`（Agent Notes，41 篇）；证据在
`README.md` 的 E1-E41；本表只做对照。

## 核心范式

| DSH 原则（出处） | VidHarness 落地 | 验证证据 |
|---|---|---|
| 一切皆插件（Cordis 插件化） | 适配器加载即注册：seams/providers/consumers 三层角色（Python 缩放版） | E1-E10 三模型换代零核心改动；测试 TestRegistry |
| 能力缝 = SD+Provider+Consumer 三角色，完整不残缺 | seams/（Protocol+数据结构）/ providers/（@register）/ consumers/（编排闭环） | 角色映射笔记；seam 一致性元测试 |
| 孪生适配器：两个真实实现暴露词汇缺口 | generator×2（H3 本地/API）+ judge×2（vLLM/DeepSeek 文本）+ script×2（官方/通用兼容） | E15 暴露 modalities 缺口→模态守卫；E20 暴露拆分缺口 |
| 声明式提供者目录（参数/能力/单价） | param_schema（类型/choices/必需）+ capabilities schema + cost_rates_usd_per_s | 元测试锁定"声明==签名"；vh adapters --verbose |
| 按请求/能力路由 | resolve_provider + generator.route + judge.stages 阶段路由 | 测试 TestRegistry/TestPerStageJudgeRouting |
| 配置平面归属（harness/适配器/用户） | core/config.py 任务 schema；params 归适配器；--query/--brief/--label 归用户 | E11（Bug#1-4）；配置校验测试 |
| fail loud：配置错误最早点响亮失败 | schema 校验/instantiate 参数校验/能力校验/GPU 显存预检/fl2va keyframe 守卫 | E11/E18/E20/E29；148 测试中 ~30 个失败路径 |
| 显式 > 隐式：缺省是拥有方显式步骤 | run_judge 统一结算（权重/阈值归消费者）；cost 口径声明化；ratio 归 context | Bug#1 修复（E11）；E18 |
| 事件溯源：模型可见⟺日志 | events.jsonl 权威 + manifest 投影 + 配置快照 + 裁判产物归档 | E16 全链路；E29 SIGKILL 实证；重放测试 |
| package-owned 运行时不变量 | core/invariants.py + finalize 挂钩 + vh doctor | E12 旧布局现形；E29 配对检查；7 类违规测试 |
| Agent Notes 决策记忆 | .agents/notes/{proposed,implemented}/（41 篇）+ README 格式规范 | 本仓库的 47 轮演进全靠它沉淀 |
| 实验即证据（evidence-driven） | experiments/（events+manifest+eval）+ E1-E41 + leaderboard 基线 | E8/E10 衔接结论；E21 修复路径复现 |
| 基准矩阵（一次只变一个变量） | vh bench + 规划期全格校验 + 成本预估 + 格级断点续跑 | E19 首次真实矩阵；E22 格跳过冒烟 |
| 会话持久化/断点续跑 | 产物缓存 + 配置快照守卫 + query 守卫 + 进程级恢复 | E28 幂等；E29 kill -9 恢复 |
| 跨模型对比与成本口径 | report 聚合唯一正源 + leaderboard + 校准 + 回归套件 | E24-E26 量化审计；vh regress 4 任务 ✅ |

## 尚未移植（有意的缩放取舍）

| DSH 有 | VidHarness 现状 | 理由（Agent Note） |
|---|---|---|
| Cordis 服务注入/事件总线 | 显式函数调用 + 注册表 | Python 单进程规模；seam 角色映射笔记 |
| 多包独立发布 | 单仓库目录分层 | 拆包样板成本 > 收益（能力缝笔记） |
| 子进程 seam / 沙箱 | 直接调用（ffmpeg 预检） | 单机实验环境；E29 的预检替代 |
| Web 配置面/权限 | CLI + YAML | 单用户实验场景；配置平面笔记 |
| 多机并行 workflow | bench 串行 + 跨格缓存复用 | GPU 预算现实（双卡任务占满 2 卡）；bench 笔记 |

## 演进节奏

47 轮（2026-08-14 → 08-16）每轮 = 差距分析 → 修复/增强 → 测试 → 真实验证
→ Agent Note → 提交。六类证据环环相扣：范式（本表）→ 决策（notes）→
证据（E 系列）→ 代码（seams/core/consumers）→ 回归（vh regress）→
基线（leaderboards/）。
