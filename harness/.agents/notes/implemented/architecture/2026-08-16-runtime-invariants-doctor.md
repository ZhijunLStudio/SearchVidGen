# Agent Note: 运行时不变量（check_experiment + vh doctor）

Status: implemented

## Problem

实验证据的价值取决于完整性，但 2026-08-16 前没有任何机制断言证据之间的关系：
manifest 总额被篡改/写坏、产物文件被误删、config.yaml 被改动、评测文件损坏、
事件流与投影漂移——全部静默存在，直到有人偶然发现数字对不上。

对齐 deepseek-harness 的 package-owned invariant 原则（只断言**多个观测之间的关系**，
提交错误状态前验证，违规抛出），为实验目录建立不变量检查。

## Decision

`core/invariants.py::check_experiment(root)` 返回违规列表（空 = 通过），断言
manifest ↔ 文件系统 ↔ 事件流之间的关系：

1. manifest 总额 == 各产物 meta 累计（成本/耗时）；
2. manifest 记录的每个产物文件真实存在；
3. eval/*.json 可解析且每条记录是 dict（judge_* 旧版布局给迁移提示）；
4. config.yaml 存在且 sha256 与 config.snapshotted 事件一致（防篡改）；
5. retries 计数非负整数；
6. 事件流完整时：重放投影与 manifest 一致（产物条目/路径/总额/重试/query）；
7. 事件流完整时：重放评测明细与 eval/*.json 集合一致。

挂钩点：`Experiment.finalize()` 收尾时执行，违规即抛 RuntimeError（证据不完整
的 run 不允许静默收官）；CLI `vh doctor <run_dir>` 供事后体检（旧 run 也可查）。

配套修复（本轮发现的同类静默问题）：director 的 ffmpeg/ffprobe 调用此前在工具
缺失时裸崩 FileNotFoundError、assemble 的时长探测失败静默回退 5.0s——
统一改为 `consumers/tools.py::require_tool` 响亮失败。

## Alternatives considered

- **中央 invariant 服务**：否决。DSH 的 package-owned 是分布式版本；本仓库规模
  下一个 `core/invariants.py` + finalize 挂钩即可，无需服务抽象。
- **违规仅告警不阻断**：否决。证据完整性受损的 run 混进对比报告会污染基准；
  响亮失败是 fail-loud 纪律的延伸。
- **用 JSON Schema 校验 manifest**：否决。Schema 校验形状，不校验**关系**
  （总额 vs 累计、投影 vs 事件流）；本决策断言的是关系。

## Consequences

- 新增实验状态字段/事件类型时，必须同步更新不变量（否则关系检查会漏掉新状态）。
- 旧 run（无事件流）只受不变量 1-5 约束；事件流检查自动跳过。
- finalize 从"永远成功"变为"可能因证据违规抛错"——这是有意的契约变化。
