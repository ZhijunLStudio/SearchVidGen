# Agent Note: 变体回归套件（vh regress）

Status: implemented

## Problem

四个 check 任务（story_smoke/canvas/fl2va_check/ref2va_check）是拆分重构
与适配器改动的回归防线（E18-E27 的载体），但"防线"本身没有统一视图：
查各任务最新 run 要手工翻目录；任务文件改了之后旧 run 是否还代表当前
代码，完全没有信号——**配置漂移会让"已回归过"变成假象**。

## Decision

`tasks/regression.yaml` 套件清单 + `vh regress`：

- **状态表**：每任务一行 = 最新完成 run + 关键评分（段级/跨段维度均分）
  + **配置漂移检测**（run 的 config.yaml 快照 vs 当前任务文件；
  漂移 = "需要重跑"的硬信号）。有漂移或未跑过 → exit 2。
- **执行模式**（--run）：逐任务跑，已完成跳过、未完成/漂移重跑
  （与 bench 格级断点续跑同语义）；ref2va 需 h3int8 环境、其余需
  torch 环境，分两轮跑（清单注释写明）。
- 任务路径双惯例兼容：先按写入值解析，不存在则相对 spec 文件解析。

## Alternatives considered

- **pytest 集成 GPU 回归**：否决。GPU 回归是分钟级-小时级的重资产运行，
  不适合进单测循环；状态表 + 显式 --run 是人审+资产分离的正确形状。
- **自动检测漂移即自动重跑**：否决。重跑成本高（GPU），自动触发会烧卡；
  漂移报警 + 人工/CI 决策是刻意的门。

## Consequences

- "变体回归"从口头约定变成可查可跑的清单：`vh regress` 一行看到全部
  变体的最新证据状态；新变体/新适配器加回归任务 = 清单加一行。
- 套件清单与 check 任务文件一起构成"资产回归基线"，leaderboard 基线的
  对应物。
