# Agent Note: bench 格级断点续跑（格身份三要素）

Status: implemented

## Problem

bench 逐格串行执行，第 N 格崩溃则前 N-1 格的 GPU 时间全部沉没（重跑整个矩阵
= 每格重新生成）。单 run 有断点续跑，矩阵没有——长矩阵（如 4×3=12 格）的
实用性被崩溃成本卡死。

## Decision

`bench_cell_status(base_dir, task_name, label, cfg, query)` 判定每格的续跑状态：

- **格身份三要素**：bench_cell 标签 + config.yaml 快照 + query——三者同才算
  同一格；换 query 是不同实验，**不得跳过/续跑旧格**（与单 run 的
  bind_query 守卫同口径）。
- 已完成格 → 跳过（打印 run_id）；未完成格 → `run_task(resume=...)` 续跑
  （快照守卫保证配置一致）；无匹配 → 全新跑。
- 复用范围仍是 bench 进程内的 adapters_cache（跨进程无法复用已加载模型）。

真实冒烟：E19 的 bench_ratio 矩阵重跑，两格均在秒级被识别并跳过，
零 GPU 消耗。

## Alternatives considered

- **格状态写进 spec 文件**：否决。状态是运行产物，spec 是实验设计；
  状态从 run 目录推导（manifest+config.yaml）保持单一正源。
- **只按标签匹配**：否决。标签由矩阵取值拼接，不同矩阵可能同标签；
  且 query 变化必须视为新实验（query 守卫的存在理由）。
- **每格独立子进程 + 全局锁**：否决。串行执行已经简单可靠；
  并行化留给多机/多卡未来轮次。

## Consequences

- 长矩阵可以"跑一段、崩溃、接着跑"；bench 的实用半径从短矩阵扩展到
  任意长度。
- 格身份语义进入文档：修改矩阵取值、query 或 base 配置都会形成新格
  （这正是对比实验想要的隔离）。
