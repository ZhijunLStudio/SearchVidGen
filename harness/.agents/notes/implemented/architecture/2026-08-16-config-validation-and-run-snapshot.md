# Agent Note: 任务配置 fail-loud 校验与实验配置快照

Status: implemented

## Problem

任务 YAML 是 harness 的配置平面，但 2026-08-16 前没有校验：拼错的键
（`segmant`、`genrator`）、写错的衔接策略、畸形的评测维度都会被
`cfg.get(...)` 静默吞成默认值——配置错误既不报错也不可见。
同时实验目录不冻结任务配置：`scripts/compare_chains.py` 无法得知每个 run 用的
chain_mode，只能硬编码 run_id 映射来猜（Bug#4）；断点续跑还可能把旧产物与新配置
混跑，破坏"实验 = 可复现证据"。

## Decision

对齐 DeepSeek Harness 的配置平面纪律（misconfiguration fails loud；
配置归属分层），缩放落地：

1. **`core/config.py` 的任务 schema 校验**：未知顶层/pipeline 键、未知
   chain_mode、缺 adapter 的组件块、缺 name/question 的评测维度、类型错误——
   全部在启动时抛 `ConfigError`（带路径定位），拒绝启动而非静默默认。
   归属边界：harness 拥有任务结构；params 由 `registry.instantiate` 按构造
   签名校验（未知参数/缺必需参数 fail loud）；`--query/--brief/--segments`
   是运行时用户覆盖。
2. **实验配置快照**：`Experiment.snapshot_config(cfg)` 在 run 开始时把有效配置
   冻结为 `<run>/config.yaml`，manifest 记录 `config_file`。快照是续跑守卫：
   配置不一致 = 两个不同的实验，拒绝混跑（报错提示开新实验）。
   compare_chains.py 由此改为从快照读 chain_mode 自动聚合，删除硬编码映射。
3. **可重建性补充**（"模型可见 ⟺ 日志"的最小落地）：script 提供者把完整
   template 记入 meta.params；生成器把完整 prompt 记入 meta.params；
   judge 把 criteria 规格与媒体清单落盘。

## Alternatives considered

- **JSON Schema 库做校验**：否决。单层 YAML 规模用 ~150 行显式校验器即可，
  引入依赖换来的嵌套 schema 表达力暂不需要；ConfigError 的定位信息更好定制。
- **只警告不拒绝**：否决。静默吞配置错误的模式正是本笔记要消灭的；
  警告在批量实验里必被忽略。
- **快照存 manifest 内嵌（而非独立 config.yaml）**：否决。manifest 是运行产物
  元信息，配置是输入；独立文件让 compare 脚本与人类都能直接 diff 两个实验。
- **续跑时允许部分配置覆盖**：否决。续跑的语义是"同一实验接着跑"；
  要改配置就开新实验——这是对比实验有效性的前提。

## Consequences

- 新增任务 YAML 键必须在 `config.py` 的 schema 里登记（配置协议演进走显式清单）。
- 2026-08-16 前的旧 run 没有 config.yaml，compare_chains 归入 "?" 并提示。
- 续跑必须用与首次运行完全相同的有效配置（含 --brief/--segments 覆盖值）；
  query 作为实验变量由 `Experiment.bind_query()` 记录并守卫（剧本按 query 生成，
  续跑换 query 会静默复用旧剧本）。
