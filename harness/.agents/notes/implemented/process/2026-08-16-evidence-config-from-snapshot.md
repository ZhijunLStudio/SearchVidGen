# Agent Note: 证据脚本的配置正源是实验快照（禁止脚本内硬编码端点）

Status: implemented

## Problem

`scripts/collect_evidence.py` 硬编码裁判端点与模型名（
`base_url="http://127.0.0.1:8030/v1", model="judge-qwen3.5-27b"`）。这违反两条
已确立的原则：配置正源是任务配置+实验快照（config-validation 笔记）、
"无硬编码 tunable"（AGENTS 级约定）。后果是真实的：judge 服务换端口/换模型后，
证据收集静默用旧端点失败或评出不可比口径，而 run 快照里明明存着正确配置。

## Decision

`collect_evidence.load_judge_from_run(run_dir)` 从 run 的 `config.yaml` 快照
读取 judge 适配器与参数并经 `instantiate()` 校验实例化——证据与运行同口径。
无快照（2026-08-16 前的旧 run）或快照缺 judge 配置：响亮失败并给出指引
（重跑任务生成快照），而不是回退到任何"默认"端点。

## Alternatives considered

- **保留硬编码作为 fallback**：否决。fallback 就是第二个正源，端口迁移后
  会静默用旧端点；"证据与运行同口径"要求 fail loud。
- **从 manifest 读 judge 参数**：否决。manifest 不记录 judge 配置（它是
  投影不是配置正源）；快照才是。
- **把证据收集并入 harness（vh evidence 命令）**：否决。脚本当前足够；
  若证据收集成为高频流程再考虑升格为命令（不预防性建设）。

## Consequences

- 所有 scripts/ 下的工具必须以 run 快照为配置正源；新增脚本评审时检查
  硬编码端点（与"能力键必须登记 schema"同类的显式协议）。
- 旧 run 必须先重跑（或手工补 config.yaml）才能收集证据——这是为同口径
  付的代价。
