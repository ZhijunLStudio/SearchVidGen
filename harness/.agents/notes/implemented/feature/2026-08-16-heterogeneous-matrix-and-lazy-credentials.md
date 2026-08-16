# Agent Note: 异构矩阵轴（多键配对）与凭据延迟解析

Status: implemented

## Problem

H3 API 对比实验是路线图的最后一块大实验，但卡在外部凭据（MINIMAX_API_KEY
20+ 轮未配置）。审计发现 harness 侧其实还有两块没到位：①bench 矩阵只支持
单键轴（{路径: [取值]}），无法表达"adapter 与 params 成对切换"的异构格
（本地格需要 model_path/gpu，API 格需要 resolution/duration——两者
参数集不同，单键轴必然让某一格参数非法）；②适配器在**构造期**就解析
凭据（_load_minimax_key/_load_token），导致 dry-run 规划也依赖 key。

## Decision

1. **多键轴（异构格）**：expand_matrix 支持 `{路径1: [...], 路径2: [...]}`
   多键轴——各键取值列表等长、按位配对覆写（adapter 与 params 成对切换）；
   格标签只取非 dict 值（adapter 名）拼接。单键轴语义不变。
2. **凭据延迟解析**：MiniMaxH3API 与 deepseek_script 的凭据从构造期移到
   首次生成调用前（fail loud 仍保持：无凭据时第一次 I/O 报错）。
   规划期（bench dry-run / instantiate 校验）不再依赖凭据。
3. `tasks/bench_api_local.yaml`：本地/API 双格对比 spec（多键轴），
   dry-run 无 key 实测通过（本地 $0.24 / API $0.67 预估）——key 到位后
   一条命令即跑（RUNBOOK 指引）。

## Alternatives considered

- **两个独立 bench spec 分跑**：否决。对比实验必须同 spec、同批执行
  （控制变量）；多键轴是矩阵语言的必要扩展。
- **凭据构造期解析保持，dry-run 特判跳过**：否决。特判制造"规划可用、
  运行不可用"的假象；延迟解析是统一口径（fail 点移到 I/O）。

## Consequences

- 异构格成为一等公民：未来任何"换后端对比"（多模型基准）都可复用
  多键轴写法。
- 凭据语义变化：instantiate 成功 ≠ 有凭据；首次 generate 才是凭据
  校验点（错误信息不变）。
- API 对比实验的准备度从"harness 就绪"升到"一键可跑"：只差 key。
