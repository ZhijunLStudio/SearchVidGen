# Agent Note: 提供者参数声明目录（param_schema）

Status: implemented

## Problem

任务 YAML 传给适配器的 params 此前只按构造签名内省校验：能拦住未知参数与缺
必需参数，但拦不住**类型错误**（如 `steps: "30"` 字符串）、**取值错误**
（如 `variant: fl2va2` 拼错），也提供不了任何自助文档。对齐 deepseek-harness
的"声明式提供者目录"（declared provider catalog：声明字段、类型、默认值、
可选值，目录是建议性发现、校验权威归被选中的适配器）。

## Decision

每个提供者类声明 `param_schema`：`{参数名: {type, required?, default?, choices?, help?}}`，
type ∈ str/path/secret/int/float/bool/list。`registry.instantiate()` 校验顺序：
声明目录（权威）→ 构造签名内省（未声明者的兜底）。校验规则（fail loud）：

- 未声明参数 → 报错并列出声明；
- 类型不符（YAML 来的 bool 不当作 int）→ 报错；
- choices 违规 → 报错并列出可选值；
- 缺 required 参数 → 报错并带 help。

CLI `vh adapters --verbose` 展示每个提供者的能力 + 参数声明目录（自助文档）。
`vh adapters` 默认输出附带能力声明。

4 个内置提供者全部声明（minimax-h3 local/api、deepseek_script、vllm_judge、
sensevoice）；元测试锁定"声明 == 构造签名"，防 drift。

## Alternatives considered

- **只靠构造签名**：否决。Python 签名只有名字与默认值，没有类型/取值语义；
  本轮前的方案正是它，拦不住 `steps: "30"` 这类 YAML 真实会出现的错误。
- **JSON Schema 声明**：否决。单层参数目录用轻量 dict 即可；引入 schema 库
  换来的 $ref/嵌套表达力暂不需要（与任务配置校验同款判断）。
- **声明进 YAML 任务文件**：否决。参数语义属于提供者（换提供者换语义），
  任务文件只引用参数；声明跟着提供者走才不漂移。

## Consequences

- 新增提供者必须声明 param_schema（元测试强制：声明 ≠ 签名即失败）；
  新增参数字段 = 同时更新签名与声明。
- `secret` 类型是预留语义（未来脱敏/日志红线时使用），当前校验等同 str。
- 声明目录是建议性发现：类型/取值校验在 instantiate 处权威执行，
  目录本身不做请求白名单（对齐 DSH 的目录定位）。
