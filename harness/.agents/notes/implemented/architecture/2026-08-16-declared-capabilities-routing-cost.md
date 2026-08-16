# Agent Note: 声明式能力目录、按能力路由与成本口径

Status: implemented

## Problem

能力（capabilities）是 harness 的路由与校验基础，但 2026-08-16 前它是自由 dict：
键名拼错会静默绕过能力校验；路由只能按名字硬指（YAML 显式 adapter）或手写
fallback 链；能力要求硬编码在消费者里（SegmentDirector 写死 first_last_frame，
与 chain_mode 无关的提供者被误拒，Bug#3）；成本口径靠字符串嗅探
（`"local" in adapter 名`）而非声明。

## Decision

对齐 DeepSeek Harness 的"声明式提供者目录"（provider-routed 模式），缩放落地：

1. **能力词汇是协议**：`SEAM_CAPABILITY_SCHEMAS` 按 seam 声明允许的能力键与类型；
   `@register` 在注册点校验（未知键/类型错误/未知 seam → 响亮失败）。
   新能力键 = 显式协议演进（先登记 schema）。
2. **按能力路由**：`resolve_provider(seam, required)` 从已注册提供者中选出唯一
   满足者；无人满足报错并列出各候选不满足原因；多候选同时满足时拒绝替用户
   做决定，要求显式指定 adapter。任务 YAML 支持
   `generator: {route: {audio: true, max_duration_s: 10}}` 替代 adapter 名。
3. **能力校验按任务推导**：SegmentDirector 按 chain_mode 推导要求
   （hard→first_last_frame / ref→refs≥1 / none→无），校验对象是**实例**
   （fallback 等合成提供者的能力是实例级并集，类上没有）。
4. **成本口径靠声明**：`backend: local|api` 是 generator 的能力声明。
   API 提供者在 ArtifactMeta.cost_usd 声明费用；本地产物按 backend=local
   的耗时计 GPU 时间，按任务配置 `cost.gpu_price_usd_per_hour`（默认 1.2）
   折算。finalize 消费声明，不再按名字嗅探。

## Alternatives considered

- **能力键保持自由 dict + 文档约定**：否决。拼写错误（如 max_duraton）会静默
  绕过能力校验，正是 fail-loud 要防的坑；schema 注册点校验的成本极低。
- **路由自动挑"最优"候选（打分排序）**：否决。能力满足是布尔语义；
  "最优"引入不可解释的偏好，多候选时让用户显式选择更符合"显式 > 隐式"。
- **GPU 成本按 `params.backend` 记录**：否决。params 是运行参数不是能力声明，
  历史产物里的 params 各写各的；声明式能力可被未来计费/调度复用。
- **删掉 hard 能力要求（chain none 用不到）**：否决。要求必须跟随任务语义
  （chain_mode）而非删除——错误在于硬编码，不在校验本身。

## Consequences

- 新增能力键必须先登记 schema（协议演进走 registry，不走自由 dict）。
- `resolve_provider` 是"新模型即插即路由"的基础：未来多生成器并存时，
  任务只需写 route 要求。
- 旧 manifest 的成本口径不变（H3-local 的 backend 声明与原嗅探结果一致）。
