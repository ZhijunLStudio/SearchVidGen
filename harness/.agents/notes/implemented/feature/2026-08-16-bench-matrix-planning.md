# Agent Note: 基准矩阵与规划期校验（vh bench）

Status: implemented

## Problem

对比实验是 harness 的护城河（模型换代 3-6 个月一轮，统一评测/对比/成本统计
越值钱），但此前"一次只变一个变量"完全靠手工：手动改 YAML、手动跑、手动记
哪个 run 是哪个变体（E8/E10 的 hard/none/ref 对比就是手工流程，compare_chains
甚至一度靠硬编码 run_id 猜模式）。没有机制把矩阵展开、逐格校验、成本预估
制度化。

## Decision

`vh bench <spec.yaml>`：bench spec 的 `bench:` 段声明 `base`（基础任务配置）+
`matrix`（变量轴列表，点路径 → 取值列表），展开为笛卡尔积格子，每格一次
完整实验 run。

- **规划期全格校验**（对齐 DSH "配置错误在最早可解析点响亮失败"）：
  `plan()` 对每一格先做 配置 schema 校验 + 三个提供者的参数声明校验 +
  能力要求校验；任何一格不合法即整体失败——不花一分钟 GPU 就发现配置错误。
  `--dry-run` 只做规划与预估。
- **格标签**：各轴取值拼接（如 `20.0.7`），经 `Experiment.bind_label()` 写入
  manifest.bench_cell，报告按格分组对比。
- **成本预估（规划口径，显式假设）**：API 后端 = 段数 × 时长 × 提供者声明的
  `cost_rates_usd_per_s` 能力（按分辨率）；本地后端 = 段数 × `local_min_per_seg`
  （E4 规划常数 12 分钟，spec 可覆盖）× GPU 单价。预估≠结算：结算口径仍在
  finalize（按实际耗时与声明 backend）。
- 报告消费层同步升级：`collect()` 输出分阶段耗时/成本分解（stages_elapsed_s /
  stages_cost_usd）与 bench_cell；完整性过滤改以 `finished_at` 为准
  （旧 run 用成片存在兜底）。

## Alternatives considered

- **CLI 参数传矩阵（--matrix 'a=x,b=y'）**：否决。矩阵是实验设计的一部分，
  应像任务配置一样以文件形式固化、可评审、可复现（spec 文件进 git）。
- **并行执行格子**：否决。GPU/裁判服务是共享瓶颈，并行只放大排队抖动，
  破坏"同一环境控制变量"；先串行，需要时再加。
- **预估用历史 run 的中位数**：否决。引入隐式数据依赖（历史集变化预估就变）；
  显式常数 + 声明单价更可解释，也更接近"预估"的诚实口径。

## Consequences

- 新变量轴直接写 spec 文件即可；矩阵路径必须从配置根写全（点路径），
  结构不匹配在规划期报错。
- `cost_rates_usd_per_s` 成为 generator 能力 schema 的正式键；新 API 提供者
  要参与预估必须声明单价。
- bench 逐格串行执行时经验记忆（_memory.jsonl）跨格共享——跨格学习是特性
  而非污染（环境反馈本来就该跨 run 累积），但对比解读时需注意后续格
  可能受益于前格经验。
