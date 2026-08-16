# Changelog

VidHarness 0.2.1（2026-08-16 晚）—— 0.2.0 后的持续优化（rounds 31-49）。
完整决策记录见 `.agents/notes/`（43 篇 Agent Note：42 implemented + 1 proposed），实验证据见
`README.md` E1-E43，范式对照见 `docs/paradigm.md`。

## v0.2.1（2026-08-16 晚）

### 新增

- 异构矩阵轴（adapter+params 成对切换）+ 凭据延迟解析——API 对比实验
  一键可跑（E 准备轮）
- 语义聚类 `vh memory-consolidate`（LLM 归纳 + 归并 + 提升）
- 成本报表 `vh costs`（跨任务 API/GPU/总计聚合）
- bench repeats（每格 N 次重复，统计比较基建）
- Makefile 健康门禁（make check）
- **run 标题自动生成**：run 完成时 script 提供者提炼 ≤12 字标题
  （manifest.title 经事件流），leaderboard/报告/详情页三处渲染
  （report.collect 单一正源；`SegmentDirector.run(before_finalize=...)`
  钩子保证 finalize 前挂入，E40 真实 API 实证）
- **裁判重复评测工具** `scripts/judge_repeatability.py`：同视频同口径
  重复评测（口径读 run 快照零硬编码，结果落事件溯源实验），E42 实证
- **种子控制与确定性工具链（E43）**：`GenRequest.seed` 缝级字段 +
  `kw.seed > req.seed > 构造参数` 逐调用覆盖（meta 记录有效种子）；
  `scripts/compare_seed_runs.py`（跨 run 像素差）、
  `scripts/seed_determinism_direct.py`（固定提示直测）、
  `tasks/bench_seed.yaml`——配对种子 A/B 的完整工具链
- **script 提供者 kw.system 覆盖**：变换任务（标题/归纳）自拥系统指令，
  摆脱导演人格；meta.params["system"] 落盘可审计（E43）

### 修复

- iterdir 双重拼接（costs 与 export_all 扫描静默为空）
- consolidate 无标签条目丢弃（数据保全）+ 同名经验重复去重
- 死代码清理（parse_judge_output/recent_feedback/report_costs）
- 测试文件 ruff 残留（E741/F401）清零——`make lint` 门禁范围外但同标修复
- **静默行为全量审计（E41）**：43 个异常处理器判级，6 处修复——
  优化器裁判不可用不再=0 分+噪声进经验记忆（整轮全挂响亮失败）、
  损坏 manifest 续跑 fail-loud、save_eval 从事件流重建、finalize 成本
  口径 warning、报告损坏 JSON 可见、script 裁判 outage 落 error 记录、
  中段末帧失败可见（新增 "warning" 事件通道）
- **旧布局裁判推断标注强化（E42）**："judge.openai-compat（推断：旧布局
  未记录，裁判版本未知，跨期不可比）"——同名 adapter 跨代不可比的
  口径警示进 leaderboard/报告
- **异构生成器参数格切换 OOM（E43）**：adapters_cache 双模型驻留显存
  → `MiniMaxH3Local.dispose()`（资源归提供者）+ `bench.evict_generators()`
  + cmd_bench 参数变化接线；兼解除 bench_api_local 异构矩阵潜在阻塞
- **标题静默失败（E43）**：4 个 bench 格全部无标题——提供者导演人格
  在温度 0/0.7/1.0 下压倒用户指令；kw.system 覆盖修复（温度 0 适配器
  实测产出"橘猫窗台观雨"）

### 实验证据（E31-E43）

bench 缓存复用实测（cell2 省 53%）、记忆噪声清洗与迁移、语义聚类首次
自动提升、优化预算消融（3×3 档 +0.52）、自学习闭环端到端（E34 方向 +
E37 机制级因果 A/B：经验注入 +0.70，旁白自然 +2.2 对应被教导维度）、
GPU 级下游效应诚实负结果（E38）、API 适配器 mock 协议验证（E39）、
session title 全链路落地（E40，真实 API $0.000017/次）、静默行为审计
（E41，6 处吞异常路径可见化/响亮化 + 真实 run 副本验证）、裁判重复
评测方差分解（E42：本代裁判 sd=0 确定性，臂间方差全部来自生成；
8-14 裁判与今日跨代不可比，实证 5.0→10.0）、种子控制确定性
（E43：同种子 MAE=0.0 像素级相同 vs 异种子 69.67——配对种子 A/B
生成层无噪声，E38 样本量需求坍塌；连带修复 OOM 与标题静默失败）。

## v0.2.0（2026-08-16）

### 新增

- **实验事件溯源**：events.jsonl 权威事件流 + manifest 投影 + 崩溃重放恢复
  （SIGKILL 故障演练实证，E29）
- **运行时不变量**：check_experiment + `vh doctor`（单 run / --all 全量）
- **任务配置校验**：config schema fail-loud + 提供者参数声明目录
  （param_schema）+ 能力 schema 注册点校验 + 按能力路由（resolve_provider）
- **基准矩阵**：`vh bench`（规划期全格校验/成本预估/格级断点续跑/
  跨格适配器缓存复用）
- **评测体系**：judge 缝双实现（vLLM VLM + DeepSeek 文本裁判）+
  阶段级裁判路由（judge.stages）+ 模态守卫 + 评分解析兜底 +
  JSON 模式 + 可操作反馈 + **跨裁判校准**（calibration/ + leaderboard
  --calibrated 换算）
- **script 缝双实现**：DeepSeek 官方 + 通用 OpenAI 兼容；提示/解析契约
  归 Service Definition；剧本优化闭环（温度轮转候选多样性）
- **报告与基线**：单 run 详情页（`vh report --run`）、分阶段分解、
  leaderboard 基线（JSON+MD+diff）、index.html 总览、混用裁判警告
- **变体回归套件**：`vh regress`（状态表 + 配置漂移检测 + --run 执行）
- **提供者脚手架**：`vh scaffold` + cookbook 接入指南
- **工程化**：pyproject 分发（vh 入口）、ruff/mypy 零问题、core 覆盖 93%
- **经验记忆**：提升规则修复（重复反馈自动提升）+ 记录版本化

### 修复（真实 bug）

- Bug#1 评测权重在协议传递中丢失（E11）
- Bug#2 优化器段数读 manifest 幽灵字段（E11）
- Bug#3 能力校验硬编码 first_last_frame（E11）
- Bug#4 实验不冻结任务配置（E11）
- Bug#5 经验记忆提升规则从未生效（E14）
- Bug#6 双卡 t2va 画布参数静默失效（E18）
- Bug#7 fl2va 条件化自 E6/E7 起静默失效（E20）+ 段 2 OOM（块级 offload）
- mypy 审计：OpenAI 响应 None 崩溃路径、变量遮蔽等 3 处（E 系列之外）

### 实验证据（E1-E30）

三变体实测配方（t2va/fl2va 双卡、ref2va int8 单卡）、衔接策略三模式
对比与修复路径复现（E8/E10/E21）、成本实盘（E4/E16/E19）、none 模式
方差三观测（E16/E23）、优化闭环量化（E24/E26）、校准 k=10 稳定偏移（E30）。

## v0.1（2026-08-14/15）

- 初版：seams/providers/consumers 三层、Judge 闭环、SegmentDirector、
  H3 本地双卡 + ref2va int8 配方（E1-E10）
