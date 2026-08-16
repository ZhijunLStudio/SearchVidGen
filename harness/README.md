# VidHarness —— 面向未来的视频生成 Harness

> SearchVidGen v2 的核心。不是"又一个一键出片工具"，而是一套**视频生成流水线的实验与评测 Harness**。
> 对标 LLM 生态的 `lm-evaluation-harness`：适配器是更新单元，任务是评测目标，实验产出可复现证据。

## 快速上手（5 分钟）

```bash
pip install -e .                                  # 安装（vh 命令入口）
vh adapters --verbose                             # 看能力与参数声明目录
vh run tasks/story_smoke.yaml --query "雨夜，一只小猫在旧书店的橱窗前躲雨" \
   --output experiments                           # 最小真实端到端（2 段×19 步，~25 分钟 GPU）
vh doctor experiments/story_smoke/<run_id>        # 运行时不变量体检
vh leaderboard story_smoke                        # 导出基线（leaderboards/ 入库追踪）
vh regress --output experiments                   # 变体回归套件状态（配置漂移检测）
vh bench tasks/bench_ablation.yaml --query "..." --dry-run   # 基准矩阵规划（不花 GPU）
make check                                        # 健康门禁：pytest+coverage+mypy+ruff
```

接入新模型：`vh scaffold generator my-model` + [cookbook](docs/cookbook/adding-a-provider.md)。
文档导航：[范式对照](docs/paradigm.md) / [决策记忆](.agents/notes/README.md) /
[变更日志](CHANGELOG.md) / [操作手册](RUNBOOK.md)。

## 为什么是 Harness 而不是"流水线"（2026-08 关键判断）

2026 年的全模态模型（MiniMax H3 等）正在把"脚本→图像→视频→TTS→字幕"这一整条分离式流水线
**吸收进单次生成**（文字+参考 → 视频+原生音频，一次调用）。因此：

| 会被模型吸收（不做厚） | 不会被模型替代（做厚，这是 harness 的护城河） |
|---|---|
| 图像/一致性组件（IP-Adapter、InstantCharacter 类） | **评测层**：模型不会给自己做质检；生成越"一体化"，验证越稀缺 |
| 独立 TTS / 音效阶段 | **跨片段编排**：模型单次 ≤15s，30-60s 故事片的多段拼接与上下文携带是持久需求 |
| 字幕 ASR（自产旁白无需转写） | **跨段一致性验证**：跨调用的角色/场景延续性检查 |
| 固定 7 步业务流水线 | **实验与基准**：模型换代越快（3-6 个月一代），统一评测/对比/成本统计越值钱 |

## 架构（对齐 deepseek-harness 的"一切皆插件"哲学）

参考实现：`/data/lizhijun/work/Harness/deepseek-harness`（Cordis 插件化 agent harness）。
本 harness 取其精髓、缩放到 Python 规模：
**能力缝 = 服务定义(seams/) + 提供者(providers/) + 消费者(consumers/)**，核心只留注册表与实验管理。

```
harness/vidharness/
├── seams/                   # 服务定义（协议与数据结构，不绑模型）
│   ├── generator.py         #   MediaGenerator + GenRequest（全模态生成）
│   ├── script.py            #   ScriptGenerator（故事规划）
│   └── judge.py             #   Judge + JudgeCriteria + RetryPolicy（评测）
├── providers/               # 提供者（模型适配器，加载即注册，更新单元）
│   ├── deepseek_script.py   #   剧本：DeepSeek V4 Flash API
│   ├── minimax_h3.py        #   生成：本地 H3-Base(diffusers) / 官方 H3 API(2K)
│   └── vllm_judge.py        #   评测：OpenAI 兼容 VLM（本地 vLLM Qwen3.5-27B）
├── consumers/               # 消费者（编排与闭环）
│   ├── judge_loop.py        #   生成→评测→失败反馈重试
│   ├── segment_director.py  #   剧本→逐段生成(首尾帧衔接)→跨段评测→总装
│   └── assemble.py          #   FFmpeg 拼接 + 字幕烧录
├── core/
│   ├── registry.py          # 注册表（@register 能力schema校验 / instantiate 参数声明校验 / 按能力路由）
│   ├── config.py            # 任务配置 schema 校验（fail loud：拼错键拒绝启动）
│   ├── experiment.py        # 实验：events.jsonl 事件溯源 + manifest 投影/缓存/断点续跑/成本
│   ├── invariants.py        # 运行时不变量（manifest↔文件↔事件流关系）+ vh doctor
│   ├── memory.py            # 经验记忆（环境反馈积累）
│   └── report.py            # 实验对比报告
├── tasks/story.yaml         # 组合配置（对应 cordis.yml：选提供者/评测维度/重试）
├── .agents/notes/           # 决策记忆（Agent Notes：为什么这样做、否决了什么）
└── cli.py
```

## 核心机制

1. **能力声明式路由**：每个生成适配器声明 capabilities（时长/音频/参考数/分辨率/后端），
   harness 据此做能力校验、参数降级与成本预估。新模型 = 新适配器文件。
2. **上下文携带**（模型替代不了的编排）：
   - 角色/风格锚点：全片共享参考图（Ref2VA 模式）
   - 段间衔接：段 i 末帧 → 段 i+1 首帧条件（H3 首尾帧模式，结构性连续）
3. **评测闭环**：逐段评测（指令一致/质量缺陷/音画同步）+ 跨段评测（角色/场景延续）。
   评分 < 阈值 → judge 反馈注入下一次生成指令 → 自动重试。
4. **实验即产物**：manifest.json 记录模型版本/参数/seed/耗时/成本；同任务可跑多模型对比。
   每次运行把**有效任务配置冻结**为 run 目录内 config.yaml（可重建 + 续跑一致性守卫）。
5. **事件溯源 + 不变量**：`events.jsonl` 是权威记录（追加式，崩溃可重放重建
   manifest 投影）；`vh doctor <run>` 断言 manifest↔文件↔事件流的关系，
   finalize 时自动校验，证据不完整响亮失败。
6. **声明式提供者目录**：每个适配器声明 capabilities（能力 schema 校验）与
   param_schema（参数类型/取值/必需），`vh adapters --verbose` 即自助文档；
   支持 `route:` 按能力路由与 `generator.fallback` 降级链。
7. **基准矩阵（benchmark matrix）**：`vh bench <spec>` 把"一次只变一个变量"
   制度化——矩阵展开、规划期全格校验（配置/参数/能力，错误不花 GPU）、
   成本预估（API 声明单价 / 本地 E4 常数）、逐格执行并按格标签对比。
8. **聚合唯一正源 + leaderboard 基线**：report.collect() 是唯一聚合面
   （全局/分 stage 分数、通过率、模型、成本分解）；`vh leaderboard <task>`
   导出可入库基线（JSON+MD+增量 diff），`vh doctor --all` 全量体检。
9. **孪生适配器纪律**：judge 缝双实现（vLLM VLM 媒体裁判 + DeepSeek 文本裁判）、
   script 缝双实现（DeepSeek 官方 + 通用 OpenAI 兼容端点），提示/解析契约
   归 Service Definition 所有；模态声明可强制——媒体评测配到 text-only
   裁判在第一次调用即响亮失败，抽帧失败记错误记录而不是假装评测。
   **阶段级裁判路由**（judge.stages）：文本评测阶段可覆盖为 DeepSeek 文本
   裁判，消除对 VLM 服务就绪的依赖。
10. **新模型 = 新文件（带脚手架）**：`vh scaffold <seam> <name>` 生成提供者
    骨架（能力骨架来自 seam schema）；接入清单见
    [docs/cookbook/adding-a-provider.md](docs/cookbook/adding-a-provider.md)。
11. **session 标题**：run 完成时自动用 script 提供者提炼 ≤12 字标题
    （manifest.title，经事件流），leaderboard/报告直接可读；
    finalize 前经 `SegmentDirector.run(before_finalize=...)` 钩子挂入，
    保证落盘与不变量校验覆盖（E40）。

## 决策记忆（Agent Notes）

`harness/.agents/notes/` 存放设计决策：代码与 README 无法承载的**为什么**与
**放弃了什么**（移植自 deepseek-harness 的 Agent Notes 机制，格式见
[.agents/notes/README.md](.agents/notes/README.md)）。实验证据（E 系列）支撑决策，
决策笔记记录理由；经验记忆（_memory.jsonl）是运行时环境反馈，三者分工不同。

## 当前部署（本机实验环境）

| 角色 | 模型 | 方式 | GPU |
|---|---|---|---|
| 剧本 | DeepSeek V4 Flash | API (deepseek-chat) | - |
| 剧本评审（文本裁判） | DeepSeek V4 Flash | API，judge.deepseek-text（不占 GPU） | - |
| 生成 | MiniMax H3-Base (2026-07 开源, 33B, 音视频一体) | diffusers@minimax-h3 分支, bf16 | GPU 6 |
| 评测 | Qwen3.5-27B (多模态, 视频理解) | vLLM OpenAI 兼容服务 :8030 | GPU 7 |
| 总装 | FFmpeg 9.0 | 拼接+字幕 | - |

## 路线（实验驱动）

1. ✅ 协议/注册/实验管理/Judge 闭环/SegmentDirector（8-14）
2. ✅ H3-Base 本地端到端 + 三模式衔接对比（E1-E10，8-15）
3. ✅ 范式对齐轮（8-16）：任务配置 fail-loud 校验、能力 schema + 按能力路由、
   实验配置快照与续跑守卫、评测结算归属消费者（weight/min_score 生效）、
   成本口径声明化；Bug#1-#4 修复
4. ✅ 事件溯源轮（8-16）：events.jsonl 权威事件流 + 崩溃重放恢复、
   运行时不变量 + vh doctor、提供者参数声明目录、裁判原始输出归档 artifacts/judge
5. ✅ 基准轮（8-16）：vh bench 矩阵对比（规划期全格校验 + 成本预估 + 逐格执行）、
   报告分阶段耗时/成本分解、证据脚本配置正源改为实验快照（删除硬编码端点）
6. ✅ 基线轮（8-16）：leaderboard 基线导出（聚合唯一正源 + git 追踪 + 增量 diff）、
   doctor --all 全量体检、seam 一致性元测试
7. ✅ 记忆与复盘轮（8-16）：经验记忆提升规则修复（E14，机制曾形同虚设）+
   记录格式版本化；单 run 详情页 vh report --run（概览/配置/产物/评测/事件流）
8. ✅ 孪生适配器轮（8-16）：judge 缝第二实现 judge.deepseek-text（真实 API 冒烟
   通过）+ 模态守卫 + 抽帧失败 fail-visible；script 缝第二实现
   script.openai-compat + 提示/解析契约上移 SD + 阶段生命周期事件与配对不变量
9. ✅ 真实端到端轮（8-16）：story_smoke 首次真实 GPU 全链路验证（E16），
   vLLM 裁判服务恢复（GPU 7）——H3 API 对比实验只差 MINIMAX_API_KEY
10. ✅ 画布修复轮（8-16）：双卡 t2va 画布静默失效修复（E18，1:1 实测 768×768）+
   帧抽取唯一实现点 + ratio 归位任务上下文
11. ✅ bench 真跑轮（8-16）：ratio 消融矩阵首次真实 GPU 执行（E19，两格
   16:9/1:1 分辨率确凿、分组与聚合全链路正确）+ MiniMax 单价单一正源
12. ✅ fl2va 修复轮（8-16）：双卡 fl2va 静默失效四件套修复（E20 拆分路由/
   首段锚点/fail loud/块级 offload）+ hard 结论复现（E21）
13. ✅ 巩固轮（8-16）：bench 格级断点续跑 + 评分解析可操作反馈（E22）、
   none 方差第三数据点（E23）、优化闭环量化（E24）、跨裁判校准 + JSON
   模式（E25）、优化增益诚实审计（E26）
14. ✅ 变体回归套件（8-16）：vh regress 状态表（最新 run/关键分/配置漂移
   检测）+ --run 执行；四变体 check 任务全部 ✅ 一致（E27 收尾）
15. ✅ leaderboard 页面化（8-16）：vh leaderboard --all 导出全部任务 +
   index.html 总览（每任务最新 run/混用裁判警告/校准摘要）
12. ⏳ H3 API 适配器（2K 完整流程）+ 本地/API 对比实验
13. 未来：多模型基准对比（JoyAI-Echo/MOVA/未来模型）、公开 leaderboard、任务库扩展

## 环境踩坑记录（H3 本地部署，2026-08-14 实测）

1. **权重获取**：Wan2.5 权重已从 HF/魔搭下架；MiniMax H3（2026-07 开源）在
   `MiniMaxAI/MiniMax-H3`（未 gated，hf-mirror 直连）。仅下载 diffusers 推理所需子目录
   （transformer/transformer_ref/vae/audio_vae/text_encoder/tokenizer/processor/scheduler/
   audio_scheduler + modular_model_index.json）≈196GB，跳过 FL2VA/Ref2VA 原始格式。
2. **diffusers 版本**：PyPI 0.39.0 无 H3 支持；需装 GitHub main（Gitee 镜像
   `gitee.com/mirrors/diffusers` 可用）。依赖链修复：xformers 0.0.32 硬编码
   flash-attn ≤2.8.2（patch 到 2.8.3）、transformers 需 ≥5.15（HybridCache 已移除，
   需同步升级 peft）。
3. **加载**：`MiniMaxH3ModularPipeline.from_pretrained(path, components_manager=manager)`
   不传 `workflow`（传了会二次 get_workflow 报错）；`load_components(workflow=..., dtype=bf16,
   pretrained_model_name_or_path=本地路径)` 必须显式调用并覆盖组件源。
4. **纯文本生成必须显式传 canvas**（height/width，32 倍数）；首帧模式由首帧推断。
5. **时长约束**：5-15s（120-360 帧 @24fps）。
6. **性能**：A800-80GB 单卡 bf16 + auto_cpu_offload（主机需 ~100GB RAM 供换页），
   49 步约 16 分钟/段，30 步约 12 分钟/段。

## 实验发现（evidence-driven findings）

### E1：全模态模型一次生成 = 分离式流水线被吸收（2026-08-14）
MiniMax H3 单次调用产出"视频+原生音频"，原 7 步流水线的图像/TTS/音效阶段被模型内化。
harness 因此只保留三个能力缝：生成/剧本/评测（+转写）。

### E2：评测是 harness 的护城河，但思考型裁判需要工程约束
Qwen3.5-27B 思考文本可吃光 token 预算导致评分 JSON 未输出（实测触发段3/4误判 5.0）。
修复：enable_thinking=false + max_tokens 4096 + 维度别名正则。裁判质量本身很高
（对段1的描述与评分与人工判断一致：9.17/9.67）。

### E3：画面级连续 ≠ 好的剪辑（跨段衔接策略发现）
首尾帧硬衔接（fl2va）使跨段一致性满分 10.0，但裁判指出衔接帧"几乎一模一样"——
产生冻结帧跳切。新增"叙事推进"评测维度（同画面复制应扣分），并提供
chain_mode=ref（ref2va 参考图软衔接）对比实验。

### E4：成本实盘（A800×2，30步，8s@768p）
- 单段生成 ~18 分钟（去噪 17.5min + 编解码），4 段成片总 GPU 时长 ~81 分钟
- 本地算力成本 ≈ $1.6/部（$1.2/卡时），对比 H3 官方 API 768P 约 ¥0.3/秒 ≈ 32s 成片 ¥9.6
- 判决：本地模式适合批量/隐私场景；API 模式单价更低但需按量计费

### E5：ref2va（参考模式）单/双卡部署均 OOM（2026-08-15）
ref2va 将参考 latent 拼入打包序列，生成侧（transformer_ref 62GB 常驻 + 参考/视频/音频
token 激活）超过单卡 80GB；480p 画布与 5s 帧数均不足以缓解（瓶颈是权重常驻而非帧数）。
官方 int8 路径（TransformersTorchAoConfig Int8WeightOnlyConfig + block_level group offload，
~75GB 主机内存）是正确解，留待下轮实现。
暂用 chain_mode=none（纯文本延续）替代软参考衔接做对比实验。

### E6：ref2va 双卡拆分要点
- 条件侧必须合并 before_encode + text_encoder 两个子块（SequentialPipelineBlocks.from_blocks_dict），
  因为参考编码器在 text_encoder 内、依赖 before_encode 的 normalized_references
- before_encode 需要 num_frames（视频参考归一化）
- 参考图参数名是 references（MiniMaxH3ImageReference.from_file），不是 reference_images

### E7：ref2va int8 配方落地（2026-08-15）
官方 int8 配方在本机成功：独立环境 torch 2.13+torchao 0.18（0.15+ 才支持
Int8WeightOnlyConfig(version=2) 且要求 torch>=2.9，与旧环境 vllm 0.11 冲突→独立环境）。
要点：diffusers 新版要求 low_cpu_mem_usage=True（与官方文档相反）；torchao 量化加载后
需 pipe._device 兜底执行设备；ffmpeg 需在 PATH。效果：ref2va 单卡显存从 ~78GB 降至
**38GB**（量化权重+块级流式 offload），5s 测试 342s 完成（含 ~4.5min 加载）。

### E8：衔接策略对比（hard vs none，同一故事"雨夜小猫"）
| 模式 | 跨段一致性 | 叙事推进 | 结论 |
|---|---|---|---|
| hard（fl2va 首帧硬衔接） | 10.0 | **1.0（冻结帧）** | 画面连续但叙事冻结 |
| none（t2va 文本延续） | 10.0 | **10.0** | 切镜自由 + 文本角色描述保一致 |
| ref（ref2va 参考软衔接） | 运行中 | 运行中 | int8 后单卡可跑 |

叙事推进维度由跨段评测（相邻段首尾帧对比 + 裁判判定"是否为合理新镜头"）量化。

### E9：裁判解析可靠性审计（33 次真实调用，2026-08-15）
- JSON 直接解析 29/33（88%），正则兜底 4/33（12%），完全失败 0
- 修复（思考关闭+别名解析+token 预算）后，评测闭环的评分提取 100% 可用
- 分数分布 min=0(证据收集的一行描述) max=10 mean=7.52，重试阈值 6 具有区分度

### E10：三模式衔接对比完成（2026-08-15，同一故事控制变量）
| 模式 | 跨段一致性 | 叙事推进 | 段均分 | 结论 |
|---|---|---|---|---|
| hard（fl2va 首帧硬衔接） | 10.0 | **1.0 冻结帧** | 10.0 | 过度约束 |
| none（t2va 文本延续） | 10.0 | 10.0 | 10.0 | 本故事最优 |
| ref（ref2va 参考软衔接） | 9.2（首段衔接 5.0，锚点图弱） | **10.0** | 10.0 | 允许切镜；一致性强依赖参考图质量 |

结论：段间衔接不应做帧级硬约束。默认 none（文本延续）；当有高质量角色/风格参考图时
ref 是更强的一致化工具（锚点质量决定一致性下限）。

### E11：范式对齐审计发现 4 个静默配置 bug（2026-08-16）
对照 deepseek-harness 的"显式 > 隐式 / fail loud / 模型可见⟺日志"原则审计发现：
- Bug#1 评测权重丢失：YAML 的 weight=1.2/min_score=5 在 judge 协议传递中被默认值
  替换（加权分与阈值判定均与配置不符）；
- Bug#2 优化器段数读 manifest 幽灵字段（manifest 从未写入 segments）→ 恒为默认 4；
- Bug#3 能力校验硬编码 first_last_frame，与 chain_mode 无关的提供者被误拒；
- Bug#4 实验不冻结任务配置，对比脚本只能硬编码 run_id 猜衔接模式。
全部修复并加回归测试（30 passed），决策沉淀在 .agents/notes/implemented/2026-08-16-*。
经验：**"配置从产物回读"、"字符串嗅探"、"自由 dict 能力键"是静默 bug 的三类温床**。

### E12：不变量体检让旧实验的布局问题现形（2026-08-16）
`vh doctor` 上线首跑即对 8-15 的实验目录报出 16 条违规：裁判原始输出
（judge_*.json，dict 格式）与评测明细（记录列表）混放在 eval/ 目录。
根因：裁判 workdir 指向 eval_dir。修复：裁判原始输出迁至 artifacts/judge/
（作为产物经事件流归档），eval/ 只保留评测明细记录；doctor 对旧布局给出迁移提示。
经验：**关系不变量（"此目录只放哪种文件"）能抓住 schema 校验看不见的结构污染**。

### E13：证据脚本的硬编码端点审计（2026-08-16）
审计 scripts/ 发现 collect_evidence.py 硬编码裁判端点与模型名——违反"配置正源
= 实验快照"原则；judge 服务换端口后证据会静默失效或口径不可比。修复：裁判配置
从 run 的 config.yaml 快照读取（无快照 fail loud）。同日 bench dry-run 对真实
story.yaml 的 4 格矩阵（步数×温度）全部规划期校验通过，本地口径预估 $0.96/格、
总 $3.84——**配置错误与成本暴露都发生在 GPU 启动之前**。

### E14：经验记忆的提升规则从未生效（2026-08-16）
审计真实 _memory.jsonl：8 条条目、3 条经验行全部来自 CLI 手动 add_experience；
`add()` 只涨 count 从不置 promoted，而 experience_lines() 要求两条件同时成立——
裁判重复反馈一条都没被自动提升过（文档承诺与代码行为脱节，E11 同款问题）。
修复：提升发生在 add() 内（到达阈值即置位）+ 记录格式版本化 + 坏行记入
load_warnings + sources 上限 5。回归测试锁定提升语义。
经验：**"文档宣称的自学习机制"必须有用真实数据验证的回归测试，否则它就是
装饰品**。

### E15：judge 缝第二实现落地，孪生适配器暴露词汇缺口（2026-08-16）
按 DSH 孪生适配器原则为 judge 缝交付第二个真实实现 judge.deepseek-text
（DeepSeek 官方 API 文本裁判）：真实 API 冒烟通过（7.0 / passed / 反馈可用，
计费入 meta）。孪生化立即暴露两个单实现时代被掩盖的缺口：①modalities 声明
不可强制——text-only 裁判误配到媒体评测会静默丢弃媒体；②抽帧失败返回 None
会被静默传给裁判。修复：run_judge 模态守卫（第一次媒体调用即响亮失败）+
cross_consistency 抽帧失败记错误记录。
经验：**单一实现会掩盖协议缺口；第二个真实实现是协议完备性的探测器**。

### E16：新基建首次真实 GPU 端到端验证（2026-08-16）
story_smoke 任务（2 段×19 步，GPU 4/6 双卡生成 + GPU 7 vLLM 裁判，全真实）：
- **事件溯源/快照/不变量/详情页在真实运行中全部工作**：24 条事件
  （run.created→query→config→stage×8→artifact×6→eval×3→finalized），
  doctor ✅ 零违规，finalize 自动不变量通过，单 run 详情页时间线完整
- 段级真实裁判：与指令一致性 10.0 / 画面质量 10.0（两段均 passed）；
  跨段一致性 2.0（chain_mode=none 镜头切换大——对比 E8 同模式 10.0，
  **none 模式的一致性方差大、与具体剧本强相关**，值得后续量化）
- 成本实盘：2 段共 29.9 分钟（19 步/段 ~12 分钟）、0.492 GPU 卡时、
  估算 $0.59；成片 16s@1344×768 含原生音轨
- 暴露两个待办：①剧本评测在 vLLM 未就绪时被跳过（容错降级有效但缺该维度
  评价；judge.deepseek-text 可承接文本评测消除此依赖）；②diffusers 双卡
  t2va conditioner 报 ignored-input 警告（height/width/num_frames），
  画布最终仍正确（1344×768），留待与上游确认

### E17：阶段级裁判路由落地（2026-08-16，E16 待办①关闭）
judge.stages 阶段路由上线：story_smoke 的 script_judge 覆盖为
judge.deepseek-text。真实冒烟：剧本评测走 DeepSeek 文本裁判，两次尝试
（第 1 次空评分 → 反馈重试 → 第 2 次 叙事完整 5.0/可生成性 7.0），
judge 产物 adapter 记录正确（judge.deepseek-text）。经验：**评测选型是
任务配置的一部分，与生成选型同等对待——按阶段显式路由，而不是共享单一
裁判实例**。

### E18：双卡 t2va 画布参数静默失效（Bug#6）修复实锤（2026-08-16）
E16 的 diffusers ignored-input 警告溯源到真 bug：双卡拆分把画布参数传给
不声明它们的条件侧，t2va 画布静默回落到模型默认 16:9——E16 的 1344×768
是默认值巧合，任何非 16:9 请求都会被静默产出 16:9。修复（variant 感知
拆分 + ratio 归位 pipeline.context）后实测：story_canvas 1:1 任务产出
**768×768**（修复前必为 1344×768）。经验：**"被忽略的输入"警告是静默
降级的烟雾弹，必须溯源到参数消费点；修复要提取为纯函数并用测试锁定**
（首次修复曾因未提取函数而在真机 NameError）。

### E19：bench 矩阵首次真实 GPU 执行（2026-08-16）
`vh bench tasks/bench_ratio.yaml`（ratio 消融 2 格，真实 GPU + 真实裁判）：
- 两格全部完成：16:9 → 1344×768、1:1 → **768×768**——E18 画布修复在
  矩阵下再现，bench_cell 标签/leaderboard 分组/report 分阶段聚合全链路正确
- 段级真实裁判两格均 10.0/10.0 passed；剧本裁判（deepseek-text）叙事完整
  3.0-4.0——单段故事难构成完整起伏，低分可解释（1 段是画布验证的最小单元）
- 成本实盘：0.228+0.117 GPU 卡时 ≈ $0.41；逐格串行每格重载模型
  （含 ~5 分钟加载）——bench 的跨格模型复用列为已知优化点

### E20：fl2va 条件化自 E6/E7 起静默失效（Bug#7）修复实锤（2026-08-16）
真机回归发现 fl2va 双卡路径在 diffusers 深处 torch.cat 空列表崩溃。溯源：
E6/E7 为 ref2va 把 before_encode 拆到条件侧后，image 未随条件侧传入且
get_workflow("fl2va") 的 prepare_condition_latents 无条件执行——E8 时代
全集 pipeline 的 t2va 静默回退消失，此后 fl2va 一直静默失效（E10 的 hard
数据复用了 E8 旧 run，掩盖了此事）。修复三层：拆分按声明契约路由 image 到
条件侧、hard 首段以锚点首图为首帧、无 keyframe 在最早点响亮失败（不再
深处崩溃）。段 2+ 的 OOM（auto offload 下 78.2/79.25GB）按 E7 配方解决：
生成侧 transformer 块级流式 offload，显存 78GB → **24GB**。最终
story_fl2va_check 完整跑通：锚点条件化段 1（差 20.15/255）、段间衔接
25.59/255、跨段一致性 10.0、段级 10.0/10.0。经验：**"复用了旧实验数据"
的基准结论会掩盖路径级 bug；条件路径变更后必须真机重跑，且基准数据要
标注生成路径版本**。

### E21：修复后 hard 衔接质量重测——E8 结论在正确路径上成立（2026-08-16）
修复 fl2va 条件化后，用与 E8 相同的故事（雨夜小猫）重跑 hard 衔接 2 段，
补齐"叙事推进"维度：

| 指标 | E8 旧路径（bug 时代） | E21 修复路径 |
|---|---|---|
| 跨段一致性 | 10.0 | 10.0 |
| 叙事推进 | 1.0（冻结帧） | **1.0（冻结帧判定复现）** |
| 段间衔接像素差 | 5.68/255 | 21.34/255（几乎一样但有演化） |

结论：**E8"hard = 过度约束"的结论在修复后的正确条件化路径上依然成立**——
帧级硬衔接的本质就是冻结开头，与实现 bug 无关；修复路径的像素差更高
（21.34 vs 5.68），冻结程度略轻但裁判判定一致。E8 旧数据因全链路 bug
曾受质疑，现已获得修复路径的独立复现。
另：ref2va int8 单管路径经代码级审计确认自 E7 以来零改动（本轮改动仅
触及双卡拆分），无需真机重跑。

### E22：bench 格级断点续跑与裁判解析失败的可操作反馈（2026-08-16）
- **格身份三要素**（标签+配置快照+query）：bench 重跑时已完成格秒级跳过、
  未完成格续跑（真实冒烟：E19 矩阵两格 0 GPU 消耗跳过）；换 query 是
  新实验，不跳过旧格
- **评分解析失败反馈**（E21 暴露：script_judge 两次空评分）：裁判输出
  不可解析时 feedback 注入结构化指令 + 原文上下文，重试循环重新获得信号
- 全库体检：12 个 run doctor 全扫（6 条违规均为 8-15 旧布局提示），
  CLI 七条命令全通过，测试 98 passed

### E23：none 模式第三数据点——一致性方差实锤 + 反馈重试首个真实生效（2026-08-16）
同一故事（雨夜小猫）none 模式第三跑（label none-r3，2 段）：

| 指标 | E8 none | E16 none | E23 none-r3 |
|---|---|---|---|
| 跨段一致性 | 10.0 | 2.0 | **3.0** |
| 叙事推进 | 10.0 | 未测 | **3.0** |
| 段间像素差 | ~ | ~ | 59.18（大跳切） |

三个数据点确认 E16 观察：**none 模式跨段维度方差大（2.0-10.0），与剧本的
镜头设计强相关**——文本延续的一致性由剧本质量决定，不是 harness 保证。
另两个实证：①剧本裁判反馈重试**首次真实生效**（4.0 未过 → 反馈注入 →
6.0 passed，E22 修复后的首个闭环证据）；②`vh run --label` 冒烟成功
（manifest.bench_cell=none-r3）。

### E24：剧本优化闭环量化——均分 2.65→5.46 + 裁判口径差异警报（2026-08-16）
`scripts/compare_script_optimize.py`（纯 API，无 GPU，各 N=3）：
- **optimize 开/关**：剧本均分 2.65 → **5.46**（+2.81）、通过率 1/3 → 2/3、
  成本 $0.0003 → $0.0009/试（2轮×2候选 ≈ 4 次生成）
- 5.46 仍低于 6.0 阈值——当前优化预算不足，候选/轮次/温度待调参
- **裁判口径差异警报**：同一批剧本在 vLLM 裁判（E8 时代）≈9-10 分 vs
  deepseek-text 2.65-5.46——混用裁判的 leaderboard 数据不可直接对比；
  跨裁判校准（同批剧本双裁判打分 → 校准系数）列为必做项，未校准前
  报告须标注裁判来源

### E25：跨裁判校准落地——口径差异大半是解析失败伪影（2026-08-16）
校准机制（scripts/calibrate_judges.py + calibration/ 入库）首跑即修正
E24 警报：
- **JSON 模式**（response_format=json_object）把 deepseek-text 解析成功率
  40% → **80%**（E24 的巨大分差大半来自解析失败→缺失维度计 0，而非
  纯尺度差）
- 校准实测（同批 5 剧本双裁判，n=4/维）：叙事完整 +0.5 / 旁白自然 +0.5 /
  **可生成性 -1.5**（deepseek-text 更宽松）——两裁判从"不可比"收窄到
  "可标注修正"
- leaderboard 增加裁判列 + 混用裁判警告（引用 calibration/ 数据）
经验：**评测口径疑云先查解析链路，再查尺度——解析失败会把缺失维度
记 0 分，制造出假的口径差异**。

### E26：优化增益的诚实审计——E24 的 +2.81 大半是解析修复被误记（2026-08-16）
JSON 模式（E25）落地后重跑同一量化（N=3，同 query）：
- **off 基线 2.65 → 7.94**：E24 的 off 低分绝大部分是解析失败伪影；
  E24 记录的"+2.81 优化增益"实为"解析修复增益"被误记为优化增益
- 同温候选 on == off（7.94，多样性不足时优化零增益）；
  温度轮转（0.6/0.9/1.2，optimizer 新能力）on = **8.06**（+0.12，4× 成本）
- 结论：单次生成在修复后已 7.9/10 且 3/3 通过；script_optimize 默认预算
  的 ROI 弱、天花板有限——预算调参是微调空间，不是主线
经验：**机制增益必须用修复后的基线重测——解析 bug 会把修复收益记到
上层机制的账上**。

### E27：ref2va 真机回归——三变体复测闭环完成（2026-08-16）
拆分重构后最后一个未复测变体 ref2va（h3int8 单卡 int8，chain_mode=ref，
锚点参考，2 段）完整跑通：
- 段级 10.0/10.0 ×2 passed；剧本 6.0/8.0 一次通过；doctor ✅
- **跨段一致性 10.0 + 叙事推进 10.0**（本次剧本双满分）——与 E10 的
  ref 结论一致：允许切镜 + 一致性强
- 锚点 vs 段1首帧差 88.95/255——参考软衔接不冻结首帧（ref 是"参考"而非
  "条件首帧"），主体特征相似但画面自由演化，设计语义成立
- int8 单卡画像复现 E7（~30GB 常驻、22.7 分钟/2 段）
至此 t2va/fl2va/ref2va 三变体全部具备拆分重构后的真实运行证据。

### E28：真实数据闭环验证——续跑幂等 + 反馈→记忆→提示全链路（2026-08-16）
两个此前只有单元测试的逻辑用真实数据走通：
- **已完成 run 的续跑幂等**：对 story_smoke none-r3 执行 --resume，全部
  阶段命中缓存、秒级完成、零生成、finalize 不变量再次通过——"断点续跑"
  对已完成 run 是幂等操作（快照/query 守卫全过）
- **用户反馈闭环**：`vh feedback` 写入经验 → director 记忆加载
  （1 条经验）→ stage_script 的剧本模板含该反馈文本（真实 DeepSeek 生成
  路径）——"环境反馈 → 跨任务经验 → 注入提示"链路端到端成立
经验：**已交付机制的最终验收是"用真实数据走一遍"，单元测试是必要条件
不是充分条件**。

### E31：bench 跨格缓存复用真实验证（2026-08-16）
真实 2 格矩阵（同参数、异 query 的新实验）实测：
- cell 1（含模型加载）**12.4 分钟**；cell 2（adapters_cache 复用）**5.8 分钟**
  ——跨格复用省 **~6.6 分钟/格（53%）**，第 11 轮的缓存机制从单测走向实测
- 两格 doctor 零违规、leaderboard 基线更新；格级跳过（E22）+ 缓存复用
  （E31）共同构成 bench 长矩阵的完整效率语义

### E32：真实记忆审计——JSON 噪声与聚类粒度卡死提升机制（2026-08-16）
审计 29 条生产记忆：7 条 complaint 是整段/多段 JSON（解析反馈原文混入）、
~20 条"叙事缺乏起承转合"的语义近重复因措辞不同永不提升——E14 机制正确
但被噪声与粒度卡死。修复：clean_feedback_text 正则抽取内层 feedback +
解析指令过滤；记忆加载迁移（清洗+键重算+同键合并+补提升）。真实迁移
29 条 → **0 残留 JSON**、3 提升。语义聚类（嵌入）记录为未来项。
经验：**自学习机制的验收要用生产数据做账——单测通过≠在真实反馈分布
上工作**。

### E33：语义聚类兑现——"叙事缺乏起承转合"首次自动提升（2026-08-16）
E32 的未来项落地：`vh memory-consolidate` 用 script 提供者（DeepSeek）
把语义近重复反馈归纳为规范短语并归并。过程中两个关键发现：①裁判的
系统纪律强制 feedback="pass"，无法做自由变换任务——**归纳用 script
提供者而非裁判**（能力选择按任务形状）；②第一版 consolidate 丢弃
无标签条目（数据丢失）→ 修复为原样保留。真实运行：23 条 → 4 组归并、
**"叙事缺乏起承转合"自动提升为跨任务经验**、16 条保留待下次。
至此自学习闭环五环全部有真实数据证据：反馈 → 清洗（E32）→ 聚类
（E33）→ 提升 → 注入（E28）。

### E34：自学习闭环的端到端影响——经验注入后的第一个对照 run（2026-08-16）
E33 提升的"叙事缺乏起承转合"经验注入同 query 的 none 模式 run
（label none-learned，控制变量对照 E23 none-r3）：

| 指标 | E23（无该经验） | E34（4 条经验注入） |
|---|---|---|
| 剧本叙事完整 | 4.0→6.0（重试才过） | **6.0 一次通过** |
| 跨段一致性 | 3.0 | **10.0** |
| 叙事推进 | 3.0 | **10.0** |

- 注入链路实锤：剧本模板含 4 条经验（含新提升条目）——反馈→学习→
  行为改变的最后一环在真实数据上成立
- 诚实边界：n=1 对照、方向与学习预期一致（E8 的 none 也曾双 10.0），
  **因果确认需要更多样本**——列为后续实验
- 附带：Makefile 一键健康门禁（make check = pytest+coverage+mypy+ruff）

### E35：优化预算消融——(3,3) 是真实增益档而非微调（2026-08-16）
`compare_script_optimize --rounds/--candidates` 参数化后扫描（N=3）：
- 2×2（E26 基线）：off 7.94 → on 8.06（+0.12）
- **3×3：off 7.81 → on 8.33（+0.52）**，18 次调用/3 试（early-stop
  在 target 8.0 处截断，预算未满即停）——高预算 + 早停 = 增益与成本
  兼得
- story.yaml 默认预算升级为 3×3 / target 8.0
结论修正 E26：**预算调参不是纯微调——(3,3) 档有显著增益，且
early-stop 使成本可控**。

### E36：vh costs 成本报表 + iterdir 双重拼接 bug（2026-08-16）
新增 `vh costs` 跨任务成本聚合（成本护城河的最后一角），首跑即抓到
隐蔽路径 bug：`d.iterdir()` 返回全路径，`(d / r / ...)` 双重拼接使
相对基座下扫描静默为空；同款 bug 潜伏在 export_all（round 21 起，
被旧基线文件掩盖）。单元测试没抓住它——pytest tmp_path 是绝对路径，
pathlib 对绝对 r 替换而非拼接，恰好绕过 bug 形态。修复两处 +
**相对基座回归测试**。真实报表：5 任务 18 run、**14.5 GPU 卡时、
总估算 $17.44**（38 轮会话的真实算力成本）。

### E37：经验记忆的机制级因果 A/B——效应集中在被教导的维度（2026-08-16）
E34 因果问题的最近端验证（纯 API，N=5/臂，同 query/裁判/生成器，
唯一变量 = 注入经验）：
- 总分：A（4 条经验）**8.56** vs B（空记忆）7.86（**+0.70**）
- 维度分解：**旁白自然 6.8 → 9.0（+2.2）**、叙事完整 +0.2、可生成性 -0.6
- 效应集中性 = 因果证据：记忆里恰有 2 条旁白经验（最重），旁白维度
  提升最大——不是随机波动，是"被教导的维度变好"
- 下游效应（跨段一致性）仍需 GPU 实验：bench repeats 基建已就绪
至此自学习闭环具备机制级因果证据（E28 注入 → E32 清洗 → E33 聚类 →
E34 方向 → E37 因果）。

### E38：GPU 级因果 A/B——下游效应低于噪声水平（诚实负结果，2026-08-16）
4 格 A/B（同 query/同 seed，唯一变量 = memory.path，n=2/臂）：
- 臂均分：A（经验注入）跨段一致性 4.5 vs B（空记忆）7.5——方向甚至
  反号，但**臂内方差 ±3.5 淹没臂间差 3.0**（E16/E23 的 none 模式方差
  是主导因素），n=2 无法检出效应
- 科学结论：经验在**剧本层**有效（E37 因果 +0.70），在下游**画面层**
  的效应低于噪声水平（检出 3 分效应需 n≈8-10/臂 ≈ 200+ 分钟 GPU）
- 经验：**上游机制有效 ≠ 下游效应可测——效应在每层传播中衰减，
  层间方差决定了需要的最小样本量**。E34 的因果问题以此诚实负结果
  收官：学习的价值在剧本质量，画面层效应交给未来大样本研究

### E40：run 标题自动生成——session title 全链路落地（2026-08-16）
- 机制：`vh run`/`vh bench` 完成后、finalize 之前，用 script 提供者把
  query 提炼成 ≤12 字标题 → artifacts/title/ 产物 + manifest.title
  （经事件流，重放可恢复）；两轮尝试收紧提示，失败静默降级
- 真实 API 实证：DeepSeek 生成"雨夜暖面"（4 字），成本 $0.000017，
  doctor 干净；leaderboard MD/对比 HTML/run 详情页三处渲染同源
  （report.collect 单一正源）
- 老 run 无 title → 表格显示 "-"，新旧兼容

### E41：静默行为全量审计——6 处吞异常路径可见化/响亮化（2026-08-16）
- 43 个异常处理器逐个判级（吞掉什么/有无测试/有无文档/逃生通道），
  6 处修复：①优化器裁判不可用=候选 0 分+噪声进经验记忆（BUG）→
  error 记录+整轮全挂响亮失败+零记忆污染；②续跑损坏 manifest 被
  静默当全新 run（BUG）→ fail-loud；③save_eval 损坏文件静默清空 →
  从事件流重建+warning 事件；④finalize 能力解析失败的 GPU 时间被
  静默排除 → warning 可见；⑤报告静默跳 run → stderr 提示；
  ⑥script 裁判不可用只 print → 落 error 记录；中段末帧失败记
  error 记录（E16 同口径）
- 新 "warning" 事件类型（scope/msg）：重放忽略、不变量不比对，
  纯可见通道；剩余 37 处判定为有意的 best-effort
- 回归测试 +7（155 total）；三条修复路径在真实 run 副本上验证

### E29：kill -9 故障演练——崩溃恢复实证 + 僵尸进程预检（2026-08-16）
对真实生成中的 run 执行 SIGKILL 演练：
- 事件溯源按设计工作：kill 后 events.jsonl 完好、投影一致、无 finalized
- **发现**：kill -9 父进程遗留 diffusers 僵尸子进程占住 62.7GB 显存，
  首次续跑在 torch 深处 OOM 且报错不可读
- 修复：适配器加载前 GPU 显存预检（check_gpu_free，<40GB 响亮失败 +
  "pkill -f vidharness" 指引）；RUNBOOK 记录规范终止姿势
- 演练收尾：清僵尸 → 二次续跑 → 缓存命中 + 段2重生成 → finalize →
  不变量通过——**进程级崩溃后的断点续跑获得真实实证**
