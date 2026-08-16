# VidHarness —— 面向未来的视频生成 Harness

> SearchVidGen v2 的核心。不是"又一个一键出片工具"，而是一套**视频生成流水线的实验与评测 Harness**。
> 对标 LLM 生态的 `lm-evaluation-harness`：适配器是更新单元，任务是评测目标，实验产出可复现证据。

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
