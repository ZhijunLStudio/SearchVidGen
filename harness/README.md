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
│   ├── registry.py          # 注册表（@register，能力校验 fail loud）
│   └── experiment.py        # 实验：manifest/产物/缓存/断点续跑/成本
├── tasks/story.yaml         # 组合配置（对应 cordis.yml：选提供者/评测维度/重试）
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

## 当前部署（本机实验环境）

| 角色 | 模型 | 方式 | GPU |
|---|---|---|---|
| 剧本 | DeepSeek V4 Flash | API (deepseek-chat) | - |
| 生成 | MiniMax H3-Base (2026-07 开源, 33B, 音视频一体) | diffusers@minimax-h3 分支, bf16 | GPU 6 |
| 评测 | Qwen3.5-27B (多模态, 视频理解) | vLLM OpenAI 兼容服务 :8030 | GPU 7 |
| 总装 | FFmpeg 9.0 | 拼接+字幕 | - |

## 路线（实验驱动）

1. ✅ 协议/注册/实验管理/Judge 闭环/SegmentDirector（本轮）
2. ⏳ H3-Base 本地跑通 + DeepSeek 剧本 + 评测闭环，第一个端到端实验（本轮）
3. ⏳ H3 API 适配器（2K 完整流程）+ 本地/API 对比实验（下轮）
4. 未来：多模型基准对比（JoyAI-Echo/MOVA/未来模型）、公开 leaderboard、任务库扩展

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
