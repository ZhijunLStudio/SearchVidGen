# SearchVidGen 战略发展综合报告（2026-08）

> 依据：①仓库本地审计（一手）；②三份专项调研（docs/ 下 market_research / model_stack_research / 开源增长实战报告，star 数均为 2026-08-14 实测）；③本人交叉验证的公开信息。所有关键来源已在三份报告中附链接。

## 一、先回答：目标是否合适？

**结论：合适，但在当前状态下直接推进不合适；须满足三个前提，并明确三个"不做"。**

### 为什么合适
1. 需求侧已验证且仍在增长：2026 年 WAIC 定调"从一键成片到一人剧组"；AI 短剧/漫剧/宠物短剧变现案例频出（《归墟》播放破亿、宠物短剧博主月入 50 万）；"想法→成片"类项目（Pixelle-Video ≈2.7万★、VideoClaw ≈1.7k★、Toonflow 1.39万★）持续涌现且增长快。
2. 差异化空白真实存在：头部要么是"检索式素材混剪"（MoneyPrinterTurbo 10.3万★），要么依赖云 API/云 ComfyUI（Pixelle-Video），要么是重平台（Toonflow/BigBanana）。**"100% 本地开源直连模型的生成式流水线 + 角色一致性 + 搜索意图场景"目前无人占据**。Wan 2.7 转为不开源反而强化本地开源流水线的价值。
3. 底子可用：7 步脚本齐全、Gradio 雏形、demo 成片、MIT、双语 README；12★ 无历史包袱。

### 三个前提（缺一不可）
- **前提1·定位收敛**：GitHub 描述（"搜索意图→原生广告视频"）与 README（通用故事视频）自相矛盾，且"关键词→短视频"心智已被 MoneyPrinterTurbo 占据。必须收敛为差异化定位（见第五节），主动与混剪类划界。
- **前提2·体验底线**：新用户（只会 Python）10-30 分钟跑出第一条片子。现状：README 引用的 requirements.txt 不存在 + 12 处硬编码个人路径 + 7 步手工脚本——**12★ 的根因不是没推广，是跑不起来**。
- **前提3·技术栈跟上 2026**：Wan2.1/FLUX.1/InstantCharacter 均落后一代；InstantCharacter 最后 push 2025-05。旧栈没人用也没人看教程（ShortGPT 即前车之鉴）。

### 三个"不做"
- 不刷星：先产品后宣发（P0 不完成不宣发）。
- 不追新模型本身：与官方仓库比模型无意义，做可插拔接入层。
- 不承诺"电影级全自动"：主打"稳定可发的短叙事/营销片"，管理预期。

## 二、仓库现状审计（为什么是 12★）

| 维度 | 现状 | 影响 |
|---|---|---|
| 更新 | 最后实质提交 2025-07-24，停更 13 个月 | 错过 Wan2.2/2.5、HunyuanVideo 1.5、FLUX.2、LTX-2、MiniMax H3 等全部发布 |
| 技术栈 | Wan2.1(vendor) + FLUX.1-dev + InstantCharacter(停更) + Kokoro + o4-mini | 落后一代；核心一致性组件停更 |
| 流水线 | UI 只有 4 个 tab，视频/音频/字幕不在 UI；字幕阶段仓库内未实现 | "一键成片"名不副实 |
| 上手 | README 引用不存在的 requirements.txt；12 处 `/data/home/lizhijun` 硬编码；2 处硬编码 IP | 新用户第一步就失败 |
| 代码质量 | 三个重复的 ImageGenerator 类 + try/except TypeError hack | 维护负担重、贡献者难以下手 |
| 工程设施 | 无 CI/测试/Docker/uv/HF Space/ComfyUI 节点/GitHub topics | 无信任感、无生态入口 |
| 资产 | MIT、双语 README、2 张架构图、demo 成片、Gradio 雏形 | 可复用 |

## 三、市场与竞争（详见 docs/market_research_2026-08.md）

- **直接竞品**：Pixelle-Video ≈27k★（ComfyUI 工作流引擎 + FLUX 图像 + Wan2.1 视频 + edge-tts，RunningHub 云，火柴人风格规避一致性，Windows 整合包）；VideoClaw ≈1.7k★（AI 导演系统，模型可切换，WebUI 一键安装，官方 B站/YouTube）。
- **心智占位者**：MoneyPrinterTurbo 103k★（混剪式，非生成式）；Toonflow 13.9k★、BigBanana 1.7k★、LocalMiniDrama 1.3k★（短剧平台）。
- **上游**：Wan2.1 16.8k★ / Wan2.2 17.1k★、HunyuanVideo 12.4k★、LTX-Video 10.9k★、Open-Sora 29.3k★、ComfyUI 127k★（分发渠道）。
- **一致性组件**：InstantCharacter 1k★（停更）、PhotoMaker 10k★、ACE++ 1.4k★、UMO 190★（新）。
- **新赛道**：视频智能体（UniVA、MovieAgent、南洋理工分层 agent、vivago R1）；长视频（JoyAI-Echo 5 分钟成片）。

## 四、技术栈升级方案（2026 版，详见 docs/model_stack_research_2026-08.md）

核心原则：**可插拔适配器架构，不绑定单一模型；每阶段三档（轻量本地/高质本地/云 API 兜底）。**

| 阶段 | 现栈 | 2026 推荐（主线/备选） | 许可证 |
|---|---|---|---|
| 剧本 LLM | DeepSeek/GPT-4 | DeepSeek V3.2-Exp/V4-Flash API；本地 Qwen3-32B | MIT/Apache |
| 图像+一致性 | FLUX.1-dev + InstantCharacter(停更) | **Qwen-Image（20B，中文强）或 HiDream-I1（17B）或 FLUX.2 klein 4B（Apache-2.0，原生多参考编辑）** + LoRA/PuLID 一致性路线；一致性层抽象接口，观察 UMO | Apache-2.0/MIT（FLUX.2-dev 非商用勿选） |
| 图生视频 | Wan2.1-I2V-14B-480P | **Wan 2.5 I2V-14B-720P（确定 Apache-2.0，生态最稳，VACE 支持音频条件/续写）**；观察 Wan2.6（开源状态冲突）与 MiniMax H3（音视一体但限地域+8卡）；低配 Wan2.5-5B/HunyuanVideo 1.5 | Apache-2.0 |
| 音效/音画同生 | 无 | MiniMax H3（受限）/ MOVA（Apache-2.0，360p 起步）/ LTX-2（商用受限）三选一；退而求其次 Wan2.5 VACE 音频条件 + 音效轨（MusicGen/Elefant） | 视方案 |
| 提示词增强 VLM | o4-mini/qwen2.5-vl | **Qwen3-VL-8B/32B 本地**，o4-mini 兜底 | Apache-2.0 |
| TTS | Kokoro-82M | **Qwen3-TTS（Apache-2.0，流式）或 CosyVoice 3（4GB 显存）**；极致中文情感选 IndexTTS-2（自定义许可需确认） | Apache-2.0 |
| 字幕（补缺失阶段） | 无 | **SenseVoice 或 Qwen3-ASR**（中文快准）+ whisper-large-v3-turbo 兜底 | Apache-2.0 |
| 总装/长视频 | FFmpeg 脚本 | FFmpeg 保留；**评估 JoyAI-Echo**（5 分钟长片、角色不崩）替代纯拼接 | 待核实 |

两大待核实项（落地前确认）：① Wan 2.6 权重是否真开源（信源冲突）；② MiniMax H3 / JoyAI-Echo 许可证全文与显存。

## 五、建议定位（"吸引更多人"的根本）

> **"本地优先 · 生成式的故事/营销视频编排框架"——从搜索词到成片的开源垂直视频生成智能体；画面真正 AI 生成（非混剪）、100% 本地可跑、模块可插拔、每步可复现、文档教学化。**

- 一句话口径："输入搜索词，本地自动生成一部角色一致、有旁白字幕的营销/故事短片。"
- 三张差异牌：① 生成式 vs MoneyPrinterTurbo 混剪；② 全本地 vs Pixelle/VideoClaw 云依赖；③ 轻量透明 vs Toonflow/BigBanana 重平台。
- 叙事升级：从"流水线"讲到"垂直视频生成智能体"，对齐 2026"一人剧组"趋势。

## 六、90 天路线图（详见 docs/SearchVidGen-开源增长实战报告-2026-08.md）

- **P0（第1-2周，不完成不宣发）**：main.py 一键 + config.yaml/.env + setup.py 模型自动下载（含 ModelScope/hf-mirror 镜像）+ Gradio 全流程 + 补字幕阶段 + 去硬编码/三合一 ImageGenerator + requirements/uv。
- **P1（第3-6周）**：技术栈升级（Wan2.5 接入 + 一致性层抽象 + Qwen3-TTS/SenseVoice）+ 3-5 条成片 demo + README 重构（首屏视频/badges/FAQ/News）+ HF Space & 魔搭创空间在线 demo + 首轮中英文宣发（Show HN/Reddit/X + V2EX/知乎/B站）+ 提交收录清单。
- **P2（第7-13周）**：差异化内容（角色一致性对比视频、vs MPT 划界）+ ComfyUI 工作流版 + 一键安装（uv/docker/Pinokio，Windows 整合包量力）+ 社区（Discord/微信群/CONTRIBUTING）+ 每 1-2 周 changelog + 持续接入新模型。
- **务实指标**：star 12 → 300-1000+；demo 访问 >1万；B站教程播放 >10万；UGC 成片 ≥20 条。

## 七、风险
- 竞品已占位且迭代快（Pixelle 半年 2.2 万★靠"整合包+教程+更新频率"三件套），90 天内必须做出差异化，否则宣发=给竞品送对比素材；
- 上游停更/换代风险（InstantCharacter 已发生）→ 适配器架构是唯一解；
- 显存门槛 → 三档配置 + 魔搭创空间免费 GPU（Ada 48G 可申请）；
- "开源+音画同生+全地域商用"三者当前不可兼得 → 音画同生先做可选适配器，主线用"视频+独立音频"管线；
- 个人精力 → 定位"小而实"，拒绝平台化军备竞赛。
