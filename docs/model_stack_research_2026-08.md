# SearchVidGen 2026 开源技术栈调研报告（截至 2026-08）

> 说明：所有日期/许可证/参数量均来自检索信源并附链接；硬件数字部分为社区实测口径（标注"约/估计"），未查到的明确写"未查到"。**信源冲突处已特别注明。**

## ⚡ 重点问题先行回答

### Q1：Wan 2.6 / 2.7 到底哪个是开源权重？——**信源冲突，结论如下**

| 版本 | 发布时间 | 开源权重状态 | 依据 |
|---|---|---|---|
| **Wan 2.5** | 2025-08（[官方发布页](https://tongyi.aliyun.com/news?id=pxwhvf%2Fsuodqg%2Fusofpg66drzae5s1)） | ✅ 确定开源，**Apache 2.0**，T2V/I2V 14B-480P/720P + 5B 系列（[AI Wiki](https://aiwiki.ai/wiki/wan_2_5)、[Wan 综合页](https://vantaige.io/ai-tool/wan)） | 最稳的"确定开源"基线 |
| **Wan 2.6** | 2025-12-16（[新华社/千问APP接入](https://www.news.cn/tech/20251217/e6386eb39f4546b081511973fd7fa00c/c.html)、[阿里官方新闻稿](https://www.alibabacloud.com/zh/press-room/alibaba-unveils-wan2-6-series-enabling-everyone)） | ⚠️ **多数第三方资料称"开源权重/开源"**（[VizStudio](https://vizstudio.art/zh/wan-2-6)、[Vibedex 评测](https://vibedex.ai/blog/wan-26-review-2026)、[llmreference 标注 "multimodal, open weights"](https://www.llmreference.com/model/wan-2.6)、[Cliprise](https://www.cliprise.app/news/alibaba-wan-2-2-2-6)、[Flowith](https://flowith.io/blog/wan-2-6-3-0-open-weight-architecture-democratize-ai-film-production/)）；**但有信源直接质疑**（[wan27.org "Is Wan 2.6 Open Source?"](https://wan27.org/blog/wan-2-6-open-source-guide)），且本次未能直接抓取到 HF/ModelScope 权重页验证 | **倾向"已开源"但存在冲突**，落地前务必上 [Wan-Video 官方 GitHub](https://github.com/Wan-Video) / HF 复核仓库是否存在 |
| **Wan 2.7** | 2026 年发布（首帧控制、15 秒片段，[Seedance20 介绍](https://seedance20.net/zh/blog/wan-2-7-video-generator)） | ❌ **未开源**——头条信源明确"**这次不开源了**"（[通义千问发布 WAN 2.7 图生视频](https://m.toutiao.com/article/7624570432846299658/)）；wan27.org / seedance20 等营销站称"未来会开源"，不可靠 | **2.7 不可用** |

### Q2：LTX-2.5 的开源状态与许可？
**未查到可靠信源**。官方已发布的是 **LTX-2**（2026-01-06 开源权重，[官方新闻稿](https://www.globenewswire.com/fr/news-release/2026/01/06/3213304/0/en/Lightricks-Open-Sources-LTX-2-the-First-Production-Ready-Audio-and-Video-Generation-Model-With-Truly-Open-Weights.html)）和 **LTX-2.3**（2026-03 发布，[Gigazine](https://gigazine.net/gsc_news/en/20260306-ltx-2-3-video-generation-ai/)、[HF 仓库](https://huggingface.co/Lightricks/LTX-2.3)）。"LTX-2.5" 仅见于 [Pexo 一篇标题](https://pexo.ai/blog/ltx-2-5-alternatives-5267)，内容不可信，**判定为未查到/未证实**。

**LTX-2 许可证（重要提醒）**：宣传口径是 "truly open weights"，但 **HF 社区明确标注"因商用限制不属于开源/开放权重，'shared weights' 更合适"**（[HF commit](https://huggingface.co/Lightricks/LTX-2/commit/1ebd7cfe3855674870913a1e158f9b60b2759ab8)、[HF discussion #21](https://huggingface.co/Lightricks/LTX-2/discussions/21/files)）→ **自定义社区许可、商用受限，非 Apache/MIT**（[WaveSpeed 许可分析](https://wavespeed.ai/blog/posts/blog-ltx-2-license-commercial-use/)）。

---

## 1. 视频生成（重点）

### 候选模型

| 模型 | 发布时间 | 参数量 | 许可证 | 硬件门槛 | 备注 |
|---|---|---|---|---|---|
| **Wan 2.5** | 2025-08 | 14B（T2V/I2V，480P/720P）+ 5B | **Apache 2.0** | 5B 单卡 16-24GB；14B 需 40GB+ 或多卡 | 当前最稳开源基线；ComfyUI/diffusers 生态成熟；中文提示词好；VACE 支持音频条件、视频续写（[ComfyUI WanVideoWrapper](https://deepwiki.com/kijai/ComfyUI-WanVideoWrapper/11.2-vace-context-extension)） |
| **Wan 2.6** | 2025-12 | 未精确核实（14B 级） | 多数信源称开源（冲突见上） | 未查到 | 多模态、电影级叙事方向（[pconline](https://www.pconline.com.cn/ai/2038/20387813.html)） |
| **Wan 2.2** | 2025-05 | 14B/5B | Apache 2.0 | 同 Wan2.5 | 曾用于 VACE 系列；已被 2.5 取代 |
| **HunyuanVideo 1.5** | 2025-11-21（[腾讯新闻](https://news.qq.com/rain/a/20251121A03RFX00)） | 轻量级（未精确核实） | **腾讯混元社区许可**（[已抓取全文](https://gitserver.onethingai.com/ai-models/HunyuanVideo-1.5/raw/commit/abcdc0cc5a6dc75ae65978d9cbea2f859411d4b9/LICENSE)：EU/UK/韩国禁用；**月活<1 亿可免费商用**；禁止用于改进其他 AI 模型） | **消费级显卡可部署**（[InfoQ](https://www.infoq.cn/article/SVO2eO7PSqhV91jTskOr)） | 支持 T2V+I2V、多分辨率；轻量是其卖点 |
| **LTX-2** | 2026-01-06 | 未精确核实 | **自定义社区许可，商用受限**（非 OSI 开源，见 Q2） | 4K、宣称生产级（[官方](https://www.globenewswire.com/fr/news-release/2026/01/06/3213304/0/en/)）；社区称消费级可跑 | 原生**视频+音频**联合生成，是其最大差异化 |
| **LTX-2.3** | 2026-03 | 未精确核实 | 同 LTX-2（自定义） | 官方推免费桌面应用（[Gigazine](https://gigazine.net/gsc_news/en/20260306-ltx-2-3-video-generation-ai/)） | 新增 I2V 等能力 |
| **SkyReels-V2** | 2025-04 | I2V-1.6B-480P / I2V·T2V-**14B-720P**（diffusers 版） | **Apache 2.0**（[HF 元数据](https://huggingface.co/api/resolve-cache/models/phazei/phazei-SkyReels-V2-fp8-e5m2/...)） | 1.6B 单卡消费级；14B 约 40-80GB | **扩散强迫（diffusion forcing）+无限时长**，30/40 秒片段；[GitHub](https://github.com/SkyworkAI/SkyReels-V2)、[量子位](https://www.qbitai.com/2025/04/275232.html) |
| **MiniMax H3**（2026 重磅） | 2026-07 前后开源（[官方公告](https://www.minimaxi.com/news/minimax-h3-open-source)、[36氪](https://www.36kr.com/p/3922353035433603)） | **33B 联合视频+音频扩散（全模态）** | 开源但**限制美/欧/英使用**（[mer.vin](https://mer.vin/news/minimax-h3-open-weight-video-model-blocked-us-eu-uk/)） | 实测 **8×RTX 6000D + vLLM**（[CSDN 部署实测](https://blog.csdn.net/b_bencom/article/details/163557582)）；量化/MLX 变体降低门槛（[MLX 移植](https://github.com/PipeNetwork/minimax-h3-mlx)） | 2K 视频+音频同步；"视频界的 DeepSeek 时刻"（[雷峰网](https://www.leiphone.com/category/industrynews/aiUMBoeUYbi8fX4x.html)） |
| **HappyHorse-1.0/1.1**（阿里） | 2026-04 开源（[DoNews](https://www.donews.com/news/detail/1/6503717.html)、[1.1 动态](https://developer.aliyun.com/article/1743009)） | 未查到 | 未查到（阿里系惯例 Apache） | 未查到 | 强调"会动的故事板"输入 |
| **Mochi 1** | 2024-10 | 10B | Apache 2.0（[GitHub](https://github.com/genmoai/mochi)） | 80GB 级 | 已过时；Mochi 2 **未查到** |
| **CogVideoX 1.5 / 5B-I2V** | 2025 下半年 | 5B 级 | Apache 2.0 | 5B 单卡 24GB 级 | 1.5 支持 **4K + 音效联动**（[GitCode 介绍](https://blog.gitcode.com/505c4e2e33302aea457c7e9369f65cdc.html)）；**CogVideoX 2.0 未查到** |
| **Step-Video-T2V / T2V-Turbo / TI2V** | 2025-02 起 | **30B** | 开源（许可证未在本次检索确认） | 30B 需多卡 80GB 级 | 曾为"全球最大开源视频模型"（[BAAI](https://hub.baai.ac.cn/view/43465)）；TI2V 支持可控运镜（[GitCode](https://blog.gitcode.com/cabde4e5c6142dd4c1a7e707d6142972.html)） |
| **Open-Sora 2.x** | 2024-2025 | 11B 级 | Apache 2.0（[HF](https://huggingface.co/hpcai-tech/Open-Sora-v2)） | 多卡 | 2025 后**未见重大更新** |

### ✅ 推荐（视频生成）
**主选 Wan 2.5（Apache 2.0，I2V-14B-720P，保留 torchrun 多卡或换 ComfyUI）**；这是目前"许可证干净 + 中文好 + I2V/首尾帧/VACE 音频条件 + 生态最全"的确定开源选择，且能无缝衔接现有 InstantCharacter 角色图方案。
**升级观察位**：① Wan 2.6（先核实权重是否真开源）；② MiniMax H3（33B 音视频一体，但需评估许可证地域限制与 8 卡成本）；③ SkyReels-V2（做 30s+ 长片段）。HunyuanVideo 1.5 适合低硬件预算，但注意其"禁止用于改进其他 AI 模型+地域限制"条款。

---

## 2. 音视频协同

### 候选

| 方案 | 类型 | 许可证 | 硬件 | 备注 |
|---|---|---|---|---|
| **MiniMax H3** | 视频+音频**联合原生生成**（33B） | 开源但美/欧/英受限 | 8×RTX 6000D 级 | 2026 最强音视频一体开源（[官方](https://www.minimaxi.com/news/minimax-h3-open-source)、[AMD 日 0 支持](https://www.amd.com/en/developer/resources/technical-articles/2026/day-0-support-for-minimax-h3-on-amd-gpus.html)） |
| **MOVA（复旦 OpenMOSS）** | 视频+音频联合生成（360p/720p） | **Apache 2.0**（[HF 元数据](https://huggingface.co/OpenMOSS-Team/MOVA-360p)） | 360p 较友好；720p 需更高显存（未查到精确值） | "开源版 Seedance 2.0"，国内首个开源高质量音视频模型（[官网博客](https://openmoss.ai/blog/cn/mova/)、[专访](https://www.163.com/dy/article/KMR3P2AP055040N3.html)） |
| **LTX-2 / 2.3** | 视频+音频联合生成（4K） | 自定义许可、商用受限 | 消费级可跑（社区口径） | 与 MOVA/H3 并列的"音画同生"三选一（[WaveSpeed 对比](https://wavespeed.ai/blog/posts/mova-vs-wan-sora-seedance-video-audio-comparison-2026/)） |
| **Wan 2.5 VACE 音频条件** | 音频驱动/口型（非联合生成） | Apache 2.0 | 同 Wan2.5 | 音频→视频/对口型，ComfyUI 教程成熟（[Apatero](https://www.apatero.com/blog/wan-2-5-audio-driven-video-generation-complete-comfyui-guide-2025)、[唇形同步解读](https://wan3api.com/zh/blog/lip-sync-technology-wan-2-5)） |
| **JavisDiT++** | 音视频联合生成（研究） | 未查到（论文开源） | 未查到 | ICLR 2026（[论文页](https://iclr.cc/virtual/2026/poster/10008062)） |
| **TalkVerse** | 音频驱动分钟级视频 | 开源（研究） | 未查到 | CVPR 2026（[论文](https://openaccess.thecvf.com/content/CVPR2026F/html/Wang_TalkVerse_Democratizing_Minute-Long_Audio-Driven_Video_Generation_CVPRF_2026_paper.html)） |
| **音乐/音效生成** | MusicGen（MIT）、Stable Audio Open（SAO 非商用许可）、**Elefant**（HF，1B）、**YuE**（开源音乐生成）、ACE-Step（字节） | 混用 | 单卡 24GB 级 | 2026 可部署清单（[Spheron](https://www.spheron.network/blog/deploy-open-source-ai-music-generation-gpu-cloud-2026/)）；许可"地震"综述见 [TraeAI](https://learn.traeai.com/t/ai-engineering/phases/06-speech-and-audio/09-music-generation) |

### ✅ 推荐
**若接受多卡成本：MiniMax H3 一步到位（视频+配音+音效原生同步）**；否则用 **Wan 2.5 VACE 音频条件做口型/驱动 + 单独音效轨（MusicGen/Elefant）+ FFmpeg 混音**的管线式方案，MOVA 作为 Apache 2.0 下的折中尝试（360p 起步）。注意：**截至检索时"开源 + 音画同生 + 全地域商用"三者不可兼得**（H3 限地域、LTX-2 限商用、MOVA 画质偏低）。

---

## 3. 图像生成与角色一致性

### 候选

| 模型 | 发布时间 | 许可证 | 硬件 | 备注 |
|---|---|---|---|---|
| **FLUX.2-dev** | 2025-11 | **FLUX.2-dev 非商用许可：年收入 <$100 万可商用，非 OSI 开源**（[HF LICENSE](https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/main/LICENSE.md)、[BFL 官方许可说明](https://help.bfl.ai/articles/9272590838-self-serve-dev-license-overview-pricing)） | 32B 级；fp8 约 24GB 可推理（社区口径，未逐项核实） | 文本渲染/编辑强 |
| **Qwen-Image / Edit** | 2025-08 | **Apache 2.0**，20B（[GitHub](https://github.com/QwenLM/Qwen-Image)） | 20B fp8 约 24GB | 中文文本渲染最好之一；**2026 已出 Qwen-Image 2.0**（生成+编辑合一，[aitop100](https://www.aitop100.cn/infomation/details/33310.html)） |
| **SD3.5** | 2024-10 | Stability 社区许可（自定义、商用限制） | Medium 2.5B 单卡 | 2026 已非前沿（[对比文](https://aifoss.dev/blog/flux-vs-sdxl-vs-sd35-2026/)） |
| **HiDream-I1** | 2025-03 | **MIT**（[Hivebook](https://hivebook.wiki/wiki/hidream-i1-open-17b-sparse-dit-image-model-mit)） | 17B 稀疏 DiT，24GB 级 | ComfyUI 原生支持（[官方工作流](https://docs.comfy.org/zh/tutorials/image/hidream/hidream-i1)） |
| **InstantCharacter**（腾讯+InstantX） | 2025-04 | 开源（腾讯/InstantX 许可） | 依赖底座模型 | IP-Adapter 方案，多图一致性强（[量子位](https://www.qbitai.com/2025/04/276754.html)、[ComfyUI 节点](https://comfyai.run/custom_node/ComfyUI-InstantCharacter)）；**2026 维护动态未查到** |
| **IP-Adapter for FLUX.2** | — | — | — | **未查到** InstantX 官方 FLUX.2 适配版 |
| **OmniGen2 / ACE++ / PhotoMaker** | — | — | — | **均未查到 2026 新进展**（OmniGen 评测分 68/100，[nolist](https://nolist.ai/item/omnigen)） |
| **UMO（字节，CVPR 2026）** | 2026 | 开源（[GitHub](https://github.com/bytedance/UMO)） | 未查到 | 多身份一致性图像定制，匹配奖励训练，含 ComfyUI workflow |

### ✅ 推荐
**图像底座：Qwen-Image（Apache 2.0，中文文本强）或 HiDream-I1（MIT）**；FLUX.2-dev 效果更好但许可证门槛（<$100 万营收）需评估。**角色一致性**：2026 社区主流已从"IP-Adapter 单方案"转向 **LoRA 训练 + PuLID/InstantID/多参考图**组合（[2026 年中角色一致性横评](https://nowaythisisai.com/blog/character-consistency-fictional-characters-mid-2026)）；InstantCharacter 仍可用但建议并列评估 UMO 与 LoRA 路线。

---

## 4. TTS（中文优先）

| 模型 | 发布时间 | 许可证 | 硬件/成本 | 备注 |
|---|---|---|---|---|
| **Qwen3-TTS** | 2025-12 | **Apache 2.0**（[HF 元数据](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)） | 0.6B 单卡 8-16GB；**97ms 流式延迟**（[chinaz](https://m.chinaz.com/ainews/24883.shtml)） | 3 秒克隆+一句话设计音色；全家桶开源（[IT之家](https://m.ithome.com/html/915616.htm)） |
| **CosyVoice 3（Fun-CosyVoice3-0.5B）** | 2025-12 | **Apache 2.0**（[HF](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)） | **4GB 低显存**（[V2EX 实测](https://global.v2ex.co/t/1179310)） | 多语种零样本克隆（[头条](https://m.toutiao.com/article/7584110459121402402/)） |
| **IndexTTS-2** | 2025 年末 | 自定义 INDEX_MODEL_LICENSE（[许可文件](https://github.com/tabortao/index-tts2/blob/main/INDEX_MODEL_LICENSE)） | 单卡 24GB 级 | 中文效果顶级：情绪/停顿/多音字控制（[GitHub](https://github.com/index-tts/index-tts)） |
| **F5-TTS** | 2024-2025 | MIT | 单卡消费级 | 零样本克隆，中文尚可（[GitCode](https://blog.gitcode.com/f755d07e4800ebcfa72393f811feb92e.html)） |
| **Kokoro-82M（现栈）** | 2024-2025 | Apache 2.0 | 极轻量 CPU/小显存 | 中文支持有限，仅适合低成本场景（[2026 TTS 指南](https://tts.ai/blog/open-source-text-to-speech-guide-2026/)） |
| **ChatTTS / GPT-SoVITS** | 2024-2025 | ChatTTS 模型权重非商用（CC BY-NC）/ GPT-SoVITS MIT | 单卡消费级 | 仍是中文克隆常用备选（[实战指南](https://blog.csdn.net/2600_94960196/article/details/159417976)） |

### ✅ 推荐
**主选 Qwen3-TTS（Apache 2.0 + 流式 + 中文强）或 CosyVoice 3（更省显存）**；追求极致中文情感/多音字可上 IndexTTS-2（注意查自定义许可商用条款）。Kokoro 中文可以退休。

---

## 5. ASR / 字幕

| 方案 | 许可证 | 硬件 | 备注 |
|---|---|---|---|
| **Qwen3-ASR（2026-01 开源）** | Apache 2.0（官方称"可免费商用"） | 轻量（未精确核实） | **52 语种/方言，1.7B 版达 SOTA**（[IT之家](https://m.ithome.com/html/917675.htm)、[C114](https://www.c114.net.cn/ainews/56546.html)） |
| **FunASR / Paraformer（阿里）** | Apache 2.0 | CPU 可跑/单卡 | 中文工业级，实时/离线齐全；持续维护（[GitHub](https://github.com/modelscope/FunASR)、[vs Whisper 实测讨论](https://github.com/modelscope/FunASR/discussions/2947)） |
| **SenseVoice（阿里）** | Apache 2.0 | CPU/小显存，极快 | 中日韩语转写性价比之王（[whispernotes 评测](https://whispernotes.app/blog/sensevoice-fastest-cjk-transcription)） |
| **Whisper large-v3-turbo + faster-whisper / whisper.cpp / transcribe.cpp** | MIT | CPU/单卡 | 仍是通用字幕标准工具（[2026 中文教程](https://most.tw/posts/ainews/whisper-speech-to-text-guide/)、[transcribe.cpp](https://github.com/handy-computer/transcribe.cpp/blob/main/docs/models/whisper-large-v3-turbo.md)） |

### ✅ 推荐
**中文生成字幕用 SenseVoice 或 Qwen3-ASR（快+准+免费商用）；对白转写/多语种用 whisper large-v3-turbo（faster-whisper）兜底**。VideoCaptioner 类工具仍可用作包装层。

---

## 6. LLM / VLM 编排

| 模型 | 发布时间 | 许可证 | 备注 |
|---|---|---|---|
| **DeepSeek V3.2-Exp** | 2025-12（[知定](https://maker.zhiding.cn/2025/1202/3174667.shtml)） | MIT 系（DeepSeek 一贯；未单独核实） | 线性注意力长文本+深度思考 |
| **DeepSeek V4-Pro / V4-Flash** | **2026-08** | **MIT**（[OSFY](https://www.opensourceforu.com/2026/08/deepseek-open-sources-v4-flash/)、[AgentRQ](https://agentrq.com/blog/deepseek-v4-pro)）；个别信源写 Apache 2.0（[ofox](https://ofox.ai/blog/deepseek-v4-release-guide-2026/)）——**MIT/Apache 冲突待核实，但均为宽松许可** | 1.6T MoE、100 万上下文、原生多模态（[腾讯云解读](https://developer.cloud.tencent.cn/article/2663824)） |
| **Qwen3.8-Max（阿里）** | 2026-08-04（[至顶网](https://www.zhiding.cn/models/2026/0804/3195272.shtml)） | **2.4T 参数，官方称"下周开源权重"** | 长程任务/自动编程定位 |
| **Qwen3 / Qwen3-VL（2B/4B/8B/32B/235B-A22B）** | 2025-09 | Apache 2.0 | 本地部署性价比标杆；32B 版可 vLLM/SGLang 部署（[SiliconFlow 模型页](https://www.siliconflow.com/zh-tw/models/qwen3-vl-32b-instruct)、[刘聪博客](https://blog.csdn.net/csdn_xmj/article/details/152091918)） |

### ✅ 推荐
- **剧本生成（LLM）**：API 走 **DeepSeek V3.2-Exp / V4-Flash**（便宜、中文强）；本地 24-48GB 单机走 **Qwen3-32B/235B-A22B**（Apache 2.0）。
- **"看图写视频提示词"（VLM）**：本地 **Qwen3-VL-8B/32B** 性价比最高（Apache 2.0）；追求上限再用 o4-mini 等闭源 API。**结论：qwen2.5-vl 可升级到 Qwen3-VL**。

---

## 7. 视频一致性 / 长视频（2026 新方案）

| 方案 | 类型 | 备注 |
|---|---|---|
| **JoyAI-Echo（京东，2026 开源）** | **长音视频框架：5 分钟成片、角色不崩、声音稳定、对话式局部编辑**（[GitHub](https://github.com/jd-opensource/JoyAI-Echo)、[HF](https://huggingface.co/jdopensource/JoyAI-Echo)、[C114](https://www.c114.net.cn/industry/88062.html)） | **当前最贴合"多镜头角色一致+长视频"的开源框架**；许可证/参数量未在本次检索核实 |
| **SkyReels-V2 扩散强迫** | 无限时长/30-40s 片段 | 长片段生成（见第 1 节） |
| **MiniMax H3** | 2K 音视频长生成 | 见第 1 节 |
| **DiTCtrl（港中文+腾讯）** | MM-DiT 多提示无训练长视频（[arXiv 2412.18597](https://ar.library.dctabudhabi.ae/eds/detail?db=edsarx&an=edsarx.2412.18597)、[51CTO](http://51cto.com/aigc/3504.html)） | Wan 系底座，多镜头切换 |
| **ConsisID（北大，CVPR 2025 Highlight）** | 单图身份保持 T2V、免微调（[GitHub](https://github.com/PKU-YuanGroup/ConsisID)） | 仍被引用；2026 更新未查到 |
| **EchoShot（2506.15838）/ FlashPortrait（CVPR 2026）** | 多镜头人像 / 无限时长+6×加速（[arXiv](https://ar5iv.labs.arxiv.org/html/2506.15838)、[GitHub](https://github.com/Francis-Rings/FlashPortrait)） | 研究级新方案 |
| **UMO（字节，CVPR 2026）** | 多身份一致性（图像侧） | 见第 3 节 |

### ✅ 推荐
**2026 年"多镜头角色一致+长视频"不再靠拼装：优先评估 JoyAI-Echo 框架**；退而求其次用 **Wan2.5 + VACE 视频续写 + InstantCharacter/LoRA 角色图** 的管线，配合 DiTCtrl 思路做多提示切换。StoryMaker 等 2024 年项目**未查到 2026 更新**。

---

## 📌 给 SearchVidGen 的总升级建议（速览）

| 阶段 | 现栈 | 建议升级 | 许可证变化 |
|---|---|---|---|
| 剧本 LLM | DeepSeek/GPT-4 | DeepSeek V3.2-Exp / V4-Flash API，或 Qwen3-32B 本地 | 不变（MIT/Apache） |
| 图像/角色 | FLUX.1-dev + InstantCharacter | **Qwen-Image 或 HiDream-I1 + LoRA/PuLID 组合**（FLUX.2-dev 需评估营收门槛） | FLUX.1/2-dev 均非 OSI 开源 |
| 图生视频 | Wan2.1-I2V-14B-480P | **Wan 2.5 I2V-14B-720P**（Apache 2.0，生态最稳）；观察 Wan 2.6 权重与 MiniMax H3 | Wan2.5 确定 Apache |
| 提示词增强 | o4-mini/qwen2.5-vl | **Qwen3-VL-8B/32B 本地** | Apache 2.0 |
| TTS | Kokoro-82M | **Qwen3-TTS 或 CosyVoice 3（均 Apache 2.0）**；重情感选 IndexTTS-2 | 更宽松 |
| 字幕 | VideoCaptioner/未实现 | **SenseVoice 或 Qwen3-ASR + whisper-turbo 兜底** | Apache 2.0 |
| 拼接/长视频 | FFmpeg | 评估 **JoyAI-Echo**（5 分钟长片框架）替代纯拼接；音画同步可选 MOVA/H3 | 待核实 |

**两大待核实项**：① Wan 2.6 权重是否真开源（信源冲突，以官方 Wan-Video GitHub/HF 为准）；② MiniMax H3 与 JoyAI-Echo 的许可证全文与显存需求（本次仅见二手信源）。

---

> 调研方法：20+ 次中英文 web_search（覆盖 Wan/HunyuanVideo/LTX/Open-Sora/SkyReels/Step/Mochi/CogVideoX/MiniMax H3/HappyHorse/MOVA/JoyAI-Echo、VACE 音频、FLUX.2/Qwen-Image/SD3.5/HiDream/InstantCharacter/UMO、Qwen3-TTS/CosyVoice3/IndexTTS-2/F5-TTS/Kokoro、Qwen3-ASR/SenseVoice/FunASR/Whisper、DeepSeek V3.2/V4/Qwen3-VL、DiTCtrl/ConsisID/EchoShot/FlashPortrait）+ 一次许可证原文抓取（HunyuanVideo-1.5 = 腾讯混元社区许可全文）。未查到的项均已如实标注"未查到"。
