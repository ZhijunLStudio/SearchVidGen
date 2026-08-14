# SearchVidGen 市场定位调研报告（2026-08）

> 调研时间：2026年8月14日。star 数为当日通过 GitHub API / 页面抓取所得（精确值），版本/日期来自 web 搜索，均附来源链接。凡未查到者明确标注"未查到"。

---

## 一、竞品全景表

### A. 官方模型仓库（模型层，SearchVidGen 的上游供应商而非直接竞品）

| 项目 | GitHub | star* | 定位/能力 | 许可证 | 活跃度 |
|---|---|---|---|---|---|
| Wan2.1（阿里通义万相） | [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) | 16,821 | T2V/I2V 旗舰开源模型，2025年发布 | Apache-2.0 | 2026-03 仍更新 |
| Wan2.2 | [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) | 17,116 | 续作（含 S2V 数字人、[Wan2.2-Animate 动作生成](https://tech.ifeng.com/c/8mmBBYgpjCY?ch=ttsearch)等） | Apache-2.0 | 2026-03 仍更新 |
| Wan 2.5 Preview | [百度百科：通义万相Wan2.5 Preview](https://baike.baidu.com/item/%E9%80%9A%E4%B9%89%E4%B8%87%E7%9B%B8Wan2.5%20Preview/67267632) | — | 更高质量续代（[通义App免费生成10秒视频](https://www.php.cn/faq/1917244.html)），精确发布时间未查到 | — | 2026 上半年热点 |
| HunyuanVideo（腾讯混元） | [Tencent-Hunyuan/HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo) | 12,422 | 视频生成系统框架；[1.5 版 2025-11-21 开源，消费级显卡（约14G显存）可跑](https://news.qq.com/rain/a/20251121A03RFX00?suid=&media_id=) | 腾讯自定义 | 2026-06 仍更新 |
| HunyuanVideo-I2V | [Tencent-Hunyuan/HunyuanVideo-I2V](https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V) | 1,839 | 图生视频 | 自定义 | 2026-04 更新 |
| LTX-Video（Lightricks） | [Lightricks/LTX-Video](https://github.com/Lightricks/LTX-Video) | 10,850 | 快速视频生成 | Apache-2.0 | 2026-01 更新 |
| LTX-2 | [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2) | ~8,982 | [2025-10 发布，音视频一体、4K 生成](https://www.opensourceforu.com/2025/10/lightricks-launches-ltx-2-to-democratise-4k-ai-video-production/) | 未查到 | 活跃 |
| Open-Sora | [hpcaitech/Open-Sora](https://github.com/hpcaitech/Open-Sora) | 29,273 | 复刻 Sora、高效视频生产 | Apache-2.0 | 2026-04 更新 |
| CogVideoX（智谱） | [zai-org/CogVideo](https://github.com/zai-org/CogVideo)（原 THUDM/CogVideo 已迁移） | 12,954 | [v1.5 支持带声音视频](https://www.163.com/dy/article/JGG2VOLN0511B8LM.html?referFrom=) | Apache-2.0 | 2025-11 更新 |
| SkyReels 系列（昆仑万维） | [SkyworkAI/SkyReels-V2](https://github.com/SkyworkAI/SkyReels-V2)（V1: 2,692★ / V2: 7,332★ / A1: 582★ / [V3: 535★](https://github.com/SkyworkAI/SkyReels-V3)） | 见左 | [V2 主打 AI 短剧、无限时长电影，2025-04 开源](http://stock.10jqka.com.cn/20250422/c667649195.shtml)；[V3 多模态全能，2026-01-29 开源](https://finance.eastmoney.com/news/1354,202601293635635435.html) | 自定义 | V2 2026-01 更新 |
| Mochi 1 | [genmoai/mochi](https://github.com/genmoai/mochi) | 3,705 | Genmo 开源视频模型 | Apache-2.0 | 2025-11 更新 |
| VACE（阿里） | [ali-vilab/VACE](https://github.com/ali-vilab/VACE) | 3,916 | 一站式视频创建与编辑（[README](https://raw.githubusercontent.com/ali-vilab/VACE/master/README.md)） | 未查到 | 活跃 |
| ComfyUI（生态底座） | [comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI) | ~127,466 | 节点化 AI 生成工作流引擎 | GPL-3.0 | 极活跃 |

### B. 编排/成片流水线（应用层，SearchVidGen 的直接竞争带）

**B1. AI 短剧 / 漫剧 / 故事视频类**

| 项目 | GitHub | star* | 定位 | 上手方式 | 许可证 |
|---|---|---|---|---|---|
| Toonflow | [HBAI-Ltd/Toonflow-app](https://github.com/HBAI-Ltd/Toonflow-app) | 13,862 | 开源一站式 AI 短剧创作：小说/剧本 → 动画短剧，含编剧、智能分镜、角色与场景资产 | 桌面/Web 应用 | Apache-2.0 |
| BigBanana AI Director | [shuyu-labs/BigBanana-AI-Director](https://github.com/shuyu-labs/BigBanana-AI-Director) | 1,744 | 工业级 AI 短剧/漫剧导演平台，"一句话生成完整短剧，剧本到成片全自动化"，强调角色一致性与场景连续性 | 平台/工作流 | 未标注标准许可 |
| LocalMiniDrama | [xuanyustudio/LocalMiniDrama](https://github.com/xuanyustudio/LocalMiniDrama) | 1,274 | 本地 AI 短剧&漫剧，已接入 Seedance2，数据不出本机 | 本地工具 | MIT |
| CineGen-ShortDrama | [UllrAI/CineGen-ShortDrama](https://github.com/UllrAI/CineGen-ShortDrama)（原 Will-Water/CineGen-AI） | 491 | AI 短剧生成 | 未查到 | 未查到 |
| Yihen-Drama | [CszYihen/Yihen-Drama](https://github.com/CszYihen/Yihen-Drama) | 213 | AI 短剧生成平台，前后端+一键 Docker 部署 | Web | 未查到 |
| ai-shotlive | [sorker/ai-shotlive](https://github.com/sorker/ai-shotlive) | 277 | 小说→剧本→分镜→关键帧→视频→AI 剪辑一站式（改自 BigBanana/CutOS/CineGen/Toonflow） | Web | 未标注 |
| AIYOU | [yubowen123/AIYOU_open-ai-video-drama-generator](https://github.com/yubowen123/AIYOU_open-ai-video-drama-generator) | 130 | AI 短剧平台，36 天 VibeCoding，接入 5 个中继 API | Web | 未标注 |
| Koma | [M-JYuan/Koma](https://github.com/M-JYuan/Koma) | 86 | AI 短剧发布仓库 | 未查到 | GPL-3.0 |
| plotloom | [T0UGH/plotloom](https://github.com/T0UGH/plotloom) | 7 | AI 短剧生产 CLI，repo-first 工作流 | CLI | MIT |
| MovieAgent（研究） | [showlab/MovieAgent](https://github.com/showlab/MovieAgent) | 353 | [多智能体 CoT 规划的自动电影生成（论文）](https://huggingface.co/papers/2503.07314) | 研究代码 | 未查到 |

**B2. 营销 / 解说 / 推文短视频类**

| 项目 | GitHub | star* | 定位 | 上手方式 | 许可证 |
|---|---|---|---|---|---|
| MoneyPrinterTurbo | [harry0703/MoneyPrinterTurbo](https://github.com/harry0703/MoneyPrinterTurbo) | 103,249 | "根据主题或关键词一键生成高清短视频"——素材检索+文案+TTS 混剪式（非生成式画面） | WebUI/API | MIT |
| ShortGPT | [RayVentura/ShortGPT](https://github.com/RayVentura/ShortGPT) | 7,814 | YouTube Shorts/TikTok 频道自动化框架 | Python/CLI | MIT |
| short-video-factory | [YILS-LIN/short-video-factory](https://github.com/YILS-LIN/short-video-factory) | 5,120 | 产品营销/泛内容短视频，AI 批量剪辑，桌面端 | 桌面端 | AGPL-3.0 |
| AI_novel | [tyxben/AI_novel](https://github.com/tyxben/AI_novel) | 251 | 小说一键转短视频（有声书+AI 配图），面向抖音/小红书 | 未查到 | MIT |
| Video-Ad-Gen-Agent | [GS-GOAT/Video-Ad-Gen-Agent](https://github.com/GS-GOAT/Video-Ad-Gen-Agent) | 1 | 低资源环境下的广告视频生成 agent | 未查到 | 未查到 |

### C. 角色一致性（组件层，SearchVidGen 的依赖技术）

| 项目 | GitHub | star* | 说明 |
|---|---|---|---|
| InstantCharacter（腾讯） | [Tencent-Hunyuan/InstantCharacter](https://github.com/Tencent-Hunyuan/InstantCharacter) | 1,046 | [2025-04-18 开源](https://news.qq.com/rain/a/20250418A04BSI00?suid=&media_id=)，基于 FLUX DiT 的参考图角色定制，支持姿势/表情/场景控制；有第三方 ComfyUI 节点（[ComfyUI-InstantCharacter](https://github.com/jax-explorer/ComfyUI-InstantCharacter)）；**最后 push 2025-05，近一年未更新** |
| PhotoMaker | [TencentARC/PhotoMaker](https://github.com/TencentARC/PhotoMaker) | 10,093 | 早期角色一致方案，生态成熟 |
| ACE++ | [ali-vilab/ACE_plus](https://github.com/ali-vilab/ACE_plus) | 1,365 | 阿里统一编辑/定制框架 |
| UMO（字节） | [bytedance/UMO](https://github.com/bytedance/UMO) | 190 | CVPR 2026 多身份一致性新方案（新、尚小） |

### D. 视频智能体（2025-2026 新赛道，方向性对手）

| 项目 | 链接 | star/热度 | 说明 |
|---|---|---|---|
| UniVA | [univa-agent/univa](https://github.com/univa-agent/univa) | 518 | [通用视频智能体框架（论文 2511.08521）](https://ar5iv.labs.arxiv.org/html/2511.08521)，"告别抽卡、散装工具拼凑" |
| 南洋理工分层 Agent | [36氪报道](https://eu.36kr.com/zh/p/3827028472763269) / [智源社区](https://hub.baai.ac.cn/view/55015) | 2026 年热点 | "一句话生成完整短剧"，AI 短剧生产走向标准化 |
| 智象未来 vivago R1 | [新华网](http://www.xinhuanet.com/tech/20260719/5e9da195bf374535800f50c8d766699d/c.html) | WAIC 2026 发布 | 无限时长内容创作智能体（商业） |
| Paper2Video | [Gen-Verse/Paper2Video](https://github.com/Gen-Verse/Paper2Video) | ICCV 2025 | 论文→视频 agent 系统 |

> **对照组：SearchVidGen 自身** [ZhijunLStudio/SearchVidGen](https://github.com/ZhijunLStudio/SearchVidGen)：12★，MIT，2026-08 仍更新。定位"搜索词/一句话 → 完整短视频"，100% 开源栈：DeepSeek/LLM 剧本 → InstantCharacter（FLUX）关键帧 → Wan2.1 图生视频 → Kokoro TTS → FFmpeg+VideoCaptioner 字幕成片，Gradio UI + 分步脚本。

\* star 数采集于 2026-08-14（GitHub API / 页面抓取，四舍五入展示）；LTX-2 为页面抓取近似值。

---

## 二、重点竞品分析

### 1. MoneyPrinterTurbo（10.3万★）—— "关键词→视频"词条的最强占据者
- 功能：输入主题/关键词 → LLM 写文案 → 检索现成图片/视频素材 → TTS 配音 → 字幕 → 一键成片（[GitHub 自述](https://github.com/harry0703/MoneyPrinterTurbo)）；[曾被媒体称为"GitHub 7万星印钞机"](https://cloud.tencent.cn/developer/article/2697476?policyId=1004)，2026-08 已破 10 万星，仍在持续发版（[v1.3.3](https://github.com/harry0703/MoneyPrinterTurbo/releases/tag/v1.3.3)）。
- 目标用户：自媒体口播、营销号、带货视频批量生产者；以云 API 为主，几乎不用本地大模型。
- **关键差异**：它是"**检索式素材混剪**"（PPT 式图文+口播，画面非 AI 生成）；SearchVidGen 是"**生成式**"（LLM 剧本+AI 生成画面+角色一致）。这是两类产品，但用户心智上容易混为一谈——**必须主动划清界限**。
- 生态：衍生 fork 极多，已有 [MoneyPrinterAICreate](https://github.com/q1uki/MoneyPrinterAICreate) 把 Wan2.1 文生视频/图生视频接进去（说明"生成式升级 MPT"是社区已验证的需求方向）。

### 2. Toonflow（1.39万★）—— AI 短剧赛道的开源头部
- 一站式"小说→动画短剧"：AI 编剧、智能分镜、角色与场景资产、批量生产，Apache-2.0，桌面/Web 形态，工程完整度高。
- 目标用户：想做 AI 漫剧/短剧变现的创作者（分账、推文赛道）。
- 对 SearchVidGen 的启示：它证明"**完整成片工具（而非散装模型）**"有巨大需求，但也意味着**重平台型短剧工具赛道已被头部占据**，新项目拼工程完整度很难赢。

### 3. BigBanana AI Director（1,744★）—— 与 SearchVidGen 定位最接近的对手
- "一句话生成完整短剧，从剧本到成片全自动化"，主打印刷式"Script-to-Asset-to-Keyframe"工业化工作流、角色一致性、场景连续性（[README](https://github.com/shuyu-labs/BigBanana-AI-Director/blob/main/README.md)），2026-08 仍在活跃更新。
- 差异点：它面向**多角色、多集短剧的工业化生产**（重资产、平台化）；SearchVidGen 面向**单角色、短叙事、轻量流水线**，模块更少、更透明、更易复现——"轻"可以成为卖点，但功能上会被 BigBanana 覆盖，需靠"本地化/教学/可 hack"错位。

### 4. LocalMiniDrama（1,274★）—— 本地化路线的最直接对标
- "本地 AI 短剧&漫剧，从故事到成片一站式，数据不出本机"，已接入 Seedance2，MIT，2026-08 仍活跃。**它和 SearchVidGen 几乎是同一细分定位的竞品，且已跑在前面**（star 1.3k vs 12）。

### 5. ShortGPT（7,814★）—— 前车之鉴
- 曾经的"Youtube/TikTok 频道自动化"明星项目，但[最后活跃停在 2025-02](https://github.com/RayVentura/ShortGPT)，基本停滞。教训：**流水线类项目如果只做"写文案+拼素材"，会被 MPT 等后来者碾压；停留在旧技术栈=死亡**。

### 6. SkyReels（昆仑万维）—— 短剧向的开源模型层
- [V2（2025-04）主打"AI 短剧、无限时长电影"，单卡可部署](https://hub.baai.ac.cn/view/43461)；[V3（2026-01）多模态全能](https://finance.eastmoney.com/news/1354,202601293635635435.html)。模型层不断往"短剧"场景卷，SearchVidGen 这类应用层项目应**持续跟进接入新模型**而非绑定 Wan2.1。

### 7. HunyuanVideo 1.5（腾讯）—— 轻量化趋势的信号
- [2025-11-21 开源，约 14G 显存消费级显卡可跑 5-10 秒高清视频](https://www.tmtpost.com/nictation/7776109.html)。说明 2026 年"**消费级硬件本地跑视频生成**"已不是幻想，本地化应用层的门槛在降低——这对"本地一键成片"定位是利好。

### 8. InstantCharacter（1,046★）—— SearchVidGen 的命门组件
- 角色一致性是 SearchVidGen 的核心卖点，但 InstantCharacter 本身 star 不高、**近一年未更新**（最后 push 2025-05）；同类替代（PhotoMaker 1.0万★、ACE++ 1,365★、UMO 190★）都在演进。**依赖一个停滞组件=技术栈过时风险**，需要抽象出"可插拔角色一致层"并跟踪新方案（[UMO 已带 ComfyUI 工作流](https://github.com/bytedance/UMO)）。

### 9. 视频智能体（UniVA / MovieAgent / vivago R1 / 南洋理工框架）—— 2026 年的大方向
- WAIC 2026 官方定调"**从一键成片到一人剧组**"（[CCTV](https://xwzs.cctv.cn/2026/07/31/ARTIhG6oJjf3V5AO776N9ISl260730.shtml)）；开源侧 [UniVA（518★）](https://hub.baai.ac.cn/view/50695)、[南洋理工分层 Agent（一句话生成完整短剧）](https://eu.36kr.com/zh/p/3827028472763269) 都在做"agent 编排多个模型成片"。**SearchVidGen 的"LLM→图→视频→音频→剪辑"编排，本质上就是一个垂直视频生成 agent——这是 2026 年正确的赛道，但叙事要从"流水线"升级为"智能体"**。

### 10. ComfyUI（12.7万★）—— 最大的流量与分发渠道
- 所有开源模型（Wan2.1、LTX、CogVideoX、InstantCharacter）都有一堆[教程与工作流](https://docs.comfy.org/zh/tutorials/video/wan/wan-video)，"ComfyUI 工作流"是中文社区学 AI 视频的默认入口；但 ComfyUI 是**节点编排**，不是"一键成片"，用户仍要自己拼装。SearchVidGen 可考虑以 ComfyUI 节点作为分发渠道之一（非主定位）。

---

## 三、用户需求与痛点（2025-2026）

### 谁在用、怎么用
- **中文创作者主流工具链 = 商业云服务 + ComfyUI 混合**：即梦（字节，[2025-09 开放 API](https://www.stdaily.com/web/gdxw/2025-09/02/content_394033.html)）、可灵、豆包等负责出画面，ComfyUI 做本地精调/工作流，DeepSeek 写文案——B站/小红书上大量"ComfyUI+豆包+即梦全流程"教程即证明（[例1](https://www.bilibili.com/video/BV15hk6BkEXQ/)、[例2](https://www.bilibili.com/video/BV1wzc4zvEh9/)）。
- **本地全开源管线反而是少数派**：多数人用云 API 是因为省事、免显存；本地部署教程集中在"单模型跑通"（如 [Wan2.1 最低 8G 显存可跑 1.3B/480P](https://blog.csdn.net/king14bhhb/article/details/148509647)、[14B 需 24G 级显存优化](https://blog.51cto.com/u_15177056/14725201)），"从词到成片的完整本地管线"几乎没有被教程化——**这是 SearchVidGen 的内容红利**。
- **变现需求真实且旺盛**：[AI 短剧/漫剧赛道月入数万、播放数亿的案例频出](https://m.jiemian.com/article/13347554.html)（如[AI 宠物短剧博主月入 50 万](https://w.dzwww.com/p/pczBDuPIx4.html)）；"狂刷2亿播放、副业月入2万"成为标题党常态（[澎湃](https://thirdpage.thepaper.cn/h5/jrtt/31611958)）；AI 短剧出海分账破百万（[界面](https://m.jiemian.com/article/14245318.html)）。**需求侧强劲，缺的是"低门槛、能出稳定成片"的工具**。

### 核心痛点（按频次排序）
1. **角色一致性**：人物跨镜头变脸是最大抱怨（"AI视频人物总变脸"[今日头条教程](https://m.toutiao.com/article/7617779248144646694/)；B站大量"超强一致性工作流"视频，[例](https://www.bilibili.com/video/BV1pKN96iEso/)）。InstantCharacter/PhotoMaker/ACE++/UMO 都在解决它，但**"一键成片场景下的角色一致性"仍是空白点**（这些组件都要求用户自己搭工作流）。
2. **转场与分镜连贯性**：单段模型只能出 5-10 秒，多段拼接的镜头衔接、场景连续性需要额外编排（正是短剧工具们的战场）。
3. **时长与叙事**：单段太短 → 需要"剧本→分镜→多段→拼接"的编排层，普通创作者搞不定。
4. **音频**：配音情绪、口型、音效、字幕对齐（Kokoro 等 TTS 已解决基础问题，但成片级音频仍是痛点）。
5. **上手门槛**：环境配置、依赖安装、报错、路径修改劝退小白——教程里满是"保姆级/避坑指南"（[例](https://blog.csdn.net/threejs5artist/article/details/151789580)），说明门槛真实存在。
6. **显存/成本**：14B 级模型需 24G+；8G 只能跑轻量档且质量打折；云 API 按量计费。HunyuanVideo 1.5 轻量化说明硬件门槛在降，但"全本地+高质量"仍两难。
7. **内容合规/平台限流**：AI 内容标识、版权、平台审核是中文平台的现实约束（未在本次搜索中获得具体政策数据，属常识性风险）。

### 中文社区热点做法
- **小说推文/漫画推文**：抖音/小红书爆款赛道，AI 一键生成（有声书+配图）工具已成一类产品（[AI_novel](https://github.com/tyxben/AI_novel)、[lpanda](https://blog.gitcode.com/95b1b2b4a2d6d3c3c233c89991100559.html)、TypeTale 等）。
- **AI 漫剧/短剧工业化**：Toonflow、BigBanana、CineGen 等开源平台+付费课程遍地；魔搭社区模型速递/月报持续推视频模型（[魔搭 25 年 8 月发布月报](https://developer.aliyun.com/article/1677925)）。
- **"一人剧组"叙事**：WAIC 2026 官方趋势（[CCTV](https://xwzs.cctv.cn/2026/07/31/ARTIhG6oJjf3V5AO776N9ISl260730.shtml)），长视频智能体（vivago R1）入场。
- **API 化**：即梦/Seedance 开放 API（[科技日报](https://www.stdaily.com/web/gdxw/2025-09/02/content_394033.html)），创作者倾向"本地编排+云端出片"的混合模式。

---

## 四、市场定位结论与差异化建议

### 结论 1：定位本身仍成立，但"关键词→视频"这个词已经被占
- 2026 年"从一句话/关键词到成片"是真实且增长的需求（WAIC"一人剧组"定调、短剧变现案例、UniVA 等智能体项目涌现），**方向没错**。
- 但 **MoneyPrinterTurbo（10.3万★）已占据"关键词→短视频"的用户心智**；Toonflow（1.39万★）、BigBanana（1,744★）、LocalMiniDrama（1,274★）占据了"短剧/漫剧成片"。SearchVidGen（12★）如果只说"搜索词一键生成短视频"，会被当成 MPT 的仿品而被淹没。

### 结论 2：差异化空间在"生成式 × 本地化 × 可复现"，而非"一键成片"本身
三个可打、且目前没有头部占据的交叉点：
1. **生成式画面**（vs MPT 的素材混剪）：画面是 AI 生成的、有统一角色——"真正用生成模型讲故事"。
2. **100% 开源本地栈 + 单角色短叙事**（vs Toonflow/BigBanana 的重平台/云依赖）：透明、可复现、可 hack，教学与研究友好。
3. **编排层即智能体叙事**（借 2026 年"视频智能体"风口）：把 LLM→关键帧→图生视频→TTS→剪辑抽象成可插拔 agent/pipeline，而不是又一个 ComfyUI 大礼包。

### 建议主打定位（组合拳）
> **"本地优先、生成式的故事/营销视频编排框架"——从搜索词到成片的开源垂直视频生成智能体，模块可插拔、每步可复现、文档教学化。**

- **主标签**：`生成式（非混剪）` + `全开源本地` + `模块化编排/视频智能体`，避开与官方模型仓库（Wan/Hunyuan/Open-Sora 等）和 MPT 的词面竞争。
- **目标用户分层**：
  - 开发者/研究者/学生（主）：要一个透明、可扩展的端到端参考实现，拿去改造成自己的产品；
  - 个人创作者（次）：要"一个关键词→能发的短片"，重点服务小红书/抖音口播、AI 短剧试水、营销物料。
- **反面教材规避**：不要做成第二个 ShortGPT（停留在旧技术栈、被 MPT 碾压）；不要拼 Toonflow 式的工程完备度（工程投入打不过头部）。

### 三条落地行动建议
1. **技术栈升级与可插拔化（立即做，决定生死）**
   - Wan2.1 → 支持 Wan2.2/Wan2.5 与 HunyuanVideo 1.5（轻量档利好 8-14G 用户），提供 `轻量本地(1.3B/5B) / 高质本地(14B) / 云API(即梦/Seedance)` 三档配置；
   - 角色一致层抽象出接口：InstantCharacter（现状）→ 评估替换/并存 UMO、ACE++、PhotoMaker；**不要绑定一个停滞组件**；
   - 把"分步脚本+手动改路径"升级为**一键 CLI + Gradio UI 完善化**（现 UI 已有雏形），这是对抗"教程型 ComfyUI 工作流"的关键卖点。
2. **叙事升级：从"流水线"讲到"智能体"**
   - README/标题突出"垂直视频生成智能体（LLM 编排 → 成片）"，对齐 UniVA/南洋理工框架的 2026 年叙事，但强调"小而实、本地可跑"；
   - 发布"SearchVidGen vs MoneyPrinterTurbo / Toonflow / BigBanana / LocalMiniDrama"对比表（生成式 vs 混剪、本地 vs 云端、轻量 vs 重平台），主动划界。
3. **内容与社区营销（低成本拉 star 的主渠道）**
   - 在 B站/小红书/知乎/即刻发"从搜索词到成片"的全本地复现教程（痛点 3、5 的内容红利）；提交魔搭创空间/ModelScope 社区（[魔搭月报](https://developer.aliyun.com/article/1677925)显示视频模型是社区热点）；
   - 提供 ComfyUI 节点版作为分发渠道之一（借 12.7 万 star 生态的流量），但主定位仍是"成片框架"而非节点集；
   - 每集成一个新模型就发一篇教程+对比评测（Wan2.2/2.5、HunyuanVideo 1.5、LTX-2 都是现成话题）。

### 风险提示
- **依赖组件停滞**（InstantCharacter 2025-05 后无更新）与**模型代际更迭**（Wan2.1→2.5、LTX-2、HunyuanVideo 1.5）是最大技术风险，需建立"模型接入层"。
- **被误认为 MPT 仿品**：必须在一切宣传口径上与"检索式混剪"划清界限。
- **显存门槛**：全本地 14B 需要 24G+；需提供轻量档与云 API 档降低门槛。
- **平台合规**：AI 内容标识与平台限流风险（本次未查到具体政策数据，建议运营时关注）。
- 未查到项：Wan 2.5 精确发布时间、LTX-2 精确 star 数/许可证、部分小项目（CineGen-ShortDrama、Yihen-Drama、Koma 等）许可证与上手方式、Seedance 是否有官方开源仓库（即梦以 API 形态提供）。
