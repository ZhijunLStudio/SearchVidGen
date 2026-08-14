# SearchVidGen 开源增长实战报告（2026 年 8 月）

> 面向对象：SearchVidGen（[github.com/ZhijunLStudio/SearchVidGen](https://github.com/ZhijunLStudio/SearchVidGen)，MIT）作者。
> 数据说明：star 数为 **2026-08-14 通过 GitHub API / shields.io 实测**（未用代理，个别仓库经 CDN 镜像核验）；所有案例与引述均附来源链接，查不到确切日期的标注"日期未查到"。

## 现状诊断（先看这一节）

| 指标 | SearchVidGen | 同赛道对标项目 |
|---|---|---|
| star（2026-08-14 实测） | **12** | [Pixelle-Video](https://github.com/ATH-MaaS/Pixelle-Video)（阿里系）≈**2.7 万**；[VideoClaw](https://github.com/HITsz-TMG/VideoClaw)（哈工大张民团队+阿里）≈**1.7k**；[MoneyPrinterTurbo](https://github.com/harry0703/MoneyPrinterTurbo) **10.3 万** |
| 创建时间 | 2025-05-15 | Pixelle-Video 2025-11 前后起量，2026-06 即被报道"2.2 万 Star"（[python88](https://www.python88.com/topic/197711)、[CSDN](https://blog.csdn.net/caoli201314/article/details/162035045)）；VideoClaw 2026-03-27 发布（README News 区） |
| 一键运行 | ❌ 需手动预装 4 个外部项目、逐个脚本改路径 | ✅ 均提供 WebUI / 整合包 / 一键脚本 |
| 在线 Demo | ❌ 无 | ✅ HF Space / 创空间 / Replicate |

**结论先行**：SearchVidGen 的赛道（"一句话/搜索词 → 完整短视频"）已被验证为 2025-2026 年最热的开源细分，且出现两个强竞品——阿里系 [Pixelle-Video](https://github.com/ATH-MaaS/Pixelle-Video)（"AI 全自动短视频引擎"，2.7 万 star）与哈工大 [VideoClaw](https://github.com/HITsz-TMG/VideoClaw)（"Chat an Idea. Get a Film"，1.7k star）。好消息是赛道被它们教育好了（用户已被种草"一句话出片"）；坏消息是**当前 12 个 star 的根因不是没推广，而是"跑不起来"**——README 的快速开始要求用户自行安装 InstantCharacter、Wan2.1、Kokoro、VideoCaptioner 四个项目并逐脚本改路径（[README 原文](https://github.com/ZhijunLStudio/SearchVidGen)）。本报告所有增长动作都建立在"先把流水线打通成一键可跑"这一前提之上。

---

## 一、成功案例复盘：2024-2026 AI 生成爆款项目的增长打法

### 1.1 案例总表（star 为 2026-08-14 实测，创建时间来自 GitHub API）

| 项目 | star | 创建 | 一句话定位 | 核心增长动作 |
|---|---|---|---|---|
| [ComfyUI](https://github.com/Comfy-Org/ComfyUI) | 127,466 | 2023-01 | 节点式 AI 生成工作流 | 生态杠杆：自定义节点/工作流分享、每个新模型都有 kijai 等 wrapper；2026-04 估值 5 亿美元（[TechCrunch](https://techcrunch.com/2026/04/24/comfyui-hits-500m-valuation-as-creators-seek-more-control-over-ai-generated-media/)） |
| [MoneyPrinterTurbo](https://github.com/harry0703/MoneyPrinterTurbo) | 103,249 | 2024-03 | 一句话/关键词一键生成短视频 | "躺赚/印钞机"人设 + 中文全网教程轰炸 + Gradio/Streamlit 双 UI + 多平台一键分发（[腾讯云社区分析](https://cloud.tencent.com/developer/article/2697476)、[今日头条](https://m.toutiao.com/article/7651908923997569571/)） |
| [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) | 60,844 | 2024-01 | 5 秒克隆声音、1 分钟微调 | 上线两天 1.4k star（[来源](https://www.e-com-net.com/article/1751376327107166208.htm)）；Windows 整合包 + B站教程生态，从 4.5 万（[来源](https://cloud.tencent.com/developer/article/2536489)）涨至 5.9 万（[来源](http://www.caieglobal.com/ainews/753.html)） |
| [Fooocus](https://github.com/lllyasviel/Fooocus) | 52,231 | 2023-08 | 新手版 Stable Diffusion，Midjourney 级画质 | 极简定位 + 一张图对比"SD WebUI 的复杂 vs Fooocus 的简单"（[Wikipedia](https://en.wikipedia.org/wiki/Fooocus)、[ToolPilot](https://toolpilot.tools/tools/fooocus)） |
| [Open-Sora](https://github.com/hpcaitech/Open-Sora) | 29,273 | 2024-02 | 让视频生产平民化 | Sora 热度借势 + 阶梯式发布 1.0→1.2→2.0 + HF 权重托管（[OSS AI Hub](https://ossaihub.com/tool/hpcaitech-open-sora/#main)） |
| [Pixelle-Video](https://github.com/ATH-MaaS/Pixelle-Video) | ≈27,000 | 2025 末 | 输入一句话全自动出短视频，零剪辑零门槛 | B站视频教程 badge + **Windows 整合包** + 独立文档站 + README 内 changelog + 模块化任意换模型（[README 原文](https://github.com/ATH-MaaS/Pixelle-Video)、[B站](https://www.bilibili.com/video/BV19ERvBbEdU/)） |
| [Wan2.1](https://github.com/Wan-Video/Wan2.1) | 16,821 | 2025-02 | 开源文生/图生视频，消费级显卡可跑 | 发布即 VBench 总分第一 + 免费商用 + 官方 ComfyUI 原生支持 + 全媒体共振（[阿里集团新闻](https://www.alibabagroup.com/zh-HK/document-1831486012178563072)、[BAAI Hub](https://hub.baai.ac.cn/view/43727)、[新华社](http://www.bj.xinhua.org/20250226/b3ac6715160049b98fa974ff46e3ad3c/b.html)） |
| [VideoCaptioner](https://github.com/WEIFENG2333/VideoCaptioner) | ≈15,600 | — | 开箱即用的智能字幕助手 | "开箱即用"定位 + HelloGitHub 收录（[star-history](https://www.star-history.com/weifeng2333/videocaptioner/)、[HelloGitHub](https://hellogithub.com/repository/9aa3cc57a6774f45b4d865f150a64f18)） |
| [HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo) | 12,422 | 2024-11 | 130 亿参数、720p 开源视频模型 | 开源即上科技媒体头条（[一财](https://www.yicai.com/news/102386257.html)、[品玩](https://www.pingwest.com/a/300654)、[澎湃](https://www.thepaper.cn/newsDetail_forward_29535163)） |
| [LTX-Video](https://github.com/Lightricks/LTX-Video) | 10,850 | 2024-11 | 轻量实时视频生成 | 发布当天同步 **Replicate 在线 demo + HF Space**，点开即玩（[HF commit](https://huggingface.co/Lightricks/LTX-Video/commit/bfef8dbeecac8c098997f349f93e2522b727eec2)、[HF Space](https://huggingface.co/spaces/Lightricks/ltx-video-distilled)） |
| [Kokoro](https://github.com/hexgrad/kokoro) | ≈8,400 | — | 82M 超轻量高质量 TTS | 小模型 + 实时合成 + 评测文遍地（[VisionStory 评测](https://www.visionstory.ai/zh-cn/open-source/kokoro-tts)） |
| [ShortGPT](https://github.com/comphy/ShortGPT) | 7,814 | 2023-06 | 自动化 YouTube Shorts/TikTok 频道 | "自动化躺赚"叙事先行，但维护停滞（最后推送 2025-02） |
| [VideoClaw](https://github.com/HITsz-TMG/VideoClaw) | ≈1,700 | 2026-03 | 一句话拍完一部剧，AI 导演系统 | **README 内嵌成片视频 + B站/YouTube 官方号 + WebUI 一键安装 + OpenClaw 集成 + Trendshift badge**（[README 原文](https://github.com/HITsz-TMG/VideoClaw)、[Trendshift](https://trendshift.io/repositories/24295)） |
| [InstantCharacter](https://github.com/Tencent-Hunyuan/InstantCharacter) | 1,046 | 2025-04 | 一张参考图保持角色一致 | 大厂背书 + 社区 ComfyUI wrapper 接力（[ComfyUI-InstantCharacter](https://github.com/jax-explorer/ComfyUI-InstantCharacter)、[站长之家](https://m.chinaz.com/ainews/17308.shtml)） |

### 1.2 可复用的 8 条打法（按杠杆大小排序）

1. **"一句话卖点" > 功能清单。** MoneyPrinterTurbo 的"印钞机/躺赚"、Fooocus 的"新手版 SD"、Pixelle-Video 的"零门槛、零剪辑经验"、VideoClaw 的"Chat an Idea. Get a Film."——这些项目的 star 曲线和它们的"一句话人设"强相关。SearchVidGen 现有的"认知型视频合成引擎"过于学术，建议对外口径改为 **"输入搜索词，自动出一部带剧情、角色一致、有旁白字幕的营销短视频"**（保持真实，但去掉黑话）。

2. **发布节奏：里程碑式发布 + 媒体共振。** Wan2.1（[2025-02 发布即屠榜](https://hub.baai.ac.cn/view/43727)）、HunyuanVideo（[2024-12 开源即头条](https://www.yicai.com/news/102386257.html)）都证明了：模型/工具级开源要在发布窗口内集中引爆，而不是零散 commit。VideoClaw 在 README 顶部维护"News"时间线（2026-03-27 发布 → 4/9 短剧优化 → 5/8 一键安装…），把每个版本都变成可传播的新闻点。

3. **Demo 即传播：在线 demo 是 star 的放大器。** LTX-Video 发布当天上 Replicate + HF Space；Wan2.1 靠 VBench 榜单夺冠制造"权威背书"；魔搭创空间案例显示一个冷门学术项目上线免费 demo 后获得 **45,789 次访问、15,798 独立访客、26,978 次真实推理**（[搬家手记](https://modelscope.csdn.net/6a7551a810ee7a33f2978640.html)）。"点开即玩"几乎等于"转发素材"。

4. **生态杠杆：接入 ComfyUI 节点 / 被大工具集成。** Wan2.1、HunyuanVideo、InstantCharacter 全部第一时间被社区做成 ComfyUI wrapper（[kijai 的 wrapper 案例](https://ossaihub.com/tool/kijai-comfyui-hunyuanvideowrapper/#main)、[ComfyUI-InstantCharacter](https://github.com/jax-explorer/ComfyUI-InstantCharacter)），ComfyUI 的 12.7 万 star 生态反向给模型导流。SearchVidGen 作为"流水线"项目，应反过来做：**发布 ComfyUI 工作流版**（把剧本→分镜→出片做成 workflow JSON），吃 ComfyUI 的用户池。

5. **一键安装是中文社区的生命线。** GPT-SoVITS 与 Pixelle-Video 都靠 **Windows 整合包** 拿下大量非程序员用户；Pinokio 为 ComfyUI、Wan2GP 提供一键启动（[comfyui.pinokio](https://github.com/cocktailpeanut/comfyui.pinokio)、[wan2gp pinokio launcher](https://github.com/umsfeuer-bot/wan2gp/commit/2c2b396acdca0a9239951ecd236f1a93f9192262)）；MoneyPrinterTurbo 有社区 Colab 版（[MoneyPrinterTurbo-Colab](https://github.com/YasinC2/MoneyPrinterTurbo-Colab)）。三者任选其一，都比"手动装四个项目"强。

6. **教程生态 = 免费增长团队。** GPT-SoVITS 的增长与 B站"整合包部署教程"的数量正相关；Wan2.1 发布后 CSDN/B站立即出现"保姆级本地部署教程"（[示例](https://blog.csdn.net/king14bhhb/article/details/148509647)）。教程是别人替你做的营销物料。

7. **宣发渠道分层。** 英文侧：Show HN + Reddit（r/StableDiffusion、r/LocalLLaMA）+ X；有研究证实 HN 发布对 AI 工具 star 有可测量影响（[Launch-Day Diffusion，arXiv 2511.04453](https://ar5iv.labs.arxiv.org/html/2511.04453)）；Reddit 上 Wan 系列有持续的社区情绪讨论（[Wan 3.0 Reddit 舆情](https://wan27.org/blog/wan-3-0-reddit)）。中文侧：V2EX 问答式引流 4 天拿 1.5k star 的经典案例（[复盘全文](https://github.com/hogani/telegram-groups/blob/main/%E5%A6%82%E4%BD%95%E5%9C%A84%E5%A4%A9%E5%86%85%E8%8E%B7%E5%BE%97%E4%B8%80%E4%B8%AA1.5k%2B%20Star%E7%9A%84Github%E9%A1%B9%E7%9B%AE%20-%20%E7%AD%96%E7%95%A5%E4%B8%8E%E5%8F%8D%E6%80%9D.md)）+ B站教程 + 魔搭社区。

8. **差异化定位是存亡问题。** Pixelle-Video（2.7 万 star）与 VideoClaw（1.7k star）已占据"一句话出片"叙事。SearchVidGen 必须打差异点：**① 搜索意图驱动（营销/原生广告场景，竞品是通用内容）；② 全开源、可完全本地部署（Pixelle 偏 API 直连、整合包闭源集成）；③ 以角色一致性（InstantCharacter）为卖点的剧情片**。若不做差异化，宣发做得再好也是给竞品做对比素材。

---

## 二、开发者体验最佳实践：让新用户 10 分钟跑起来

### 2.1 "易上手程度"与 star 的关系（有据可查）

- **同赛道对照**：MoneyPrinterTurbo（一键跑通，10.3 万 star）vs ShortGPT（自动化叙事强但维护停滞、需自行配置一堆 API，7.8k star，[仓库](https://github.com/comphy/ShortGPT)）；VideoCaptioner 靠"开箱即用"四个字做到 1.5 万 star（[HelloGitHub 介绍](https://hellogithub.com/repository/9aa3cc57a6774f45b4d865f150a64f18)）。
- **awesome 类清单的收录标准就是"好用"**：awesome-generative-ai 的 CONTRIBUTING 要求"被广泛使用、维护活跃"（[原文](https://github.com/steven2358/awesome-generative-ai/blob/main/CONTRIBUTING.md)）；awesome-go 社区对"人气与质量标准"有专门讨论（[Issue #4244](https://github.com/avelino/awesome-go/issues/4244)）。**跑不通的项目连收录门槛都过不了**。
- **GitHub Trending 的公开拆解**（[dev.to](https://dev.to/iris1031/how-to-get-on-github-trending-the-algorithm-the-tactics-and-the-real-data-o5b)、[gingiris](https://gingiris.tools/blog/2026/04/06/how-to-get-on-github-trending/)）：README 质量、demo 直观度、issue 响应速度都在权重之内。

### 2.2 2025-2026 年"新用户 10 分钟跑起来"的通行做法清单

按优先级排列，直接对照 SearchVidGen 现状（README 要求用户手动装 4 个项目 + 改 6 处脚本路径）：

1. **一键主脚本**：`python main.py --topic "xxx"` 串起全流程（对应作者 Roadmap 第一项）。中间产物可断点续跑。
2. **集中配置**：`config.yaml`（模型路径、API key、目录）+ `.env.example`。杜绝"改脚本里的路径"。
3. **模型自动下载脚本**：`python setup.py` 自动检测并下载全部依赖权重，**必须支持国内镜像（ModelScope / hf-mirror）**——这是中文用户能不能装上的分水岭。
4. **README 结构**（参考 [GitHub 社区最佳实践讨论](https://github.com/orgs/community/discussions/160970) 与 [readme-guidelines](https://github.com/maximosovsky/readme-guidelines)）：
   - 顶部 15 秒 demo GIF/内嵌视频（VideoClaw 直接把成片视频嵌进 README，[原文](https://github.com/HITsz-TMG/VideoClaw)）；
   - badges（license/version/star/依赖模型）；
   - "快速开始"三段式：`git clone → setup → python main.py`；
   - News/changelog 时间线（Pixelle-Video 每两周更新一次，[原文](https://github.com/ATH-MaaS/Pixelle-Video)）；
   - 常见报错 FAQ（模型下载失败、显存不足、API key 配置）。
5. **在线 demo**：HuggingFace Space（`gradio_demo.py` 即插即用，参考 [LTX-Video 的 Space](https://huggingface.co/spaces/Lightricks/ltx-video-distilled)）+ **ModelScope 创空间**（国内访问快，可申请免费 GPU，详见第三节）。
6. **一键安装形态（按投入从低到高）**：uv/poetry + setup 脚本 → docker-compose（模型挂 volume 预下载）→ Pinokio json（[例子](https://github.com/cocktailpeanut/comfyui.pinokio)）→ Windows 整合包（Pixelle-Video 的做法，最吃香但最耗精力）。
7. **ComfyUI 工作流导出**：把流水线做成 workflow JSON，让 12.7 万 star 的 ComfyUI 用户"导入即用"。
8. **错误处理**：每个子步骤 try/except + 明确的中文错误提示（模型没下、显存不够、网络超时分别给什么建议）。

> **底线标准**：把 README 发给一个只装过 Python 的朋友，他 10 分钟内没跑出第一条视频，就继续打磨，不要开始宣发。

---

## 三、中文社区增长路径

### 3.1 魔搭社区（ModelScope）与创空间：被低估的中文流量池

- 魔搭开源模型数量已达 **17 万个**（2026-03 报道，[腾讯新闻](https://news.qq.com/rain/a/20260322A045G000)），服务 **2500 万开发者**（[解放日报](https://www.jfdaily.com/wx/detail.do?id=1085141)），16 个月暴涨 1600 万用户（[CSDN](https://blog.csdn.net/ympzuelx3aiap7q/article/details/149059904)）。
- **创空间实战数据**：一个"二次元立绘拆分层"的小众学术项目，上线创空间后拿到 45,789 次访问、15,798 独立访客、26,978 次真实推理，且**免费申请到 Ada 48G 显存机器**（xGPU 计划），魔搭小编会主动在 Twitter 上找 GitHub 开源项目邀约（[上线手记](https://modelscope.csdn.net/6a7551a810ee7a33f2978640.html)）。
- 对 SearchVidGen 的价值：① 权重和脚本发布到模型库（国内直连）；② 创空间免费 GPU 跑在线 demo；③ 魔搭官方频道是中文 AI 圈的核心分发渠道。

### 3.2 B站：AI 视频项目的中文主战场

- B站已开源自己的视频生成模型 Index-AniSora（"动漫版 Sora"，代码权重全开放 + 保姆级教程）（[CSDN 教程](https://blog.csdn.net/m0_58581576/article/details/153048168)）。
- 教程类 up 主的涨粉神话：**"AI 动画电影制作，12 个作品涨粉 50 万"**（[B站](https://www.bilibili.com/video/BV1ZYTh6iEWB/)）、**"AI 西游记取经 Vlog，5 个作品涨粉 100 万"**（[B站](https://www.bilibili.com/video/BV1a6tbzdEMV/)）、"中专生 10 天手搓 AI 短片，1200 万播放"（[头条](https://m.toutiao.com/article/7641503324881043995/)）。这些 up 主就是 AI 开源工具最便宜的 KOL 池。
- Pixelle-Video 的 README 顶部直接挂 B站视频教程 badge（[原文](https://github.com/ATH-MaaS/Pixelle-Video)）；VideoClaw 有官方 B站/YouTube 账号（[README](https://github.com/HITsz-TMG/VideoClaw)）。**"README 里有 B站教程"本身就是中文增长标配。**

### 3.3 AI 短剧市场热度：需求端已被验证

- 《归墟》：90 后奶爸一人手搓 AI 短剧，**播放破亿**，总投入约 20 万、每分钟算力成本约 2000 元（[华龙网](https://www.cqnews.net/app/content_1534986313233797120.html)、[南方网](https://news.southcn.com/node_08203b6b14/c9b45a67a2.shtml)、[网易](https://m.163.com/dy/article/L3J7SJP40514D3UH.html)）。
- AI 宠物短剧赛道"狂刷 2 亿播放"（[界面](https://m.jiemian.com/article/13347554.html)），"比熊权倾天下"账号月入 50 万（[封面新闻](https://m.thecover.cn/news_details.html?from=web&eid=nm3G5c/I5U2H90qSdq8Jkw==)）。
- 含义：**"剧情连贯 + 角色一致"是市场公认的付费点**，SearchVidGen 的卖点恰好命中，但需要把"我能生成一部完整短剧"变成可见的 demo 证据。

### 3.4 其他中文渠道

- **知乎/公众号**：技术专栏 + 投稿科技媒体（CSDN/InfoQ/机器之心等），配合"教程式长文"（MoneyPrinterTurbo 的 7 万 star 报道即来自腾讯云开发者社区等，[示例](https://cloud.tencent.com/developer/article/2697476)）。
- **小红书**：AI 工具种草笔记 + 成品视频片段，适合"效果合集"类内容；已有大量 AI 工具笔记被 AI 运营工具批量生产（[示例项目](https://github.com/xuboboo/xiaohongshu-viral-note-agent-skill)）。
- **微信群/即刻/OpenClaw 生态**：OpenClaw 案例显示开源项目靠群运营"一天暴涨 9000 星"（[腾讯云](https://cloud.tencent.cn/developer/article/2659136?from=15425&frompage=seopage)）；V2EX 问答引流 4 天 1.5k star（[复盘](https://github.com/hogani/telegram-groups/blob/main/%E5%A6%82%E4%BD%95%E5%9C%A84%E5%A4%A9%E5%86%85%E8%8E%B7%E5%BE%97%E4%B8%80%E4%B8%AA1.5k%2B%20Star%E7%9A%84Github%E9%A1%B9%E7%9B%AE%20-%20%E7%AD%96%E7%95%A5%E4%B8%8E%E5%8F%8D%E6%80%9D.md)）。
- **垂直清单**：中文 AI 媒体工具清单 [awesome-ai-media-cn](https://github.com/JuneYaooo/awesome-ai-media-cn)（150+ 项目、每周更新）、HelloGitHub、AI 工具集（[ai-bot.cn](https://ai-bot.cn/aitoearn/)）——收录即流量。

---

## 四、宣发素材策略

### 4.1 五类"钩子"内容（按投入产出排序）

| 素材类型 | 说明 | 参考案例 | 用途 |
|---|---|---|---|
| **成品效果合集（最强钩子）** | 用项目生成 3-5 条完整短片（含旁白字幕），剪辑成 30-60 秒合集 | VideoClaw README 内嵌"程序员被裁收购公司"8 集短剧（[原文](https://github.com/HITsz-TMG/VideoClaw)）；《归墟》成片（[华龙网](https://www.cqnews.net/app/content_1534986313233797120.html)） | README 首屏、B站/小红书/抖音、Reddit/X |
| **对比视频** | 同主题：竞品/闭源 vs SearchVidGen；有/无角色一致性；有/无剧情连贯 | Fooocus "复杂 vs 简单"对比叙事（[ToolPilot](https://toolpilot.tools/tools/fooocus)）；Wan2.1 VBench 榜单对比（[阿里新闻](https://www.alibabagroup.com/zh-HK/document-1831486012178563072)） | 差异化话术的可视化证据 |
| **逐帧生成过程** | 录屏流水线各阶段：剧本→分镜图→图生视频→TTS→字幕→合成 | Pixelle-Video README 的 workflow 图（[原文](https://github.com/ATH-MaaS/Pixelle-Video)） | "技术含量"背书，技术社区传播 |
| **教程视频** | 从零到出片保姆级教程（B站向） | GPT-SoVITS 整合包教程生态（[示例](https://cloud.tencent.cn/developer/article/2587831#1)） | 拉非程序员用户，同时喂给 KOL |
| **一键成片直播/录屏** | 输入一句话→60 秒内出片全流程无剪辑 | MoneyPrinterTurbo 演示（[腾讯云分析](https://cloud.tencent.com/developer/article/2697476)） | "魔法时刻"钩子，转化率最高 |

### 4.2 渠道优先级（务实排序）

- **英文（拿全球 star）**：① Show HN（发布窗口 24 小时内，有研究支持其与 star 的相关性，[arXiv](https://ar5iv.labs.arxiv.org/html/2511.04453)）；② Reddit r/StableDiffusion + r/LocalLLaMA + r/SideProject；③ X/Twitter AI 圈（发效果视频 + 生成过程）；④ Product Hunt（开源项目上榜案例多，[Giselle 上榜复盘](https://www.codenote.net/en/posts/giselle-product-hunt-2nd-daily/)）。
- **中文（拿口碑与教程流量）**：① 魔搭社区（模型库+创空间+官方号）；② B站教程/效果视频；③ V2EX 问答式引流；④ 知乎专栏 + 公众号投稿；⑤ 即刻/微信群。
- **收录清单（低投入高回报）**：awesome-ai-media-cn、awesome-generative-ai、HelloGitHub、Trendshift、各类 AI 工具集站点。

### 4.3 发布节奏建议

- **大版本 = 发布事件**：每完成一个里程碑（一键脚本、在线 demo、整合包、ComfyUI 工作流），做一轮"README 更新 + 效果视频 + 多平台发布"的完整动作，而非随手 push。
- **每周小步更新**：changelog 保持 1-2 周可见更新（Pixelle-Video 模板，[原文](https://github.com/ATH-MaaS/Pixelle-Video)），喂给关注者"项目还活着"的信号。
- **素材复用矩阵**：1 条成片 → README 内嵌 + B站 + 小红书 + 抖音 + Reddit + X + 知乎，一套素材七处发布；教程视频剪成 3 个短视频钩子。

---

## 五、90 天增长行动计划

> 原则：**先产品后宣发**。前 30 天不碰推广，把"10 分钟跑通"做出来；中间 30 天 demo + 首轮宣发；后 30 天生态 + 社区。竞品 Pixelle-Video 从 2025-12 到 2026-06 拿到 2.2 万 star 的核心是"整合包 + 教程 + 更新频率"三件套（[报道](https://blog.csdn.net/caoli201314/article/details/162035045)），SearchVidGen 用 90 天复刻这套打法的"个人版"。

### P0｜第 1-2 周：把流水线打通成一键（不完成不宣发）

| # | 动作 | 预期效果 | 投入 |
|---|---|---|---|
| 1 | `main.py` 一键脚本串联现有 7 步（对应 Roadmap 第一项），支持断点续跑 | 用户从"装 4 个项目改 6 个路径"变成 `python main.py --topic "..."` | 高（1-2 周全职） |
| 2 | `config.yaml` + `.env.example` 集中管理路径/密钥/模型选择（对应 Roadmap 第二项） | 消除 README 里最大的劝退点 | 中 |
| 3 | `setup.py` 模型自动下载（支持 ModelScope/hf-mirror 国内镜像 + 断点续传） | 中文用户可安装的关键 | 中 |
| 4 | Gradio WebUI 端到端跑通（已有 app.py 雏形），展示每个中间产物 | 为在线 demo 和录屏打基础 | 中 |

### P1｜第 3-6 周：demo 上线 + 首轮宣发

| # | 动作 | 预期效果 | 投入 |
|---|---|---|---|
| 5 | 用项目生成 3-5 条完整中文短片，剪成 30-60 秒效果合集 + 逐帧过程录屏 | 宣发核心素材 | 中 |
| 6 | README 重构：首屏视频/GIF + badges + 三段式快速开始 + News 时间线 + FAQ | 转化率（访问→star）直接翻倍的关键 | 低 |
| 7 | HuggingFace Space + **ModelScope 创空间**上线在线 demo（申请免费 GPU） | 参照冷门项目也能拿到 4.5 万次访问（[案例](https://modelscope.csdn.net/6a7551a810ee7a33f2978640.html)） | 中 |
| 8 | 英文首轮：Show HN + Reddit（r/StableDiffusion、r/LocalLLaMA）+ X，附效果视频与生成过程 | 首批海外 star 与 issue 反馈 | 低 |
| 9 | 中文首轮：V2EX 问答引流 + 知乎专栏 + 公众号投稿（CSDN/InfoQ 等）+ 魔搭社区发布 | 中文口碑与首批"自来水"教程 | 低 |
| 10 | 提交收录：awesome-ai-media-cn、awesome-generative-ai、HelloGitHub、Trendshift | 长尾流量与背书 | 低 |

### P2｜第 7-13 周：生态 + 社区 + 差异化

| # | 动作 | 预期效果 | 投入 |
|---|---|---|---|
| 11 | **差异化定位内容**："搜索词→营销短视频（原生广告）"场景 + 角色一致性对比视频（InstantCharacter 卖点 vs 无一致性） | 与 Pixelle-Video/VideoClaw 划清边界，避免被当"仿品" | 中 |
| 12 | 发布 ComfyUI 工作流 JSON 版（剧本→分镜→出片） | 吃 ComfyUI 12.7 万 star 生态流量 | 中 |
| 13 | 一键安装落地（按资源任选）：uv 脚本 → docker-compose → Pinokio json → Windows 整合包 | 拿下非程序员用户群（GPT-SoVITS/Pixelle 已验证） | 高（整合包） |
| 14 | 社区搭建：Discord + 微信群，issue/PR 模板，CONTRIBUTING，discussions | 留存与贡献者转化（GitHub Trending 拆解强调 issue 响应，[参考](https://dev.to/iris1031/how-to-get-on-github-trending-the-algorithm-the-tactics-and-the-real-data-o5b)） | 中 |
| 15 | KOL 合作：联系 3-5 位 B站 AI 教程 up 主，提供素材包（成片+教程稿+整合包） | 参考"12 作品涨粉 50 万"类 up 主的带货能力（[B站](https://www.bilibili.com/video/BV1ZYTh6iEWB/)） | 中 |
| 16 | 持续迭代：每 1-2 周 changelog + 版本 tag；跟进 Wan2.2/2.3、LTX-2.5（[2026-08 开源](https://gigazine.net/gsc_news/en/20260812-ltx-2-5-video-generation-ai/)）、千亿 MoE 视频模型（[2026-08](https://news.qq.com/rain/a/20260806A08PYM00)）等生态更新并适配 | 保持"活跃"信号，登上 Trending 的必要条件 | 低 |

### 预期指标（90 天，务实基准）

- star：12 → **300-1000+**（参考：V2EX 单渠道 4 天 1.5k star 的案例说明单点引爆力，[复盘](https://github.com/hogani/telegram-groups/blob/main/%E5%A6%82%E4%BD%95%E5%9C%A84%E5%A4%A9%E5%86%85%E8%8E%B7%E5%BE%97%E4%B8%80%E4%B8%AA1.5k%2B%20Star%E7%9A%84Github%E9%A1%B9%E7%9B%AE%20-%20%E7%AD%96%E7%95%A5%E4%B8%8E%E5%8F%8D%E6%80%9D.md)；无大厂资源的个人项目取保守值）；
- 在线 demo 访问 >1 万、真实生成次数 >2000（创空间冷门项目基准 4.5 万访问/2.7 万推理，[案例](https://modelscope.csdn.net/6a7551a810ee7a33f2978640.html)）；
- B站教程/效果视频总播放 >10 万；社区群（Discord+微信）>500 人；
- 用户生成成品视频（UGC 案例）≥ 20 条——这是后续"效果合集"再传播的弹药。

### 风险与提醒

- **同赛道竞品已存在**（Pixelle-Video 2.7 万 star、VideoClaw 1.7k star），如果 90 天内不做出差异化（搜索意图/全开源/角色一致性），增长红利会被竞品持续收割；
- 算力成本：在线 demo 和成片生成需要 GPU，优先申请魔搭创空间免费资源，避免自购；
- 英文 README 与中文 README 需同步维护（现状已有 [README.en-US.md](https://github.com/ZhijunLStudio/SearchVidGen)），海外渠道（HN/Reddit）对中文 README 的项目转化率极低。

---

## 附：主要数据来源

- star 数：2026-08-14 GitHub API 实测（[ComfyUI](https://github.com/Comfy-Org/ComfyUI)、[MoneyPrinterTurbo](https://github.com/harry0703/MoneyPrinterTurbo)、[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)、[Fooocus](https://github.com/lllyasviel/Fooocus)、[Open-Sora](https://github.com/hpcaitech/Open-Sora)、[Wan2.1](https://github.com/Wan-Video/Wan2.1)、[HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo)、[LTX-Video](https://github.com/Lightricks/LTX-Video)、[ShortGPT](https://github.com/comphy/ShortGPT)、[InstantCharacter](https://github.com/Tencent-Hunyuan/InstantCharacter)、[SearchVidGen](https://github.com/ZhijunLStudio/SearchVidGen)）；shields.io 实测（[Pixelle-Video](https://github.com/ATH-MaaS/Pixelle-Video)、[VideoClaw](https://github.com/HITsz-TMG/VideoClaw)、[Kokoro](https://github.com/hexgrad/kokoro)）；[VideoCaptioner 15.6k](https://www.star-history.com/weifeng2333/videocaptioner/)。
- 各案例传播事实与报道链接见正文各节；部分报道的具体发布日期未查到，已如实标注。
