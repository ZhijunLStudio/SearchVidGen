# Agent Note: fl2va 双卡条件化自 E6/E7 起静默失效（Bug#7/E20）

Status: implemented

## Problem

2026-08-16 的 fl2va 真机回归（story_fl2va_check，hard 衔接 2 段）在段生成时
崩溃：`torch.cat(): expected a non-empty list of Tensors`（before_denoise 的
condition_latents 为空）。

溯源三连：①diffusers 的 `get_workflow("fl2va")` 子块 inputs 声明显示
`before_encode` 声明 `image/last_image/height/width`，而双卡拆分把
before_encode 拆到了**条件侧**、image 却一直留在**生成侧**——keyframes
无从产生，生成侧 vae_encoder 拿到空 condition_latents；②git 考古：E8 时代
（8f3d38d）只拆了 text_encoder，before_encode 还在生成侧、image 直达、
条件化正常（实测 E8 run 的段 2 首帧与段 1 末帧平均绝对差仅 5.68/255，
冻结帧证据成立）；③E6/E7 为 ref2va 把 before_encode 一并拆到条件侧之后，
**fl2va 双卡路径就一直静默失效**——E10 的 hard 数据复用了 E8 旧 run，
此后再无人重跑过 fl2va。

## Decision

完整修复三层（对应三个事实）：

1. **拆分按声明契约路由**（`split_dual_card_kwargs`）：fl2va 条件侧收
   `prompt/image/last_image/height/width`，ref2va 收
   `prompt/references/height/width`（num_frames 共享），t2va 只收 prompt。
2. **fl2va 每段都需要 keyframe**：get_workflow("fl2va") 的
   prepare_condition_latents 无条件执行，无 keyframe 的段必崩（E8 时代
   全集 pipeline 靠 t2va 回退块静默降级，workflow 选择化后此回退消失）。
   因此：hard 衔接的首段以 **anchor_refs 首图为首帧**（有锚点时）；
   无锚点/无首帧时适配器在最早点**响亮失败**并给出可操作指引
   （加 anchor 或换 t2va/none），不再在 diffusers 深处 torch.cat 崩溃。
3. 纯函数 + 三变体单测锁定；真机回归（story_fl2va_check，锚点 + hard
   衔接 2 段）作为 E20 证据。

## Alternatives considered

- **恢复 E8 时代的全集 pipeline（workflow=None 双 transformer）**：否决。
  全集需同时加载两个 61.7GB transformer 分区（diffusers 文档明言），
  E6/E7 正是为此迁移到 workflow 选择；静默降级本身也是要消灭的隐式行为。
- **恢复单块拆分（只拆 text_encoder）**：否决。ref2va 需要 before_encode
  在条件侧（E6 的 normalized_references 依赖），单块拆分会重新打破 ref2va；
  按变体拆分是两者兼得的唯一路径。
- **image 同时传两侧**：否决。条件侧需要它、生成侧不声明它（ignored+警告）；
  按声明路由才是干净口径。
- **首段无锚点自动换 t2va 变体**：否决。同一任务内混变体需要加载两套
  pipeline（显存翻倍）；显式配置（锚点或换变体）才是可审计的口径。

## Consequences

- fl2va/hard 衔接路径恢复可用（E20 真机验证：锚点条件化的段 1 成功产出，
  首帧与锚点平均绝对差 20.15/255——强条件化）；E8 的"冻结帧/叙事推进 1.0"
  结论重新获得有效基础——且提示：**基准数据必须标注生成路径版本，
  路径级 bug 会悄悄使旧结论失效**。
- 残留容量约束（E20 已解决）：双卡 fl2va 的段 2+（视频末帧作 keyframe）在
  auto offload 下 OOM（78.2/79.25GB，条件行额外显存）。同日按 E7 配方解决：
  生成侧 `rest.transformer.enable_group_offload(block_level,
  num_blocks_per_group=1)` + VAE 常驻 + `_device` 兜底，且**不用 auto
  manager**（两套放置机制打架）。实测生成侧显存 78GB → **24GB**，
  story_fl2va_check 完整跑通（段间衔接差 25.59/255，跨段一致性 10.0），
  代价是每步变慢（块级流式）。
- E6 笔记的"条件侧必须合并 before_encode+text_encoder"补记为 ref2va 专属，
  并加 fl2va 拆分警告。
- 拆分口径对 diffusers 子块结构敏感：升级 diffusers 后需重跑
  story_fl2va_check + 检查 get_workflow 的 inputs 声明。
