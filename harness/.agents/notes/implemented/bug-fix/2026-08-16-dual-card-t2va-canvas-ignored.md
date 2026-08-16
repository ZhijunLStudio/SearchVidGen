# Agent Note: 双卡 t2va 画布参数被静默忽略（Bug#6）

Status: implemented

## Problem

E16 的 diffusers 警告（"Unexpected input dict_keys(['height','width','num_frames'])
provided. This input will be ignored"）指向一个真 bug：双卡拆分把
height/width/num_frames 传给了条件侧（before_encode+text_encoder），而 t2va
下条件侧**不声明这些输入**（diffusers 源码注释："a text-only request (t2va)
skips this block, and the layout step falls back to MiniMax-H3's own 16:9
canvas"）。画布在生成侧 PrepareLayoutStep 解析，条件侧没拿到 → 回落到模型
默认 16:9。E16 的成片恰好 1344×768 是**默认值巧合**（我们的默认比例正好
16:9）——任何非 16:9 请求（1:1/9:16/21:9）都会被静默产出 16:9。

## Decision

1. **variant 感知的参数拆分**（`split_dual_card_kwargs`，可单测的纯函数）：
   - t2va：条件侧只吃 prompt；height/width/num_frames 全部走生成侧；
   - fl2va/ref2va：条件侧独占 references/height/width、num_frames 两侧共享
     ——与修复前行为完全一致（E6/E7 的 ref2va 配方不受影响）。
   现状更新（E20）：修复前的 fl2va 拆分本身已因 E6/E7 的 before_encode
   上移而静默失效（image 未随条件侧传入）；2026-08-16 真机回归暴露并按
   声明契约修正为 fl2va 条件侧收 image/last_image——详见 fl2va 笔记。
2. **ratio 归位到任务上下文**：`pipeline.context.ratio`（默认 16:9）。
   此前 ratio 藏在 generator.params 里且被 harness 消费（local 适配器的
   param_schema 里根本没有它）——配置平面越界。API 适配器的 ratio 仍是
   适配器参数（那是 API 协议的字段），两处语义不同。
3. **拆分逻辑可测**：上次修复（E16 待办②的直接尝试）曾因未提取函数而在
   真机上 NameError——纯函数 + 单测杜绝此类回归。

## Alternatives considered

- **把 height/width 同时传两侧**：否决。不声明的那侧仍 ignored+警告，
  噪声与静默并存；按 variant 拆分是 diffusers 声明的输入契约。
- **接受 16:9 限制并在 capabilities 声明**：否决。画布是任务自由度的
  一部分（RATIO_CANVAS 表已支持 6 种比例），放弃它等于降级功能。
- **ratio 保留在 generator.params**：否决。见上——它被 harness 消费却不
  属于适配器参数，是配置平面越界的既有问题（本笔记一并修正）。

## Consequences

- 非 16:9 任务在双卡 t2va 下画布真实生效（验证：story_canvas 1:1 任务，
  E18 实测）。
- 双卡路径的输出对 diffusers 内部输入契约敏感：升级 diffusers 时需重验
  本拆分口径（E7 的 int8 配方同类风险）。
- fl2va/ref2va 行为字节级不变；纯函数测试锁定拆分语义。
