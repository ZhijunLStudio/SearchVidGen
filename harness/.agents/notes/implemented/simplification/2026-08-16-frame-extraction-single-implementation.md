# Agent Note: 帧抽取唯一实现点（consumers/tools.py）

Status: implemented

## Problem

帧抽取逻辑有三个各自为政的实现：director 的首/尾帧抽取（`_extract_frame`/
`_extract_last_frame`）、vllm_judge 的 n 帧采样（`_extract_frames`，含
imageio 兜底）、collect_evidence 的内联 ffmpeg——同样的 ffprobe/ffmpeg
调用、同样的缓存命名约定，却各写一遍。对齐 DSH 的"对称值偏好对称实现"
（未解释的不对称通常意味着漏掉的抽取），合并为唯一实现点。

## Decision

`consumers/tools.py` 成为**唯一的媒体工具实现点**：
`extract_frame`（单帧，缓存 stem_t{:.2f}.jpg）/
`extract_last_frame`（末帧，时长-0.5s）/
`sample_frames`（n 帧均匀采样，ffmpeg 失败退 imageio）。

三个消费方全部改为委托：director 保留静态方法薄包装（测试 monkeypatch
表面不变）；vllm_judge._extract_frames 变为一行委托；collect_evidence
改调 tools 并在抽帧失败时记录 `{segment, error}`（与 cross_consistency
的 fail-visible 同口径）。

缓存检查先于工具检查：缓存命中不需要 ffmpeg（断点续跑与无 ffmpeg
环境均可工作）。

## Alternatives considered

- **抽帧逻辑归 providers/vllm_judge 并让 director 反向 import**：否决。
  消费者依赖提供者是角色倒置（seam 角色映射笔记的原则）。
- **做成独立 seam**：否决。媒体工具是纯函数工具集，不是可替换能力
  （没有第二实现的需求，没有独立变化速率）。

## Consequences

- 帧抽取语义（缓存命名/失败返回 None/兜底策略）只在一处演进；
  三处消费方的行为差异被消除（此前 director 与 judge 的失败语义
  就不同：一个吞 ValueError 一个吞所有异常）。
- 新增媒体工具（如视频信息探测）一律进 tools.py。
