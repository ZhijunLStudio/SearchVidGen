# Agent Note: API 适配器的 mock 协议级验证（最后一块未验证代码）

Status: implemented

## Problem

`generator.minimax-h3-api` 是唯一从未执行过的适配器（MINIMAX_API_KEY 长期
缺失，真实验证被阻塞 30+ 轮）。无验证的适配器 = "API 对比实验一键可跑"
承诺里的最大不确定项：请求格式、轮询、下载、成本口径都可能错。

## Decision

**mock 官方 API 的协议级端到端测试**（tests 内嵌 HTTPServer）：

- 模拟三个端点：`/v1/files/upload`（返回文件 URL）、
  `/v2/video_generation`（返回 task_id）、
  `/v2/query/video_generation`（返回 succeeded + 内容 URL）、
  视频下载端点（真实 1 秒 mp4，ffmpeg lavfi 生成）；
- 验证链路：upload（refs 场景）→ 创建 → 轮询 → 下载 →
  Artifact(kind=video) 落盘 → meta（adapter/resolution/cost_usd>0）。
- 零外部依赖、零凭据：测试可随时跑。

## Alternatives considered

- **等真 key 再验证**：否决。外部阻塞不可控；协议级验证把适配器的
  不确定性从"请求/轮询/下载/口径"收窄到只剩"官方 API 的真实响应
  格式差异"——真实 key 到位后的风险面大幅缩小。
- **vcr 录播真请求**：否决。无真实请求可录；mock 服务器已覆盖协议
  形状，且可断言细节（比录播更可控）。

## Consequences

- 四个适配器全部具备可执行验证（H3 本地=真机 E 系列、API=mock
  协议测试、裁判=真 API 冒烟+校准、script=真 API 量化）。
- 真实 key 到位后：跑 `vh bench tasks/bench_api_local.yaml`，
  若响应格式与 mock 假设有差异，错误会集中在解析点（fail loud）。
