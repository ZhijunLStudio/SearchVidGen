# Agent Note: 成本报表与目录扫描双重拼接 bug（vh costs，E36）

Status: implemented

## Problem

新增 `vh costs`（跨任务成本聚合——护城河"成本统计"的最后一角）时，
首个真实冒烟返回空表。溯源发现一个**隐蔽的路径 bug**：
`d.iterdir()` 返回的是基座拼接后的全路径，扫描代码再写
`(d / r / "manifest.json")` 造成双重拼接——相对基座下所有任务静默
扫描为空。同款 bug 潜伏在 leaderboard.export_all 的任务扫描里
（round 21 起），被已存在的基线文件掩盖（render_index 读的是磁盘
旧文件，扫描失败不可见）。单元测试没抓住它：pytest 的 tmp_path 是
绝对路径，`d / r` 遇绝对 r 时 pathlib 会**替换而非拼接**——测试
恰好绕过了 bug 形态。

## Decision

1. 两处扫描修正：`(r / "manifest.json")`（r 本身已是 run 全路径）；
   加注释说明 iterdir 语义。
2. `vh costs`（core/costs.py + CLI --gpu-price 参数）：
   按任务聚合 runs / API 成本 / GPU 卡时 / 估算 GPU 成本 / 总计，
   markdown 表 + JSON。
3. **相对路径回归测试**：monkeypatch.chdir(tmp_path) + Path(".") 基座
   ——绝对路径测试恰好绕过双重拼接 bug 形态，必须用相对基座锁定。
4. 真实报表：5 任务 18 run、**14.5 GPU 卡时、总估算 $17.44**。

## Alternatives considered

- **从 leaderboard 基线读成本**：否决。基线可能过期（本次 bug 正是
  掩盖源）；live 扫描 collect() 是唯一正源。
- **只修 costs 不修 export_all**：否决。同 bug 形态必须同修
  （对称值偏好对称实现）。

## Consequences

- 成本视图三件套齐备：单 run（manifest/详情页）→ 任务（leaderboard）
  → 全局（vh costs）。
- 目录扫描的路径纪律：iterdir 返回全路径、勿再拼基座；新扫描代码
  必须有相对基座回归测试。
