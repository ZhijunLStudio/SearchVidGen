# Agent Note: 工程化基线——覆盖纪律、lint 与可安装分发

Status: implemented

## Problem

harness 一直在"仓库内 sys.path"方式运行：无 pyproject（不可 pip 安装/分发，
无命令入口）、无 lint 门禁、无覆盖率基线。对齐 DSH 的工程纪律（CI 覆盖
门禁、lint、可分发包），需要给 Python 缩放版一套轻量基线。

## Decision

1. **pyproject.toml**（setuptools）：`pip install -e .` 可安装；
   `vh` console script 入口（= vidharness.cli:main）；依赖分层
   （core 轻依赖 / local / judge / dev extras）。
2. **ruff 门禁**（E/F 规则，E501 豁免）：首跑 20 条问题全部清零
   （17 个未用 import + 歧义变量名等）。
3. **覆盖纪律**：pytest-cov 基线——core/* 模块 **≥90% 达标**
   （report 96% / config 96% / experiment 96% / memory 94% / registry 92%
   / invariants+leaderboard 90% / bench+regress 89%）。GPU/媒体提供者
   （minimax/sensevoice/vllm）低覆盖是**有意策略**：它们的 I/O 路径由
   真实验证（vh regress 四任务 + E16-E27）而非 mock 单测覆盖——覆盖
   数字不该为了好看而 mock 模型权重。
4. 顺带补的校验缺口：`brief` 字段此前无类型校验（覆盖测试发现），
   已加 str 校验。

## Alternatives considered

- **全仓 100% 覆盖门禁（DSH 原文）**：否决。DSH 的每文件 100% 是纯 TS
  逻辑的产物；本仓库的 GPU 提供者只能靠真实验证，硬套会催生大量
  假 mock。core ≥90% + 提供者真实验证 是本仓库的等价纪律。
- **mypy 类型门禁**：否决。先 ruff 卫生与覆盖；类型标注已有
  （from __future__ import annotations + 全量注解），mypy 留给
  未来轮次评估收益。

## Consequences

- `vh` 命令与 `python -m vidharness.cli` 双入口等价；多机分发路径打开
  （未来并行 bench 的子进程执行可直接用 vh run）。
- RUNBOOK 增加工程命令（pytest-cov / ruff / pip install -e）。
- 覆盖率变化可审查：新增核心逻辑若跌破 90% 即提示补测试。
