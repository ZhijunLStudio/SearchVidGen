# Agent Notes（VidHarness 决策记忆）

本目录存放 VidHarness 的设计决策记录：代码与 README 无法承载的**为什么**与**放弃了什么**。
机制移植自 [DeepSeek Harness 的 Agent Notes](https://github.com/deepseek-ai/deepseek-harness)
（`/data/lizhijun/work/Harness/deepseek-harness/.agents/notes/`），缩放到本仓库规模：
单语（中文）、无格式门禁、按 commit 粒度维护。

## 布局与命名

每份笔记有两个维度，都编码在**路径**中：`{lifecycle}/{class}/yyyy-mm-dd-topic.md`

- **生命周期**（顶层文件夹）：
  - `proposed/`：实施前评审的提案（尚未构建或仅部分构建）。
  - `implemented/`：决策已交付，记录做了什么、否决了什么，并随代码演进同步事实。
  - `rejected/`：提案被否决。仅当决策依据仍能避免一种诱人且影响重大的错误时保留。
- **类别**（嵌套文件夹）：`architecture`（交付源码的结构性决策）/
  `process`（围绕源码的工具与工作流）/ `feature` / `bug-fix` / `simplification` / `testing`。

活跃生命周期目录树就是工作清单；不设集中式 INDEX。

## 何时写一份

每个**非平凡变更**写一份：改了行为、架构、跨模块约定、存储/协议/配置格式，
或维护者可能重新审视的决策。纯机械/局部编辑豁免。已交付的决策从 `implemented/` 开始；
对未来工作的提案从 `proposed/` 开始。更新已有笔记即可，不新建重复记录；
**永不把旧笔记改写成相反决策**（用新笔记取代并互链）。

## 文件格式

前三行严格为：

```markdown
# Agent Note: <title>

Status: implemented
```

正文骨架（implemented）：

```markdown
## Problem
## Decision
…bespoke sections…
## Alternatives considered
## Consequences
```

`## Alternatives considered` 必写：每个真实落选方案一段加粗引导 + 落选原因。
proposed 用 `## Proposal` / `## Acceptance criteria` / `## Risks`；rejected 保留提案骨架，
结论写在 `Status: rejected — <一句话原因>` 行。

## 与实验证据的分工

- **Agent Notes**：决策与理由（为什么这样做、否决了什么）。
- **实验证据（E1-E10，见 ../README.md 与 experiments/）**：决策所依据的可复现数据。
- **经验记忆（experiments/_memory.jsonl）**：运行时环境反馈，注入生成提示，不是决策记录。
