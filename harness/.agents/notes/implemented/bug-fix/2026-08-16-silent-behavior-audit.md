# Agent Note: 静默行为全量审计——6 处吞异常路径可见化/响亮化（E41）

Status: implemented

## Problem

方法论笔记把"静默行为审计"列为最高 ROI 轮型（E11/E14/E18/E20/E32/E36
全是审计挖出的）。本轮委托子代理对全部 43 个异常处理器做全量审计
（判定：吞掉什么 / 有无测试 / 有无文档 / 逃生通道），产出 6 处值得修：

1. **script_optimizer 裁判不可用 = 候选 0 分 + 基础设施噪声进记忆**
   （BUG）：judge 挂了 → verdict={"score":0,"feedback":"评测不可用: X"}
   → 经 extract_feedback_text 写进 ExperienceMemory——E32 明确要消灭的
   噪声源头，且 0 分候选继续烧 API/GPU；整轮全挂会悄悄返回空剧本让
   下游空跑。
2. **续跑时损坏 manifest 被静默当作全新 run**（BUG）：manifest.json
   存在但不可解析 → except pass → 重建空 manifest + 追加 run.created，
   损坏证据被永久抹掉。
3. save_eval 读旧 eval 文件失败 → 静默 merged=[] 丢弃全部旧记录。
4. finalize 解析适配器能力失败 → backend=""，该产物本地 GPU 时间被
   静默排除在成本之外。
5. report._load_json 把损坏 manifest/eval 静默跳过 → 报告少掉整个 run
   无任何提示。
6. script 阶段裁判不可用只 print 到 stdout（不进事件流），未评剧本被
   静默接受。
（相邻发现：stage_segments 中段末帧抽取失败时，hard/ref 衔接条件
静默缺失/被锚点顶替。）

## Decision

修复原则与 DSH 范式一致："证据不完整响亮失败；UX 增强才允许静默"：

1. **优化器裁判异常**：该候选记 error 记录（score=None，不参与选优），
   **绝不写"评测不可用"进记忆**；整轮全部不可用 → save_eval 落盘后
   raise（续跑可恢复），拒绝空跑。
2. **损坏 manifest fail-loud**：manifest.json 存在但不可解析 →
   RuntimeError（无事件流时尤其关键；事件流完整时重放本就是权威，
   不会走到这步——真实 run 验证了两条路径）。
3. **save_eval 从事件流重建**：eval 文件损坏 → replay_eval_records
   重建旧记录 + 落 warning 事件（事件流是权威的又一实证）。
4. **finalize 成本口径**：能力解析失败的适配器 → warning 事件列出，
   GPU 时间不再无声消失。
5. **report 可见化**：损坏 JSON 打印 stderr 路径提示。
6. **script 裁判不可用**：save_eval("script_judge", error 记录)，
   剧本继续（质量门非硬依赖），但"未评测"本身进证据流。
7. **中段末帧失败可见**：hard/ref 模式下记 segments error 记录
   （E16 同口径：记错误记录，而不是假装衔接还在）。

## Alternatives considered

- **裁判异常一律 fail-loud**：否决。优化器场景下部分候选失败应降级
  继续（记录 error），整轮全挂才响亮失败——失败粒度与恢复成本匹配
  （E22/E29 的断点续跑语义）。
- **warning 进 manifest**：否决。warning 是过程可见性不是投影字段；
  事件流已保证"模型可见⟺日志"（重放忽略未知事件类型，前向兼容）。

## Consequences

- 新事件类型 "warning"（scope/msg）：重放忽略、不变量不比对，纯可见
  通道，后续所有"可恢复异常"共用。
- 经验记忆污染源消灭：裁判基础设施故障不再以"反馈"身份进记忆，
  自学习闭环（E32→E33→E37）的数据纯度提升。
- 回归测试 +7（155 total）：损坏 manifest、save_eval 重建、finalize
  warning、script 裁判 outage、末帧失败可见、优化器整轮/部分 outage；
  真实 run 副本上验证三条修复路径。
- 审计方法沉淀：43 个处理器逐个判级（OK/RISKY/BUG + 测试 + 文档 +
  逃生通道），剩 37 处判定为有意的 best-effort（均有逃生通道或
  fail-toward-rerun 语义），不防御性加噪音。
- 老 run 兼容：warning 事件只在新事件里出现；doctor/regress 无变化。
