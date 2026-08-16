# Agent Note: 种子控制与生成确定性检验——配对 A/B 前提 + 两个连带 bug（E43）

Status: implemented

## Problem

E38 的诚实负结果：下游效应被臂内方差淹没，检出 3 分效应需 n≈8-10/臂。
E42 已证明裁判侧 sd=0——剩余方差全部在生成侧。若生成侧能被**种子对齐**
（同种子→同视频），配对 A/B 就能把方差从"臂内随机波动"变成"配对差"，
样本量需求坍塌式下降。前提从未验证：H3 双卡 t2va 同种子是否确定性。

## Decision

1. **种子优先级链**：`kw.seed > GenRequest.seed > 构造参数`——GenRequest
   补 seed 字段（缝级声明），minimax_h3 逐调用覆盖（E26 同源），
   一个已加载实例可做多种子生成（种子消融/证据脚本免重载）。
2. **bench 矩阵实测**（tasks/bench_seed.yaml）：基座任务剧本温度 0 +
   无剧本评审 → 唯一变量 = generator seed；2 种子 × 2 重复 = 4 格。
   像素比较用 scripts/compare_seed_runs.py（ffmpeg 等距抽帧 + MAE）。
3. **三个连带发现与修复**（本轮真实运行逮住）：
   - **OOM bug**：异构生成器参数格切换时，adapters_cache 里旧实例与新
     实例同时驻留显存（每个本地模型 ~78GB）→ 第二格加载 OOM。
     修复：`MiniMaxH3Local.dispose()`（资源由提供者拥有）+ 
     `bench.evict_generators()`（参数变化时释放旧生成器）+ cmd_bench
     接线。附带解除 bench_api_local（本地 vs API 异构矩阵）的潜在阻塞。
   - **标题静默失败**：4 个 bench 格全部无标题。根因：script 提供者的
     导演人格（system 提示）在 temperature 0/0.7/1.0 下都压倒用户指令、
     返回分镜 JSON——E40 独立验收用温度 0.3 碰巧成功掩盖了此缺陷。
     修复：script 提供者支持 `kw.system` 覆盖（变换任务的系统指令归
     任务自身拥有）+ 标题调用传"标题编辑"人格 + temperature 0.3；
     meta.params["system"] 落盘可审计。实测温度 0 适配器产出
     "橘猫窗台观雨"（doctor 干净）。
   - **缝级声明缺失**：首版代码引用 `GenRequest.seed` 却未声明——真实
     运行 fail loud（AttributeError）逮住，缝补字段 + 优先级链单测。
4. **直测实验**（scripts/seed_determinism_direct.py）：bench 版残留
   混淆（DeepSeek API 温度 0 下四次剧本仍互不相同——音频措辞差异），
   用逐调用种子覆盖做固定提示直测：同进程/同模型/逐字相同提示，
   2 种子 × 2 重复，同种子对 vs 异种子对 MAE 分解。

## Alternatives considered

- **bench 加 repeats 已是最贴近生产的路径**：先跑；发现剧本混淆后
  再上直测（本 note 的直测脚本），证据等级从"矩阵承诺"升级到
  "生成器孤立检验"。
- **script.static 固定剧本提供者**：否决。为实验加生产代码不值得；
  直测脚本绕过剧本阶段更干净。

## Consequences

- bench 矩阵版实测：同种子对 MAE=13.0/26.3，异种子对 MAE=62.2-64.2
  （4-5× 分离）——种子强控制生成，但同种子非 0。
- **直测版（决定性）**：固定提示 + 同进程 + 逐调用种子覆盖：
  **同种子对 MAE=0.0（像素级相同），异种子对 MAE=69.67**（两对完全
  一致）——H3 t2va 在固定种子下**完全确定性**；bench 版的 13-26 残留
  全部来自剧本措辞混淆（DeepSeek API 温度 0 下四次剧本仍互不相同）。
- **方法论结论**：E42（裁判 sd=0）+ E43（同种子生成 sd=0）→
  **配对种子 A/B 在生成层无噪声**：固定 query + 剧本温度 0/无剧本评审
  + 臂间匹配种子，任何视频层差异 = 处理效应。E38 的 n≈8-10/臂需求
  在生成层效应上坍塌为小样本配对设计。跨进程同种子复现性（vh run
  重跑场景）留待后续单点验证。
- 异构参数矩阵从此安全（dispose+evict）；同参数格缓存复用不变。
- 变换任务（标题/归纳）与规划任务在 script 缝上正式分野：
  kw.system 覆盖 + 可审计落盘；E33 的 canonicalize 可受益同通道。
- 新工具 3 个：compare_seed_runs.py / seed_determinism_direct.py /
  tasks/bench_seed.yaml（+story_seed_check.yaml）；测试 +5（160 total）。
