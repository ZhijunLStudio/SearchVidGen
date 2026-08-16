# VidHarness 操作手册（本机实验环境）

## 环境

| 用途 | 环境 | 关键依赖 |
|---|---|---|
| 默认运行（fl2va/t2va 双卡、CLI、脚本、评测客户端） | `anaconda3/envs/torch` | torch 2.8+cu128, diffusers main(0.40.0.dev0), transformers 5.15, peft, edge-tts, funasr, modelscope, pytest |
| ref2va int8（单卡参考模式） | `anaconda3/envs/h3int8` | torch 2.13+cu130, torchao 0.18, diffusers main |
| 裁判服务 | `anaconda3/envs/vllm` | vllm 0.16rc2，GPU 7，端口 8030 |

依赖补丁（已应用）：
- `xformers/ops/fmha/flash.py`：flash-attn 上限 2.8.2 → 2.8.3（torch 环境）
- `diffusers/quantizers/quantization_config.py`：torchao 门槛 0.15 → 0.12（torch 环境，已弃用，int8 走 h3int8）

## 模型（/data-ssd/lizhijun/models/）

| 模型 | 位置 | 用途 |
|---|---|---|
| MiniMax H3（196GB，diffusers 子目录） | MiniMax-H3/ | 全模态生成（fl2va/t2va/ref2va） |
| Qwen3.5-27B（多模态） | Qwen/Qwen3.5-27B | 裁判（vLLM 服务） |
| SenseVoiceSmall | ~/.cache/modelscope | 音频验证 |

## 服务

```bash
# 裁判（一次启动，常驻）
CUDA_VISIBLE_DEVICES=7 /data/lizhijun/anaconda3/envs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
  --model /data-ssd/lizhijun/models/Qwen/Qwen3.5-27B --served-model-name judge-qwen3.5-27b \
  --port 8030 --max-model-len 12000 --gpu-memory-utilization 0.92 --enforce-eager
```

## 跑实验

```bash
cd harness
export PATH=/data/lizhijun/anaconda3/envs/torch/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 故事短片（默认 none 衔接 / t2va 双卡 4,6）
python -m vidharness.cli run tasks/story.yaml --query "..." --output experiments

# 断点续跑（要求与首次运行完全相同的配置；config.yaml 快照守卫会拒绝混跑）
python -m vidharness.cli run tasks/story.yaml --query "..." --output experiments --resume <run_id>

# ref2va int8（单卡，需 h3int8 环境 + ffmpeg PATH）
export PATH=/data/lizhijun/anaconda3/envs/h3int8/bin:/data/lizhijun/anaconda3/envs/torch/bin:$PATH
python -m vidharness.cli run tasks/bench_chain_ref.yaml --query "..." --output experiments

# 报告 / 对比（compare_chains 从每个 run 的 config.yaml 快照读衔接模式）
python -m vidharness.cli report story_short --output experiments
python -m vidharness.cli report story_short --run <run_id>   # 单 run 详情页（写入 run 目录）
python scripts/compare_chains.py experiments/story_short
python scripts/collect_evidence.py experiments/story_short/<run_id>

# 基准矩阵（一次只变一个变量；规划期全格校验，错误不花 GPU）
python -m vidharness.cli bench tasks/bench_ablation.yaml --query "..." --dry-run
python -m vidharness.cli bench tasks/bench_ablation.yaml --query "..."
# 格级断点续跑：已完成格自动跳过、未完成格续跑（格身份 = 标签+配置+query）；
# 中途崩溃后原命令重跑即可接着跑

# 体检 / 目录
python -m vidharness.cli doctor experiments/story_short/<run_id>   # 单 run 不变量
python -m vidharness.cli doctor --all experiments                  # 全量体检
python -m vidharness.cli adapters --verbose                        # 能力+参数声明目录

# leaderboard 基线（leaderboards/ 入库追踪；跑完实验后更新并提交）
python -m vidharness.cli leaderboard story_short

# 剧本优化闭环量化（纯 API 无 GPU；E24 基线：off 2.65 / on 5.46）
python scripts/compare_script_optimize.py --query "..." --trials 3

# 测试
python -m pytest tests/ -q
```

每个 run 目录包含：events.jsonl（权威事件流）+ manifest.json（投影，可重放重建）+
config.yaml（有效配置快照）+ artifacts/（产物，.meta.json 含完整输入，judge/ 为
裁判原始输出）+ eval/（评测明细记录）。

## 关键参数（tasks/*.yaml）

- `variant`: t2va（纯文本）/ fl2va（首尾帧）/ ref2va（参考图，int8 环境）
- `chain_mode`: none（默认，E8 实证）/ hard（冻结帧，已弃用）/ ref（参考软衔接）。
  配置校验拒绝其他值；能力校验按此推导要求（hard→first_last_frame、ref→refs）
- `generator.adapter` 可用 `generator.fallback`（params.chain 降级链）或
  `generator.route: {audio: true, ...}`（按能力路由，多候选时报错要求显式指定）
- `steps`: 去噪步数（30 约 36s/步双卡、75-125s/步单卡 int8）
- `disable_thinking`: true（防 token 燃烧；如需更严谨评分可 false）
- `min_score`/`weight`: 评测阈值与权重（由消费者统一结算，改配置即生效）
- `cost.gpu_price_usd_per_hour`: 本地 GPU 成本口径（默认 1.2）
- params 类型/取值/必需性由适配器的 param_schema 声明目录校验（fail loud）
- judge 适配器二选一：`judge.openai-compat`（本地 vLLM VLM，图像/视频评测，
  需 :8030 服务）或 `judge.deepseek-text`（DeepSeek 官方 API，文本评测如
  script_judge，不占 GPU；误配到媒体评测会被模态守卫响亮拒绝）
- 阶段级裁判路由：`judge.stages.{script_judge|script_optimize|segment_judge|cross_judge}`
  覆盖默认裁判（story_smoke 已把 script_judge 路由到 deepseek-text）
- script 适配器二选一：`script.deepseek-v4-flash`（官方 API，自带计费）或
  `script.openai-compat`（任意 OpenAI 兼容端点；未配单价时 billing=unpriced）
- 快速基建冒烟：`tasks/story_smoke.yaml`（2 段×20 步，完整管线降档时长）

## bench spec（tasks/bench_ablation.yaml 示例）

- `bench.base`: 基础任务配置 YAML（路径相对 harness/）
- `bench.matrix`: 变量轴列表，每项 {点路径: [取值...]}，路径从配置根写全
  （如 pipeline.generator.params.steps）；展开 = 笛卡尔积，格标签 = 取值拼接
- `bench.local_min_per_seg`: 本地 GPU 每段分钟数（E4 规划常数 12，可覆盖）
- 成本预估口径：API = 段数×时长×声明单价（cost_rates_usd_per_s）；
  本地 = 段数×local_min_per_seg×gpu_price；预估≠结算（结算看 manifest）

## 决策记录

设计决策与弃选方案见 `harness/.agents/notes/`（Agent Notes，格式见其 README.md）；
实验证据（E 系列）见 harness/README.md「实验发现」。

## 已知约束

- H3 单次 5-15s（120-360 帧，17n+5）；ref2va int8 单卡建议 ≤8s
- 共享服务器负载高时生成显著变慢（36s→125s/步）
- DeepSeek V4 API 不支持图像输入 → 图像/视频评测必须走本地裁判
- ffmpeg 在 torch 环境 bin 下，跑 funasr/保存视频需 PATH 包含
