# Agent Note: 并行 bench 子进程调度（设计提案）

Status: proposed

## Problem

bench 逐格串行（+ 同参数格缓存复用）已实测（E31），但混合矩阵的
总墙钟时间 = 各格之和。当矩阵里出现可并行的格（使用不同 GPU 集合，
如 ref2va 单卡 GPU2 与 t2va 双卡 GPU4,6）时，串行浪费了空闲卡。

## Proposal

`vh bench --parallel N`（N 默认为当前空闲卡可容纳的最大并发）：

1. **格 → 子进程**：每格生成临时任务 YAML（cell cfg 落盘），以
   `vh run <tmp.yaml> --query ... --label <label>` 子进程执行（vh 入口
   已在 v0.2.0 分发）。
2. **GPU 预算锁**：从格的 `pipeline.generator.params.gpu` 声明读物理卡号，
   锁文件 /tmp/vh-locks/gpu-<idx>.lock（fcntl 独占）；worker 需先取得
   格所需全部卡锁才启动。不同 env 的格（torch vs h3int8）由格配置推断
   并在协调器里选 python（env 表进 regression.yaml 同源清单）。
3. **失败与恢复**：格失败记录并继续（子进程日志入 run 目录）；
   重跑时 bench_cell_status 跳过已完成格（E22 语义不变）。
4. **不在本提案内**：跨机分发（锁文件换分布式锁）、GPU 共享分时。

## Alternatives considered

- **进程内并行（线程/多进程池）**：否决。CUDA_VISIBLE_DEVICES 是进程级
  环境；同进程多 pipeline 并行的显存管理复杂且易踩 E29 类僵尸问题。
- **当前就做**：否决。现有硬件上唯一可并行组合需要跨 env 协调，
  且 GPU3 被外部占用——**触发条件不满足**（DSH 的"不预防性建设"）。

## Acceptance criteria

- 出现真实矩阵满足：≥2 个格、格间 GPU 集不相交、总墙钟明显受串行支配；
- `--parallel 2` 实测墙钟 ≈ max(各并行组)，锁竞争测试（fcntl）通过；
- 格级断点续跑/跳过/缓存复用语义在并行下不变（回归测试）。

## Risks

- 僵尸进程问题（E29）在子进程编排下放大：协调器需处理子进程组清理；
- 两个 env 的 python 选择出错会把 ref2va 跑在 torch env（错误在 diffusers
  深处才暴露）——env 表要 fail-loud 校验。
