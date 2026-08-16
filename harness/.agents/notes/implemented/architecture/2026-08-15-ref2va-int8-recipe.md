# Agent Note: ref2va int8 单卡配方（E5/E6/E7 实证）

Status: implemented

## Problem

ref2va（参考图软衔接）把参考 latent 拼入打包序列：生成侧 transformer_ref 62GB
常驻 + 参考/视频/音频 token 激活，单卡 80GB OOM；480p 画布与 5s 帧数均不能缓解
（瓶颈是权重常驻而非帧数）。双卡拆分（E6）能跑但占用两张卡，与"单卡可比成本"的
实验目标冲突。

## Decision

采用官方 int8 配方（E7 落地，独立环境 `h3int8`）：

- torch 2.13 + torchao 0.18（0.15+ 才支持 `Int8WeightOnlyConfig(version=2)`，
  要求 torch>=2.9，与旧环境 vllm 0.11 冲突 → 独立 conda 环境）；
- transformer_ref / text_encoder 以 Int8WeightOnly 量化加载（显存减半），
  `modules_to_not_convert` 排除投影/embedding 层；
- transformer_ref 块级流式 offload（`enable_group_offload(block_level,
  num_blocks_per_group=1, use_stream=True)`），text_encoder 叶级 offload；
- VAE / audio_vae 常驻 GPU；加载后 `pipe._device` 显式兜底执行设备；
- 关键差异（相对官方文档）：`low_cpu_mem_usage=True` 必须显式传（文档相反）。

效果：ref2va 单卡显存 78GB → **38GB**，5s 测试 342s 完成（含 ~4.5min 加载）。

配套经验（E6，双卡拆分的通用要点，int8 同样适用）：
- **ref2va** 条件侧必须合并 before_encode + text_encoder 两个子块
  （`SequentialPipelineBlocks.from_blocks_dict`）：参考编码器在 text_encoder 内、
  依赖 before_encode 的 normalized_references；
- before_encode 需要 num_frames（ref2va 视频参考归一化）；
- 参考图参数名是 `references`（`MiniMaxH3ImageReference.from_file`）；
- 参考图默认按 2048 短边编码导致视觉 token 爆炸 OOM → 缩到 768 短边
  （token 数降 ~7 倍），保留主体特征。
- **fl2va 拆分注意**（2026-08-16 真机回归发现，E20）：before_encode 拆到
  条件侧后，其声明的 `image`/`last_image` 必须随条件侧传入——否则 keyframes
  无从产生，生成侧 vae_encoder 拿到空 condition_latents 崩溃。拆分口径以
  `get_workflow(variant)` 各子块的 inputs 声明为权威契约
  （split_dual_card_kwargs，见 minimax_h3.py）。

## Alternatives considered

- **双卡拆分常驻（E6）**：否决。两卡成本翻倍且与 API 对比实验的公平性冲突；
  条件/生成两侧 ~62GB 各占一卡，算力利用率低。
- **降低画布/帧数缓解**：否决。E5 实测瓶颈是权重常驻而非帧数，
  降帧数不解决 62GB 权重。
- **等待官方 fp8/小权重版本**：否决。不可控；int8 是官方配方，先落地。

## Consequences

- ref 衔接实验依赖 `h3int8` 环境（RUNBOOK 记录）；环境矩阵多一行维护成本。
- int8 配方对模型权重版本敏感：H3 升级后需重验量化路径
  （modules_to_not_convert 名单与权重命名耦合）。
