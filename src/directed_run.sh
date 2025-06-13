#!/bin/bash
set -euo pipefail

##########################
# 1. 配置区 — 全部参数都写在这里！#
##########################

# 原始 LLM 输出内容目录（必须是真实存在的目录，含 generated_images, img2vid_video_prompts.txt）
LLM_OUTPUT_BASE_DIR="/data/home/lizhijun/opensource/SearchVidGen/generated_video_content/春天在哪里_20250614_002954"

# 要生成的片段编号列表（逗号分隔）
SCENES_LIST="5"

# 重复 runs 次数
NUM_RUNS=1

# 模型路径、分辨率、DARGS
VIDEO_MODEL_PATH="/data/home/lizhijun/llm/flux-hf/model/Wan-AI/Wan2.1-I2V-14B-480P"
VIDEO_SIZE="832*480"
DARGS="--dit_fsdp --t5_fsdp --ring_size 4 --sample_shift 4 --sample_guide_scale 6"

##########################
# 2. 脚本主体，不要动下面内容 #
##########################

# 解析 SCENES_LIST 到数组
IFS=',' read -r -a SCENES_ARR <<< "$SCENES_LIST"

# 输入目录校验
IMAGE_DIR="$LLM_OUTPUT_BASE_DIR/generated_images"
PROMPTS_FILE="$LLM_OUTPUT_BASE_DIR/img2vid_video_prompts.txt"
if [[ ! -d "$IMAGE_DIR" ]]; then
  echo "❌ 找不到图片目录：$IMAGE_DIR" >&2
  exit 1
fi
if [[ ! -f "$PROMPTS_FILE" ]]; then
  echo "❌ 找不到 prompts 文件：$PROMPTS_FILE" >&2
  exit 1
fi

# 读取并清洗 prompts
mapfile -t RAW_PROMPTS < "$PROMPTS_FILE"
VIDEO_PROMPTS=()
for p in "${RAW_PROMPTS[@]}"; do
  p="${p//$'\r'/}"
  [[ -n "$p" ]] && VIDEO_PROMPTS+=("$p")
done
PROMPT_COUNT=${#VIDEO_PROMPTS[@]}
if (( PROMPT_COUNT==0 )); then
  echo "❌ 没有读取到任何 prompt" >&2
  exit 1
fi

echo ">>> 读取到 $PROMPT_COUNT 个 scene prompt"
echo ">>> 本次只处理 scenes: ${SCENES_ARR[*]}"
echo ">>> 重复 runs=$NUM_RUNS 次"
echo ">>> 输出根目录: $LLM_OUTPUT_BASE_DIR/video_runs"
echo

# 输出根目录
OUT_ROOT="$LLM_OUTPUT_BASE_DIR/video_runs"
mkdir -p "$OUT_ROOT"

# 函数：计算下一个 run 索引
get_next_run_idx(){
  local max=0
  for d in "$OUT_ROOT"/run_*; do
    [[ -d "$d" ]] || continue
    num="${d##*/run_}"
    num="${num##0}"
    (( num>max )) && max=$num
  done
  echo $((max+1))
}

# 主循环：每次 run
for (( run_i=1; run_i<=NUM_RUNS; run_i++ )); do
  idx=$(get_next_run_idx)
  run_name=$(printf "run_%03d" "$idx")
  run_dir="$OUT_ROOT/$run_name"
  videos_dir="$run_dir/videos"
  mkdir -p "$videos_dir"

  echo "===== 开始 $run_name ====="

  for scene in "${SCENES_ARR[@]}"; do
    # 编号合法性检查
    if ! [[ "$scene" =~ ^[0-9]+$ ]]; then
      echo "⚠️ 非法 scene 编号：$scene，跳过"
      continue
    fi
    if (( scene<1 || scene>PROMPT_COUNT )); then
      echo "⚠️ scene=$scene 超出范围 1..$PROMPT_COUNT，跳过"
      continue
    fi

    prompt="${VIDEO_PROMPTS[$scene-1]}"
    pad=$(printf "%02d" "$scene")
    img=$(find "$IMAGE_DIR" -maxdepth 1 -type f -name "scene${pad}*.png" | sort | head -n1)
    if [[ -z "$img" ]]; then
      echo "⚠️ 未找到 scene${pad} 图片，跳过"
      continue
    fi

    out_mp4="$videos_dir/scene_${pad}.mp4"
    ring=$(echo "$DARGS" | grep -oP '--ring_size \K\d+' || echo 4)

    echo "---- $run_name scene=$scene ----"
    echo "  图片  = $img"
    echo "  prompt= ${prompt:0:80}..."
    echo "  输出  = $out_mp4"

    torchrun --nproc_per_node="$ring" src/generate.py \
      --task i2v-14B \
      --size "$VIDEO_SIZE" \
      --ckpt_dir "$VIDEO_MODEL_PATH" \
      --image "$img" \
      --prompt "$prompt" \
      --save_file "$out_mp4" \
      $DARGS \
      && echo "✅ scene $scene 完成" \
      || echo "❌ scene $scene 失败"
  done

  # 拼接本次 run
  echo ">>> 拼接 $run_name 中的视频 ..."
  listf="$videos_dir/filelist.txt"
  : > "$listf"
  for scene in "${SCENES_ARR[@]}"; do
    pad=$(printf "%02d" "$scene")
    [[ -f "$videos_dir/scene_${pad}.mp4" ]] && \
      echo "file 'scene_${pad}.mp4'" >> "$listf"
  done

  if [[ -s "$listf" ]]; then
    final="$videos_dir/final_story_${run_name}.mp4"
    (cd "$videos_dir" && ffmpeg -f concat -safe 0 -i filelist.txt -c copy "$final" -y)
    echo "✅ $run_name 拼接完成：$final"
  else
    echo "⚠️ $run_name 没有可拼接的视频"
  fi

  echo "===== 完成 $run_name ====="
  echo
done

echo "所有 runs 完成，输出在：$OUT_ROOT"
