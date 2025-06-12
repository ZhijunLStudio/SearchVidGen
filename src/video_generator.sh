#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

# --- Configuration Section ---
# IMPORTANT: Adjust these paths and parameters as needed!

# Base directory where your LLMClient output folders are located
# This should point DIRECTLY to the specific content folder.
LLM_OUTPUT_BASE_DIR="/data/home/lizhijun/opensource/SearchVidGen/generated_video_content/卡皮巴拉的一天_20250612_163521"

# Path to your Wan-AI video generation model checkpoint directory
VIDEO_MODEL_PATH="/data/home/lizhijun/llm/flux-hf/model/Wan-AI/Wan2.1-I2V-14B-480P"

# Resolution for generated videos
VIDEO_SIZE="832*480"

# DARGS for generate.py (space-separated, no quotes needed here)
DARGS="--dit_fsdp --t5_fsdp --ring_size 4 --sample_shift 8 --sample_guide_scale 6"

# Default seed (can be fixed or removed if not needed by generate.py)
BASE_SEED=42

# --- End Configuration Section ---

# --- Determine Input/Output Directories ---
echo "Using specified content directory: '$LLM_OUTPUT_BASE_DIR'"
INPUT_CONTENT_DIR="$LLM_OUTPUT_BASE_DIR"

# Define subdirectories for images and videos
IMAGES_SUBDIR="generated_images"
VIDEOS_SUBDIR="videos"
FINAL_VIDEO_NAME="final_story_video.mp4"

# Full path to the directory containing input images
INPUT_IMAGES_DIR="$INPUT_CONTENT_DIR/$IMAGES_SUBDIR"
# Full path for the output videos
OUTPUT_VIDEOS_DIR="$INPUT_CONTENT_DIR/$VIDEOS_SUBDIR"

# Check if the images directory exists
if [ ! -d "$INPUT_IMAGES_DIR" ]; then
    echo "❌ Error: Image directory not found at '$INPUT_IMAGES_DIR'."
    echo "Please ensure you have run 'ImageGenerator.py' first to generate images into this folder."
    exit 1
fi

# Create output video directory if it doesn't exist
mkdir -p "$OUTPUT_VIDEOS_DIR"
echo "Generated videos will be saved to: '$OUTPUT_VIDEOS_DIR'"

# --- Load Video Generation Prompts ---
PROMPTS_FILE="$INPUT_CONTENT_DIR/img2vid_prompts.txt"
if [ ! -f "$PROMPTS_FILE" ]; then
    echo "❌ Error: Video prompts file not found at '$PROMPTS_FILE'."
    exit 1
fi

readarray -t VIDEO_PROMPTS < "$PROMPTS_FILE"
VIDEO_PROMPTS=("${VIDEO_PROMPTS[@]///$'\r'/}") 

FILTERED_VIDEO_PROMPTS=()
for prompt in "${VIDEO_PROMPTS[@]}"; do
    if [ -n "$prompt" ]; then
        FILTERED_VIDEO_PROMPTS+=("$prompt")
    fi
done
VIDEO_PROMPTS=("${FILTERED_VIDEO_PROMPTS[@]}")

NUM_SCENES=${#VIDEO_PROMPTS[@]}
if [ "$NUM_SCENES" -eq 0 ]; then
    echo "❌ Error: No video prompts found in '$PROMPTS_FILE'."
    exit 1
fi
echo "Loaded $NUM_SCENES video prompts from '$PROMPTS_FILE'."

# --- Main Video Generation Loop ---
echo ""
echo "--- Starting Video Generation for All Scenes ---"
for i in $(seq 0 $((NUM_SCENES - 1))); do
    SCENE_NUM=$((i + 1))
    PROMPT="${VIDEO_PROPROMPTS[$i]}"
    
    # <<< MODIFIED LOGIC TO FIND IMAGE DYNAMICALLY >>>
    # This part now searches for any file starting with "sceneXX" and picks the first one.
    # It handles names like "scene01.png" and "scene01_run01.png" automatically.
    
    SCENE_NUM_PADDED=$(printf "%02d" $SCENE_NUM)
    # Find the first matching image file for the current scene, sorted alphabetically
    IMG_PATH=$(find "$INPUT_IMAGES_DIR" -maxdepth 1 -type f -name "scene${SCENE_NUM_PADDED}*.png" | sort | head -n 1)

    # Check if an image was found for the scene
    if [ -z "$IMG_PATH" ]; then
        echo "⚠️  Warning: No image found for scene $SCENE_NUM (pattern: scene${SCENE_NUM_PADDED}*.png). Skipping."
        continue 
    fi
    # <<< END OF MODIFIED LOGIC >>>
    
    OUTPUT_FILE="$OUTPUT_VIDEOS_DIR/scene_$(printf "%02d" $SCENE_NUM).mp4"
    
    RING_SIZE=$(echo "$DARGS" | grep -oP '--ring_size \K\d+' | head -1)
    if [ -z "$RING_SIZE" ]; then
        RING_SIZE=4
    fi

    echo ""
    echo "=== Generating Scene $SCENE_NUM ==="
    echo "  Image: '$IMG_PATH'" # This will now show the actual found image path
    echo "  Prompt: '${PROMPT:0:100}...' (truncated)"
    echo "  Output: '$OUTPUT_FILE'"
    
    torchrun --nproc_per_node=$RING_SIZE src/generate.py \
      --task i2v-14B \
      --size "$VIDEO_SIZE" \
      --ckpt_dir "$VIDEO_MODEL_PATH" \
      --image "$IMG_PATH" \
      --prompt "$PROMPT" \
      --save_file "$OUTPUT_FILE" \
      --base_seed "$BASE_SEED" \
      $DARGS
      
    if [ $? -eq 0 ]; then
      echo "✅ Scene $SCENE_NUM completed."
    else
      echo "❌ Scene $SCENE_NUM FAILED."
    fi
done

echo ""
echo "--- All Scene Videos Generated ---"

# --- Concatenate Final Video ---
echo ""
echo "=== Concatenating Final Story Video ==="

FILELIST="$OUTPUT_VIDEOS_DIR/filelist.txt"
> "$FILELIST"

for i in $(seq 1 $NUM_SCENES); do
  VIDEO_FILE="$OUTPUT_VIDEOS_DIR/scene_$(printf "%02d" $i).mp4"
  if [ -f "$VIDEO_FILE" ]; then
    echo "file '$VIDEO_FILE'" >> "$FILELIST"
  else
    echo "Warning: Video file '$VIDEO_FILE' not found, skipping in final concatenation."
  fi
done

if [ ! -s "$FILELIST" ]; then
    echo "❌ Error: No video files found to concatenate. Final video will not be created."
    exit 1
fi

FINAL_OUTPUT_PATH="$OUTPUT_VIDEOS_DIR/$FINAL_VIDEO_NAME"

ffmpeg -f concat -safe 0 -i "$FILELIST" -c copy "$FINAL_OUTPUT_PATH" -y

if [ $? -eq 0 ]; then
  echo ""
  echo "✅ Concatenation complete!"
  echo "Final story video: '$FINAL_OUTPUT_PATH'"
else
  echo "❌ Error: Video concatenation failed. Check FFmpeg output above for details."
fi

echo ""
echo "--- All Tasks Completed ---"
echo "You can find your generated videos in: '$OUTPUT_VIDEOS_DIR'"
echo "The final combined video is: '$FINAL_OUTPUT_PATH'"