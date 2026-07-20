#!/bin/bash
# Run VideoMMMU eval with Qwen2.5-VL-7B on lmms-eval across multiple GPUs (data parallel).
# Each GPU loads its own copy of the model and evaluates a shard of the data (~N x throughput).
#
# Usage:
#   bash run_videommmu_multigpu.sh [NUM_GPUS] [TASKS] [EXTRA_ARGS]
# Examples:
#   bash run_videommmu_multigpu.sh 4                       # 4 GPUs, full video_mmmu
#   bash run_videommmu_multigpu.sh 4 video_mmmu_perception # 4 GPUs, one subtask
#   bash run_videommmu_multigpu.sh 4 video_mmmu "--limit 8"# quick smoke test on 4 GPUs
set -euo pipefail

export HF_HOME="/mnt/lustre/work/kuehne/kqr828/.cache/huggingface"
# torchcodec cannot load (missing ffmpeg/libavutil); force decord backend which works
export FORCE_QWENVL_VIDEO_READER=decord
# Raise decord's EOF retry limit so it doesn't error on long 4K videos and fall back to
# torchvision.read_video (which loads ALL frames into RAM and OOM-kills the job cgroup).
export DECORD_EOF_RETRY_MAX=40960
# The cached HF token is invalid; dataset+videos are already downloaded, so run fully offline
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# Force the HF datasets cache to the (already-populated) lustre cache instead of an empty
# node-local scratch dir, so offline load_dataset can reuse the cached arrow data.
export LMMS_EVAL_DATASETS_CACHE="$HF_HOME/datasets"
export TOKENIZERS_PARALLELISM=false

ACCEL=/mnt/lustre/work/kuehne/kqr828/.conda/lmms_eval/bin/accelerate
cd /mnt/lustre/work/kuehne/kqr828/WorkSpace/MMEvals/lmms-eval

NUM_GPUS="${1:-4}"
TASKS="${2:-video_mmmu}"
EXTRA_ARGS="${3:-}"           # e.g. --limit 8
PORT="${MAIN_PROCESS_PORT:-$((12300 + RANDOM % 200))}"

# Restrict to the first NUM_GPUS visible devices unless the caller already set it.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
fi

echo "Launching on $NUM_GPUS GPU(s) [CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES], tasks=$TASKS, port=$PORT"

$ACCEL launch --num_processes="$NUM_GPUS" --main_process_port="$PORT" -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_pixels=12845056,attn_implementation=sdpa \
    --tasks "$TASKS" \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/ \
    $EXTRA_ARGS
