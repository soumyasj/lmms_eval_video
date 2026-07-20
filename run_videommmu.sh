#!/bin/bash
# Run VideoMMMU eval with Qwen2.5-VL-7B on lmms-eval
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

PYBIN=/mnt/lustre/work/kuehne/kqr828/.conda/lmms_eval/bin/python
cd /mnt/lustre/work/kuehne/kqr828/WorkSpace/MMEvals/lmms-eval

TASKS="${1:-video_mmmu}"
LIMIT_ARG="${2:-}"   # e.g. --limit 2

$PYBIN -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_pixels=12845056,attn_implementation=sdpa \
    --tasks "$TASKS" \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/ \
    $LIMIT_ARG
