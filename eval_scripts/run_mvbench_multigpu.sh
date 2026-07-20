#!/bin/bash
# Run MVBench eval with Qwen2.5-VL-7B on lmms-eval across multiple GPUs (data parallel).
# Each GPU loads its own copy of the model and evaluates a shard of the data (~N x throughput).
#
# This script is self-bootstrapping. MVBench's "video" revision is ~16 GB spread over ~3900
# individual .mp4 files, and lmms-eval's in-eval downloader aborts the whole run on the first
# transient Hub HTTP error. So we download the dataset ONCE up front in a single process with a
# resumable retry loop (anonymous, since MVBench is public and the cached HF token is invalid),
# drop a marker file, then run every subsequent time fully offline.
#
# Usage:
#   bash run_mvbench_multigpu.sh [NUM_GPUS] [TASKS] [EXTRA_ARGS]
# Examples:
#   bash run_mvbench_multigpu.sh 4                       # 4 GPUs, full mvbench group
#   bash run_mvbench_multigpu.sh 4 mvbench_action_count  # 4 GPUs, one subtask
#   bash run_mvbench_multigpu.sh 4 mvbench "--limit 8"   # quick smoke test on 4 GPUs
set -euo pipefail

export HF_HOME="/mnt/lustre/work/kuehne/kqr828/.cache/huggingface"
# torchcodec cannot load (missing ffmpeg/libavutil); force decord backend which works
export FORCE_QWENVL_VIDEO_READER=decord
# Raise decord's EOF retry limit so it doesn't error on long 4K videos and fall back to
# torchvision.read_video (which loads ALL frames into RAM and OOM-kills the job cgroup).
export DECORD_EOF_RETRY_MAX=40960
# Force the HF datasets cache to the (already-populated) lustre cache instead of an empty
# node-local scratch dir, so offline load_dataset can reuse the cached arrow data.
export LMMS_EVAL_DATASETS_CACHE="$HF_HOME/datasets"
export TOKENIZERS_PARALLELISM=false

# The cached HF token is invalid. MVBench is public, so force ANONYMOUS Hub access:
#   HF_TOKEN_PATH -> nonexistent file makes token lookup return None (verified on hub 0.36.2).
export HF_TOKEN=""
export HF_TOKEN_PATH="/nonexistent-force-anonymous"

PY=/mnt/lustre/work/kuehne/kqr828/.conda/lmms_eval/bin/python
ACCEL=/mnt/lustre/work/kuehne/kqr828/.conda/lmms_eval/bin/accelerate
cd /mnt/lustre/work/kuehne/kqr828/WorkSpace/MMEvals/lmms-eval

NUM_GPUS="${1:-4}"
TASKS="${2:-mvbench}"
EXTRA_ARGS="${3:-}"           # e.g. --limit 8
PORT="${MAIN_PROCESS_PORT:-$((12300 + RANDOM % 200))}"

# ---------------------------------------------------------------------------
# Step 1: ensure the full MVBench dataset is cached (single process, resumable)
# ---------------------------------------------------------------------------
READY_MARKER="$HF_HOME/.mvbench_video_download_complete"
if [ ! -f "$READY_MARKER" ]; then
    echo "MVBench not fully downloaded yet -> fetching ~16 GB (~3900 files, anonymous, resumable)."
    attempt=0
    max_attempts=8
    until HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 "$PY" -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='OpenGVLab/MVBench', revision='video', repo_type='dataset',
                  token=False, max_workers=8, etag_timeout=60)
print('MVBench snapshot complete.')
"; do
        attempt=$((attempt + 1))
        if [ "$attempt" -ge "$max_attempts" ]; then
            echo "Download did not complete after $attempt attempts. Aborting." >&2
            exit 1
        fi
        echo "Download interrupted (attempt $attempt/$max_attempts) — retrying in 15s (resumes where it left off)..."
        sleep 15
    done
    touch "$READY_MARKER"
    echo "MVBench download complete; marker written to $READY_MARKER"
else
    echo "MVBench already downloaded (marker present) -> skipping download."
fi

# ---------------------------------------------------------------------------
# Step 2: run the eval fully offline (data is guaranteed present at this point)
# ---------------------------------------------------------------------------
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

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
