#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Runner for hf_inference/qwenvl.py  (standalone Qwen2.5-VL VideoMME eval).
#
# HOW TO USE:
#   Uncomment exactly ONE command block below (the setting you want to run),
#   leave the rest commented, then:  bash hf_inference/run_qwenvl.sh
#
# See hf_inference/README.md for what each flag does.
# ---------------------------------------------------------------------------
set -euo pipefail

# --- Shared config (edit as needed) ----------------------------------------
cd "$(dirname "$0")/.."                       # repo root
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
JSON="random_debug/videomme_subset_200.json"
OUTDIR="/mnt/lustre/work/kuehne/kqr828/WorkSpace/MMEvals/lmms-eval/hf_inference/results"
mkdir -p "$OUTDIR"

# ===========================================================================
# SETTINGS — uncomment ONE block.
# ===========================================================================

# --- (1) Mirror lmms_eval exactly: lmms prompt, 4 frames, max_pixels=602112 -
# python hf_inference/qwenvl.py \
#   --model "$MODEL" \
#   --json "$JSON" \
#   --prompt-style lmms \
#   --nframes 4 \
#   --max-pixels 602112 \
#   --output "$OUTDIR/lmms_nframes4.json"

# --- (2) Original standalone prompt, 4 frames ------------------------------
# python hf_inference/qwenvl.py \
#   --model "$MODEL" \
#   --json "$JSON" \
#   --prompt-style standalone \
#   --nframes 4 \
#   --max-pixels 602112 \
#   --output "$OUTDIR/standalone_nframes4.json"

# --- (3) lmms prompt, fps-based sampling (1 fps) ---------------------------
python hf_inference/qwenvl.py \
  --model "$MODEL" \
  --json "$JSON" \
  --prompt-style lmms \
  --fps 1.0 \
  --max-pixels 602112 \
  --output "$OUTDIR/lmms_fps1.json"

# --- (4) Quick smoke test: first 2 questions only --------------------------
# python hf_inference/qwenvl.py \
#   --model "$MODEL" \
#   --json "$JSON" \
#   --prompt-style lmms \
#   --nframes 4 \
#   --max-pixels 602112 \
#   --limit 2 \
#   --output "$OUTDIR/smoke_test.json"

# --- (5) More frames (16), flash-attention ---------------------------------
# python hf_inference/qwenvl.py \
#   --model "$MODEL" \
#   --json "$JSON" \
#   --prompt-style lmms \
#   --nframes 16 \
#   --max-pixels 602112 \
#   --flash-attn \
#   --output "$OUTDIR/lmms_nframes16_flash.json"

echo "Nothing ran — uncomment a command block in $0 first."
