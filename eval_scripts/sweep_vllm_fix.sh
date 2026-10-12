#!/usr/bin/env bash
# vLLM with the .mp4 forwarded AS A VIDEO (model plugin `vllm_fix`, eval_scripts/runners/vllm_fix/)
# against the stock vllm backend and the hf reference, for ONE Qwen2.5-VL model on ONE task.
#
# Why: lmms_eval's stock `vllm` backend decodes the video client-side and sends N frames as N
# separate images -> Qwen2.5-VL sees N x <|image_pad|> (no 2-frame temporal merge, no video
# position ids, no timestamps). `vllm_fix` hands vLLM a file:// URL, vLLM decodes it with the
# model's own video processor -> one <|video_pad|> block, the input the hf backend builds.
# See FIXES.md ("vLLM backend never sends a video").
#
# Usage (from the repo root):
#   bash eval_scripts/sweep_vllm_fix.sh --model Qwen/Qwen2.5-VL-7B-Instruct [--task metav] [--dry_run]
#
# Grid (per NFRAMES value):
#   hf        reference: batch 1, SEED                       -> <out>/hf/nf<N>_hf_bs1_seed<S>
#   vllm      stock backend, frames as images, batch-invariant, BATCH_SIZES x SEEDS
#   vllm_fix  server-side video,                batch-invariant, BATCH_SIZES x SEEDS
#
# Knobs (env): NFRAMES="4"  BATCH_SIZES="1 4 8 16 32"  SEED=22  SEEDS=22,42,72  MAX_PIXELS=602112
#   BATCH_INVARIANT=1 (0 = plain vllm; 1 needs vllm >= 0.11.1: LMMS_EVAL_ENV_BIN=.../lmms_eval2/bin)
#   NUM_GPUS=1  RESUME=1  OUT_ROOT=logs/sweep_vllm_fix  ONLY="hf vllm vllm_fix"
# Output: OUT_ROOT/<task>/<model>/{hf,vllm,vllm_fix}/<run tag>/ ; resume = skip run dirs holding *_results.json.
set -u
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

MODEL_ARG="Qwen/Qwen2.5-VL-7B-Instruct"; FAM_ARG=""; TASK="metav"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)   MODEL_ARG="$2"; shift 2;;
    --family)  FAM_ARG="$2"; shift 2;;
    --task)    TASK="$2"; shift 2;;
    --dry_run) DRY_RUN=1; shift;;
    -h|--help) sed -n '2,24p' "$0"; exit 0;;
    *) echo "error: unknown arg '$1'" >&2; exit 1;;
  esac
done
setup_model "$MODEL_ARG" "$FAM_ARG"
[[ "$FAM" == qwen25vl ]] || { echo "error: the vllm_fix plugin sets Qwen-VL processor kwargs (max_pixels/min_pixels); only the qwen25vl family is supported (got ${FAM})" >&2; exit 1; }
has_task "$TASK" || { echo "error: task '$TASK' is not registered under lmms_eval/tasks" >&2; exit 1; }

NFRAMES="${NFRAMES:-4}"; BATCH_SIZES="${BATCH_SIZES:-1 4 8 16 32}"
SEED="${SEED:-22}"; SEEDS="${SEEDS:-22,42,72}"
BATCH_INVARIANT="${BATCH_INVARIANT:-1}"
ONLY="${ONLY:-hf vllm vllm_fix}"
OUT_ROOT="${OUT_ROOT:-logs/sweep_vllm_fix}"
OUT="${OUT_ROOT}/${TASK}/${MODEL}"
IFS=',' read -r -a SEED_LIST <<< "$SEEDS"
BI_TAG=""; [[ "$BATCH_INVARIANT" == 1 ]] && BI_TAG="_bi"
HF_RUNNER="$RUNNERS/run_video_qa_hf_multigpu.sh"
FIX_RUNNER="$RUNNERS/run_video_qa_vllm_fix.sh"

echo "sweep_vllm_fix: model=${MODEL} task=${TASK} nframes=[${NFRAMES}] batch_sizes=[${BATCH_SIZES}] seeds=${SEEDS} max_pixels=${MP} batch_invariant=${BATCH_INVARIANT} out=${OUT} env=${FAM_ENV##*/.conda/}"
for n in $NFRAMES; do
  if [[ " $ONLY " == *" hf "* ]]; then
    RUNNER=$HF_RUNNER
    run_cfg "${OUT}/hf" "nf${n}_hf_bs1_seed${SEED}" "exp=vllm_fix task=${TASK} backend=hf nframes=${n} max_pixels=${MP} batch_size=1 seed=${SEED}" \
      --backend hf --video_reader torchcodec --max_pixels "$MP" --tasks "$TASK" --nframes "$n" --batch_size 1 --seed "$SEED"
  fi
  for bs in $BATCH_SIZES; do for s in "${SEED_LIST[@]}"; do
    if [[ " $ONLY " == *" vllm "* ]]; then
      RUNNER=$HF_RUNNER
      run_cfg "${OUT}/vllm" "nf${n}_vllm${BI_TAG}_bs${bs}_seed${s}" "exp=vllm_fix task=${TASK} backend=vllm nframes=${n} max_pixels=${MP} batch_size=${bs} batch_invariant=${BATCH_INVARIANT} seed=${s}" \
        --backend vllm --video_reader torchcodec --max_pixels "$MP" --tasks "$TASK" --nframes "$n" --batch_size "$bs" --seed "$s" --batch_invariant "$BATCH_INVARIANT"
    fi
    if [[ " $ONLY " == *" vllm_fix "* ]]; then
      RUNNER=$FIX_RUNNER
      run_cfg "${OUT}/vllm_fix" "nf${n}_vllmfix${BI_TAG}_bs${bs}_seed${s}" "exp=vllm_fix task=${TASK} backend=vllm_fix nframes=${n} max_pixels=${MP} batch_size=${bs} batch_invariant=${BATCH_INVARIANT} seed=${s}" \
        --max_pixels "$MP" --tasks "$TASK" --nframes "$n" --batch_size "$bs" --seed "$s" --batch_invariant "$BATCH_INVARIANT"
    fi
  done; done
done
summary
