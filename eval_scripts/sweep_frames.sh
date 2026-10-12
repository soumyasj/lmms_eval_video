#!/usr/bin/env bash
# Frame-budget protocol (v1.2) for ONE model on the hf backend: exp5 (raw .mp4, exact frame
# count) and exp7 (the same items as <task>_frames<N> lists of pre-extracted JPEGs), one seed,
# over one or more tasks. This is the grid every model of the paper was run on.
#
# Usage (from the repo root):
#   bash eval_scripts/sweep_frames.sh --model Qwen/Qwen2.5-VL-7B-Instruct [--family qwen25vl] [--exp 5|7] [--dry_run]
#   bash eval_scripts/sweep_frames.sh --model Video-R1/Video-R1-7B --family qwen25vl
#
# Grid (per task):
#   exp5_number_of_frames   task=<TASK>            nframes in NFRAMES
#   exp7_video_vs_frames    task=<TASK>_frames<N>  nframes in NFRAMES  (needs eval_scripts/make_frames_task.py --task <TASK>)
#
# Families (inferred from the id; --family for fine-tunes) and what they get on hf:
#   qwen25vl / qwen3vl   torchcodec decode, family max_pixels (602112 / 524288)
#   qwen35               idem, thinking off, max_pixels 131072
#   internvl             batch 1; both exps x MAX_PATCHES tiles per frame. exp7: stock internvl_hf
#                        image tiler. exp5: INTERNVL_EXP5=tiled (default) = internvl_hf_tiled, the
#                        model card's load_video(max_num=M) recipe; INTERNVL_EXP5=hf = stock video
#                        processor (ignores max_patches, pinned to 1 tile)
#   glm / gemma          exp7 only (their hf wrappers take frame lists, not a frame-budgeted video)
#
# Batch size: 16 for 4 frames, 8 for 8, 4 for 16 and 32, 1 for 64 and 128 (InternVL hf: always 1);
# BATCH_SIZE=<n> overrides the table. Batching is score-neutral on hf (exp2 of sweep_params.sh).
#
# Knobs (env): TASKS="videomme mvbench tempcompass_multi_choice vsibench mmvu_val"  NFRAMES="4 8 16 32 64 128"
#   SEED=22  NUM_GPUS=1  RESUME=1  MAX_PIXELS=<family default>  MAX_PATCHES="1 2 4 8 12"  INTERNVL_EXP5=tiled|hf
#   MAX_CTX_TOK=40960 (InternVL: skip configs whose vision tokens exceed it; 0 = off)
#   OUT_ROOT=logs/sweep_frames  LMMS_EVAL_ENV_BIN=<env/bin>
# Output: OUT_ROOT/<task>/<model>/exp{5,7}_*/<run tag>/ ; resume = skip run dirs holding *_results.json.
set -u
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

MODEL_ARG=""; FAM_ARG=""; EXP=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)   MODEL_ARG="$2"; shift 2;;
    --family)  FAM_ARG="$2"; shift 2;;
    --exp)     EXP="$2"; shift 2;;
    --dry_run) DRY_RUN=1; shift;;
    -h|--help) sed -n '2,30p' "$0"; exit 0;;
    *) echo "error: unknown arg '$1'" >&2; exit 1;;
  esac
done
[[ -n "$MODEL_ARG" ]] || { echo "error: --model is required" >&2; exit 1; }
[[ -z "$EXP" || "$EXP" == 5 || "$EXP" == 7 ]] || { echo "error: --exp must be 5 or 7" >&2; exit 1; }
setup_model "$MODEL_ARG" "$FAM_ARG"

TASKS="${TASKS:-videomme mvbench tempcompass_multi_choice vsibench mmvu_val}"
NFRAMES="${NFRAMES:-4 8 16 32 64 128}"
SEED="${SEED:-22}"
BATCH_SIZE="${BATCH_SIZE:-}"
MAX_PATCHES="${MAX_PATCHES:-1 2 4 8 12}"
INTERNVL_EXP5="${INTERNVL_EXP5:-tiled}"
MAX_CTX_TOK="${MAX_CTX_TOK:-40960}"
OUT_ROOT="${OUT_ROOT:-logs/sweep_frames}"
[[ "$INTERNVL_EXP5" == tiled || "$INTERNVL_EXP5" == hf ]] || { echo "error: INTERNVL_EXP5 must be tiled or hf" >&2; exit 1; }

batch_for() {  # batch_for <nframes>
  local n=$1 bs=4
  [[ "$n" -le 8 ]] && bs=8; [[ "$n" -le 4 ]] && bs=16; [[ "$n" -ge 64 ]] && bs=1
  [[ -n "$BATCH_SIZE" ]] && bs=$BATCH_SIZE
  [[ "$FAM" == internvl ]] && bs=1
  echo "$bs"
}

# run_one <exp> <task> <nframes> [<max_patches>]
run_one() {
  local exp=$1 task=$2 n=$3 mp=${4:-} base_task=${task%_frames*}
  local bs; bs=$(batch_for "$n")
  local out="${OUT_ROOT}/${base_task}/${MODEL}/${exp}"
  local backend=hf tag="nf${n}_hf_bs${bs}_seed${SEED}" kv="exp=${exp} task=${task} nframes=${n} batch_size=${bs} seed=${SEED}"
  local -a args
  if [[ "$FAM" == internvl ]]; then
    [[ "$exp" == exp5_* && "$INTERNVL_EXP5" == tiled ]] && backend=hf_tiled
    if [[ "$MAX_CTX_TOK" -gt 0 ]]; then
      local tiles=$mp; [[ "$mp" -gt 1 ]] && tiles=$(( mp + 1 ))
      local est=$(( n * 256 * tiles + 1024 ))
      if [[ "$est" -gt "$MAX_CTX_TOK" ]]; then echo "--- SKIP ${exp} ${task} nf=${n} tiles=${mp}: ~${est} tokens > MAX_CTX_TOK=${MAX_CTX_TOK}"; return 0; fi
    fi
    tag="nf${n}_tiles${mp}_${backend/hf_tiled/hftiled}_bs${bs}_seed${SEED}"; kv="${kv} backend=${backend} max_patches=${mp}"
    read -r -a args <<< "$(fam_args "$FAM" "$backend")"; args+=(--max_patches "$mp")
    [[ "$backend" == hf ]] && args=(--backend hf --max_patches "$mp" --min_patches 1)
  else
    kv="${kv} backend=hf max_pixels=${MP}"
    read -r -a args <<< "$(fam_args "$FAM" hf)"
  fi
  [[ "$task" == *_frames* ]] && kv="${kv} frames_task=1"
  run_cfg "$out" "$tag" "$kv" "${args[@]}" --tasks "$task" --nframes "$n" --batch_size "$bs" --seed "$SEED"
}

echo "sweep_frames (protocol v1.2, hf): model=${MODEL} family=${FAM} tasks=[${TASKS}] nframes=[${NFRAMES}] seed=${SEED} exp=${EXP:-5+7} max_pixels=${MP:-n/a} internvl tiles=[${MAX_PATCHES}] exp5=${INTERNVL_EXP5} out=${OUT_ROOT} env=${FAM_ENV##*/.conda/}"
for task in $TASKS; do
  has_task "$task" || { echo "--- SKIP ${task}: not a registered task"; continue; }
  # exp5: the raw video with an exact frame count
  if [[ -z "$EXP" || "$EXP" == 5 ]]; then
    if ! fam_has "$FAM" hf_video; then
      echo "--- SKIP exp5 ${task}: the ${FAM} hf wrapper takes frame lists only"
    elif [[ "$FAM" == internvl ]]; then
      mps=$MAX_PATCHES
      [[ "$INTERNVL_EXP5" == hf ]] && { mps=1; echo "--- NOTE exp5 ${task}: INTERNVL_EXP5=hf pins max_patches to 1 (the stock video processor ignores it)"; }
      for n in $NFRAMES; do for mp in $mps; do run_one exp5_number_of_frames "$task" "$n" "$mp"; done; done
    else
      for n in $NFRAMES; do run_one exp5_number_of_frames "$task" "$n"; done
    fi
  fi
  # exp7: the same items as N pre-extracted JPEGs
  if [[ -z "$EXP" || "$EXP" == 7 ]]; then
    for n in $NFRAMES; do
      ft="${task}_frames${n}"
      has_task "$ft" || { echo "--- SKIP exp7 ${task} nf=${n}: no task ${ft} (python eval_scripts/make_frames_task.py --task ${task})"; continue; }
      if [[ "$FAM" == internvl ]]; then for mp in $MAX_PATCHES; do run_one exp7_video_vs_frames "$ft" "$n" "$mp"; done
      else run_one exp7_video_vs_frames "$ft" "$n"; fi
    done
  fi
done
summary
