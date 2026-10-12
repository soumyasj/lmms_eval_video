#!/usr/bin/env bash
#
# Protocol runner for Gemma 3 (google/gemma-3-*-it). There was no Gemma runner
# in eval_scripts before this file; it follows the shape of the other
# *_protocol.sh runners (--seed/--seeds, DRY_RUN, results check,
# include path). Driven by eval_scripts/sweep_*.sh.
#
#   bash eval_scripts/runners/run_video_qa_gemma_protocol.sh --tasks videomme_frames32 --nframes 32
#   --include_path <dir>   optional extra task YAML dir (every task, incl. metav and *_frames<N>, is in-tree);
#
# Model: lmms_eval `gemma3` (lmms_eval/models/simple/gemma3.py), HF
# transformers only. What that wrapper can and cannot do:
#   * FRAME-LIST TASKS ONLY. For a raw .mp4 it just hands the path to the
#     processor's chat template and never applies max_num_frames, so the
#     frame count is not under our control. This runner therefore refuses a
#     task that is not a *_frames<N> variant (exp5 does not exist for Gemma).
#   * Each frame is one SigLIP image = 256 tokens regardless of max_pixels
#     (no pan-and-scan in the wrapper), so 128 frames ~ 32k tokens, inside
#     the 128k context of the 4B/12B/27B models.
#   * batch_size > 1 is supported by the wrapper.
# No task yaml has a gemma prompt block: all tasks use their `default:`.
set -euo pipefail

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export LMMS_EVAL_DATASETS_CACHE="${LMMS_EVAL_DATASETS_CACHE:-$HF_HOME/datasets}"
if [[ -n "${LMMS_EVAL_ENV_BIN:-}" ]]; then export PATH="${LMMS_EVAL_ENV_BIN}:${PATH}"; fi   # else: the current environment
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF="${SELF_DIR}/$(basename "${BASH_SOURCE[0]}")"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # repo root

# ---- --seeds a,b,c : rerun this script once per seed ----------------------
ARGS=("$@")
for i in "${!ARGS[@]}"; do
  if [[ "${ARGS[$i]}" == "--seeds" ]]; then
    REST=("${ARGS[@]:0:$i}" "${ARGS[@]:$((i+2))}")
    IFS=',' read -r -a SEED_LIST <<< "${ARGS[$((i+1))]}"
    for s in "${SEED_LIST[@]}"; do
      echo "===== seed ${s} ====="
      bash "$SELF" "${REST[@]}" --seed "$s"
    done
    exit 0
  fi
done

# ---- defaults -------------------------------------------------------------
MODEL="google/gemma-3-12b-it"
NFRAMES=32                   # must match the *_frames<N> task; also passed as max_num_frames
MAX_PIXELS=1605632           # wrapper default (896x896 SigLIP input is fixed anyway)
MIN_PIXELS=200704
ATTN=""                      # sdpa | flash_attention_2 | eager | empty = model default
DEVICE_MAP="auto"
TASKS="videomme_frames32"
BATCH_SIZE=1
MAX_NEW_TOKENS=""
TEMPERATURE=""
SYSTEM_PROMPT=""             # empty = wrapper default ("You are a helpful assistant.")
LIMIT=""
SEED=""
RUN_TAG=""
OUTPUT_ROOT="./logs/normalized_runs/video_qa_gemma"
NUM_GPUS=""
MAIN_PORT=""
ALLOW_VIDEO_TAG="${ALLOW_VIDEO_TAG:-0}"   # 1 = let a raw-video task through anyway (frame count uncontrolled)

usage() { sed -n '2,22p' "$0"; exit 0; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)          MODEL="$2"; shift 2;;
    --nframes)        NFRAMES="$2"; shift 2;;
    --max_pixels)     MAX_PIXELS="$2"; shift 2;;
    --min_pixels)     MIN_PIXELS="$2"; shift 2;;
    --attn)           ATTN="$2"; shift 2;;
    --device_map)     DEVICE_MAP="$2"; shift 2;;
    --tasks)          TASKS="$2"; shift 2;;
    --batch_size)     BATCH_SIZE="$2"; shift 2;;
    --max_new_tokens) MAX_NEW_TOKENS="$2"; shift 2;;
    --temperature)    TEMPERATURE="$2"; shift 2;;
    --system_prompt)  SYSTEM_PROMPT="$2"; shift 2;;
    --limit)          LIMIT="$2"; shift 2;;
    --seed)           SEED="$2"; shift 2;;
    --run_tag)        RUN_TAG="$2"; shift 2;;
    --include_path)   INCLUDE_PATH_ARG="$2"; shift 2;;
    --output_root)    OUTPUT_ROOT="$2"; shift 2;;
    --num_gpus)       NUM_GPUS="$2"; shift 2;;
    --main_port)      MAIN_PORT="$2"; shift 2;;
    -h|--help)        usage;;
    *) echo "unknown option: $1" >&2; exit 1;;
  esac
done

if [[ "$TASKS" != *_frames* && "$ALLOW_VIDEO_TAG" != "1" ]]; then
  echo "error: gemma3 has no frame-count control for raw video; use a *_frames<N> task" >&2
  echo "       (e.g. --tasks ${TASKS}_frames${NFRAMES}), or ALLOW_VIDEO_TAG=1 to run anyway." >&2
  exit 1
fi
if [[ "$TASKS" == *_frames* && "$TASKS" != *_frames${NFRAMES} ]]; then
  echo "error: --nframes ${NFRAMES} does not match task '${TASKS}'" >&2; exit 1
fi

# ---- derived --------------------------------------------------------------
if [[ -z "$NUM_GPUS" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    NUM_GPUS=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
  else
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
  fi
fi
[[ "${NUM_GPUS:-0}" -lt 1 ]] && NUM_GPUS=1
[[ -z "$MAIN_PORT" ]] && MAIN_PORT=$(( 29500 + $$ % 1000 ))

TEMP_TAG=""; [[ -n "$TEMPERATURE" ]] && TEMP_TAG="_t${TEMPERATURE//./p}"
SEED_TAG=""; [[ -n "$SEED" ]] && SEED_TAG="_seed${SEED}"
# Tag layout <model>_<task>_nf<N>_..._seed<S>_<ts> is what the batch driver's
# have_results() globs on (*_nf<N>_*_seed<S>_*). Keep it.
if [[ -z "$RUN_TAG" ]]; then
  RUN_TAG="${MODEL##*/}_${TASKS}_nf${NFRAMES}_mp${MAX_PIXELS}_hf_bs${BATCH_SIZE}${TEMP_TAG}${SEED_TAG}_$(date +%Y%m%d_%H%M%S)"
fi
OUTPUT_PATH="${OUTPUT_ROOT}/${RUN_TAG}"

GEN_KWARGS=""
[[ -n "$MAX_NEW_TOKENS" ]] && GEN_KWARGS="${GEN_KWARGS:+${GEN_KWARGS},}max_new_tokens=${MAX_NEW_TOKENS}"
[[ -n "$TEMPERATURE" ]]    && GEN_KWARGS="${GEN_KWARGS:+${GEN_KWARGS},}temperature=${TEMPERATURE}"

MODEL_ARGS="pretrained=${MODEL},max_pixels=${MAX_PIXELS},min_pixels=${MIN_PIXELS},max_num_frames=${NFRAMES},device_map=${DEVICE_MAP}"
[[ -n "$ATTN" ]]          && MODEL_ARGS="${MODEL_ARGS},attn_implementation=${ATTN}"
[[ -n "$SYSTEM_PROMPT" ]] && MODEL_ARGS="${MODEL_ARGS},system_prompt=${SYSTEM_PROMPT}"

echo "run: gemma3 | nframes=${NFRAMES} (via ${TASKS}) | bs=${BATCH_SIZE} | limit=${LIMIT:-all}"
echo "     max_pixels=${MAX_PIXELS} attn=${ATTN:-<default>} gpus=${NUM_GPUS} seed=${SEED:-<none>}"
echo "     -> ${OUTPUT_PATH}"

# ---- launch ---------------------------------------------------------------
if [[ "$NUM_GPUS" -gt 1 ]]; then
  LAUNCHER=(accelerate launch --num_processes "$NUM_GPUS" --main_process_port "$MAIN_PORT" -m lmms_eval)
else
  LAUNCHER=(python -m lmms_eval)
fi
INCLUDE_PATH="${INCLUDE_PATH_ARG:-}"   # optional extra task dir; all tasks are in-tree
[[ -z "$INCLUDE_PATH" || -d "$INCLUDE_PATH" ]] || { echo "error: include_path dir not found: ${INCLUDE_PATH}" >&2; exit 1; }

CMD=("${LAUNCHER[@]}" --model gemma3 --model_args "$MODEL_ARGS"
     --tasks "$TASKS" ${INCLUDE_PATH:+--include_path "$INCLUDE_PATH"} --batch_size "$BATCH_SIZE"
     --log_samples --log_samples_suffix "$RUN_TAG" --output_path "$OUTPUT_PATH")
[[ -n "$LIMIT" ]]      && CMD+=(--limit "$LIMIT")
[[ -n "$SEED" ]]       && CMD+=(--seed "$SEED")
[[ -n "$GEN_KWARGS" ]] && CMD+=(--gen_kwargs "$GEN_KWARGS")

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'DRY_RUN:'; printf ' %q' "${CMD[@]}"; printf '\n'; exit 0
fi
"${CMD[@]}"

# lmms_eval catches evaluation errors and still exits 0; require a results file.
if ! compgen -G "${OUTPUT_PATH}/*/*_results.json" > /dev/null; then
  echo "error: no *_results.json under ${OUTPUT_PATH} -- the evaluation did not finish." >&2
  exit 1
fi
