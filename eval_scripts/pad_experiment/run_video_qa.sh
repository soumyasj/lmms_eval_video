#!/usr/bin/env bash
# Configurable video-QA runner. Pass options as flags, e.g.:
#   bash eval_scripts/pad_experiment/run_video_qa.sh --fps 1 --batch_size 8 --pad video
set -euo pipefail
export LMMS_EVAL_DATASETS_CACHE=/tmp/lmms_eval_hf_datasets_$USER
export PATH="/mnt/lustre/work/kuehne/kqr828/.conda/lmms_eval/bin:${PATH}"
cd /mnt/lustre/work/kuehne/kqr828/WorkSpace/MMEvals/lmms-eval

# ---- defaults ----
MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
SAMPLING="nframes"        # nframes | fps  (mutually exclusive)
NFRAMES=2
FPS=1
PAD="image"               # image | video
MAX_PIXELS=602112
MAX_FRAME_NUM=2
MAX_MODEL_LEN=32768       # cap KV reservation; Qwen default (128k) OOMs a 40GB GPU
GPU_MEM=0.9
TASKS="videomme"
BATCH_SIZE=1
LIMIT=""                  # empty = all samples
RUN_TAG=""
OUTPUT_ROOT="./logs/normalized_runs/video_qa"

# ---- parse flags (passing --fps sets fps mode; --nframes sets nframes mode) ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)          MODEL="$2"; shift 2;;
    --sampling)       SAMPLING="$2"; shift 2;;
    --nframes)        NFRAMES="$2"; SAMPLING="nframes"; shift 2;;
    --fps)            FPS="$2"; SAMPLING="fps"; shift 2;;
    --pad)            PAD="$2"; shift 2;;
    --max_pixels)     MAX_PIXELS="$2"; shift 2;;
    --max_frame_num)  MAX_FRAME_NUM="$2"; shift 2;;
    --max_model_len)  MAX_MODEL_LEN="$2"; shift 2;;
    --gpu_mem)        GPU_MEM="$2"; shift 2;;
    --tasks)          TASKS="$2"; shift 2;;
    --batch_size)     BATCH_SIZE="$2"; shift 2;;
    --limit)          LIMIT="$2"; shift 2;;
    --run_tag)        RUN_TAG="$2"; shift 2;;
    --output_root)    OUTPUT_ROOT="$2"; shift 2;;
    -h|--help)        sed -n '/^# ARGUMENTS/,/^# END ARGS/p' "$0"; exit 0;;
    *) echo "unknown option: $1" >&2; exit 1;;
  esac
done

# ---- derived ----
if [[ "$SAMPLING" == "fps" ]]; then
  SAMPLE="fps=${FPS}";       SAMPLE_TAG="fps${FPS//./p}"
else
  SAMPLE="nframes=${NFRAMES}"; SAMPLE_TAG="nf${NFRAMES}"
fi
# log_samples_suffix / run tag auto-derived from the config
[[ -z "$RUN_TAG" ]] && RUN_TAG="${MODEL##*/}_${TASKS}_${SAMPLE_TAG}_${PAD}_bs${BATCH_SIZE}"
OUTPUT_PATH="${OUTPUT_ROOT}/${RUN_TAG}"
[[ "$PAD" == "video" ]] && { ENTRY=(python eval_scripts/pad_experiment/run_pad_experiment.py); NAME=vllm_video_url; } \
                        || { ENTRY=(python -m lmms_eval); NAME=vllm; }

MODEL_ARGS="model=${MODEL},${SAMPLE},max_pixels=${MAX_PIXELS},max_frame_num=${MAX_FRAME_NUM},max_model_len=${MAX_MODEL_LEN},gpu_memory_utilization=${GPU_MEM},max_num_seqs=1,enforce_eager=True,limit_mm_per_prompt={\"video\":1\,\"image\":${MAX_FRAME_NUM}}"

# rough per-prompt vision-token estimate (image path): frames x (max_pixels/784)
EST_TOK=$(( MAX_FRAME_NUM * MAX_PIXELS / 784 ))
echo "run: ${NAME} | ${SAMPLE} | pad=${PAD} | tasks=${TASKS} | batch=${BATCH_SIZE} | limit=${LIMIT:-all}"
echo "     max_frame_num=${MAX_FRAME_NUM} max_pixels=${MAX_PIXELS} max_model_len=${MAX_MODEL_LEN} ~vision_tok/prompt≈${EST_TOK} -> ${OUTPUT_PATH}"

# ARGUMENTS (flag ............. possible values ............... default)
#   --model ................... any HF model id .............. Qwen/Qwen2.5-VL-7B-Instruct
#   --sampling ................ nframes | fps ................ nframes
#   --nframes ................. even int (2,4,8,...) ......... 2        (also sets --sampling nframes)
#   --fps ..................... float (0.5, 1, 2, ...) ....... 1        (also sets --sampling fps)
#   --pad ..................... image | video ................ image    (image=<|image_pad|>, video=<|video_pad|>)
#   --max_pixels .............. int (e.g. 602112, 1605632) ... 602112
#   --max_frame_num ........... int .......................... 2
#   --tasks ................... videomme | videomme,mvbench .. videomme
#   --batch_size .............. int | auto | auto:N .......... 1
#   --limit ................... int or float(0-1) | <empty> .. <all>
#   --run_tag ................. any string ................... <model>_<sampling>_<pad>
#   --output_root ............. path ......................... ./logs/video_qa
# END ARGS
"${ENTRY[@]}" \
  --model "$NAME" \
  --model_args "$MODEL_ARGS" \
  --tasks "$TASKS" \
  --batch_size "$BATCH_SIZE" \
  --log_samples --log_samples_suffix "$RUN_TAG" \
  --output_path "$OUTPUT_PATH" \
  ${LIMIT:+--limit "$LIMIT"}
