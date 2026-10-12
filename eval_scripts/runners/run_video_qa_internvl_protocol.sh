#!/usr/bin/env bash
#
# Protocol runner for the InternVL family (InternVL3 / InternVL3.5, HF format).
# Flag-driven counterpart of run_video_qa_internvl_multigpu.sh (left untouched)
# with a vllm backend, --seed / --seeds, DRY_RUN, a results-file check and the
# optional include path. Driven by eval_scripts/sweep_*.sh.
#
#   bash eval_scripts/runners/run_video_qa_internvl_protocol.sh --backend vllm --batch_invariant 1 --tasks videomme --nframes 32
#   bash eval_scripts/runners/run_video_qa_internvl_protocol.sh --backend hf --tasks videomme_frames8 --nframes 8 --seeds 22,42
#   --include_path <dir>   optional extra task YAML dir (every task, incl. metav and *_frames<N>, is in-tree);
#
# Backends
#   vllm (protocol default): lmms_eval's generic `vllm` model; vllm resolves the
#     HF-format checkpoint (InternVLForConditionalGeneration -> its InternS1
#     implementation). NOTE how that wrapper feeds video: it decodes the clip
#     with qwen_vl_utils (nframes uniform, max_pixels resize,
#     FORCE_QWENVL_VIDEO_READER) and sends the frames as base64 IMAGES -- vllm
#     never sees a video, so exp5 (.mp4) and exp7 (*_frames<N>) both reach the
#     model as N images. Same as the Qwen vllm runs.
#     Tiles: InternVL cuts every image into up to `max_patches` 448x448 tiles
#     (256 tok each) chosen by aspect ratio, so the token count per frame is
#     not fixed unless the tile budget is pinned. The runner passes
#     mm_processor_kwargs={"max_patches":M,"min_patches":m} (lmms_eval's vllm
#     wrapper JSON-decodes {...} model args and hands them to vllm) and sizes
#     the qwen_vl_utils resize to M tiles' worth of pixels.
#     Batch invariance: the language model is dense Qwen3 / Qwen2.5 attention,
#     so VLLM_BATCH_INVARIANT=1 applies (FLASH_ATTN needs an A100-class GPU);
#     the vision tower is pinned to SDPA like the Qwen runners. MoE ids
#     (30B-A3B, 241B-A28B) depend on vllm's MoE kernel -- unverified.
#   hf: lmms_eval `internvl_hf` (chat/internvl_hf.py). batch_size must be 1
#     (asserted in its __init__); frames via num_frames (exact) or fps; tiles
#     via min/max_patches; --video_size for a square resize. NOTE: for a raw
#     video input transformers' InternVLVideoProcessor ignores min/max_patches
#     (always one 448 tile per frame) -- the tile flags only affect images.
#   hf_tiled: `internvl_hf_tiled` from eval_scripts/runners/internvl_tiled/
#     (out-of-tree, subclass of internvl_hf). Reproduces the model card's
#     load_video(): frames at segment midpoints (decord), every frame through
#     the *image* tiler (min/max_patches = max_num, thumbnail when >1 tile,
#     ImageNet normalisation) and a "Frame{i}: <img>...</img>" prompt. This is
#     the backend where --max_patches actually changes a video run.
#     --video_reader here selects decord (default) or pyav for frame decoding.
# Context: InternVL3.5-8B-HF text max_position_embeddings = 40960, so 128
# frames must stay at 1 tile (128 x 256 = 32768 tok). No task yaml has an
# internvl prompt block: all tasks use their `default:`.
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
MODEL="OpenGVLab/InternVL3_5-8B-HF"
BACKEND="vllm"               # vllm | hf | hf_tiled
VIDEO_READER="torchcodec"    # vllm (qwen_vl_utils decoder): decord | torchvision | torchcodec; hf_tiled: decord | pyav
SAMPLING="nframes"           # nframes | fps (fps: hf only)
NFRAMES=32
FPS=1
MIN_PATCHES=1
MAX_PATCHES=1                # 448x448 tiles per frame, 256 tok each; 12 = model's image default
MAX_PIXELS=""                # vllm only, qwen_vl_utils resize; empty = MAX_PATCHES x 448x448
VIDEO_SIZE=""                # hf only: px, square resize of sampled frames
ATTN=""                      # hf only: sdpa | flash_attention_2 | eager | empty = model default
DEVICE_MAP="auto"            # hf only
TASKS="videomme"
BATCH_SIZE=1                 # hf: must be 1
MAX_NEW_TOKENS=""
TEMPERATURE=""
LIMIT=""
SEED=""
RUN_TAG=""
OUTPUT_ROOT="./logs/normalized_runs/video_qa_internvl"
NUM_GPUS=""
MAIN_PORT=""
TENSOR_PARALLEL_SIZE=1       # vllm only
GPU_MEMORY_UTILIZATION=0.9   # vllm only
MAX_MODEL_LEN=""             # vllm only; empty = derived from the token estimate
MAX_NUM_SEQS=8               # vllm only
BATCH_INVARIANT=0            # vllm only: 1 = VLLM_BATCH_INVARIANT

usage() { sed -n '2,35p' "$0"; exit 0; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)          MODEL="$2"; shift 2;;
    --backend)        BACKEND="$2"; shift 2;;
    --video_reader)   VIDEO_READER="$2"; shift 2;;
    --nframes|--num_frames) NFRAMES="$2"; SAMPLING="nframes"; shift 2;;
    --fps)            FPS="$2"; SAMPLING="fps"; shift 2;;
    --min_patches)    MIN_PATCHES="$2"; shift 2;;
    --max_patches)    MAX_PATCHES="$2"; shift 2;;
    --max_pixels)     MAX_PIXELS="$2"; shift 2;;
    --video_size)     VIDEO_SIZE="$2"; shift 2;;
    --attn)           ATTN="$2"; shift 2;;
    --device_map)     DEVICE_MAP="$2"; shift 2;;
    --tasks)          TASKS="$2"; shift 2;;
    --batch_size)     BATCH_SIZE="$2"; shift 2;;
    --max_new_tokens) MAX_NEW_TOKENS="$2"; shift 2;;
    --temperature)    TEMPERATURE="$2"; shift 2;;
    --limit)          LIMIT="$2"; shift 2;;
    --seed)           SEED="$2"; shift 2;;
    --run_tag)        RUN_TAG="$2"; shift 2;;
    --include_path)   INCLUDE_PATH_ARG="$2"; shift 2;;
    --output_root)    OUTPUT_ROOT="$2"; shift 2;;
    --num_gpus)       NUM_GPUS="$2"; shift 2;;
    --main_port)      MAIN_PORT="$2"; shift 2;;
    --tensor_parallel_size)   TENSOR_PARALLEL_SIZE="$2"; shift 2;;
    --gpu_memory_utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2;;
    --max_model_len)          MAX_MODEL_LEN="$2"; shift 2;;
    --max_num_seqs)           MAX_NUM_SEQS="$2"; shift 2;;
    --batch_invariant)        BATCH_INVARIANT="$2"; shift 2;;
    -h|--help)        usage;;
    *) echo "unknown option: $1" >&2; exit 1;;
  esac
done

# ---- validate -------------------------------------------------------------
[[ "$BACKEND" == "hf" || "$BACKEND" == "hf_tiled" || "$BACKEND" == "vllm" ]] || {
  echo "error: --backend must be hf, hf_tiled or vllm, got '${BACKEND}'" >&2; exit 1; }
case "$VIDEO_READER" in
  decord|torchvision|torchcodec) ;;
  pyav) [[ "$BACKEND" == "hf_tiled" ]] || { echo "error: --video_reader pyav is hf_tiled-only" >&2; exit 1; };;
  *) echo "error: --video_reader must be decord|torchvision|torchcodec|pyav, got '${VIDEO_READER}'" >&2; exit 1;;
esac
if [[ "$BACKEND" == "hf_tiled" ]]; then
  case "$VIDEO_READER" in decord|pyav) ;; *) VIDEO_READER=decord;; esac   # torchcodec/torchvision are qwen_vl_utils names
  [[ -d "${SELF_DIR}/internvl_tiled" ]] || { echo "error: plugin dir not found: ${SELF_DIR}/internvl_tiled" >&2; exit 1; }
  export PYTHONPATH="${SELF_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
fi
export FORCE_QWENVL_VIDEO_READER="$VIDEO_READER"

[[ "$BACKEND" == "vllm" && "$SAMPLING" == "fps" ]] && {
  echo "error: --fps is hf-only here; the vllm path uses --nframes." >&2; exit 1; }
[[ "$BACKEND" != "vllm" && "$BATCH_SIZE" != "1" ]] && {
  echo "error: internvl_hf(_tiled) only supports batch_size=1, got '${BATCH_SIZE}'" >&2; exit 1; }

[[ "$BATCH_INVARIANT" == "0" || "$BATCH_INVARIANT" == "1" ]] || {
  echo "error: --batch_invariant must be 0 or 1, got '${BATCH_INVARIANT}'" >&2; exit 1; }
if [[ "$BATCH_INVARIANT" == "1" ]]; then
  [[ "$BACKEND" == "vllm" ]] || {
    echo "error: --batch_invariant 1 is vllm-only (it sets VLLM_BATCH_INVARIANT; hf/transformers has no such mode)." >&2
    exit 1; }
  python - <<'PY' || exit 1
import importlib
try:
    bi = importlib.import_module("vllm.model_executor.layers.batch_invariant")
    assert any(hasattr(bi, f) for f in ("vllm_is_batch_invariant", "init_batch_invariance", "enable_batch_invariant_mode"))
    import vllm.envs as envs
    assert hasattr(envs, "VLLM_BATCH_INVARIANT")
except Exception as e:  # noqa: BLE001
    raise SystemExit(f"error: this vllm has no batch-invariance support (needs >= 0.11.1): {e!r}")
PY
  export VLLM_BATCH_INVARIANT=1
fi

# ---- derived --------------------------------------------------------------
[[ -z "$MAX_PIXELS" ]] && MAX_PIXELS=$(( MAX_PATCHES * 448 * 448 ))
TOK_PER_FRAME=$(( 256 * MAX_PATCHES ))
# image tiler adds a thumbnail tile when the grid has >1 tile (upper bound:
# the aspect-ratio grid search may pick fewer tiles than MAX_PATCHES)
[[ "$BACKEND" == "hf_tiled" && "$MAX_PATCHES" -gt 1 ]] && TOK_PER_FRAME=$(( 256 * (MAX_PATCHES + 1) ))
if [[ "$SAMPLING" == "fps" ]]; then
  SAMPLE_ARGS="fps=${FPS}"; SAMPLE_TAG="fps${FPS//./p}"; FRAME_BUDGET="$NFRAMES"
else
  SAMPLE_ARGS="num_frames=${NFRAMES}"; SAMPLE_TAG="nf${NFRAMES}"; FRAME_BUDGET="$NFRAMES"
fi
EST_TOK=$(( FRAME_BUDGET * TOK_PER_FRAME ))

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
BI_TAG="";   [[ "$BATCH_INVARIANT" == "1" ]] && BI_TAG="_bi"
# Tag layout <model>_<task>_nf<N>_..._seed<S>_<ts> is what the batch driver's
# have_results() globs on (*_nf<N>_*_seed<S>_*). Keep it.
if [[ -z "$RUN_TAG" ]]; then
  if [[ "$BACKEND" == "vllm" ]]; then
    RUN_TAG="${MODEL##*/}_${TASKS}_${SAMPLE_TAG}_mp${MAX_PATCHES}_vllm${BI_TAG}_vr${VIDEO_READER}_bs${BATCH_SIZE}${TEMP_TAG}${SEED_TAG}_$(date +%Y%m%d_%H%M%S)"
  elif [[ "$BACKEND" == "hf_tiled" ]]; then
    RUN_TAG="${MODEL##*/}_${TASKS}_${SAMPLE_TAG}_mp${MAX_PATCHES}_hftiled_vr${VIDEO_READER}_bs${BATCH_SIZE}${TEMP_TAG}${SEED_TAG}_$(date +%Y%m%d_%H%M%S)"
  else
    RUN_TAG="${MODEL##*/}_${TASKS}_${SAMPLE_TAG}_mp${MAX_PATCHES}_hf_bs${BATCH_SIZE}${TEMP_TAG}${SEED_TAG}_$(date +%Y%m%d_%H%M%S)"
  fi
fi
OUTPUT_PATH="${OUTPUT_ROOT}/${RUN_TAG}"

GEN_KWARGS=""
[[ -n "$MAX_NEW_TOKENS" ]] && GEN_KWARGS="${GEN_KWARGS:+${GEN_KWARGS},}max_new_tokens=${MAX_NEW_TOKENS}"
[[ -n "$TEMPERATURE" ]]    && GEN_KWARGS="${GEN_KWARGS:+${GEN_KWARGS},}temperature=${TEMPERATURE}"

# ---- model args -----------------------------------------------------------
if [[ "$BACKEND" == "vllm" ]]; then
  if [[ -z "$MAX_MODEL_LEN" ]]; then
    MAX_MODEL_LEN=$(( EST_TOK + 4096 ))
    [[ "$MAX_MODEL_LEN" -lt 8192 ]] && MAX_MODEL_LEN=8192
  fi
  # chat/vllm.py defaults nframes=32 regardless of max_frame_num: set both.
  MODEL_ARGS="model=${MODEL},max_frame_num=${FRAME_BUDGET},nframes=${FRAME_BUDGET},max_pixels=${MAX_PIXELS}"
  MODEL_ARGS="${MODEL_ARGS},mm_processor_kwargs={\"max_patches\":${MAX_PATCHES},\"min_patches\":${MIN_PATCHES}}"
  MODEL_ARGS="${MODEL_ARGS},data_parallel_size=${NUM_GPUS},tensor_parallel_size=${TENSOR_PARALLEL_SIZE}"
  MODEL_ARGS="${MODEL_ARGS},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
  MODEL_ARGS="${MODEL_ARGS},max_model_len=${MAX_MODEL_LEN},max_num_seqs=${MAX_NUM_SEQS}"
  # Batch-invariant mode forces FLASH_ATTN globally; pin the vision tower to SDPA.
  [[ "$BATCH_INVARIANT" == "1" ]] && MODEL_ARGS="${MODEL_ARGS},mm_encoder_attn_backend=TORCH_SDPA"
  MODEL_NAME="vllm"
else
  MODEL_ARGS="pretrained=${MODEL},${SAMPLE_ARGS},min_patches=${MIN_PATCHES},max_patches=${MAX_PATCHES},device_map=${DEVICE_MAP}"
  [[ -n "$ATTN" ]]       && MODEL_ARGS="${MODEL_ARGS},attn_implementation=${ATTN}"
  [[ -n "$VIDEO_SIZE" ]] && MODEL_ARGS="${MODEL_ARGS},video_size=${VIDEO_SIZE}"
  if [[ "$BACKEND" == "hf_tiled" ]]; then
    MODEL_ARGS="${MODEL_ARGS},video_backend=${VIDEO_READER}"
    MODEL_NAME="internvl_hf_tiled"
  else
    MODEL_NAME="internvl_hf"
  fi
fi

# ---- banner ---------------------------------------------------------------
echo "run: internvl (${BACKEND}) | ${SAMPLE_ARGS} | tasks=${TASKS} | bs=${BATCH_SIZE} | limit=${LIMIT:-all} | seed=${SEED:-<none>}"
echo "     max_patches=${MAX_PATCHES} min_patches=${MIN_PATCHES} (${TOK_PER_FRAME} tok/frame) ~vision_tok/prompt~=${EST_TOK}"
if [[ "$BACKEND" == "vllm" ]]; then
  echo "     vllm: max_pixels=${MAX_PIXELS} vr=${VIDEO_READER} dp=${NUM_GPUS} tp=${TENSOR_PARALLEL_SIZE} max_model_len=${MAX_MODEL_LEN} max_num_seqs=${MAX_NUM_SEQS} batch_invariant=${BATCH_INVARIANT}  [frames-as-images via qwen_vl_utils]"
elif [[ "$BACKEND" == "hf_tiled" ]]; then
  echo "     hf_tiled: frames -> image tiler (max_num=${MAX_PATCHES}, +thumbnail if >1) vr=${VIDEO_READER} attn=${ATTN:-<default>} gpus=${NUM_GPUS}  [plugin: internvl_tiled]"
else
  echo "     hf: video_size=${VIDEO_SIZE:-<native>} attn=${ATTN:-<default>} gpus=${NUM_GPUS}  [max_patches is inert for video input]"
fi
echo "     -> ${OUTPUT_PATH}"

# ---- launch ---------------------------------------------------------------
if [[ "$BACKEND" == "vllm" ]]; then
  WORLD_SIZE=$(( TENSOR_PARALLEL_SIZE * NUM_GPUS ))
else
  WORLD_SIZE="$NUM_GPUS"
fi
ENTRY=lmms_eval
[[ "$BACKEND" == "hf_tiled" ]] && ENTRY=internvl_tiled   # registers internvl_hf_tiled, then lmms_eval's CLI
if [[ "$WORLD_SIZE" -gt 1 ]]; then
  LAUNCHER=(accelerate launch --num_processes "$WORLD_SIZE" --main_process_port "$MAIN_PORT" -m "$ENTRY")
else
  LAUNCHER=(python -m "$ENTRY")
fi
INCLUDE_PATH="${INCLUDE_PATH_ARG:-}"   # optional extra task dir; all tasks are in-tree
[[ -z "$INCLUDE_PATH" || -d "$INCLUDE_PATH" ]] || { echo "error: include_path dir not found: ${INCLUDE_PATH}" >&2; exit 1; }

CMD=("${LAUNCHER[@]}" --model "$MODEL_NAME" --model_args "$MODEL_ARGS"
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
