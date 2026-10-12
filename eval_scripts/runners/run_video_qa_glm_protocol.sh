#!/usr/bin/env bash
#
# Protocol runner for the GLM-V family (GLM-4.6V / GLM-4.6V-Flash / GLM-4.5V).
# Flag-driven counterpart of run_video_qa_glm_multigpu.sh (left untouched) with
# a vllm backend, --seed / --seeds, DRY_RUN, a results-file check and the
# optional include path. Driven by eval_scripts/sweep_*.sh.
#
#   bash eval_scripts/runners/run_video_qa_glm_protocol.sh --backend vllm --batch_invariant 1 --tasks videomme --nframes 32
#   bash eval_scripts/runners/run_video_qa_glm_protocol.sh --backend hf --tasks videomme_frames32 --nframes 32
#   --include_path <dir>   optional extra task YAML dir (every task, incl. metav and *_frames<N>, is in-tree);
#
# Backends
#   vllm (protocol default): lmms_eval's generic `vllm` model. NOTE how that
#     wrapper feeds video: it decodes the clip with qwen_vl_utils (nframes
#     uniform, max_pixels resize, FORCE_QWENVL_VIDEO_READER) and sends the
#     frames as a list of base64 IMAGES -- vllm never sees a video. So a raw
#     .mp4 task (exp5) and a *_frames<N> task (exp7) both reach the model as N
#     images; they differ only in decoder + resize. Same as the Qwen vllm runs.
#     Batch invariance: GLM-4.6V-Flash is dense attention, so vllm's
#     VLLM_BATCH_INVARIANT=1 applies (FLASH_ATTN needs an A100-class GPU); the
#     vision tower is pinned to SDPA like the Qwen runners. MoE ids (GLM-4.6V,
#     GLM-4.5V) depend on vllm's MoE kernel declaring support -- unverified.
#     Thinking: the GLM chat template thinks by default and the wrapper cannot
#     pass chat_template_kwargs, so --chat_template defaults to
#     chat_templates/glm4v_nothink.jinja (verbatim template with
#     enable_thinking preset to false: "/nothink" suffix + empty <think/>).
#   hf: lmms_eval `glm4v` (simple/glm4v.py). Frame-list tasks only: it has no
#     nframes/fps/max_pixels args and consumes decoded PIL frames, so a raw
#     video task is refused on this backend. Needs transformers >= 5.
# Tokens per frame on vllm: GLM uses a 14-px patch with 2x2 merge -> 28-px
# grid, 1 token = 784 px. Default max_pixels 401408 = 512 tok/frame, matching
# the Qwen3.5 protocol's 512 tok. Context: 131072 (config.json).
# No task yaml has a glm prompt block: all tasks use their `default:`.
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
MODEL="zai-org/GLM-4.6V-Flash"
BACKEND="vllm"               # vllm | hf
VIDEO_READER="torchcodec"    # vllm only (qwen_vl_utils decoder): decord | torchvision | torchcodec
NFRAMES=32
MAX_PIXELS=401408            # vllm only: 512 tok/frame on the 28-px grid
CHAT_TEMPLATE="${SELF_DIR}/chat_templates/glm4v_nothink.jinja"   # vllm only; "" = model default (thinking ON)
ATTN=""                      # hf only: sdpa | flash_attention_2 | eager | empty = model default
DEVICE_MAP="auto"            # hf only
TASKS="videomme"
BATCH_SIZE=1
MAX_NEW_TOKENS=""
TEMPERATURE=""
SYSTEM_PROMPT=""             # hf only
LIMIT=""
SEED=""
RUN_TAG=""
OUTPUT_ROOT="./logs/normalized_runs/video_qa_glm"
NUM_GPUS=""
MAIN_PORT=""
TENSOR_PARALLEL_SIZE=1       # vllm only
GPU_MEMORY_UTILIZATION=0.9   # vllm only
MAX_MODEL_LEN=""             # vllm only; empty = derived from the token estimate
MAX_NUM_SEQS=8               # vllm only
BATCH_INVARIANT=0            # vllm only: 1 = VLLM_BATCH_INVARIANT
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"

usage() { sed -n '2,33p' "$0"; exit 0; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)          MODEL="$2"; shift 2;;
    --backend)        BACKEND="$2"; shift 2;;
    --video_reader)   VIDEO_READER="$2"; shift 2;;
    --nframes)        NFRAMES="$2"; shift 2;;
    --max_pixels)     MAX_PIXELS="$2"; shift 2;;
    --chat_template)  CHAT_TEMPLATE="$2"; shift 2;;
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
[[ "$BACKEND" == "hf" || "$BACKEND" == "vllm" ]] || {
  echo "error: --backend must be hf or vllm, got '${BACKEND}'" >&2; exit 1; }
case "$VIDEO_READER" in
  decord|torchvision|torchcodec) ;;
  *) echo "error: --video_reader must be decord|torchvision|torchcodec, got '${VIDEO_READER}'" >&2; exit 1;;
esac
export FORCE_QWENVL_VIDEO_READER="$VIDEO_READER"

if [[ "$BACKEND" == "hf" && "$TASKS" != *_frames* ]]; then
  echo "error: on --backend hf, glm4v only takes precomputed frames; use a *_frames<N> task (e.g. ${TASKS}_frames${NFRAMES})," >&2
  echo "       or --backend vllm, whose wrapper decodes the video into ${NFRAMES} frames itself." >&2
  exit 1
fi
if [[ "$TASKS" == *_frames* && "$TASKS" != *_frames${NFRAMES} ]]; then
  echo "error: --nframes ${NFRAMES} does not match task '${TASKS}'" >&2; exit 1
fi

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

if [[ "$BACKEND" == "vllm" && -n "$CHAT_TEMPLATE" && ! -f "$CHAT_TEMPLATE" ]]; then
  echo "error: chat template not found: ${CHAT_TEMPLATE}" >&2; exit 1
fi

# ---- preflight: transformers >= 5 (glm4v wrapper and vllm's HF processor) --
if [[ "$SKIP_PREFLIGHT" != "1" ]]; then
  python - <<'PY' || exit 1
try:
    import transformers
    from transformers import Glm4vForConditionalGeneration  # noqa: F401
except Exception:
    raise SystemExit(
        f"error: transformers {getattr(transformers, '__version__', '?')} has no Glm4vForConditionalGeneration;"
        " GLM-V needs transformers >= 5.0. Point LMMS_EVAL_ENV_BIN at the lmms_eval3 env, or SKIP_PREFLIGHT=1."
    )
PY
fi

# ---- derived --------------------------------------------------------------
# vllm: every frame is one image on the 28-px grid (no temporal merge).
TOK_PER_FRAME=$(( MAX_PIXELS / 784 ))
EST_TOK=$(( NFRAMES * TOK_PER_FRAME ))

if [[ -z "$NUM_GPUS" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    NUM_GPUS=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
  else
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
  fi
fi
[[ "${NUM_GPUS:-0}" -lt 1 ]] && NUM_GPUS=1
[[ -z "$MAIN_PORT" ]] && MAIN_PORT=$(( 29500 + $$ % 1000 ))

TEMP_TAG="";  [[ -n "$TEMPERATURE" ]] && TEMP_TAG="_t${TEMPERATURE//./p}"
SEED_TAG="";  [[ -n "$SEED" ]] && SEED_TAG="_seed${SEED}"
BI_TAG="";    [[ "$BATCH_INVARIANT" == "1" ]] && BI_TAG="_bi"
THINK_TAG=""; [[ "$BACKEND" == "vllm" && -z "$CHAT_TEMPLATE" ]] && THINK_TAG="_think"
# Tag layout <model>_<task>_nf<N>_..._seed<S>_<ts> is what the batch driver's
# have_results() globs on (*_nf<N>_*_seed<S>_*). Keep it.
if [[ -z "$RUN_TAG" ]]; then
  if [[ "$BACKEND" == "vllm" ]]; then
    RUN_TAG="${MODEL##*/}_${TASKS}_nf${NFRAMES}_mp${MAX_PIXELS}_vllm${BI_TAG}${THINK_TAG}_vr${VIDEO_READER}_bs${BATCH_SIZE}${TEMP_TAG}${SEED_TAG}_$(date +%Y%m%d_%H%M%S)"
  else
    RUN_TAG="${MODEL##*/}_${TASKS}_nf${NFRAMES}_hf_bs${BATCH_SIZE}${TEMP_TAG}${SEED_TAG}_$(date +%Y%m%d_%H%M%S)"
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
  MODEL_ARGS="model=${MODEL},max_frame_num=${NFRAMES},nframes=${NFRAMES},max_pixels=${MAX_PIXELS}"
  MODEL_ARGS="${MODEL_ARGS},data_parallel_size=${NUM_GPUS},tensor_parallel_size=${TENSOR_PARALLEL_SIZE}"
  MODEL_ARGS="${MODEL_ARGS},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
  MODEL_ARGS="${MODEL_ARGS},max_model_len=${MAX_MODEL_LEN},max_num_seqs=${MAX_NUM_SEQS}"
  [[ -n "$CHAT_TEMPLATE" ]] && MODEL_ARGS="${MODEL_ARGS},chat_template=${CHAT_TEMPLATE}"
  # Batch-invariant mode forces FLASH_ATTN globally; pin the vision tower to SDPA
  # (same reason as the Qwen runners: ViT head dim vs bundled flash-attn).
  [[ "$BATCH_INVARIANT" == "1" ]] && MODEL_ARGS="${MODEL_ARGS},mm_encoder_attn_backend=TORCH_SDPA"
  MODEL_NAME="vllm"
else
  MODEL_ARGS="pretrained=${MODEL},device_map=${DEVICE_MAP}"
  [[ -n "$ATTN" ]]          && MODEL_ARGS="${MODEL_ARGS},attn_implementation=${ATTN}"
  [[ -n "$SYSTEM_PROMPT" ]] && MODEL_ARGS="${MODEL_ARGS},system_prompt=${SYSTEM_PROMPT}"
  MODEL_NAME="glm4v"
fi

# ---- banner ---------------------------------------------------------------
echo "run: glm (${BACKEND}) | nframes=${NFRAMES} | tasks=${TASKS} | bs=${BATCH_SIZE} | limit=${LIMIT:-all} | seed=${SEED:-<none>}"
if [[ "$BACKEND" == "vllm" ]]; then
  echo "     max_pixels=${MAX_PIXELS} (${TOK_PER_FRAME} tok/frame) ~vision_tok/prompt~=${EST_TOK}  [frames-as-images via qwen_vl_utils, vr=${VIDEO_READER}]"
  echo "     vllm: dp=${NUM_GPUS} tp=${TENSOR_PARALLEL_SIZE} max_model_len=${MAX_MODEL_LEN} max_num_seqs=${MAX_NUM_SEQS} batch_invariant=${BATCH_INVARIANT} chat_template=${CHAT_TEMPLATE:-<model default, thinking ON>}"
else
  echo "     hf: attn=${ATTN:-<default>} gpus=${NUM_GPUS}"
fi
echo "     -> ${OUTPUT_PATH}"

# ---- launch ---------------------------------------------------------------
if [[ "$BACKEND" == "vllm" ]]; then
  WORLD_SIZE=$(( TENSOR_PARALLEL_SIZE * NUM_GPUS ))
else
  WORLD_SIZE="$NUM_GPUS"
fi
if [[ "$WORLD_SIZE" -gt 1 ]]; then
  LAUNCHER=(accelerate launch --num_processes "$WORLD_SIZE" --main_process_port "$MAIN_PORT" -m lmms_eval)
else
  LAUNCHER=(python -m lmms_eval)
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
