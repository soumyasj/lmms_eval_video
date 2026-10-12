#!/usr/bin/env bash
# Video QA on vLLM with the .mp4 forwarded AS A VIDEO (model `vllm_fix`).
#
# The stock vllm backend (run_video_qa_hf_multigpu.sh --backend vllm) decodes the
# video client-side and sends N frames as separate images -> Qwen2.5-VL gets
# N x <|image_pad|>, no temporal merge, no timestamps. This runner uses the
# out-of-tree plugin eval_scripts/runners/vllm_fix/, which hands vLLM a
# file:// URL so the server decodes it through the model's own video processor
# -> one <|video_pad|> block, like the hf backend. Nothing under lmms_eval/ is
# modified.
#
# Frame count / resolution on that path are set on the vLLM side; the plugin
# derives them from --nframes / --max_pixels / --min_pixels. fps sampling is not
# available here (vLLM's loader takes a frame count).
#
# Usage (from the repo root):
#   bash eval_scripts/runners/run_video_qa_vllm_fix.sh --nframes 32 --tasks videomme --seed 22
#   bash eval_scripts/runners/run_video_qa_vllm_fix.sh --nframes 32 --seed 22 \
#        --tasks metav
#
# ARGUMENTS (flag ............. values ...................... default)
#   --model ................... HF id ....................... Qwen/Qwen2.5-VL-7B-Instruct
#   --nframes ................. int, frames per video ....... 32
#   --max_pixels .............. int, per-frame budget ....... 602112
#   --min_pixels .............. int ......................... 200704
#   --tasks ................... task name(s) ................ videomme
#   --include_path ............ extra task YAML dir ......... <none>
#   --batch_size .............. int ......................... 1
#   --max_new_tokens .......... int | <empty> ............... <task yaml>
#   --temperature ............. float | <empty> ............. <task yaml>
#   --limit ................... int or 0-1 | <empty> ........ <all>
#   --seed .................... int or "a,b,c,d" | <empty> .. lmms_eval default
#   --seeds ................... N,N,...  one full run per seed
#   --run_tag ................. string ...................... <model>_<tasks>_nf<N>_mp<MP>_vllmfix[_bi]_bs<B>[_t<T>][_seed<S>]_<timestamp>
#   --output_root ............. path ........................ ./logs/normalized_runs/video_qa_hf
#   --num_gpus ................ int, data-parallel replicas . <autodetect>
#   --tensor_parallel_size .... int ......................... 1
#   --gpu_memory_utilization .. float ....................... 0.9
#   --max_model_len ........... int ......................... 32768
#   --max_num_seqs ............ int ......................... 8
#   --batch_invariant ......... 0 | 1 ....................... 0   (1 = VLLM_BATCH_INVARIANT, needs vllm >= 0.11.1 -> lmms_eval2 env)
#   --media_root .............. dir vLLM may read videos from  $HF_HOME
#   --main_port ............... int ......................... 29500+pid%1000
#   --dry_run ................. print the command and exit
# END ARGS
set -euo pipefail
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export LMMS_EVAL_DATASETS_CACHE="${LMMS_EVAL_DATASETS_CACHE:-$HF_HOME/datasets}"
if [[ -n "${LMMS_EVAL_ENV_BIN:-}" ]]; then export PATH="${LMMS_EVAL_ENV_BIN}:${PATH}"; fi   # else: the current environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # repo root

# ---- --seeds N,N,... : one full run per seed ----
ARGS=("$@")
for i in "${!ARGS[@]}"; do
  if [[ "${ARGS[$i]}" == "--seeds" ]]; then
    IFS=',' read -r -a SEED_LIST <<< "${ARGS[$((i+1))]}"
    REST=("${ARGS[@]:0:$i}" "${ARGS[@]:$((i+2))}")
    for s in "${SEED_LIST[@]}"; do echo "===== --seeds: seed ${s} ====="; bash "${BASH_SOURCE[0]}" "${REST[@]}" --seed "$s"; done
    exit 0
  fi
done

MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
NFRAMES=32; MAX_PIXELS=602112; MIN_PIXELS=200704
TASKS="videomme"; INCLUDE_PATH=""
BATCH_SIZE=1; MAX_NEW_TOKENS=""; TEMPERATURE=""; LIMIT=""; SEED=""
RUN_TAG=""; OUTPUT_ROOT="./logs/normalized_runs/video_qa_hf"
NUM_GPUS=""; MAIN_PORT=""; DRY_RUN=0
TENSOR_PARALLEL_SIZE=1; GPU_MEMORY_UTILIZATION=0.9; MAX_MODEL_LEN=32768; MAX_NUM_SEQS=8
BATCH_INVARIANT=0; MEDIA_ROOT="$HF_HOME"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)          MODEL="$2"; shift 2;;
    --nframes)        NFRAMES="$2"; shift 2;;
    --max_pixels)     MAX_PIXELS="$2"; shift 2;;
    --min_pixels)     MIN_PIXELS="$2"; shift 2;;
    --tasks)          TASKS="$2"; shift 2;;
    --include_path)   INCLUDE_PATH="$2"; shift 2;;
    --batch_size)     BATCH_SIZE="$2"; shift 2;;
    --max_new_tokens) MAX_NEW_TOKENS="$2"; shift 2;;
    --temperature)    TEMPERATURE="$2"; shift 2;;
    --limit)          LIMIT="$2"; shift 2;;
    --seed)           SEED="$2"; shift 2;;
    --run_tag)        RUN_TAG="$2"; shift 2;;
    --output_root)    OUTPUT_ROOT="$2"; shift 2;;
    --num_gpus)       NUM_GPUS="$2"; shift 2;;
    --main_port)      MAIN_PORT="$2"; shift 2;;
    --tensor_parallel_size)   TENSOR_PARALLEL_SIZE="$2"; shift 2;;
    --gpu_memory_utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2;;
    --max_model_len)          MAX_MODEL_LEN="$2"; shift 2;;
    --max_num_seqs)           MAX_NUM_SEQS="$2"; shift 2;;
    --batch_invariant)        BATCH_INVARIANT="$2"; shift 2;;
    --media_root)             MEDIA_ROOT="$2"; shift 2;;
    --fps|--max_frames|--sampling)
      echo "error: $1 is not available on the server-side video path (vLLM samples a frame count); use --nframes" >&2; exit 1;;
    --dry_run)        DRY_RUN=1; shift;;
    -h|--help)        sed -n '/^# ARGUMENTS/,/^# END ARGS/p' "$0"; exit 0;;
    *) echo "unknown option: $1" >&2; exit 1;;
  esac
done

[[ -d "${SCRIPT_DIR}/vllm_fix" ]] || { echo "error: plugin dir not found: ${SCRIPT_DIR}/vllm_fix" >&2; exit 1; }
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
[[ -d "$MEDIA_ROOT" ]] || { echo "error: --media_root ${MEDIA_ROOT} does not exist" >&2; exit 1; }

[[ "$BATCH_INVARIANT" == "0" || "$BATCH_INVARIANT" == "1" ]] || { echo "error: batch_invariant must be 0 or 1" >&2; exit 1; }
if [[ "$BATCH_INVARIANT" == "1" ]]; then
  python - <<'PY' || exit 1
try:
    from vllm.model_executor.layers.batch_invariant import vllm_is_batch_invariant  # noqa: F401
except ImportError:
    raise SystemExit("error: this vllm has no batch-invariance support (needs >= 0.11.1).\n"
                     "       Point LMMS_EVAL_ENV_BIN at the bin dir of an env with vllm >= 0.11.1")
PY
  export VLLM_BATCH_INVARIANT=1
fi

if [[ -z "$NUM_GPUS" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then NUM_GPUS=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
  else NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l); fi
fi
[[ "$NUM_GPUS" -lt 1 ]] && NUM_GPUS=1
[[ -z "$MAIN_PORT" ]] && MAIN_PORT=$(( 29500 + $$ % 1000 ))
DATA_PARALLEL_SIZE=$NUM_GPUS
WORLD_SIZE=$(( TENSOR_PARALLEL_SIZE * DATA_PARALLEL_SIZE ))

GEN_KWARGS=""
[[ -n "$MAX_NEW_TOKENS" ]] && GEN_KWARGS="${GEN_KWARGS:+${GEN_KWARGS},}max_new_tokens=${MAX_NEW_TOKENS}"
[[ -n "$TEMPERATURE" ]] && GEN_KWARGS="${GEN_KWARGS:+${GEN_KWARGS},}temperature=${TEMPERATURE}"

BI_TAG=""; [[ "$BATCH_INVARIANT" == "1" ]] && BI_TAG="_bi"
TEMP_TAG=""; [[ -n "$TEMPERATURE" ]] && TEMP_TAG="_t${TEMPERATURE//./p}"
SEED_TAG=""; [[ -n "$SEED" ]] && SEED_TAG="_seed${SEED//,/-}"
[[ -z "$RUN_TAG" ]] && RUN_TAG="${MODEL##*/}_${TASKS}_nf${NFRAMES}_mp${MAX_PIXELS}_vllmfix${BI_TAG}_bs${BATCH_SIZE}${TEMP_TAG}${SEED_TAG}_$(date +%Y%m%d_%H%M%S)"
OUTPUT_PATH="${OUTPUT_ROOT}/${RUN_TAG}"

# Same engine args as run_video_qa_hf_multigpu.sh --backend vllm; the plugin adds
# allowed_local_media_path / media_io_kwargs / mm_processor_kwargs from these.
MODEL_ARGS="model=${MODEL},max_frame_num=${NFRAMES},nframes=${NFRAMES},max_pixels=${MAX_PIXELS},min_pixels=${MIN_PIXELS},allowed_local_media_path=${MEDIA_ROOT},data_parallel_size=${DATA_PARALLEL_SIZE},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},max_model_len=${MAX_MODEL_LEN},max_num_seqs=${MAX_NUM_SEQS}"
# Batch-invariant mode forces FLASH_ATTN, whose bundled kernel lacks the Qwen-VL
# vision head dim (80); pin the vision encoder to SDPA as the hf runner does.
[[ "$BATCH_INVARIANT" == "1" ]] && MODEL_ARGS="${MODEL_ARGS},mm_encoder_attn_backend=TORCH_SDPA"

if [[ "$WORLD_SIZE" -gt 1 ]]; then
  LAUNCHER=(accelerate launch --num_processes "$WORLD_SIZE" --main_process_port "$MAIN_PORT" -m vllm_fix)
else
  LAUNCHER=(python -m vllm_fix)
fi
CMD=("${LAUNCHER[@]}" --model vllm_fix --model_args "$MODEL_ARGS" --tasks "$TASKS"
     --batch_size "$BATCH_SIZE" --log_samples --log_samples_suffix "$RUN_TAG" --output_path "$OUTPUT_PATH")
[[ -n "$INCLUDE_PATH" ]] && CMD+=(--include_path "$INCLUDE_PATH")
[[ -n "$LIMIT" ]]        && CMD+=(--limit "$LIMIT")
[[ -n "$SEED" ]]         && CMD+=(--seed "$SEED")
[[ -n "$GEN_KWARGS" ]]   && CMD+=(--gen_kwargs "$GEN_KWARGS")

EST_TOK=$(( NFRAMES / 2 * MAX_PIXELS / 784 ))
echo "run: vllm_fix (video_url -> <|video_pad|>) | nframes=${NFRAMES} (sampled by vLLM) | max_pixels=${MAX_PIXELS} ~vision_tok/prompt≈${EST_TOK} | tasks=${TASKS} | batch=${BATCH_SIZE} | limit=${LIMIT:-all}"
echo "     gpus=${NUM_GPUS} (data_parallel=${DATA_PARALLEL_SIZE}, tensor_parallel=${TENSOR_PARALLEL_SIZE}) batch_invariant=${BATCH_INVARIANT} media_root=${MEDIA_ROOT} -> ${OUTPUT_PATH}"
if [[ "$DRY_RUN" == "1" ]]; then printf 'DRY_RUN:'; printf ' %q' "${CMD[@]}"; echo; exit 0; fi
"${CMD[@]}"

if ! compgen -G "${OUTPUT_PATH}/*/*_results.json" > /dev/null; then
  echo "error: no *_results.json under ${OUTPUT_PATH} -- the evaluation did not finish." >&2
  exit 1
fi
