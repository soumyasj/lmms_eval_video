#!/usr/bin/env bash
#
# Video-QA runner for Qwen3-VL / Qwen3.5, multi-GPU capable.
#
# Rewritten to be flag-driven and self-contained: no YAML config layer, no
# embedded python. Every knob is a flag with a default below; the batch driver
# (eval_scripts/sweep_*.sh) passes them explicitly.
#
#   bash eval_scripts/runners/run_video_qa_qwen3vl_multigpu.sh --tasks videomme --nframes 32
#   --include_path <dir>   optional extra task YAML dir (every task, incl. metav and *_frames<N>, is in-tree);
#   bash eval_scripts/runners/run_video_qa_qwen3vl_multigpu.sh --backend vllm --nframes 32
#   bash eval_scripts/runners/run_video_qa_qwen3vl_multigpu.sh --fps 2 --max_frames 64
#
# ---------------------------------------------------------------------------
# What Qwen3-VL changes vs Qwen2.5-VL (why this is a separate runner)
# ---------------------------------------------------------------------------
# 1. TIMESTAMPS ARE TEXT. Qwen3VLProcessor prefixes every temporal patch with a
#    literal "<3.0 seconds>" string computed as frames_indices / native_fps
#    (transformers/models/qwen3_vl/processing_qwen3_vl.py). So the decode
#    backend and the sampling mode change the *prompt text*, not just pixels.
#    If video metadata is missing the processor silently assumes fps=24 and
#    every timestamp in the prompt is wrong -- grep the log for
#    "Qwen3VL requires frame timestamps" to catch that.
# 2. 16-PX PATCH GRID. simple/qwen3_vl.py passes image_patch_size=16, so
#    1 vision token = 32*32 = 1024 px (Qwen2.5-VL: 28*28 = 784). Pixel budgets
#    from the qwen2.5 scripts are NOT comparable; see --max_pixels below.
# 3. NO RANDOM FRAME SAMPLER. simple/qwen3_vl.py samples uniformly and its
#    __init__ ends with `assert kwargs == {}`, so the qwen2.5 runner's
#    frame_sampler / frame_sampler_seed args abort the run. Not accepted here.
#
# ---------------------------------------------------------------------------
# Known limitations that CANNOT be fixed without patching lmms_eval core
# ---------------------------------------------------------------------------
# * PROMPT BLOCK IS KEYED BY MODEL NAME. api/task.py resolves a task's
#   lmms_eval_specific_kwargs by the registered model name, so `--model qwen3_vl`
#   gets the task's `qwen3_vl:` block while `--backend vllm` (registered as
#   `vllm`) falls back to `default:`. An hf-vs-vllm comparison therefore also
#   changes the prompt. The runner warns loudly; it does not paper over it.
#   Clean fix if ever wanted: an out-of-tree task YAML via `--include_path`.
# * SHORT CLIPS + HIGH --nframes CRASH. qwen_vl_utils.smart_nframes raises
#   "nframes should in interval [2, total_frames]" when the request exceeds the
#   clip length (lmms-eval issue #874). Per-video clamping would need a model
#   change, so instead keep --nframes <= the shortest clip of the benchmark.
# * Only some tasks ship a `qwen3_vl:` prompt block (videomme does; mvbench,
#   mmvu, lvbench, vsibench do not). Those fall back to `default:` and will not
#   reproduce published Qwen3-VL numbers (lmms-eval issue #901).
set -euo pipefail

# Force one shared cache location. Without this, jobs that
# do not inherit the interactive shell env re-download datasets and get
# rate-limited. LMMS_EVAL_DATASETS_CACHE must not point at node-local /tmp: it
# gets reaped, after which `datasets` thinks the build finished and then fails
# to mmap a *-test.arrow that no longer exists.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export LMMS_EVAL_DATASETS_CACHE="${LMMS_EVAL_DATASETS_CACHE:-$HF_HOME/datasets}"
if [[ -n "${LMMS_EVAL_ENV_BIN:-}" ]]; then export PATH="${LMMS_EVAL_ENV_BIN}:${PATH}"; fi   # else: the current environment
# Frame-list tasks push every frame through the vision tower in one forward, which
# fragments the allocator badly (OOM tracebacks show GBs "reserved but unallocated").
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # repo root

# ---- --seeds a,b,c : rerun this script once per seed ----------------------
# Handled before the main parser so a single-seed run never sees it.
ARGS=("$@")
for i in "${!ARGS[@]}"; do
  if [[ "${ARGS[$i]}" == "--seeds" ]]; then
    REST=("${ARGS[@]:0:$i}" "${ARGS[@]:$((i+2))}")
    IFS=',' read -r -a SEED_LIST <<< "${ARGS[$((i+1))]}"
    for s in "${SEED_LIST[@]}"; do
      echo "===== seed ${s} ====="
      bash "${BASH_SOURCE[0]}" "${REST[@]}" --seed "$s"
    done
    exit 0
  fi
done

# ---- defaults -------------------------------------------------------------
MODEL="Qwen/Qwen3-VL-8B-Instruct"
BACKEND="hf"                 # hf | vllm
VIDEO_READER="torchcodec"    # decord | torchvision | torchcodec
SAMPLING="nframes"           # nframes | fps
NFRAMES=32
FPS=2                        # Qwen's own default sampling rate
MAX_FRAMES=64                # fps mode only: cap after rate sampling
# Pixel budget on the 32-px grid: tokens/frame-pair = max_pixels / 1024.
# 1048576 = 1024 tok = VIDEO_FRAME_MAX_PIXELS, the per-frame ceiling that
# qwen_vl_utils enforces for video. Upstream lmms-eval's own default is
# max_pixels=1605632, which video is clamped down to 1048576 anyway.
MAX_PIXELS=524288            # 512 tokens per temporal patch
MIN_PIXELS=131072            # 128 tok = VIDEO_MIN_TOKEN_NUM*32*32, the qwen_vl_utils video floor
# Qwen3-VL README: keep total_pixels below 24576*32*32 to avoid excessively long
# sequences. qwen_vl_utils otherwise defaults this to MODEL_SEQ_LEN*1024*0.9
# (~118M), roughly 4.7x looser than the reference setup.
TOTAL_PIXELS=""              # empty = do not pass (see the --total_pixels note below)
ATTN="auto"                  # auto | sdpa | flash_attention_2 | eager
DEVICE_MAP="auto"
TASKS="videomme"
BATCH_SIZE=1
MAX_NEW_TOKENS=""            # empty = task yaml's generation_kwargs
TEMPERATURE=""               # empty = task yaml's generation_kwargs
TOP_P=""                     # Qwen3-VL Instruct reference: 0.8 (Thinking: 0.95)
TOP_K=""                     # Qwen3-VL reference: 20
SYSTEM_PROMPT=""             # empty = model default ("You are a helpful assistant.")
ENABLE_THINKING=""           # empty | true | false
LIMIT=""
SEED=""
RUN_TAG=""
OUTPUT_ROOT="./logs/normalized_runs/video_qa_qwen3vl"
NUM_GPUS=""                  # empty = autodetect
MAIN_PORT=""
TENSOR_PARALLEL_SIZE=1       # vllm only
GPU_MEMORY_UTILIZATION=0.9   # vllm only
MAX_MODEL_LEN=""             # vllm only; empty = derived from the token estimate
MAX_NUM_SEQS=8               # vllm only
BATCH_INVARIANT=0            # vllm only: 1 = VLLM_BATCH_INVARIANT (needs vllm >= 0.11.1)

usage() { sed -n '2,60p' "$0"; exit 0; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)          MODEL="$2"; shift 2;;
    --backend)        BACKEND="$2"; shift 2;;
    --video_reader)   VIDEO_READER="$2"; shift 2;;
    --nframes)        NFRAMES="$2"; SAMPLING="nframes"; shift 2;;
    --fps)            FPS="$2";     SAMPLING="fps";     shift 2;;
    --max_frames)     MAX_FRAMES="$2"; shift 2;;
    --max_pixels)     MAX_PIXELS="$2"; shift 2;;
    --min_pixels)     MIN_PIXELS="$2"; shift 2;;
    --total_pixels)   TOTAL_PIXELS="$2"; shift 2;;
    --attn)           ATTN="$2"; shift 2;;
    --device_map)     DEVICE_MAP="$2"; shift 2;;
    --tasks)          TASKS="$2"; shift 2;;
    --batch_size)     BATCH_SIZE="$2"; shift 2;;
    --max_new_tokens) MAX_NEW_TOKENS="$2"; shift 2;;
    --temperature)    TEMPERATURE="$2"; shift 2;;
    --top_p)          TOP_P="$2"; shift 2;;
    --top_k)          TOP_K="$2"; shift 2;;
    --system_prompt)  SYSTEM_PROMPT="$2"; shift 2;;
    --enable_thinking) ENABLE_THINKING="$2"; shift 2;;
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

[[ "$BACKEND" == "vllm" && "$SAMPLING" == "fps" ]] && {
  echo "error: the vllm backend only supports nframes sampling, not fps." >&2; exit 1; }

[[ "$BATCH_INVARIANT" == "0" || "$BATCH_INVARIANT" == "1" ]] || {
  echo "error: --batch_invariant must be 0 or 1, got '${BATCH_INVARIANT}'" >&2; exit 1; }
if [[ "$BATCH_INVARIANT" == "1" ]]; then
  [[ "$BACKEND" == "vllm" ]] || {
    echo "error: --batch_invariant 1 is vllm-only (it sets VLLM_BATCH_INVARIANT; hf/transformers has no such mode)." >&2
    exit 1; }
  # Needs vllm >= 0.11.1; older vllm silently ignores the variable, which would
  # mislabel a normal run as invariant. Fail loudly instead.
  python - <<'PY' || exit 1
try:
    from vllm.model_executor.layers.batch_invariant import vllm_is_batch_invariant  # noqa: F401
except ImportError:
    raise SystemExit(
        "error: this vllm has no batch-invariance support (needs >= 0.11.1).\n"
        "       Point LMMS_EVAL_ENV_BIN at the bin dir of an env with vllm >= 0.11.1"
    )
PY
  export VLLM_BATCH_INVARIANT=1
fi

# total_pixels flips simple/qwen3_vl.py's _build_video_kwargs from `nframes` to
# `max_frames`, i.e. it silently turns an exact frame count into a cap. Refuse
# the ambiguous combination rather than mislabel the run.
if [[ -n "$TOTAL_PIXELS" && "$SAMPLING" == "nframes" ]]; then
  echo "error: --total_pixels changes nframes into a frame *cap* (max_frames)." >&2
  echo "       Use it with --fps, or drop it to keep an exact frame count." >&2
  exit 1
fi

# ---- derived --------------------------------------------------------------
if [[ "$SAMPLING" == "fps" ]]; then
  # The cap belongs in the tag: the same rate with different caps is a
  # different run and would otherwise be indistinguishable on disk.
  SAMPLE_ARGS="fps=${FPS},max_num_frames=${MAX_FRAMES}"
  SAMPLE_TAG="fps${FPS//./p}_cap${MAX_FRAMES}"
  FRAME_BUDGET="$MAX_FRAMES"
else
  SAMPLE_ARGS="max_num_frames=${NFRAMES}"
  SAMPLE_TAG="nf${NFRAMES}"
  FRAME_BUDGET="$NFRAMES"
fi

# Vision-token estimate on the 32-px grid (1 token = 1024 px).
# Video: frames are merged 2-at-a-time (FRAME_FACTOR), so (N/2) patches.
# Frame-list tasks (*_frames<N>): each frame is a separate IMAGE -- no temporal
# merge and no timestamps -- so roughly twice the tokens for the same N.
TOK_PER_PATCH=$(( MAX_PIXELS / 1024 ))
if [[ "$TASKS" == *_frames* ]]; then
  EST_TOK=$(( FRAME_BUDGET * TOK_PER_PATCH ))
  TOK_NOTE="frames-as-images: ${FRAME_BUDGET} x ${TOK_PER_PATCH} (no temporal merge, no timestamps)"
else
  EST_TOK=$(( FRAME_BUDGET / 2 * TOK_PER_PATCH ))
  TOK_NOTE="video: ${FRAME_BUDGET}/2 x ${TOK_PER_PATCH}"
fi

if [[ -z "$NUM_GPUS" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    NUM_GPUS=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
  else
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
  fi
fi
[[ "${NUM_GPUS:-0}" -lt 1 ]] && NUM_GPUS=1
[[ -z "$MAIN_PORT" ]] && MAIN_PORT=$(( 29500 + $$ % 1000 ))

# attn=auto -> flash_attention_2 when it is actually importable, else model default.
if [[ "$ATTN" == "auto" ]]; then
  if python -c "import flash_attn" >/dev/null 2>&1; then ATTN="flash_attention_2"; else ATTN=""; fi
fi

TEMP_TAG=""; [[ -n "$TEMPERATURE" ]] && TEMP_TAG="_t${TEMPERATURE//./p}"
SEED_TAG=""; [[ -n "$SEED" ]] && SEED_TAG="_seed${SEED//,/-}"
BI_TAG="";   [[ "$BATCH_INVARIANT" == "1" ]] && BI_TAG="_bi"
if [[ -z "$RUN_TAG" ]]; then
  RUN_TAG="${MODEL##*/}_${TASKS}_${SAMPLE_TAG}_mp${MAX_PIXELS}_${BACKEND}${BI_TAG}_vr${VIDEO_READER}_bs${BATCH_SIZE}${TEMP_TAG}${SEED_TAG}_$(date +%Y%m%d_%H%M%S)"
fi
OUTPUT_PATH="${OUTPUT_ROOT}/${RUN_TAG}"

# ---- gen kwargs -----------------------------------------------------------
GEN_KWARGS=""
add_gen() { [[ -n "$2" ]] && GEN_KWARGS="${GEN_KWARGS:+${GEN_KWARGS},}$1=$2"; return 0; }
add_gen max_new_tokens "$MAX_NEW_TOKENS"
add_gen temperature    "$TEMPERATURE"
add_gen top_p          "$TOP_P"
add_gen top_k          "$TOP_K"

# ---- model args -----------------------------------------------------------
if [[ "$BACKEND" == "vllm" ]]; then
  # chat/vllm.py defaults nframes=32 regardless of max_frame_num, so both must
  # be set or --nframes is silently ignored on this backend.
  if [[ -z "$MAX_MODEL_LEN" ]]; then
    # Vision tokens + headroom for the prompt, options and generation.
    MAX_MODEL_LEN=$(( EST_TOK + 4096 ))
    [[ "$MAX_MODEL_LEN" -lt 8192 ]] && MAX_MODEL_LEN=8192
  fi
  MODEL_ARGS="model=${MODEL},max_frame_num=${FRAME_BUDGET},nframes=${FRAME_BUDGET}"
  MODEL_ARGS="${MODEL_ARGS},max_pixels=${MAX_PIXELS}"
  MODEL_ARGS="${MODEL_ARGS},data_parallel_size=${NUM_GPUS},tensor_parallel_size=${TENSOR_PARALLEL_SIZE}"
  MODEL_ARGS="${MODEL_ARGS},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
  MODEL_ARGS="${MODEL_ARGS},max_model_len=${MAX_MODEL_LEN},max_num_seqs=${MAX_NUM_SEQS}"
  # vllm 0.11 keeps a 4 GB LRU cache of processed multimodal inputs shared
  # between the frontend and the engine core. Long videos (64+ frames) thrash
  # it and the engine asserts "Expected a cached item for mm_hash=...". Every
  # video here is unique so the cache never hits; turn it off.
  MODEL_ARGS="${MODEL_ARGS},mm_processor_cache_gb=0"
  # Images (the *_frames<N> tasks) bypass qwen_vl_utils on this backend, so the
  # pixel budget must reach vllm's HF processor directly. transformers 4.57
  # ignores max_pixels/min_pixels kwargs there; only `size` is honored.
  MODEL_ARGS="${MODEL_ARGS},mm_processor_kwargs={\"size\":{\"shortest_edge\":${MIN_PIXELS},\"longest_edge\":${MAX_PIXELS}}}"
  # Batch-invariant mode forces VLLM_ATTENTION_BACKEND=FLASH_ATTN globally, but
  # the Qwen-VL vision tower's head dim is not built into vllm's bundled
  # flash-attn (multiples of 32 only) and the engine crashes at profiling.
  # Pin the vision encoder to SDPA (per-item, deterministic); the LM keeps the
  # batch-invariant FLASH_ATTN path.
  [[ "$BATCH_INVARIANT" == "1" ]] && MODEL_ARGS="${MODEL_ARGS},mm_encoder_attn_backend=TORCH_SDPA"
  MODEL_NAME="vllm"
else
  MODEL_ARGS="pretrained=${MODEL},max_pixels=${MAX_PIXELS},min_pixels=${MIN_PIXELS},${SAMPLE_ARGS},device_map=${DEVICE_MAP}"
  [[ -n "$TOTAL_PIXELS" ]]    && MODEL_ARGS="${MODEL_ARGS},total_pixels=${TOTAL_PIXELS}"
  [[ -n "$ATTN" ]]            && MODEL_ARGS="${MODEL_ARGS},attn_implementation=${ATTN}"
  [[ -n "$SYSTEM_PROMPT" ]]   && MODEL_ARGS="${MODEL_ARGS},system_prompt=${SYSTEM_PROMPT}"
  [[ -n "$ENABLE_THINKING" ]] && MODEL_ARGS="${MODEL_ARGS},enable_thinking=${ENABLE_THINKING}"
  MODEL_NAME="qwen3_vl"
fi

# ---- banner ---------------------------------------------------------------
echo "run: qwen3_vl (${BACKEND}) | ${SAMPLE_ARGS} | vr=${VIDEO_READER} | tasks=${TASKS} | bs=${BATCH_SIZE} | limit=${LIMIT:-all}"
echo "     max_pixels=${MAX_PIXELS} (${TOK_PER_PATCH} tok) ~vision_tok/prompt~=${EST_TOK}  [${TOK_NOTE}]"
echo "     attn=${ATTN:-<model default>} gpus=${NUM_GPUS} -> ${OUTPUT_PATH}"
if [[ "$BACKEND" == "vllm" ]]; then
  echo "     vllm: dp=${NUM_GPUS} tp=${TENSOR_PARALLEL_SIZE} max_model_len=${MAX_MODEL_LEN} max_num_seqs=${MAX_NUM_SEQS}"
  echo "WARNING: the vllm backend registers as model name 'vllm', so tasks with a" >&2
  echo "         'qwen3_vl:' prompt block (e.g. videomme) fall back to 'default:'." >&2
  echo "         An hf-vs-vllm delta therefore includes a prompt change, not just a backend one." >&2
fi

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

# Out-of-tree task YAMLs (exp7 *_framesN variants for lvbench/longvideobench/
# mlvu/scivideobench live here instead of lmms_eval/tasks/).
INCLUDE_PATH="${INCLUDE_PATH_ARG:-}"   # optional extra task dir; all tasks are in-tree
[[ -z "$INCLUDE_PATH" || -d "$INCLUDE_PATH" ]] || { echo "error: include_path dir not found: ${INCLUDE_PATH}" >&2; exit 1; }

# DRY_RUN=1 prints the command instead of running it -- useful for checking the
# derived args (token estimate, max_model_len, run tag) without loading a model.
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'DRY_RUN:'
  printf ' %q' "${LAUNCHER[@]}" --model "$MODEL_NAME" --model_args "$MODEL_ARGS" \
    --tasks "$TASKS" ${INCLUDE_PATH:+--include_path "$INCLUDE_PATH"} \
    --batch_size "$BATCH_SIZE" --log_samples --log_samples_suffix "$RUN_TAG" \
    --output_path "$OUTPUT_PATH" ${LIMIT:+--limit "$LIMIT"} ${SEED:+--seed "$SEED"} \
    ${GEN_KWARGS:+--gen_kwargs "$GEN_KWARGS"}
  printf '\n'
  exit 0
fi

"${LAUNCHER[@]}" \
  --model "$MODEL_NAME" \
  --model_args "$MODEL_ARGS" \
  --tasks "$TASKS" \
  ${INCLUDE_PATH:+--include_path "$INCLUDE_PATH"} \
  --batch_size "$BATCH_SIZE" \
  --log_samples --log_samples_suffix "$RUN_TAG" \
  --output_path "$OUTPUT_PATH" \
  ${LIMIT:+--limit "$LIMIT"} \
  ${SEED:+--seed "$SEED"} \
  ${GEN_KWARGS:+--gen_kwargs "$GEN_KWARGS"}

# lmms_eval CATCHES evaluation errors, prints the traceback, and still exits 0
# (__main__.py cli_evaluate: `except Exception -> results_list.append(None)`).
# A zero exit is therefore NOT proof the eval ran, and the batch driver would
# happily write a "done" marker for a run that produced nothing. Require an
# actual results file instead.
if ! compgen -G "${OUTPUT_PATH}/*/*_results.json" > /dev/null; then
  echo "error: no *_results.json under ${OUTPUT_PATH}" >&2
  echo "       the evaluation did not finish -- see the traceback above." >&2
  exit 1
fi
