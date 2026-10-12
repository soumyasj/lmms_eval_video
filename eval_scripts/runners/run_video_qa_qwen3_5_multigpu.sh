#!/usr/bin/env bash
#
# Video-QA runner for Qwen3.5 (Qwen/Qwen3.5-*), multi-GPU capable.
#
# Derived from run_video_qa_qwen3vl_multigpu.sh (kept as the reference; this
# file is a separate copy so the Qwen3-VL runner stays untouched). Flag-driven
# and self-contained; the batch driver (eval_scripts/sweep_*.sh)
# passes every knob explicitly.
#
#   bash eval_scripts/runners/run_video_qa_qwen3_5_multigpu.sh --tasks videomme --nframes 32
#   --include_path <dir>   optional extra task YAML dir (every task, incl. metav and *_frames<N>, is in-tree);
#   bash eval_scripts/runners/run_video_qa_qwen3_5_multigpu.sh --backend vllm --nframes 32
#   bash eval_scripts/runners/run_video_qa_qwen3_5_multigpu.sh --fps 2 --max_frames 64
#
# ---------------------------------------------------------------------------
# What Qwen3.5 changes vs Qwen3-VL (why this is a separate runner)
# ---------------------------------------------------------------------------
# 1. REGISTERED MODEL NAME. lmms_eval ships simple/qwen3_5.py, registered as
#    `qwen3_5` (a thin subclass of qwen3_vl with Qwen3.5 defaults). The Qwen3-VL
#    runner hardcodes --model qwen3_vl; here the hf backend uses `qwen3_5`.
#    NOTE: no task yaml in this tree has a `qwen3_5:` prompt block (videomme
#    only has `qwen3_vl:`), so under this name every task resolves to its
#    `default:` block -- same as the vllm backend does. hf and vllm therefore
#    see the same prompt here, unlike the Qwen3-VL runner.
# 2. THINKING IS ON BY DEFAULT in simple/qwen3_5.py (enable_thinking=True,
#    max_new_tokens=1024, temperature=0.7). Every checkpoint is a hybrid
#    thinking model (no separate -Instruct / -Thinking variants). For the MCQ
#    protocol sweep this runner defaults to --enable_thinking false so the
#    answer is emitted directly and the task's generation_kwargs (e.g.
#    videomme: max_new_tokens=16, temperature=0) are not fighting a <think>
#    block. Pass --enable_thinking true to opt in.
#    On the vllm backend the generic wrapper cannot pass chat_template_kwargs,
#    and the Qwen3.5-9B chat template opens a live <think> block unless
#    enable_thinking is explicitly false (the 0.8B template defaults the other
#    way) -- that made every MCQ answer a ~4k-token reasoning trace. So with
#    thinking off the runner fetches the checkpoint's own chat_template.jinja
#    (HF cache), prepends `{%- set enable_thinking = false -%}`, writes it to
#    chat_templates/generated/<model>_nothink.jinja and passes it via the
#    wrapper's `chat_template=` model arg. --chat_template <file> overrides,
#    --chat_template none keeps the model default.
# 3. MODEL CLASS. simple/qwen3_vl.py picks Qwen3_5(Moe)ForConditionalGeneration
#    from config.model_type. That class only exists in transformers >= 5.x
#    (4.57 has qwen3_vl but not qwen3_5), and vllm needs a build whose registry
#    knows Qwen3_5ForConditionalGeneration (0.11.x does not). The preflight
#    below checks both and fails loudly instead of tracebacking mid-load.
# 4. Everything else is inherited from Qwen3-VL: text timestamps ("<3.0
#    seconds>") in the prompt, the 16-px patch grid (1 token = 32*32 = 1024 px),
#    uniform sampling only (no random frame sampler), 256K native context.
#
# Known limitations (see the Qwen3-VL runner header, all still apply):
#   * short clips + high --nframes crash in qwen_vl_utils.smart_nframes;
#     keep --nframes <= the shortest clip of the benchmark.
set -euo pipefail

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export LMMS_EVAL_DATASETS_CACHE="${LMMS_EVAL_DATASETS_CACHE:-$HF_HOME/datasets}"
if [[ -n "${LMMS_EVAL_ENV_BIN:-}" ]]; then export PATH="${LMMS_EVAL_ENV_BIN}:${PATH}"; fi   # else: the current environment
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Absolute path of this script and its directory, resolved BEFORE the cd below
# so the per-seed re-invocation keeps working
# when the runner is invoked by a relative path (e.g. from a git worktree).
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
MODEL="Qwen/Qwen3.5-4B"
BACKEND="hf"                 # hf | vllm
VIDEO_READER="torchcodec"    # decord | torchvision | torchcodec
SAMPLING="nframes"           # nframes | fps
NFRAMES=32
FPS=2
MAX_FRAMES=64                # fps mode only: cap after rate sampling
# Pixel budget on the 32-px grid: tokens/frame-pair = max_pixels / 1024.
# Matches the upstream lmms-eval Qwen3.5 HF wrapper (simple/qwen3_5.py:
# max_pixels = 128*32*32 = 131072 -> 128 tok/patch, min_pixels = 64*32*32),
# and is passed to BOTH backends so vllm and hf see the same per-frame budget
# (chat/vllm.py would otherwise default to 1605632 / ~1568 tok). The Qwen3-VL
# runner uses 524288 (512 tok) -- pass --max_pixels 524288 to match that sweep.
MAX_PIXELS=131072
MIN_PIXELS=65536
TOTAL_PIXELS=""              # empty = do not pass (flips nframes into a cap, see below)
ATTN="auto"                  # auto | sdpa | flash_attention_2 | eager
DEVICE_MAP="auto"
TASKS="videomme"
BATCH_SIZE=1
MAX_NEW_TOKENS=""            # empty = task yaml's generation_kwargs
TEMPERATURE=""               # empty = task yaml's generation_kwargs
TOP_P=""                     # Qwen3.5 reference: 0.8 (non-thinking) / 0.95 (thinking)
TOP_K=""                     # Qwen3.5 reference: 20
SYSTEM_PROMPT=""
ENABLE_THINKING="false"      # true | false  (see header note 2)
CHAT_TEMPLATE=""             # vllm only: "" = derive <model>_nothink.jinja when thinking is off; "none" = model default; or a .jinja path
LIMIT=""
SEED=""
RUN_TAG=""
OUTPUT_ROOT="./logs/normalized_runs/video_qa_qwen3_5"
NUM_GPUS=""                  # empty = autodetect
MAIN_PORT=""
TENSOR_PARALLEL_SIZE=1       # vllm only
GPU_MEMORY_UTILIZATION=0.9   # vllm only
MAX_MODEL_LEN=""             # vllm only; empty = derived from the token estimate
MAX_NUM_SEQS=8               # vllm only
BATCH_INVARIANT=0            # vllm only: 1 = VLLM_BATCH_INVARIANT (needs vllm >= 0.11.1)
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"   # 1 = do not probe transformers/vllm for Qwen3.5 support

usage() { sed -n '2,45p' "$0"; exit 0; }

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
    --chat_template)  CHAT_TEMPLATE="$2"; shift 2;;
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

case "$ENABLE_THINKING" in
  true|false) ;;
  *) echo "error: --enable_thinking must be true or false, got '${ENABLE_THINKING}'" >&2; exit 1;;
esac

[[ "$BACKEND" == "vllm" && "$SAMPLING" == "fps" ]] && {
  echo "error: the vllm backend only supports nframes sampling, not fps." >&2; exit 1; }

[[ "$BATCH_INVARIANT" == "0" || "$BATCH_INVARIANT" == "1" ]] || {
  echo "error: --batch_invariant must be 0 or 1, got '${BATCH_INVARIANT}'" >&2; exit 1; }
if [[ "$BATCH_INVARIANT" == "1" ]]; then
  [[ "$BACKEND" == "vllm" ]] || {
    echo "error: --batch_invariant 1 is vllm-only (it sets VLLM_BATCH_INVARIANT; hf/transformers has no such mode)." >&2
    exit 1; }
  # Qwen3.5 = Gated DeltaNet hybrid. vllm 0.28's engine start dies with
  # "VLLM batch_invariant mode is not supported for GDN_ATTN" -- after
  # downloading and loading the full checkpoint. Fail before that instead.
  case "${MODEL##*/}" in
    Qwen3.5*|Qwen3_5*|qwen3.5*|qwen3_5*)
      echo "error: vllm batch-invariant mode is not supported for Qwen3.5 (GDN_ATTN linear-attention layers)." >&2
      echo "       Run with --batch_invariant 0 (BATCH_INVARIANT=0 in the batch driver) or use --backend hf." >&2
      echo "       Override with QWEN35_FORCE_BI=1 if a newer vllm supports it." >&2
      [[ "${QWEN35_FORCE_BI:-0}" == "1" ]] || exit 1;;
  esac
  # vllm 0.11.x exposes vllm_is_batch_invariant(); 0.28 renamed the helpers
  # (init_batch_invariance / enable_batch_invariant_mode) but keeps the
  # VLLM_BATCH_INVARIANT env flag. Accept either API.
  python - <<'PY' || exit 1
import importlib
try:
    bi = importlib.import_module("vllm.model_executor.layers.batch_invariant")
    assert any(hasattr(bi, f) for f in ("vllm_is_batch_invariant", "init_batch_invariance", "enable_batch_invariant_mode"))
    import vllm.envs as envs
    assert hasattr(envs, "VLLM_BATCH_INVARIANT")
except Exception as e:  # noqa: BLE001
    raise SystemExit(
        f"error: this vllm has no batch-invariance support (needs >= 0.11.1): {e!r}\n"
        "       Point LMMS_EVAL_ENV_BIN at the bin dir of an env with transformers >= 5"
    )
PY
  export VLLM_BATCH_INVARIANT=1
fi

if [[ -n "$TOTAL_PIXELS" && "$SAMPLING" == "nframes" ]]; then
  echo "error: --total_pixels changes nframes into a frame *cap* (max_frames)." >&2
  echo "       Use it with --fps, or drop it to keep an exact frame count." >&2
  exit 1
fi

# ---- preflight: does this env actually know the Qwen3.5 architecture? ------
# simple/qwen3_vl.py does `from transformers import Qwen3_5ForConditionalGeneration`
# once config.model_type contains "qwen3_5"; on transformers 4.57 that is an
# ImportError after the (slow) config download. vllm likewise rejects an
# architecture missing from its registry only at engine start. Probe both up
# front so a wrong env fails in seconds with a message that names the fix.
if [[ "$SKIP_PREFLIGHT" != "1" ]]; then
  python - "$BACKEND" <<'PY' || exit 1
import sys
backend = sys.argv[1]
problems = []
try:
    import transformers
    from transformers import Qwen3_5ForConditionalGeneration  # noqa: F401
except Exception:
    problems.append(
        f"transformers {getattr(transformers, '__version__', '?')} has no Qwen3_5ForConditionalGeneration "
        "(Qwen3.5 needs transformers >= 5.x)."
    )
if backend == "vllm":
    try:
        import vllm
        from vllm.model_executor.models.registry import ModelRegistry
        archs = set(ModelRegistry.get_supported_archs())
        if not any(a.startswith("Qwen3_5") for a in archs):
            problems.append(
                f"vllm {vllm.__version__} registry has no Qwen3_5* architecture "
                "(0.11.x predates Qwen3.5; needs a newer vllm)."
            )
    except Exception as e:  # noqa: BLE001
        problems.append(f"could not inspect the vllm registry: {e!r}")
if problems:
    msg = "\n".join("       - " + p for p in problems)
    raise SystemExit(
        "error: this env cannot run Qwen3.5:\n" + msg +
        "\n       Point LMMS_EVAL_ENV_BIN at an env with Qwen3.5 support, or "
        "SKIP_PREFLIGHT=1 to try anyway."
    )
PY
fi

# ---- derived --------------------------------------------------------------
if [[ "$SAMPLING" == "fps" ]]; then
  SAMPLE_ARGS="fps=${FPS},max_num_frames=${MAX_FRAMES}"
  SAMPLE_TAG="fps${FPS//./p}_cap${MAX_FRAMES}"
  FRAME_BUDGET="$MAX_FRAMES"
else
  SAMPLE_ARGS="max_num_frames=${NFRAMES}"
  SAMPLE_TAG="nf${NFRAMES}"
  FRAME_BUDGET="$NFRAMES"
fi

# Vision-token estimate on the 32-px grid (1 token = 1024 px). Video frames
# are merged 2-at-a-time; frame-list tasks (*_frames<N>) send each frame as a
# separate image (no temporal merge, no timestamps) -> ~2x the tokens.
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

if [[ "$ATTN" == "auto" ]]; then
  if python -c "import flash_attn" >/dev/null 2>&1; then ATTN="flash_attention_2"; else ATTN=""; fi
fi

TEMP_TAG="";  [[ -n "$TEMPERATURE" ]] && TEMP_TAG="_t${TEMPERATURE//./p}"
SEED_TAG="";  [[ -n "$SEED" ]] && SEED_TAG="_seed${SEED//,/-}"
BI_TAG="";    [[ "$BATCH_INVARIANT" == "1" ]] && BI_TAG="_bi"
# Thinking changes the output distribution entirely, so it belongs in the tag.
THINK_TAG=""; [[ "$ENABLE_THINKING" == "true" ]] && THINK_TAG="_think"
if [[ -z "$RUN_TAG" ]]; then
  RUN_TAG="${MODEL##*/}_${TASKS}_${SAMPLE_TAG}_mp${MAX_PIXELS}_${BACKEND}${BI_TAG}${THINK_TAG}_vr${VIDEO_READER}_bs${BATCH_SIZE}${TEMP_TAG}${SEED_TAG}_$(date +%Y%m%d_%H%M%S)"
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
  if [[ -z "$MAX_MODEL_LEN" ]]; then
    MAX_MODEL_LEN=$(( EST_TOK + 4096 ))
    [[ "$MAX_MODEL_LEN" -lt 8192 ]] && MAX_MODEL_LEN=8192
  fi
  # chat/vllm.py defaults nframes=32 regardless of max_frame_num: set both.
  MODEL_ARGS="model=${MODEL},max_frame_num=${FRAME_BUDGET},nframes=${FRAME_BUDGET}"
  # chat/vllm.py names the floor min_image_pixels (default 28 px); pass ours so
  # the vllm frame budget matches the hf wrapper's min_pixels as well.
  MODEL_ARGS="${MODEL_ARGS},max_pixels=${MAX_PIXELS},min_image_pixels=${MIN_PIXELS}"
  MODEL_ARGS="${MODEL_ARGS},data_parallel_size=${NUM_GPUS},tensor_parallel_size=${TENSOR_PARALLEL_SIZE}"
  MODEL_ARGS="${MODEL_ARGS},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
  MODEL_ARGS="${MODEL_ARGS},max_model_len=${MAX_MODEL_LEN},max_num_seqs=${MAX_NUM_SEQS}"
  # Batch-invariant mode forces FLASH_ATTN globally; the vision tower's head
  # dim is not built into vllm's flash-attn, so pin the encoder to SDPA.
  [[ "$BATCH_INVARIANT" == "1" ]] && MODEL_ARGS="${MODEL_ARGS},mm_encoder_attn_backend=TORCH_SDPA"
  # Thinking switch (header note 2): derive a per-checkpoint template with
  # enable_thinking preset to false, unless the caller supplied one.
  if [[ "$CHAT_TEMPLATE" == "none" ]]; then
    CHAT_TEMPLATE=""
  elif [[ -z "$CHAT_TEMPLATE" && "$ENABLE_THINKING" == "false" ]]; then
    CHAT_TEMPLATE="${SELF_DIR}/chat_templates/generated/${MODEL##*/}_nothink.jinja"
    python - "$MODEL" "$CHAT_TEMPLATE" <<'PY' || exit 1
import os, sys
model, out = sys.argv[1], sys.argv[2]
src = None
if os.path.isdir(model):
    p = os.path.join(model, "chat_template.jinja")
    src = open(p).read() if os.path.exists(p) else None
else:
    from huggingface_hub import hf_hub_download
    try:
        src = open(hf_hub_download(model, "chat_template.jinja")).read()
    except Exception:  # noqa: BLE001  (older repos keep it in tokenizer_config.json)
        src = None
if src is None:
    from transformers import AutoTokenizer
    src = AutoTokenizer.from_pretrained(model).chat_template
if not src or "enable_thinking" not in src:
    raise SystemExit(f"error: could not derive a thinking-off template for {model}: no enable_thinking switch in its chat template")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    f.write("{#- generated by run_video_qa_qwen3_5_multigpu.sh: " + model + " chat template with enable_thinking preset to false -#}\n")
    f.write("{%- set enable_thinking = false -%}\n")
    f.write(src)
print(f"chat template (thinking off) -> {out}")
PY
  elif [[ -n "$CHAT_TEMPLATE" && ! -f "$CHAT_TEMPLATE" ]]; then
    echo "error: chat template not found: ${CHAT_TEMPLATE}" >&2; exit 1
  fi
  [[ -n "$CHAT_TEMPLATE" ]] && MODEL_ARGS="${MODEL_ARGS},chat_template=${CHAT_TEMPLATE}"
  MODEL_NAME="vllm"
else
  MODEL_ARGS="pretrained=${MODEL},max_pixels=${MAX_PIXELS},min_pixels=${MIN_PIXELS},${SAMPLE_ARGS},device_map=${DEVICE_MAP}"
  MODEL_ARGS="${MODEL_ARGS},enable_thinking=${ENABLE_THINKING}"
  [[ -n "$TOTAL_PIXELS" ]]  && MODEL_ARGS="${MODEL_ARGS},total_pixels=${TOTAL_PIXELS}"
  [[ -n "$ATTN" ]]          && MODEL_ARGS="${MODEL_ARGS},attn_implementation=${ATTN}"
  [[ -n "$SYSTEM_PROMPT" ]] && MODEL_ARGS="${MODEL_ARGS},system_prompt=${SYSTEM_PROMPT}"
  MODEL_NAME="qwen3_5"
fi

# ---- banner ---------------------------------------------------------------
echo "run: qwen3_5 (${BACKEND}) | ${SAMPLE_ARGS} | vr=${VIDEO_READER} | tasks=${TASKS} | bs=${BATCH_SIZE} | limit=${LIMIT:-all}"
echo "     max_pixels=${MAX_PIXELS} (${TOK_PER_PATCH} tok) ~vision_tok/prompt~=${EST_TOK}  [${TOK_NOTE}]"
echo "     attn=${ATTN:-<model default>} thinking=${ENABLE_THINKING} gpus=${NUM_GPUS} -> ${OUTPUT_PATH}"
if [[ "$BACKEND" == "vllm" ]]; then
  echo "     vllm: dp=${NUM_GPUS} tp=${TENSOR_PARALLEL_SIZE} max_model_len=${MAX_MODEL_LEN} max_num_seqs=${MAX_NUM_SEQS} chat_template=${CHAT_TEMPLATE:-<model default>}"
  if [[ "$ENABLE_THINKING" == "true" && -z "$CHAT_TEMPLATE" ]]; then
    echo "NOTE: thinking=true on vllm relies on the model's default template (Qwen3.5-9B thinks by default, 0.8B does not)." >&2
  fi
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

# Out-of-tree task YAMLs (exp7 *_framesN variants) live here.
INCLUDE_PATH="${INCLUDE_PATH_ARG:-}"   # optional extra task dir; all tasks are in-tree
[[ -z "$INCLUDE_PATH" || -d "$INCLUDE_PATH" ]] || { echo "error: include_path dir not found: ${INCLUDE_PATH}" >&2; exit 1; }

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

# lmms_eval catches evaluation errors and still exits 0; require a results file.
if ! compgen -G "${OUTPUT_PATH}/*/*_results.json" > /dev/null; then
  echo "error: no *_results.json under ${OUTPUT_PATH}" >&2
  echo "       the evaluation did not finish -- see the traceback above." >&2
  exit 1
fi
