#!/usr/bin/env bash

# Video-QA runner, multi-GPU capable. Backend is selectable: hf (plain
# transformers generate) or vllm (paged attention + continuous batching,
# much higher throughput, same answers).
#
# Settings live in video_qa_hf.yaml next to this script -- edit that file and run:
#   bash eval_scripts/runners/run_video_qa_hf_multigpu.sh
#
# Flags override the config for one-off runs; --config points at another file:
#   bash eval_scripts/runners/run_video_qa_hf_multigpu.sh --nframes 8 --limit 50
#   bash eval_scripts/runners/run_video_qa_hf_multigpu.sh --backend vllm
#   bash eval_scripts/runners/run_video_qa_hf_multigpu.sh --config my_sweep.yaml
#
# Multi-GPU is data parallelism by default: one full model replica per GPU,
# each evaluating a shard of the dataset. Not model sharding -- for a model
# too big for one GPU use --backend vllm with --tensor_parallel_size > 1.
set -euo pipefail
# Force one shared cache location for the videos.
# Without this, jobs that don't inherit the interactive shell's env re-download
# the whole dataset into ~/.cache and get rate-limited by the HF hub.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# Arrow build cache, kept next to the raw downloads so it persists across nodes.
# This used to point at /tmp, which is node-local and reaped: `datasets` recorded
# the build as finished, skipped rebuilding it on the next run, then failed to
# mmap a *-test.arrow that no longer existed. Pointing it into HF_HOME means an
# already-downloaded dataset is never fetched again.
export LMMS_EVAL_DATASETS_CACHE="${LMMS_EVAL_DATASETS_CACHE:-$HF_HOME/datasets}"
if [[ -n "${LMMS_EVAL_ENV_BIN:-}" ]]; then export PATH="${LMMS_EVAL_ENV_BIN}:${PATH}"; fi   # else: the current environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # repo root

# ---- --seeds N,N,... : run this script once per seed, back to back ----
# Pulled out before the main arg parser so a single-seed run (--seed) doesn't
# need to know about looping at all.
ARGS=("$@")
for i in "${!ARGS[@]}"; do
  if [[ "${ARGS[$i]}" == "--seeds" ]]; then
    SEEDS_VALUE="${ARGS[$((i+1))]}"
    REST=("${ARGS[@]:0:$i}" "${ARGS[@]:$((i+2))}")
    IFS=',' read -r -a SEED_LIST <<< "$SEEDS_VALUE"
    for s in "${SEED_LIST[@]}"; do
      echo "===== --seeds: launching seed ${s} ====="
      bash "${BASH_SOURCE[0]}" "${REST[@]}" --seed "$s"
    done
    exit 0
  fi
done

# ---- defaults (used when a key is in neither the config nor the flags) ----
declare -A CFG=(
  [model]="Qwen/Qwen2.5-VL-7B-Instruct"
  [backend]="hf"             # hf | vllm
  [video_reader]="decord"    # decord | torchvision | torchcodec
  [sampling]="nframes"       # nframes | fps  (mutually exclusive; vllm: nframes only)
  [frame_sampler]="uniform"  # uniform | random  (which frames, not how many; hf only)
  [nframes]=4
  [fps]=1
  [max_frames]=32            # fps mode only: cap on frames after rate sampling
  [max_pixels]=301056
  [min_pixels]=200704        # 256*28*28, model default (hf only)
  [attn]=""                  # "" | sdpa | flash_attention_2 | eager (hf only)
  [device_map]="auto"        # hf only
  [tasks]="videomme"
  [include_path]=""          # extra task YAML dir for lmms_eval --include_path (optional; every task is in-tree)
  [batch_size]=1
  [max_new_tokens]=""        # empty = use the task yaml's generation_kwargs
  [temperature]=""           # empty = use the task yaml's generation_kwargs
  [limit]=""                 # empty = all samples
  [seed]=""                  # empty = lmms_eval default (0,1234,1234,1234)
  [run_tag]=""
  [output_root]="./logs/normalized_runs/video_qa_hf"
  [num_gpus]=""              # empty = autodetect
  [main_port]=""             # empty = derived from PID
  [tensor_parallel_size]=1   # vllm only
  [gpu_memory_utilization]=0.9  # vllm only
  [max_model_len]=32768      # vllm only
  [max_num_seqs]=8           # vllm only
  [batch_invariant]=0        # vllm only: 1 = VLLM_BATCH_INVARIANT (needs vllm >= 0.11.1)
)

# ---- parse flags first, so we know which keys the CLI owns ----
declare -A CLI=()
CONFIG="${SCRIPT_DIR}/video_qa_hf.yaml"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)         CONFIG="$2"; shift 2;;
    --model)          CLI[model]="$2"; shift 2;;
    --backend)        CLI[backend]="$2"; shift 2;;
    --video_reader)   CLI[video_reader]="$2"; shift 2;;
    --sampling)       CLI[sampling]="$2"; shift 2;;
    --frame_sampler)  CLI[frame_sampler]="$2"; shift 2;;
    --nframes)        CLI[nframes]="$2"; CLI[sampling]="nframes"; shift 2;;
    --fps)            CLI[fps]="$2";     CLI[sampling]="fps";     shift 2;;
    --max_frames)     CLI[max_frames]="$2"; shift 2;;
    --max_pixels)     CLI[max_pixels]="$2"; shift 2;;
    --min_pixels)     CLI[min_pixels]="$2"; shift 2;;
    --attn)           CLI[attn]="$2"; shift 2;;
    --device_map)     CLI[device_map]="$2"; shift 2;;
    --tasks)          CLI[tasks]="$2"; shift 2;;
    --include_path)   CLI[include_path]="$2"; shift 2;;
    --batch_size)     CLI[batch_size]="$2"; shift 2;;
    --max_new_tokens) CLI[max_new_tokens]="$2"; shift 2;;
    --temperature)    CLI[temperature]="$2"; shift 2;;
    --limit)          CLI[limit]="$2"; shift 2;;
    --seed)           CLI[seed]="$2"; shift 2;;
    --run_tag)        CLI[run_tag]="$2"; shift 2;;
    --output_root)    CLI[output_root]="$2"; shift 2;;
    --num_gpus)       CLI[num_gpus]="$2"; shift 2;;
    --main_port)      CLI[main_port]="$2"; shift 2;;
    --tensor_parallel_size)   CLI[tensor_parallel_size]="$2"; shift 2;;
    --gpu_memory_utilization) CLI[gpu_memory_utilization]="$2"; shift 2;;
    --max_model_len)          CLI[max_model_len]="$2"; shift 2;;
    --max_num_seqs)           CLI[max_num_seqs]="$2"; shift 2;;
    --batch_invariant)        CLI[batch_invariant]="$2"; shift 2;;
    -h|--help)        sed -n '/^# ARGUMENTS/,/^# END ARGS/p' "$0"; exit 0;;
    *) echo "unknown option: $1" >&2; exit 1;;
  esac
done

# ---- layer the config over the defaults ----
if [[ -f "$CONFIG" ]]; then
  YAML_ASSIGNMENTS=$(python - "$CONFIG" "${!CFG[@]}" <<'PY'
import shlex, sys, yaml
path, valid = sys.argv[1], set(sys.argv[2:])
data = yaml.safe_load(open(path)) or {}
if not isinstance(data, dict):
    sys.exit(f"config {path} must be a mapping of key: value")
for key, value in data.items():
    if key not in valid:
        sys.exit(f"unknown key '{key}' in {path}\nvalid keys: {', '.join(sorted(valid))}")
    if value is None:          # blank value = fall back to the script default
        continue
    if isinstance(value, bool):
        value = str(value).lower()
    print(f"CFG[{key}]={shlex.quote(str(value))}")
PY
  ) || { echo "error: could not read config ${CONFIG}" >&2; exit 1; }
  eval "$YAML_ASSIGNMENTS"
  CONFIG_NOTE="config: ${CONFIG}"
elif [[ "$CONFIG" != "${SCRIPT_DIR}/video_qa_hf.yaml" ]]; then
  echo "error: config not found: ${CONFIG}" >&2; exit 1
else
  CONFIG_NOTE="config: <none found, using built-in defaults>"
fi

# ---- flags win over the config ----
for key in "${!CLI[@]}"; do CFG[$key]="${CLI[$key]}"; done

MODEL="${CFG[model]}";           BACKEND="${CFG[backend]}";   SAMPLING="${CFG[sampling]}"
FRAME_SAMPLER="${CFG[frame_sampler]}"
VIDEO_READER="${CFG[video_reader]}"
NFRAMES="${CFG[nframes]}";       FPS="${CFG[fps]}";           MAX_FRAMES="${CFG[max_frames]}"
MAX_PIXELS="${CFG[max_pixels]}"; MIN_PIXELS="${CFG[min_pixels]}"
ATTN="${CFG[attn]}";             DEVICE_MAP="${CFG[device_map]}"
TASKS="${CFG[tasks]}";           BATCH_SIZE="${CFG[batch_size]}";  LIMIT="${CFG[limit]}"
INCLUDE_PATH="${CFG[include_path]}"
SEED="${CFG[seed]}"
MAX_NEW_TOKENS="${CFG[max_new_tokens]}"; TEMPERATURE="${CFG[temperature]}"
RUN_TAG="${CFG[run_tag]}";       OUTPUT_ROOT="${CFG[output_root]}"
NUM_GPUS="${CFG[num_gpus]}";     MAIN_PORT="${CFG[main_port]}"
TENSOR_PARALLEL_SIZE="${CFG[tensor_parallel_size]}"
GPU_MEMORY_UTILIZATION="${CFG[gpu_memory_utilization]}"
MAX_MODEL_LEN="${CFG[max_model_len]}"
MAX_NUM_SEQS="${CFG[max_num_seqs]}"
BATCH_INVARIANT="${CFG[batch_invariant]}"

# ---- merge gen-kwargs style overrides into a single --gen_kwargs string ----
GEN_KWARGS=""
[[ -n "$MAX_NEW_TOKENS" ]] && GEN_KWARGS="${GEN_KWARGS:+${GEN_KWARGS},}max_new_tokens=${MAX_NEW_TOKENS}"
[[ -n "$TEMPERATURE" ]] && GEN_KWARGS="${GEN_KWARGS:+${GEN_KWARGS},}temperature=${TEMPERATURE}"

[[ "$BACKEND" == "hf" || "$BACKEND" == "vllm" ]] || {
  echo "error: backend must be 'hf' or 'vllm', got '${BACKEND}'" >&2; exit 1; }

[[ -z "$VIDEO_READER" || "$VIDEO_READER" == "decord" || "$VIDEO_READER" == "torchvision" || "$VIDEO_READER" == "torchcodec" ]] || {
  echo "error: video_reader must be 'decord', 'torchvision', or 'torchcodec', got '${VIDEO_READER}'" >&2; exit 1; }
[[ -n "$VIDEO_READER" ]] && export FORCE_QWENVL_VIDEO_READER="$VIDEO_READER"

[[ "$SAMPLING" == "nframes" || "$SAMPLING" == "fps" ]] || {
  echo "error: sampling must be 'nframes' or 'fps', got '${SAMPLING}'" >&2; exit 1; }

# --frame_sampler picks *which* frames out of the ones the video reader decoded;
# --nframes/--fps still decide *how many*. The two are orthogonal.
[[ "$FRAME_SAMPLER" == "uniform" || "$FRAME_SAMPLER" == "random" ]] || {
  echo "error: frame_sampler must be 'uniform' or 'random', got '${FRAME_SAMPLER}'" >&2; exit 1; }

if [[ "$FRAME_SAMPLER" != "uniform" ]]; then
  # A stochastic sampler with no seed makes a run unreproducible, so require one.
  [[ -n "$SEED" ]] || {
    echo "error: --frame_sampler ${FRAME_SAMPLER} is random; pass --seed N (or --seeds N,N,...) so the run is reproducible." >&2
    exit 1; }
  [[ "$BACKEND" == "hf" ]] || {
    echo "error: --frame_sampler ${FRAME_SAMPLER} is hf-only; the vllm path does its own frame selection." >&2
    exit 1; }
fi
# lmms_eval --seed takes "random,numpy,torch,fewshot"; the frame sampler gets the first.
FRAME_SAMPLER_SEED="${SEED%%,*}"

[[ "$BACKEND" == "vllm" && "$SAMPLING" == "fps" ]] && {
  echo "error: vllm backend only supports nframes-style uniform sampling, not fps." >&2; exit 1; }

[[ "$BATCH_INVARIANT" == "0" || "$BATCH_INVARIANT" == "1" ]] || {
  echo "error: batch_invariant must be 0 or 1, got '${BATCH_INVARIANT}'" >&2; exit 1; }
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

# ---- derived ----
if [[ "$SAMPLING" == "fps" ]]; then
  # The cap belongs in the tag: fps sampling at the same rate but different
  # caps produces different runs, and without it they are indistinguishable
  # on disk (exp4 sweeps caps 32/64/128 at 1 fps).
  SAMPLE="fps=${FPS},max_num_frames=${MAX_FRAMES}"; SAMPLE_TAG="fps${FPS//./p}_cap${MAX_FRAMES}"; FRAME_BUDGET=$MAX_FRAMES
else
  # --max_frames is an fps-mode cap; passing it here would be silently ignored.
  if [[ -n "${CLI[max_frames]:-}" ]]; then
    echo "error: --max_frames only applies in fps mode (it caps frames after rate sampling)." >&2
    echo "       For ${MAX_FRAMES} uniformly-sampled frames use: --nframes ${MAX_FRAMES}" >&2
    echo "       For fps sampling capped at ${MAX_FRAMES}   use: --fps ${FPS} --max_frames ${MAX_FRAMES}" >&2
    exit 1
  fi
  SAMPLE="max_num_frames=${NFRAMES}";               SAMPLE_TAG="nf${NFRAMES}";   FRAME_BUDGET=$NFRAMES
fi

# how many data-parallel replicas: config/flag > CUDA_VISIBLE_DEVICES > all visible GPUs
if [[ -z "$NUM_GPUS" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    NUM_GPUS=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
  else
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
  fi
fi
[[ "$NUM_GPUS" -lt 1 ]] && NUM_GPUS=1
[[ -z "$MAIN_PORT" ]] && MAIN_PORT=$(( 29500 + $$ % 1000 ))

TEMP_TAG=""
[[ -n "$TEMPERATURE" ]] && TEMP_TAG="_t${TEMPERATURE//./p}"
SEED_TAG=""
[[ -n "$SEED" ]] && SEED_TAG="_seed${SEED//,/-}"
# Only tagged when non-default, so uniform runs keep the paths they always had.
FS_TAG=""
[[ "$FRAME_SAMPLER" != "uniform" ]] && FS_TAG="_fs${FRAME_SAMPLER}"
BI_TAG=""
[[ "$BATCH_INVARIANT" == "1" ]] && BI_TAG="_bi"
[[ -z "$RUN_TAG" ]] && RUN_TAG="${MODEL##*/}_${TASKS}_${SAMPLE_TAG}_mp${MAX_PIXELS}_${BACKEND}${BI_TAG}_vr${VIDEO_READER:-torchcodec}_bs${BATCH_SIZE}${FS_TAG}${TEMP_TAG}${SEED_TAG}_$(date +%Y%m%d_%H%M%S)"
OUTPUT_PATH="${OUTPUT_ROOT}/${RUN_TAG}"

if [[ "$BACKEND" == "vllm" ]]; then
  # vllm chat model (lmms_eval/models/chat/vllm.py) defaults nframes=32 unless
  # explicitly passed here -- max_frame_num alone does NOT control per-request
  # frame count, so --nframes was previously silently ignored on this backend.
  DATA_PARALLEL_SIZE=$NUM_GPUS
  MODEL_ARGS="model=${MODEL},max_frame_num=${FRAME_BUDGET},nframes=${FRAME_BUDGET},max_pixels=${MAX_PIXELS},data_parallel_size=${DATA_PARALLEL_SIZE},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},max_model_len=${MAX_MODEL_LEN},max_num_seqs=${MAX_NUM_SEQS}"
  # Batch-invariant mode forces VLLM_ATTENTION_BACKEND=FLASH_ATTN globally, but
  # the Qwen-VL vision tower's head dim (80) is not built into vllm's bundled
  # flash-attn (multiples of 32 only) and the engine crashes at profiling.
  # Pin the vision encoder to SDPA (per-item, deterministic); the LM keeps the
  # batch-invariant FLASH_ATTN path.
  [[ "$BATCH_INVARIANT" == "1" ]] && MODEL_ARGS="${MODEL_ARGS},mm_encoder_attn_backend=TORCH_SDPA"
else
  MODEL_ARGS="pretrained=${MODEL},max_pixels=${MAX_PIXELS},min_pixels=${MIN_PIXELS},${SAMPLE},device_map=${DEVICE_MAP}"
  MODEL_ARGS="${MODEL_ARGS},frame_sampler=${FRAME_SAMPLER}"
  [[ -n "$FRAME_SAMPLER_SEED" ]] && MODEL_ARGS="${MODEL_ARGS},frame_sampler_seed=${FRAME_SAMPLER_SEED}"
  [[ -n "$ATTN" ]] && MODEL_ARGS="${MODEL_ARGS},attn_implementation=${ATTN}"
  # EXTRA_MODEL_ARGS (env): appended verbatim, e.g. interleave_visuals=True (metav METAV_FRAME_SEP / METAV_LAYOUT experiments)
  [[ -n "${EXTRA_MODEL_ARGS:-}" ]] && MODEL_ARGS="${MODEL_ARGS},${EXTRA_MODEL_ARGS}"
fi

# per-prompt vision-token estimate: (frames/2) x (max_pixels/784)
# /2 = Qwen's temporal patch merge (FRAME_FACTOR); 784 = 28x28 px per token after 2x2 spatial merge
EST_TOK=$(( FRAME_BUDGET / 2 * MAX_PIXELS / 784 ))
echo "${CONFIG_NOTE}"
echo "run: qwen2_5_vl (${BACKEND}) | video_reader=${VIDEO_READER:-torchcodec(default)} | ${SAMPLE} | frame_sampler=${FRAME_SAMPLER}${FS_TAG:+ (seed ${FRAME_SAMPLER_SEED})} | tasks=${TASKS} | batch=${BATCH_SIZE} | limit=${LIMIT:-all}"
echo "     max_pixels=${MAX_PIXELS} ~vision_tok/prompt≈${EST_TOK} -> ${OUTPUT_PATH}"
if [[ "$BACKEND" == "vllm" ]]; then
  echo "     gpus=${NUM_GPUS} (vllm data_parallel_size=${DATA_PARALLEL_SIZE}, tensor_parallel_size=${TENSOR_PARALLEL_SIZE}, port ${MAIN_PORT}) batch_invariant=${BATCH_INVARIANT}"
else
  echo "     gpus=${NUM_GPUS} ($([[ "$NUM_GPUS" -gt 1 ]] && echo "data-parallel via accelerate, port ${MAIN_PORT}" || echo "single process"))"
fi

# ARGUMENTS (flag ............. possible values ............... default)
#   --config .................. path to yaml ................. <script_dir>/video_qa_hf.yaml
#   --model ................... any HF model id .............. Qwen/Qwen2.5-VL-7B-Instruct
#   --sampling ................ nframes | fps ................ nframes
#   --frame_sampler ........... uniform | random ............. uniform  (which frames; random needs --seed, hf only)
#   --nframes ................. int .......................... 4        (also sets --sampling nframes)
#   --fps ..................... float (0.5, 1, 2, ...) ....... 1        (also sets --sampling fps)
#   --max_frames .............. int, fps-mode frame cap ...... 32
#   --max_pixels .............. int (e.g. 602112, 1605632) ... 301056
#   --min_pixels .............. int .......................... 200704
#   --attn .................... sdpa|flash_attention_2|eager . <model default>
#   --device_map .............. auto | cuda:0 ................ auto
#   --tasks ................... videomme | videomme,mvbench .. videomme
#   --include_path <dir>   optional extra task YAML dir (every task, incl. metav and *_frames<N>, is in-tree);
#   --batch_size .............. int .......................... 1
#   --max_new_tokens .......... int | <empty> ................. <task yaml's generation_kwargs>
#   --temperature .............. float | <empty> ............... <task yaml's generation_kwargs>
#   --limit ................... int or float(0-1) | <empty> .. <all>
#   --seed ..................... int or "a,b,c,d" | <empty> .... lmms_eval default (0,1234,1234,1234)
#   --seeds .................... comma-separated seeds, e.g. 1,2,3 -- reruns the whole script once per seed (mutually exclusive with --seed)
#   --run_tag ................. any string ................... <model>_<tasks>_nf<N>_mp<MP>_<backend>_vr<reader>_bs<N>_<timestamp>
#   --output_root ............. path ......................... ./logs/normalized_runs/video_qa_hf
#   --num_gpus ................ int, data-parallel replicas .. <autodetect>
#   --main_port ............... int, accelerate rendezvous ... 29500+pid%1000
#   --backend ................. hf | vllm .................... hf
#   --video_reader ............ decord|torchvision|torchcodec . decord
#   --tensor_parallel_size .... int, vllm-only ............... 1
#   --gpu_memory_utilization .. float, vllm-only ............. 0.9
#   --max_model_len ........... int, vllm-only ............... 32768
#   --max_num_seqs ............ int, vllm-only ............... 8
#   --batch_invariant ......... 0 | 1, vllm-only ............. 0        (sets VLLM_BATCH_INVARIANT=1; needs vllm >= 0.11.1, e.g. LMMS_EVAL_ENV_BIN=.../lmms_eval2/bin)
# END ARGS

if [[ "$BACKEND" == "vllm" ]]; then
  # accelerate --num_processes must equal tensor_parallel_size * data_parallel_size.
  WORLD_SIZE=$(( TENSOR_PARALLEL_SIZE * DATA_PARALLEL_SIZE ))
  if [[ "$WORLD_SIZE" -gt 1 ]]; then
    LAUNCHER=(accelerate launch --num_processes "$WORLD_SIZE" --main_process_port "$MAIN_PORT" -m lmms_eval)
  else
    LAUNCHER=(python -m lmms_eval)
  fi
  MODEL_NAME=vllm
elif [[ "$NUM_GPUS" -gt 1 ]]; then
  LAUNCHER=(accelerate launch --num_processes "$NUM_GPUS" --main_process_port "$MAIN_PORT" -m lmms_eval)
  MODEL_NAME=qwen2_5_vl
else
  LAUNCHER=(python -m lmms_eval)
  MODEL_NAME=qwen2_5_vl
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
