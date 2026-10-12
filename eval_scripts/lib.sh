#!/usr/bin/env bash
# Shared plumbing for sweep_params.sh, sweep_frames.sh and sweep_vllm_fix.sh. Source it, do not run it.
#
#   * REPO_ROOT / RUNNERS, cd to the repo root (python -m lmms_eval must see this checkout)
#   * conda envs (bin dirs, all optional): ENV_A (transformers 4.57 / vllm 0.11.0) for Qwen2.5-VL,
#                 Qwen3-VL, Gemma 3; ENV_A_BI (= ENV_A with vllm 0.11.1) for their batch-invariant
#                 vllm runs; ENV_B (transformers 5.16 / vllm 0.28) for Qwen3.5, InternVL, GLM-V.
#                 Unset = the current environment. LMMS_EVAL_ENV_BIN=<env/bin> forces one env for all.
#   * resolve_model  alias -> HF id;  detect_family  HF id -> family;  fam_* family tables
#   * run_cfg  one runner call with resume (a run dir that already holds *_results.json is
#              skipped), a config.json next to the run, DRY_RUN support and a failure count.
set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNERS="eval_scripts/runners"   # relative: lib.sh cd-s to the repo root
cd "$REPO_ROOT"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# bin dirs of the conda envs; empty = run the runners in the current environment
ENV_A="${ENV_A:-}"        # transformers 4.57 / vllm 0.11.0: Qwen2.5-VL, Qwen3-VL, Gemma 3
ENV_B="${ENV_B:-}"        # transformers 5.16 / vllm 0.28:   Qwen3.5, InternVL, GLM-V
ENV_A_BI="${ENV_A_BI:-}"  # ENV_A with vllm 0.11.1 (batch-invariant mode); used for --batch_invariant 1 runs of the ENV_A families
USER_ENV="${LMMS_EVAL_ENV_BIN:-}"
DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-1}"
NUM_GPUS="${NUM_GPUS:-1}"
N_RUN=0; N_SKIP=0; N_FAIL=0

FAMILIES="qwen25vl qwen3vl qwen35 internvl glm gemma"

# ---- models --------------------------------------------------------------------------------
# resolve_model <alias|hf id>: qwen25vl_7b qwen3vl_8b qwen35_9b internvl35_8b internvl3_2b
# glm46v_flash glm46v glm45v glm41v_9b gemma3_12b -> HF id; anything else is returned as is.
resolve_model() {
  local m=$1 size
  case "$(tr '[:upper:]' '[:lower:]' <<< "$m")" in
    qwen25vl_*b)  size=${m##*_}; echo "Qwen/Qwen2.5-VL-${size%[bB]}B-Instruct";;
    qwen3vl_*b)   size=${m##*_}; echo "Qwen/Qwen3-VL-${size%[bB]}B-Instruct";;
    qwen35_*)     size=${m#*_}; size=$(tr '[:lower:]' '[:upper:]' <<< "$size"); echo "Qwen/Qwen3.5-${size//_/-}";;
    internvl35_*) size=${m#*_}; size=$(tr '[:lower:]' '[:upper:]' <<< "$size"); echo "OpenGVLab/InternVL3_5-${size//_/-}-HF";;
    internvl3_*)  size=${m#*_}; size=$(tr '[:lower:]' '[:upper:]' <<< "$size"); echo "OpenGVLab/InternVL3-${size//_/-}-hf";;
    glm46v_flash) echo "zai-org/GLM-4.6V-Flash";;
    glm46v)       echo "zai-org/GLM-4.6V";;
    glm45v)       echo "zai-org/GLM-4.5V";;
    glm41v_9b)    echo "zai-org/GLM-4.1V-9B-Thinking";;
    gemma3_1b)    echo "error: gemma-3-1b is text-only" >&2; return 1;;
    gemma3_*b)    size=${m##*_}; echo "google/gemma-3-${size%[bB]}b-it";;
    *)            echo "$m";;
  esac
}

# detect_family <hf id>: the runner family from the id; fine-tunes with other names need --family.
detect_family() {
  case "$(tr '[:upper:]' '[:lower:]' <<< "$1")" in
    *qwen2.5-vl*|*qwen2_5_vl*|*qwen2.5vl*|*video-r1*) echo qwen25vl;;
    *qwen3-vl*|*qwen3_vl*|*qwen3vl*)                   echo qwen3vl;;
    *qwen3.5*|*qwen3_5*)                               echo qwen35;;
    *internvl*)                                        echo internvl;;
    *glm*)                                             echo glm;;
    *gemma*)                                           echo gemma;;
    *) return 1;;
  esac
}

check_family() {
  case " $FAMILIES " in *" $1 "*) return 0;; esac
  echo "error: --family must be one of: $FAMILIES (got '$1')" >&2; return 1
}

# ---- family tables -------------------------------------------------------------------------
fam_runner() {
  case "$1" in
    qwen25vl) echo "$RUNNERS/run_video_qa_hf_multigpu.sh";;
    qwen3vl)  echo "$RUNNERS/run_video_qa_qwen3vl_multigpu.sh";;
    qwen35)   echo "$RUNNERS/run_video_qa_qwen3_5_multigpu.sh";;
    internvl) echo "$RUNNERS/run_video_qa_internvl_protocol.sh";;
    glm)      echo "$RUNNERS/run_video_qa_glm_protocol.sh";;
    gemma)    echo "$RUNNERS/run_video_qa_gemma_protocol.sh";;
  esac
}
fam_env() {
  case "$1" in qwen25vl|qwen3vl|gemma) echo "$ENV_A";; *) echo "$ENV_B";; esac
}
# default per-frame pixel budget (tokens/frame differ per family: 28-px grid for Qwen2.5-VL and
# GLM (784 px/token), 32-px grid for Qwen3-VL / Qwen3.5 (1024 px/token); InternVL uses tiles).
fam_max_pixels() {
  case "$1" in
    qwen25vl) echo 602112;;   # 768 tok/frame
    qwen3vl)  echo 524288;;   # 512 tok/frame
    qwen35)   echo 131072;;   # 128 tok/frame (HF wrapper default)
    glm)      echo 401408;;   # 512 tok/frame (vllm only)
    gemma)    echo 1605632;;  # ignored: SigLIP 896x896 = 256 tok/frame
    internvl) echo "";;
  esac
}
# resolution sweep grid of exp6 (max_pixels; InternVL: max_patches = 448x448 tiles per frame)
fam_res_grid() {
  case "$1" in
    qwen25vl)      echo "200704 301056 602112";;
    qwen3vl|qwen35) echo "131072 262144 524288 1048576";;
    glm)           echo "200704 401408 802816";;
    internvl)      echo "1 4 12";;
    gemma)         echo "";;
  esac
}
# fam_has <family> <capability>; capabilities:
#   vllm          the runner has a vllm backend
#   hf_video      the hf backend accepts a raw .mp4 with a frame budget (glm4v / gemma3 wrappers take frame lists only)
#   fps           rate-based sampling (--fps/--max_frames) on hf
#   video_reader  --video_reader decord|torchvision|torchcodec on hf (FORCE_QWENVL_VIDEO_READER)
#   frame_sampler --frame_sampler random (qwen2_5_vl wrapper only)
#   hf_batch      batch_size > 1 on hf (internvl_hf asserts 1)
#   max_pixels    per-frame pixel budget is a knob (InternVL: tiles instead; Gemma: fixed)
fam_has() {
  case "$1:$2" in
    qwen25vl:*) return 0;;
    qwen3vl:frame_sampler|qwen35:frame_sampler) return 1;;
    qwen3vl:*|qwen35:*) return 0;;
    internvl:vllm|internvl:hf_video) return 0;;   # no fps: its runner has no --max_frames cap
    glm:vllm|glm:hf_batch|glm:max_pixels) return 0;;
    gemma:hf_batch) return 0;;
    *) return 1;;
  esac
}
# fam_args <family> <backend>: base runner flags for that backend (MP = max_pixels of the sweep)
fam_args() {
  case "$1:$2" in
    qwen25vl:hf|qwen3vl:hf)     echo "--backend hf --video_reader torchcodec --max_pixels $MP";;
    qwen35:hf)                  echo "--backend hf --video_reader torchcodec --max_pixels $MP --enable_thinking false";;
    qwen25vl:vllm|qwen3vl:vllm) echo "--backend vllm --video_reader torchcodec --max_pixels $MP";;
    qwen35:vllm)                echo "--backend vllm --video_reader torchcodec --max_pixels $MP --enable_thinking false";;
    internvl:hf)                echo "--backend hf --max_patches 1 --min_patches 1";;
    internvl:hf_tiled)          echo "--backend hf_tiled --video_reader decord --min_patches 1";;   # + --max_patches M
    internvl:vllm)              echo "--backend vllm --video_reader torchcodec --max_patches 1 --min_patches 1";;
    glm:vllm)                   echo "--backend vllm --video_reader torchcodec --max_pixels $MP";;
    glm:hf)                     echo "--backend hf";;
    gemma:hf)                   echo "";;
    *) echo "error: family $1 has no backend $2" >&2; return 1;;
  esac
}

# ---- model / family setup used by every sweep ---------------------------------------------
# setup_model <model alias|id> <family or empty>: sets MODEL, FAM, MODEL_TAG, RUNNER, FAM_ENV, MP
setup_model() {
  MODEL=$(resolve_model "$1") || exit 1
  if [[ -n "$2" ]]; then FAM=$2; else
    FAM=$(detect_family "$MODEL") || { echo "error: cannot infer the runner family from '$MODEL'; pass --family (one of: $FAMILIES)" >&2; exit 1; }
  fi
  check_family "$FAM" || exit 1
  MODEL_TAG=${MODEL##*/}
  RUNNER=$(fam_runner "$FAM")
  FAM_ENV=$(fam_env "$FAM"); [[ -n "$USER_ENV" ]] && FAM_ENV=$USER_ENV
  MP="${MAX_PIXELS:-$(fam_max_pixels "$FAM")}"
}

# ---- one configuration ------------------------------------------------------------------------
results_of() { compgen -G "$1/*/*_results.json"; }

# run_cfg <exp_dir> <run_tag> "<k=v k=v ...>" <runner args...>
#   Runs RUNNER with --model MODEL, the args and --output_root/--run_tag/--num_gpus, so the run lands in
#   <exp_dir>/<run_tag>/. Skipped when that dir already holds a *_results.json (RESUME=1);
#   with RESUME=0 an old run dir is moved aside first. config.json records the k=v pairs
#   (read by eval_scripts/collect_results.py). A run counts as done only if the runner exits 0
#   AND a results file appeared (lmms_eval can exit 0 after a failed evaluation).
run_cfg() {
  local exp_dir=$1 tag=$2 kv=$3; shift 3
  local run_dir="${exp_dir}/${tag}"
  if [[ -n "$(results_of "$run_dir")" ]]; then
    if [[ "$RESUME" == "1" ]]; then echo "--- SKIP (done)  ${run_dir}"; N_SKIP=$((N_SKIP + 1)); return 0; fi
    mv "$run_dir" "${run_dir}.redo_$(date +%Y%m%d_%H%M%S)"
  fi
  local -a cmd=(bash "$RUNNER" --model "$MODEL" "$@" --output_root "$exp_dir" --run_tag "$tag" --num_gpus "$NUM_GPUS")
  local env=$FAM_ENV
  [[ -z "$USER_ENV" && "$env" == "$ENV_A" && " $* " == *" --batch_invariant 1 "* ]] && env=$ENV_A_BI
  echo "=== RUN  ${run_dir}"
  if [[ "$DRY_RUN" == "1" ]]; then printf '    LMMS_EVAL_ENV_BIN=%s' "${env:-<current env>}"; printf ' %q' "${cmd[@]}"; echo; return 0; fi
  mkdir -p "$run_dir"
  {
    printf '{\n  "model": "%s",\n  "family": "%s",\n  "runner": "%s",\n  "env": "%s",\n' "$MODEL" "$FAM" "${RUNNER##*/}" "$env"
    local pair
    for pair in $kv; do printf '  "%s": "%s",\n' "${pair%%=*}" "${pair#*=}"; done
    printf '  "cmd": "%s"\n}\n' "$(printf '%q ' "${cmd[@]}" | sed 's/\\/\\\\/g; s/"/\\"/g')"
  } > "${run_dir}/config.json"
  N_RUN=$((N_RUN + 1))
  LMMS_EVAL_ENV_BIN="$env" "${cmd[@]}"
  local rc=$?
  if [[ $rc -eq 0 && -n "$(results_of "$run_dir")" ]]; then
    echo "=== DONE ${run_dir}"
  else
    echo "!!! FAILED rc=${rc} ${run_dir} (no results file; retried on the next invocation)" >&2
    N_FAIL=$((N_FAIL + 1))
  fi
}

# want_exp <exp dir name> <selection>: selection "5,7,3a" matches exp5_*, exp7_*, exp3a_*; "3" also matches exp3a.
want_exp() {
  local exp=$1 sel=$2 n
  [[ -z "$sel" ]] && return 0
  IFS=',' read -r -a _sel <<< "$sel"
  for n in "${_sel[@]}"; do
    [[ "$exp" == "exp${n}_"* || "$exp" == "exp${n}"[a-z]"_"* ]] && return 0
  done
  return 1
}

# has_task <task name>: is it registered in-tree? One scan of lmms_eval/tasks/**/*.yaml per script
# run (the task/group names declared at the top level of every YAML), then a lookup.
_TASK_NAMES=""
has_task() {
  if [[ -z "$_TASK_NAMES" ]]; then
    _TASK_NAMES=$'\n'"$(grep -rhoE --include='*.yaml' "^(task|group): *[\"']?[A-Za-z0-9_.-]+" lmms_eval/tasks | sed -E "s/^(task|group): *[\"']?//" | sort -u)"$'\n'
  fi
  [[ "$_TASK_NAMES" == *$'\n'"$1"$'\n'* ]]
}

summary() { echo; echo "runs started: ${N_RUN}, skipped (already done): ${N_SKIP}, failed: ${N_FAIL}"; [[ "$N_FAIL" == 0 ]]; }
