#!/usr/bin/env bash
# Parameter sweep (exp1..exp8) for ONE model on ONE video task: which evaluation knobs move
# the score, one knob at a time, everything else fixed. Any model whose family has a runner
# (Qwen2.5-VL, Qwen3-VL, Qwen3.5, InternVL, GLM-V, Gemma 3); knobs a family's wrapper cannot
# vary are skipped with a printed reason.
#
# Usage (from the repo root):
#   bash eval_scripts/sweep_params.sh --model Qwen/Qwen2.5-VL-7B-Instruct [--task videomme]
#        [--family qwen25vl] [--exp 1,2,3a,5] [--only-backend hf|vllm] [--dry_run]
#   --model   HF id or alias (qwen25vl_7b qwen3vl_8b qwen35_9b internvl35_8b glm46v_flash gemma3_12b)
#   --family  runner family; inferred from the id, needed for fine-tunes (Video-R1-7B -> qwen25vl)
#   --task    any in-tree video task or group: videomme, mvbench, vsibench, mmvu_val, metav, ...
#   --exp     comma list of experiment numbers; "3" also selects 3a, "3a" only 3a. Default: all.
#
# Experiments (exp dir -> what varies; base point = BASE_NF frames, hf backend, batch 1, SEED):
#   exp1_hf_vs_vllm          hf (BASE_NF) vs vllm (batch-invariant, NFRAMES x SEEDS, batch EXP1_VLLM_BS)
#   exp2_batch_size          hf batch_size in EXP2_BATCH_SIZES
#   exp3_video_reader        hf decoder in EXP3_READERS (decord / torchvision / torchcodec)
#   exp3a_frame_sampler      hf random frame positions instead of uniform (Qwen2.5-VL only)
#   exp4_frame_selection     1 fps capped at EXP4_MAX_FRAMES vs the same count uniform
#   exp5_number_of_frames    hf frames in NFRAMES
#   exp6_resolution          hf max_pixels in the family's grid (InternVL: tiles/frame on hf_tiled)
#   exp7_video_vs_frames     <task>_frames<N> (pre-extracted JPEGs, eval_scripts/make_frames_task.py) for N in NFRAMES
#   exp8_temperature         hf temperature in EXP8_TEMPERATURES x SEEDS
#
# Knobs (env, all optional):
#   OUT_ROOT=logs/sweep_params   SEED=22   SEEDS=22,42,72   NUM_GPUS=1   RESUME=1 (0 redoes all)
#   NFRAMES="4 8 16 32 64 128"   BASE_NF=4   EXP1_VLLM_BS=4   EXP2_BATCH_SIZES="1 4 8 16 32"
#   EXP3_READERS="decord torchvision torchcodec"   EXP4_FPS=1   EXP4_MAX_FRAMES="32 64"
#   EXP6_GRID=<family default>   EXP8_TEMPERATURES="0.7 0.8 0.9 1.0"
#   MAX_PIXELS=<family default>  per-frame pixel budget of every non-exp6 run
#   LMMS_EVAL_ENV_BIN=<env/bin>  overrides the family's conda env
#
# Output: OUT_ROOT/<task>/<model>/<exp>/<run tag>/{config.json, <model>/<tag>_<date>_results.json, samples}
# Resume: a run dir that already holds a *_results.json is skipped.
set -u
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

MODEL_ARG=""; FAM_ARG=""; TASK="videomme"; EXP=""; ONLY_BACKEND=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)        MODEL_ARG="$2"; shift 2;;
    --family)       FAM_ARG="$2"; shift 2;;
    --task)         TASK="$2"; shift 2;;
    --exp)          EXP="$2"; shift 2;;
    --only-backend) ONLY_BACKEND="$2"; shift 2;;
    --dry_run)      DRY_RUN=1; shift;;
    -h|--help)      sed -n '2,32p' "$0"; exit 0;;
    *) echo "error: unknown arg '$1'" >&2; exit 1;;
  esac
done
[[ -n "$MODEL_ARG" ]] || { echo "error: --model is required" >&2; exit 1; }
[[ -z "$ONLY_BACKEND" || "$ONLY_BACKEND" == hf || "$ONLY_BACKEND" == vllm ]] || { echo "error: --only-backend hf|vllm" >&2; exit 1; }
setup_model "$MODEL_ARG" "$FAM_ARG"
has_task "$TASK" || { echo "error: task '$TASK' is not registered under lmms_eval/tasks" >&2; exit 1; }

OUT_ROOT="${OUT_ROOT:-logs/sweep_params}"
SEED="${SEED:-22}"; SEEDS="${SEEDS:-22,42,72}"
NFRAMES="${NFRAMES:-4 8 16 32 64 128}"; BASE_NF="${BASE_NF:-4}"
EXP1_VLLM_BS="${EXP1_VLLM_BS:-4}"
EXP2_BATCH_SIZES="${EXP2_BATCH_SIZES:-1 4 8 16 32}"
EXP3_READERS="${EXP3_READERS:-decord torchvision torchcodec}"
EXP4_FPS="${EXP4_FPS:-1}"; EXP4_MAX_FRAMES="${EXP4_MAX_FRAMES:-32 64}"
EXP6_GRID="${EXP6_GRID:-$(fam_res_grid "$FAM")}"
EXP8_TEMPERATURES="${EXP8_TEMPERATURES:-0.7 0.8 0.9 1.0}"
OUT="${OUT_ROOT}/${TASK}/${MODEL}"
IFS=',' read -r -a SEED_LIST <<< "$SEEDS"

skip() { echo "--- SKIP ${1}: ${2}"; }
backend_ok() { [[ -z "$ONLY_BACKEND" || "$ONLY_BACKEND" == "$1" ]]; }
# hf_run <exp> <tag> "<kv>" <runner args...>  -- one hf-backend run (family base args prepended)
hf_run()   { local exp=$1 tag=$2 kv=$3; shift 3; backend_ok hf   || return 0; run_cfg "${OUT}/${exp}" "$tag" "exp=${exp} task=${TASK} backend=hf max_pixels=${MP} ${kv}"   $(fam_args "$FAM" hf)   "$@"; }
vllm_run() { local exp=$1 tag=$2 kv=$3; shift 3; backend_ok vllm || return 0; run_cfg "${OUT}/${exp}" "$tag" "exp=${exp} task=${TASK} backend=vllm max_pixels=${MP} ${kv}" $(fam_args "$FAM" vllm) "$@"; }

echo "sweep_params: model=${MODEL} family=${FAM} task=${TASK} exp=${EXP:-all} max_pixels=${MP:-n/a} seed=${SEED} seeds=${SEEDS} out=${OUT} env=${FAM_ENV##*/.conda/}"

# ---- exp1: serving backend --------------------------------------------------------------------
e=exp1_hf_vs_vllm
if want_exp $e "$EXP"; then
  if fam_has "$FAM" hf_video; then
    hf_run $e "nf${BASE_NF}_hf_bs1_seed${SEED}" "nframes=${BASE_NF} batch_size=1 seed=${SEED}" --tasks "$TASK" --nframes "$BASE_NF" --batch_size 1 --seed "$SEED"
  else skip "$e (hf)" "the ${FAM} hf wrapper takes frame lists only (no raw video)"; fi
  if fam_has "$FAM" vllm; then
    for n in $NFRAMES; do for s in "${SEED_LIST[@]}"; do
      vllm_run $e "nf${n}_vllm_bi_bs${EXP1_VLLM_BS}_seed${s}" "nframes=${n} batch_size=${EXP1_VLLM_BS} batch_invariant=1 seed=${s}" \
        --tasks "$TASK" --nframes "$n" --batch_size "$EXP1_VLLM_BS" --seed "$s" --batch_invariant 1
    done; done
  else skip "$e (vllm)" "no vllm backend for ${FAM}"; fi
fi

# ---- exp2: batch size -------------------------------------------------------------------------
e=exp2_batch_size
if want_exp $e "$EXP"; then
  if fam_has "$FAM" hf_video && fam_has "$FAM" hf_batch; then
    for bs in $EXP2_BATCH_SIZES; do
      hf_run $e "nf${BASE_NF}_hf_bs${bs}_seed${SEED}" "nframes=${BASE_NF} batch_size=${bs} seed=${SEED}" --tasks "$TASK" --nframes "$BASE_NF" --batch_size "$bs" --seed "$SEED"
    done
  elif fam_has "$FAM" vllm; then
    echo "--- NOTE ${e}: ${FAM} cannot batch raw video on hf; sweeping the batch size on the vllm backend (batch-invariant) instead"
    for bs in $EXP2_BATCH_SIZES; do
      vllm_run $e "nf${BASE_NF}_vllm_bi_bs${bs}_seed${SEED}" "nframes=${BASE_NF} batch_size=${bs} batch_invariant=1 seed=${SEED}" --tasks "$TASK" --nframes "$BASE_NF" --batch_size "$bs" --seed "$SEED" --batch_invariant 1
    done
  else skip "$e" "${FAM}: no backend that batches raw video"; fi
fi

# ---- exp3: video decoder ----------------------------------------------------------------------
e=exp3_video_reader
if want_exp $e "$EXP"; then
  if fam_has "$FAM" video_reader; then
    for vr in $EXP3_READERS; do
      # the family base args carry --video_reader torchcodec; the later flag wins in every runner
      hf_run $e "nf${BASE_NF}_hf_vr${vr}_bs1_seed${SEED}" "nframes=${BASE_NF} batch_size=1 video_reader=${vr} seed=${SEED}" --tasks "$TASK" --nframes "$BASE_NF" --batch_size 1 --seed "$SEED" --video_reader "$vr"
    done
  else skip "$e" "${FAM}: the decoder is not selectable on its hf path"; fi
fi
e=exp3a_frame_sampler
if want_exp $e "$EXP"; then
  if fam_has "$FAM" frame_sampler; then
    hf_run $e "nf${BASE_NF}_hf_vrtorchcodec_fsrandom_bs1_seed${SEED}" "nframes=${BASE_NF} batch_size=1 video_reader=torchcodec frame_sampler=random seed=${SEED}" \
      --tasks "$TASK" --nframes "$BASE_NF" --batch_size 1 --seed "$SEED" --video_reader torchcodec --frame_sampler random
  else skip "$e" "random frame positions exist only in the qwen2_5_vl wrapper"; fi
fi

# ---- exp4: 1 fps vs uniform -------------------------------------------------------------------
e=exp4_frame_selection
if want_exp $e "$EXP"; then
  if fam_has "$FAM" fps; then
    for mf in $EXP4_MAX_FRAMES; do
      hf_run $e "fps${EXP4_FPS//./p}_cap${mf}_hf_bs1_seed${SEED}" "fps=${EXP4_FPS} max_frames=${mf} batch_size=1 seed=${SEED}" --tasks "$TASK" --fps "$EXP4_FPS" --max_frames "$mf" --batch_size 1 --seed "$SEED"
      hf_run $e "nf${mf}_hf_bs1_seed${SEED}" "nframes=${mf} batch_size=1 seed=${SEED}" --tasks "$TASK" --nframes "$mf" --batch_size 1 --seed "$SEED"
    done
  else skip "$e" "${FAM}: no rate-based (fps) sampling on its hf path"; fi
fi

# ---- exp5: number of frames -------------------------------------------------------------------
e=exp5_number_of_frames
if want_exp $e "$EXP"; then
  if fam_has "$FAM" hf_video; then
    for n in $NFRAMES; do
      hf_run $e "nf${n}_hf_bs1_seed${SEED}" "nframes=${n} batch_size=1 seed=${SEED}" --tasks "$TASK" --nframes "$n" --batch_size 1 --seed "$SEED"
    done
  else skip "$e" "the ${FAM} hf wrapper takes frame lists only; use exp7 or sweep_vllm_fix.sh"; fi
fi

# ---- exp6: resolution per frame ---------------------------------------------------------------
e=exp6_resolution
if want_exp $e "$EXP"; then
  if [[ "$FAM" == internvl ]]; then
    echo "--- NOTE ${e}: InternVL resolution = 448x448 tiles per frame (max_patches), on the hf_tiled backend (the only hf path where tiles change a video run)"
    for m in $EXP6_GRID; do
      backend_ok hf && run_cfg "${OUT}/${e}" "nf${BASE_NF}_tiles${m}_hftiled_bs1_seed${SEED}" "exp=${e} task=${TASK} backend=hf_tiled nframes=${BASE_NF} max_patches=${m} batch_size=1 seed=${SEED}" \
        $(fam_args "$FAM" hf_tiled) --max_patches "$m" --tasks "$TASK" --nframes "$BASE_NF" --batch_size 1 --seed "$SEED"
    done
  elif fam_has "$FAM" max_pixels && fam_has "$FAM" hf_video; then
    for mp in $EXP6_GRID; do
      hf_run $e "nf${BASE_NF}_mp${mp}_hf_bs1_seed${SEED}" "nframes=${BASE_NF} batch_size=1 seed=${SEED} max_pixels=${mp}" --tasks "$TASK" --nframes "$BASE_NF" --batch_size 1 --seed "$SEED" --max_pixels "$mp"
    done
  elif fam_has "$FAM" max_pixels && fam_has "$FAM" vllm; then
    echo "--- NOTE ${e}: ${FAM} has no raw-video hf path; sweeping max_pixels on the vllm backend (batch-invariant)"
    for mp in $EXP6_GRID; do
      vllm_run $e "nf${BASE_NF}_mp${mp}_vllm_bi_bs1_seed${SEED}" "nframes=${BASE_NF} batch_size=1 batch_invariant=1 seed=${SEED} max_pixels=${mp}" --tasks "$TASK" --nframes "$BASE_NF" --batch_size 1 --seed "$SEED" --batch_invariant 1 --max_pixels "$mp"
    done
  else skip "$e" "${FAM}: the per-frame resolution is fixed by the wrapper"; fi
fi

# ---- exp7: raw video vs pre-extracted frames --------------------------------------------------
e=exp7_video_vs_frames
if want_exp $e "$EXP"; then
  for n in $NFRAMES; do
    ft="${TASK}_frames${n}"
    if ! has_task "$ft"; then skip "$e nf=${n}" "no task ${ft}; generate it: python eval_scripts/make_frames_task.py --task ${TASK}"; continue; fi
    backend_ok hf && run_cfg "${OUT}/${e}" "frames${n}_hf_bs1_seed${SEED}" "exp=${e} task=${ft} backend=hf nframes=${n} batch_size=1 seed=${SEED} max_pixels=${MP} frames_task=1" \
      $(fam_args "$FAM" hf) --tasks "$ft" --nframes "$n" --batch_size 1 --seed "$SEED"
  done
fi

# ---- exp8: sampling temperature ---------------------------------------------------------------
e=exp8_temperature
if want_exp $e "$EXP"; then
  if fam_has "$FAM" hf_video; then
    for t in $EXP8_TEMPERATURES; do for s in "${SEED_LIST[@]}"; do
      hf_run $e "nf${BASE_NF}_t${t//./p}_hf_bs1_seed${s}" "nframes=${BASE_NF} batch_size=1 temperature=${t} seed=${s}" --tasks "$TASK" --nframes "$BASE_NF" --batch_size 1 --seed "$s" --temperature "$t"
    done; done
  else skip "$e" "the ${FAM} hf wrapper takes frame lists only (no raw video)"; fi
fi

summary
