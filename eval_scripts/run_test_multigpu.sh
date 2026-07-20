#!/usr/bin/env bash
# Multi-GPU evaluation of Qwen2.5-VL-7B with vLLM.
#
# Two parallelism modes are supported by lmms_eval's simple/vllm.py:
#   * data parallelism  (data_parallel_size): N independent model replicas,
#                        each processes a shard of the dataset. Best for models
#                        that fit on one GPU (7B does). Near-linear speedup.
#                        REQUIRES an `accelerate`/torchrun launch.
#   * tensor parallelism (tensor_parallel_size): one model split across N GPUs.
#                        For models too large for a single GPU.
#
# Rule enforced by the code: accelerate --num_processes must equal
#   tensor_parallel_size * data_parallel_size.
#
# Usage:
#   bash eval_scripts/run_test_multigpu.sh             # 8-way data parallel (default)
#   NUM_GPUS=4 bash eval_scripts/run_test_multigpu.sh  # 4-way data parallel
#   MODE=tp   bash eval_scripts/run_test_multigpu.sh   # tensor parallel across all GPUs
#
# Launch this on the node where the GPUs are actually allocated
# (e.g. inside an salloc/srun with --gres=gpu:8).

set -euo pipefail

# --- how many GPUs to use ---------------------------------------------------
# Default: honor CUDA_VISIBLE_DEVICES if set, else count all visible GPUs.
if [[ -n "${NUM_GPUS:-}" ]]; then
  N="${NUM_GPUS}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  N=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
else
  N=$(nvidia-smi -L | wc -l)
fi
MODE="${MODE:-dp}"   # dp = data parallel (default), tp = tensor parallel

echo "[info] using ${N} GPU(s) | mode=${MODE}"

# --- diagnostics ------------------------------------------------------------
echo "[diag] python  : $(which python)"
python -c "import sys, torch; print('[diag] exe    :', sys.executable); print('[diag] torch  :', torch.__version__, '| cuda avail:', torch.cuda.is_available(), '| device count:', torch.cuda.device_count())"

MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
TASKS="video_mmmu"
COMMON_ARGS="max_pixels=602112,max_model_len=32768,gpu_memory_utilization=0.9,max_num_seqs=8,max_frame_num=32,limit_mm_per_prompt={\"video\":1}"

if [[ "${MODE}" == "tp" ]]; then
  # ---- Tensor parallelism: single process, model split across N GPUs -------
  echo "[info] tensor-parallel across ${N} GPUs (single process)"
  python -m lmms_eval \
    --model vllm \
    --model_args model=${MODEL},tensor_parallel_size=${N},data_parallel_size=1,${COMMON_ARGS} \
    --tasks ${TASKS} \
    --batch_size 8

else
  # ---- Data parallelism: N replicas via accelerate -------------------------
  if ! python -c "import accelerate" 2>/dev/null; then
    echo "[error] 'accelerate' is not installed in this env; data-parallel mode needs it." >&2
    echo "        Install it:  pip install accelerate    (or run with MODE=tp)" >&2
    exit 1
  fi
  echo "[info] data-parallel: ${N} replicas via accelerate (--num_processes ${N})"
  accelerate launch --num_processes "${N}" -m lmms_eval \
    --model vllm \
    --model_args model=${MODEL},data_parallel_size=${N},tensor_parallel_size=1,${COMMON_ARGS} \
    --tasks ${TASKS} \
    --batch_size 8
fi
