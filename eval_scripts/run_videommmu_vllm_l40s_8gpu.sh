#!/bin/bash
# ===========================================================================
# SLURM batch submission for the VideoMMMU eval with Qwen2.5-VL-7B using vLLM
# (data parallel) on 8x L40S. Submit with:
#   sbatch eval_scripts/run_videommmu_vllm_l40s_8gpu.sbatch
#
# This wraps the launch script eval_scripts/run_videommmu_vllm_multigpu.sh,
# which runs 8 independent vLLM replicas (one per GPU), each evaluating a shard
# of the dataset (~8x throughput).
#
# Resource request mirrors the interactive srun:
#   srun --job-name videommmu --partition=kuehne --ntasks=1 --nodes=1 \
#        --gres=gpu:8 --mem=512G --time 2-05:30:00 --pty bash
# ===========================================================================
#SBATCH --job-name=videommmu
#SBATCH --partition=kuehne
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=2-05:30:00
#SBATCH --output=/mnt/lustre/work/kuehne/kqr828/WorkSpace/MMEvals/lmms-eval/logs/videommmu_vllm_l40s_8gpu/slurm-%j.out
#SBATCH --error=/mnt/lustre/work/kuehne/kqr828/WorkSpace/MMEvals/lmms-eval/logs/videommmu_vllm_l40s_8gpu/slurm-%j.err
#SBATCH --cpus-per-task=64

set -euo pipefail

REPO_DIR="/mnt/lustre/work/kuehne/kqr828/WorkSpace/MMEvals/lmms-eval"
# lmms_eval env has both accelerate and vllm 0.11.0 installed.
CONDA_ENV="/mnt/lustre/work/kuehne/kqr828/.conda/lmms_eval"

cd "$REPO_DIR"
mkdir -p "$REPO_DIR/logs/videommmu_vllm_l40s_8gpu"

# Activate the env via `conda activate`.
# This node's conda (v4.10.3, base /usr) ships no profile.d/conda.sh, so we enable
# the `conda` shell function with `conda shell.bash hook` instead. The env lives at a
# full path (not under the default ~/.conda/envs), so we activate BY PATH, not name.
eval "$(/usr/bin/conda shell.bash hook)"
conda activate "$CONDA_ENV"

echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Python: $(which python)  |  accelerate: $(which accelerate)"
nvidia-smi -L || true

# Reuse the launch script. Args: [NUM_GPUS] [TASKS] [EXTRA_ARGS]
bash eval_scripts/run_videommmu_vllm_multigpu.sh 8 video_mmmu
