#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --job-name=sparsity_ablation
#SBATCH --output=sparsity_ablation_slurm-%j.out
#SBATCH --error=sparsity_ablation_slurm-%j.err
#SBATCH --account=def-rrabba

# =============================================================================
# Sparsity Ablation: DART vs PPO on cartpole-swingup
# 4 thresholds × 2 methods × 10 seeds = 80 runs, single node, 80 cores
# =============================================================================

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export WANDB_MODE=offline
BENCH_DIR="${SCRATCH}/sparsity_ablation"
export WANDB_DIR="${BENCH_DIR}/wandb"
mkdir -p "${WANDB_DIR}"
mkdir -p "${BENCH_DIR}/runs"

PROJ_DIR="${SLURM_SUBMIT_DIR}"
cd "${PROJ_DIR}"
ln -sfn "${BENCH_DIR}/runs" "${PROJ_DIR}/runs"

source .env

PYTHON="${PROJ_DIR}/.venv/bin/python"

# ------ Preflight ------
echo "Python:        ${PYTHON}"
[[ -x "${PYTHON}" ]] || { echo "ERROR: venv not found at ${PYTHON}"; exit 1; }
"${PYTHON}" -c "import shimmy; import gymnasium; gymnasium.make('dm_control/cartpole-swingup-v0').close()" \
    && echo "Preflight OK" \
    || { echo "ERROR: dm_control import failed"; exit 1; }

echo "========================================================"
echo "Node:        $(hostname)"
echo "Cores:       ${SLURM_CPUS_PER_TASK}"
echo "Bench dir:   ${BENCH_DIR}"
echo "Start time:  $(date)"
echo "========================================================"

# ------ Grid ------
THRESHOLDS=(0.0 0.75 0.90 0.95)
SEEDS=(1 2 3 4 5 6 7 8 9 10)

ENV_ID="dm_control/cartpole-swingup-v0"
WANDB_PROJECT="dart_sparsity_ablation"
TOTAL_STEPS=4000000

# ------ Generate task list ------
TASKFILE="${BENCH_DIR}/tasks_sparsity_${SLURM_JOB_ID}.txt"
> "${TASKFILE}"

for thresh in "${THRESHOLDS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        # PPO
        echo "${PYTHON} cleanrl/ppo_dm_control.py \
            --env_id ${ENV_ID} \
            --seed ${seed} \
            --sparsity_threshold ${thresh} \
            --exp_name ppo_sp${thresh} \
            --total_timesteps ${TOTAL_STEPS} \
            --track \
            --wandb_project_name ${WANDB_PROJECT}" >> "${TASKFILE}"

        # DART
        echo "${PYTHON} cleanrl/dart_dm_control.py \
            --env_id ${ENV_ID} \
            --seed ${seed} \
            --sparsity_threshold ${thresh} \
            --exp_name dart_sp${thresh} \
            --dart_warmup_frac 0.4 \
            --total_timesteps ${TOTAL_STEPS} \
            --track \
            --wandb_project_name ${WANDB_PROJECT}" >> "${TASKFILE}"
    done
done

echo "Total tasks: $(wc -l < "${TASKFILE}")"  # should be 80

# ------ Run ------
JOBLOG="${BENCH_DIR}/parallel_joblog_${SLURM_JOB_ID}.txt"

parallel \
    -j "${SLURM_CPUS_PER_TASK}" \
    --joblog "${JOBLOG}" \
    --resume \
    --halt never \
    --progress \
    < "${TASKFILE}"

echo "========================================================"
echo "Completed at $(date)"
echo "To sync wandb: wandb sync ${WANDB_DIR}/*"
echo "========================================================"
