#!/bin/bash
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --job-name=dart_tune
#SBATCH --output=dart_tune_slurm-%A_%a.out
#SBATCH --error=dart_tune_slurm-%A_%a.err
#SBATCH --account=def-rrabba

# ================================================================================
# DART warmup_frac grid-search on all 50 dm_control environments.
#
# Grid: 10 values from 0.35 to 0.60 (step 0.028)
# 50 envs × 10 values × 1 seed = 500 runs, distributed across 4 nodes.
#
# After the job finishes, run:
#   python tune_dart_warmup.py --runs-dir runs
# to find the best warmup_frac.
#
# Usage:
#   sbatch tune_dart_warmup.sh
# ================================================================================

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

BENCH_DIR="${SCRATCH}/dart_warmup_tune"
mkdir -p "${BENCH_DIR}/runs"

PROJ_DIR="${SLURM_SUBMIT_DIR}"
cd "${PROJ_DIR}"

# Symlink runs/ → scratch so TensorBoard writes go to fast storage
ln -sfn "${BENCH_DIR}/runs" "${PROJ_DIR}/runs"

source .env

PYTHON="${PROJ_DIR}/.venv/bin/python"

# ------ Preflight ------
echo "Python: ${PYTHON}"
[[ -x "${PYTHON}" ]] || { echo "ERROR: venv not found"; exit 1; }

"${PYTHON}" -c "import shimmy; import gymnasium; gymnasium.make('dm_control/pendulum-swingup-v0').close()" \
    && echo "Preflight OK: dm_control verified" \
    || { echo "ERROR: dm_control import failed"; exit 1; }

NUM_NODES=4
NODE_ID=${SLURM_ARRAY_TASK_ID}

echo "========================================================"
echo "Node:       $(hostname)"
echo "Array ID:   ${NODE_ID} of ${NUM_NODES}"
echo "Cores:      ${SLURM_CPUS_PER_TASK}"
echo "Start:      $(date)"
echo "========================================================"

# ------ Environments (same 50 as benchmark) ------
ENVS=(
    dm_control/acrobot-swingup-v0
    dm_control/acrobot-swingup_sparse-v0
    dm_control/ball_in_cup-catch-v0
    dm_control/cartpole-balance-v0
    dm_control/cartpole-balance_sparse-v0
    dm_control/cartpole-swingup-v0
    dm_control/cartpole-swingup_sparse-v0
    dm_control/cartpole-two_poles-v0
    dm_control/cartpole-three_poles-v0
    dm_control/cheetah-run-v0
    dm_control/dog-stand-v0
    dm_control/dog-walk-v0
    dm_control/dog-trot-v0
    dm_control/dog-run-v0
    dm_control/dog-fetch-v0
    dm_control/finger-spin-v0
    dm_control/finger-turn_easy-v0
    dm_control/finger-turn_hard-v0
    dm_control/fish-upright-v0
    dm_control/fish-swim-v0
    dm_control/hopper-stand-v0
    dm_control/hopper-hop-v0
    dm_control/humanoid-stand-v0
    dm_control/humanoid-walk-v0
    dm_control/humanoid-run-v0
    dm_control/humanoid-run_pure_state-v0
    dm_control/humanoid_CMU-stand-v0
    dm_control/humanoid_CMU-run-v0
    dm_control/lqr-lqr_2_1-v0
    dm_control/lqr-lqr_6_2-v0
    dm_control/manipulator-bring_ball-v0
    dm_control/manipulator-bring_peg-v0
    dm_control/manipulator-insert_ball-v0
    dm_control/manipulator-insert_peg-v0
    dm_control/pendulum-swingup-v0
    dm_control/point_mass-easy-v0
    dm_control/point_mass-hard-v0
    dm_control/quadruped-walk-v0
    dm_control/quadruped-run-v0
    dm_control/quadruped-escape-v0
    dm_control/quadruped-fetch-v0
    dm_control/reacher-easy-v0
    dm_control/reacher-hard-v0
    dm_control/stacker-stack_2-v0
    dm_control/stacker-stack_4-v0
    dm_control/swimmer-swimmer6-v0
    dm_control/swimmer-swimmer15-v0
    dm_control/walker-stand-v0
    dm_control/walker-walk-v0
    dm_control/walker-run-v0
)

# ------ Warmup fractions to test (10 values: 0.350 to 0.600) ------
WARMUP_FRACS=(0.350 0.378 0.406 0.433 0.461 0.489 0.517 0.544 0.572 0.600)

SEED=42
TOTAL_STEPS=18000000
SCRIPT="cleanrl/dart_dm_control.py"

# ------ Generate task list (round-robin across nodes) ------
MY_TASKFILE="${BENCH_DIR}/tasks_tune_node${NODE_ID}_${SLURM_ARRAY_JOB_ID}.txt"
> "${MY_TASKFILE}"

TASK_IDX=0
for wf in "${WARMUP_FRACS[@]}"; do
    for env in "${ENVS[@]}"; do
        if (( TASK_IDX % NUM_NODES == NODE_ID )); then
            echo "${PYTHON} ${SCRIPT} --env-id ${env} --seed ${SEED} --total-timesteps ${TOTAL_STEPS} --dart-warmup-frac ${wf} --exp-name dart_warmup_${wf}" >> "${MY_TASKFILE}"
        fi
        TASK_IDX=$(( TASK_IDX + 1 ))
    done
done

MY_NTASKS=$(wc -l < "${MY_TASKFILE}")
echo "Tasks on this node: ${MY_NTASKS}"
echo "Parallel slots:     120"
echo ""

# ------ Run ------
JOBLOG="${BENCH_DIR}/parallel_joblog_tune_node${NODE_ID}_${SLURM_ARRAY_JOB_ID}.txt"

parallel \
    -j 120 \
    --joblog "${JOBLOG}" \
    --halt never \
    < "${MY_TASKFILE}"

echo "========================================================"
echo "Node ${NODE_ID} done at $(date)"
echo "========================================================"

# ------ On last node (node 0), run analysis after all tasks finish ------
# NOTE: This only analyzes runs from THIS node. For the full analysis,
#       run after ALL array jobs finish:
#         python tune_dart_warmup.py --runs-dir runs
echo ""
echo "To analyze results after ALL array jobs finish, run:"
echo "  python tune_dart_warmup.py --runs-dir runs"
echo ""
