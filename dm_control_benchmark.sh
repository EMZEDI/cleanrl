#!/bin/bash
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --job-name=dm_bench
#SBATCH --output=dm_bench_slurm-%A_%a.out
#SBATCH --error=dm_bench_slurm-%A_%a.err
#SBATCH --account=def-rrabba

# =============================================================================
# dm_control PPO vs DART vs PPO-Double Benchmark
# Multi-node via SLURM job array: 4 CPU nodes, each runs ~375 tasks
# Each node packs ~120 serial jobs across 192 cores with GNU Parallel
# (capped at 120 for memory safety: large envs like dog/quadruped/humanoid_CMU
#  can spike to 4GB RSS; 120 × 4GB = 480GB well within 755GB node RAM)
#
# Total: 50 envs × 10 seeds × 3 methods = 1,500 runs
# 4 nodes × 120 slots = 480 concurrent → ~4 batches per node
# Estimated wall time: ~10–14 hours per node
# =============================================================================
#
# Usage:
#   1. Install (once):  cd /path/to/cleanrl && uv sync --extra dm_control
#   2. Submit:          cd /path/to/cleanrl && sbatch dm_control_benchmark.sh
#   3. Sync wandb:      wandb beta sync -n 20 $SCRATCH/dm_control_bench/wandb/wandb/offline-run-*
#
# Code runs from the submit directory (your home/project fs — fast reads).
# All write-heavy I/O (wandb, tensorboard runs/, task files) goes to $SCRATCH.

set -euo pipefail

# ------ Prevent BLAS/OpenMP thread oversubscription ------
# Each Python process should use exactly 1 thread (physics is single-threaded)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ------ Wandb: offline mode (no internet on compute nodes) ------
export WANDB_MODE=offline
BENCH_DIR="${SCRATCH}/dm_control_bench"
export WANDB_DIR="${BENCH_DIR}/wandb"
mkdir -p "${WANDB_DIR}"
mkdir -p "${BENCH_DIR}/runs"

# Project dir = where sbatch was submitted from (home/project fs — fast reads)
# All write-heavy I/O is redirected to $SCRATCH
PROJ_DIR="${SLURM_SUBMIT_DIR}"
cd "${PROJ_DIR}"

# Symlink runs/ → $SCRATCH so tensorboard writes land on scratch
# (Python scripts write to runs/ relative to CWD — no code changes needed)
ln -sfn "${BENCH_DIR}/runs" "${PROJ_DIR}/runs"

source .env

PYTHON="${PROJ_DIR}/.venv/bin/python"

# ------ Preflight checks ------
echo "Python:        ${PYTHON}"
[[ -x "${PYTHON}" ]] || { echo "ERROR: venv not found at ${PYTHON}. Run: uv sync --extra dm_control"; exit 1; }
"${PYTHON}" -c "import shimmy; import gymnasium; gymnasium.make('dm_control/pendulum-swingup-v0').close()" \
    && echo "Preflight OK: dm_control namespace verified" \
    || { echo "ERROR: dm_control import failed. Run: uv sync --extra dm_control"; exit 1; }

NUM_NODES=4  # Must match --array=0-(N-1)
NODE_ID=${SLURM_ARRAY_TASK_ID}

echo "========================================================"
echo "Node:          $(hostname)"
echo "Array element: ${NODE_ID} of ${NUM_NODES}"
echo "Cores:         ${SLURM_CPUS_PER_TASK}"
echo "Project dir:   ${PROJ_DIR}"
echo "Bench output:  ${BENCH_DIR}"
echo "Wandb dir:     ${WANDB_DIR}"
echo "Start time:    $(date)"
echo "========================================================"

# ------ Define environments, seeds, methods ------
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

SEEDS=(1 2 3 4 5 6 7 8 9 10)

WANDB_PROJECT="dm_control_ppo_vs_dart"
TOTAL_STEPS=8000000

# Methods: script path + exp-name (indexed arrays for deterministic ordering across nodes)
# Hyperparams are baked into the scripts (lr=3e-4, ent_coef=0.01, etc.)
# Batching matches cleanrl base: num_envs=1, num_steps=2048, update_epochs=10, num_minibatches=32
METHOD_SCRIPTS=(
    "cleanrl/ppo_dm_control.py"
    "cleanrl/dart_dm_control.py"
    "cleanrl/ppo_double_dm_control.py"
)
METHOD_NAMES=(
    "ppo_dm_control"
    "dart_dm_control"
    "ppo_double_dm_control"
)
NUM_METHODS=${#METHOD_SCRIPTS[@]}

# ------ Generate this node's task list (round-robin across array elements) ------
MY_TASKFILE="${BENCH_DIR}/tasks_node${NODE_ID}_${SLURM_ARRAY_JOB_ID}.txt"
> "${MY_TASKFILE}"

TASK_IDX=0
for env in "${ENVS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for (( m=0; m<NUM_METHODS; m++ )); do
            if (( TASK_IDX % NUM_NODES == NODE_ID )); then
                echo "${PYTHON} ${METHOD_SCRIPTS[$m]} --env-id ${env} --seed ${seed} --total-timesteps ${TOTAL_STEPS} --exp-name ${METHOD_NAMES[$m]} --wandb-project-name ${WANDB_PROJECT} --track" >> "${MY_TASKFILE}"
            fi
            TASK_IDX=$(( TASK_IDX + 1 ))
        done
    done
done

NTASKS_TOTAL=$(( ${#ENVS[@]} * ${#SEEDS[@]} * NUM_METHODS ))
MY_NTASKS=$(wc -l < "${MY_TASKFILE}")
echo "Total tasks (all nodes): ${NTASKS_TOTAL}"
echo "This node's tasks:       ${MY_NTASKS}"
echo "Parallel slots:          120  (of 192 cores; capped for memory safety ~480GB peak)"
echo "Estimated batches:       $(( (MY_NTASKS + 119) / 120 ))"
echo ""

# ------ Run with GNU Parallel ------
# -j 120       : 120 concurrent jobs (memory-safe for large dm_control envs)
# --joblog     : tracks which tasks completed (enables --resume on resubmit)
# --resume     : skip already-completed tasks if resubmitted
# --progress   : show progress bar
# --halt soon,fail=10%  : stop if >10% of tasks fail (catch systemic issues)
JOBLOG="${BENCH_DIR}/parallel_joblog_node${NODE_ID}_${SLURM_ARRAY_JOB_ID}.txt"

parallel \
    -j 120 \
    --joblog "${JOBLOG}" \
    --resume \
    --halt never \
    < "${MY_TASKFILE}"

echo "========================================================"
echo "Node ${NODE_ID} completed at $(date)"
echo "Job log: ${JOBLOG}"
echo ""
echo "To sync wandb runs (from a login node with internet):"
echo "  wandb sync ${WANDB_DIR}/*"
echo "========================================================"
