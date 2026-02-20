#!/bin/bash
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --job-name=dm_bench
#SBATCH --output=dm_bench_slurm-%A_%a.out
#SBATCH --error=dm_bench_slurm-%A_%a.err
#SBATCH --account=def-rrabba

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ------ Wandb: online (Fir nodes have internet) with offline fallback ------
export WANDB_MODE=online

BENCH_DIR="${SCRATCH}/dm_control_bench2"
mkdir -p "${BENCH_DIR}/runs"

PROJ_DIR="${SLURM_SUBMIT_DIR}"
cd "${PROJ_DIR}"

ln -sfn "${BENCH_DIR}/runs" "${PROJ_DIR}/runs"

source .env

PYTHON="${PROJ_DIR}/.venv/bin/python"

# ------ Preflight ------
echo "Python: ${PYTHON}"
[[ -x "${PYTHON}" ]] || { echo "ERROR: venv not found"; exit 1; }

"${PYTHON}" -c "import shimmy; import gymnasium; gymnasium.make('dm_control/pendulum-swingup-v0').close()" \
    && echo "Preflight OK: dm_control verified" \
    || { echo "ERROR: dm_control import failed"; exit 1; }

if curl -s --max-time 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "Preflight OK: wandb API reachable — using online mode"
else
    echo "WARNING: wandb API unreachable — falling back to offline"
    export WANDB_MODE=offline
    export WANDB_DIR="${SLURM_TMPDIR}/wandb"
    mkdir -p "${WANDB_DIR}"
fi

NUM_NODES=6
NODE_ID=${SLURM_ARRAY_TASK_ID}

echo "========================================================"
echo "Node:       $(hostname)"
echo "Array ID:   ${NODE_ID} of ${NUM_NODES}"
echo "Cores:      ${SLURM_CPUS_PER_TASK}"
echo "Wandb mode: ${WANDB_MODE}"
echo "Start:      $(date)"
echo "========================================================"

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

SEEDS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)

WANDB_PROJECT="dm_control_ppo_vs_dart2"
TOTAL_STEPS=18000000

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

# ------ Generate task list (round-robin) ------
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

MY_NTASKS=$(wc -l < "${MY_TASKFILE}")
echo "Tasks on this node: ${MY_NTASKS}"
echo "Parallel slots:     120"
echo ""

# ------ Run ------
JOBLOG="${BENCH_DIR}/parallel_joblog_node${NODE_ID}_${SLURM_ARRAY_JOB_ID}.txt"

parallel \
    -j 120 \
    --joblog "${JOBLOG}" \
    --halt never \
    < "${MY_TASKFILE}"

# ------ If we fell back to offline, copy data to scratch ------
if [[ "${WANDB_MODE}" == "offline" ]]; then
    DEST="${BENCH_DIR}/wandb_node${NODE_ID}"
    mkdir -p "${DEST}"
    rsync -a "${SLURM_TMPDIR}/wandb/" "${DEST}/"
    echo "Offline wandb data saved to: ${DEST}"
    echo "Sync with: find ${DEST} -name 'offline-run-*' -type d | parallel -j 10 --delay 2 'wandb sync {}'"
fi

echo "========================================================"
echo "Node ${NODE_ID} done at $(date)"
echo "========================================================"
