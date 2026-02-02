#!/bin/bash
#SBATCH --account=aip-rrabba
#SBATCH --job-name=pendulum_ppo_vs_dart_scaled
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --exclusive                  # Claims the whole node
#SBATCH --cpus-per-task=32           # <--- FIX: Explicitly ask for 32 cores (1 per process)
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

set -e

source .env
source ~/.bashrc

echo "Node: $(hostname)"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"

# -----------------------------
# Concurrency Configuration
# -----------------------------
# We are running 32 processes (8 per GPU).
# We requested 32 CPUs, so we have 1 physical core per process.
# We set OMP threads to 2 to utilize HyperThreading without overloading.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

# -----------------------------
# Shared configuration
# -----------------------------
NUM_ENVS=256
NUM_STEPS=256
UPDATE_EPOCHS=4
NUM_MINIBATCHES=16
ENT_COEF=0.01
WANDB_PROJECT="pendulum_ppo_vs_dart"
METHODS=("ppo_pendulum_sparse.py" "dart_pendulum_sparse.py")

# -----------------------------
# Seed Generation Logic
# -----------------------------
# Node 0: Seeds 9-24
# Node 1: Seeds 25-40
SEEDS_PER_NODE=16
START_SEED=$(( 9 + SLURM_ARRAY_TASK_ID * SEEDS_PER_NODE ))
END_SEED=$(( START_SEED + SEEDS_PER_NODE - 1 ))

SEEDS=($(seq $START_SEED $END_SEED))

echo "Running seeds: ${SEEDS[@]}"

# -----------------------------
# Launch jobs
# -----------------------------
GPU_ID=0
PROC_ON_GPU=0
MAX_PROC_PER_GPU=8

for SEED in "${SEEDS[@]}"; do
  for METHOD in "${METHODS[@]}"; do

    echo "Launching ${METHOD} | seed=${SEED} | GPU=${GPU_ID}"

    CUDA_VISIBLE_DEVICES=${GPU_ID} \
    python cleanrl/${METHOD} \
        --num-envs ${NUM_ENVS} \
        --num-steps ${NUM_STEPS} \
        --update-epochs ${UPDATE_EPOCHS} \
        --num-minibatches ${NUM_MINIBATCHES} \
        --ent-coef ${ENT_COEF} \
        --wandb_project_name "${WANDB_PROJECT}" \
        --seed ${SEED} \
        --track \
        --capture_video \
        --save_model \
        &

    PROC_ON_GPU=$((PROC_ON_GPU + 1))

    # Move to next GPU after 8 processes
    if [ ${PROC_ON_GPU} -eq ${MAX_PROC_PER_GPU} ]; then
      PROC_ON_GPU=0
      GPU_ID=$((GPU_ID + 1))
      
      if [ ${GPU_ID} -ge 4 ]; then
        GPU_ID=0
      fi
    fi

  done
done

wait
echo "All experiments on node $(hostname) completed."