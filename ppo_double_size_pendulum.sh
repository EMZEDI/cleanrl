#!/bin/bash
#SBATCH --account=aip-rrabba
#SBATCH --job-name=pendulum_ppo_large_critic
#SBATCH --nodes=1                    # Optimized: We can fit all 32 seeds on 1 node now
#SBATCH --exclusive                  # Claims the whole node
#SBATCH --cpus-per-task=32           # Request 32 cores (matches your 32 processes)
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out        # Changed from %A_%a (array) to %j (single job)
#SBATCH --error=slurm-%j.err

set -e

source .env
source ~/.bashrc

echo "Node: $(hostname)"
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"

# -----------------------------
# Concurrency Configuration
# -----------------------------
# Running 32 processes on 1 node (8 per GPU).
# 32 CPUs requested / 32 Procs = 1 physical core per process.
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

# ONLY the new Large Critic PPO
METHODS=("ppo_pendulum_sparse_large_critic.py")

# -----------------------------
# Seed Generation Logic
# -----------------------------
# We run seeds 9 to 40 (32 seeds total) on this single node.
SEEDS=($(seq 9 40))

echo "Running seeds: ${SEEDS[@]}"

# -----------------------------
# Launch jobs
# -----------------------------
GPU_ID=0
PROC_ON_GPU=0
MAX_PROC_PER_GPU=8  # Keeps the high density (32 total procs / 4 GPUs)

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
      
      # Safety cycle
      if [ ${GPU_ID} -ge 4 ]; then
        GPU_ID=0
      fi
    fi

  done
done

wait
echo "All Large Critic PPO experiments on node $(hostname) completed."