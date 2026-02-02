#!/bin/bash
#SBATCH --account=aip-rrabba
#SBATCH --job-name=humanoid_ppo_vs_dart
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -e

source .env
source ~/.bashrc

echo "Node: $(hostname)"
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"

# -----------------------------
# Shared configuration
# -----------------------------
NUM_ENVS=256
NUM_STEPS=256
UPDATE_EPOCHS=4
NUM_MINIBATCHES=16
ENT_COEF=0.01
WANDB_PROJECT="humanoid_ppo_vs_dart_tuned2"

SEEDS=(1 2 3 4 5 6 7 8)
METHODS=("ppo_humanoid_sparse_large_critic.py")

# -----------------------------
# Launch jobs
# -----------------------------
GPU_ID=0
PROC_ON_GPU=0

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

    # Move to next GPU after 4 processes
    if [ ${PROC_ON_GPU} -eq 4 ]; then
      PROC_ON_GPU=0
      GPU_ID=$((GPU_ID + 1))
    fi

  done
done

wait
echo "All PPO and DART experiments completed."
