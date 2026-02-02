#!/bin/bash
#SBATCH --account=aip-rrabba
#SBATCH --job-name=humanoid_ppo_vs_dart_fast
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

source .env
source ~/.bashrc

export CUDA_DEVICE_ORDER=PCI_BUS_ID  # recommended for stable GPU ordering [page:1]

# Prevent BLAS/OpenMP oversubscription inside each run
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Node: $(hostname)"
echo "Allocated GPUs (job): $CUDA_VISIBLE_DEVICES"

WANDB_PROJECT="humanoid_ppo_vs_dart_tuned3"

# Throughput-oriented defaults (tune)
NUM_ENVS=32
NUM_STEPS=1024        # keep batch ~ 32*1024=32768; raise if GPU is still idle
UPDATE_EPOCHS=4
NUM_MINIBATCHES=8
ENT_COEF=0.01

SEEDS=(1 2 3 4 5 6 7 8)
METHODS=("ppo_humanoid_sparse.py" "dart_humanoid_sparse_opt.py")

# Build a flat list of commands (seed x method)
CMDS=()
for seed in "${SEEDS[@]}"; do
  for method in "${METHODS[@]}"; do

    CMDS+=("python cleanrl/${method} \
      --num-envs ${NUM_ENVS} \
      --num-steps ${NUM_STEPS} \
      --update-epochs ${UPDATE_EPOCHS} \
      --num-minibatches ${NUM_MINIBATCHES} \
      --ent-coef ${ENT_COEF} \
      --wandb_project_name ${WANDB_PROJECT} \
      --seed ${seed} \
      --track \
      --save_model")
  done
done

# Launch with at most 4 concurrent srun steps (one per GPU/task)
MAX_PARALLEL=4
pids=()

for cmd in "${CMDS[@]}"; do
  # Throttle to MAX_PARALLEL background steps
  while [ "${#pids[@]}" -ge "$MAX_PARALLEL" ]; do
    for i in "${!pids[@]}"; do
      if ! kill -0 "${pids[$i]}" 2>/dev/null; then
        unset 'pids[i]'
        pids=("${pids[@]}")
        break
      fi
    done
    sleep 1
  done

  echo "Launching: $cmd"
  srun --exclusive -N1 -n1 --gpus-per-task=1 --cpus-per-task=$SLURM_CPUS_PER_TASK bash -lc "$cmd" &
  pids+=("$!")
done

wait
echo "All experiments completed."
