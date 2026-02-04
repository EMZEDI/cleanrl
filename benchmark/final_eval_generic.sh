#!/bin/bash
#SBATCH --job-name=final_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --account=aip-rrabba
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/shahradm/slurm_logs/final_eval_%j.out
#SBATCH --error=/scratch/shahradm/slurm_logs/final_eval_%j.err

# ================================================================================
# Generic final evaluation: run top-N configs with 20 seeds
#
# Usage:
#   sbatch benchmark/final_eval_generic.sh <env_id> <script_path> <config_dir> [timesteps] [num_seeds]
#
# Example:
#   sbatch benchmark/final_eval_generic.sh Humanoid-v4 cleanrl/ppo_humanoid_sparse.py /scratch/shahradm/optuna_results/ppo
# ================================================================================

ENV_ID=${1:?"env_id required"}
SCRIPT_PATH=${2:?"script_path required"}
CONFIG_DIR=${3:?"config_dir required"}
TOTAL_TIMESTEPS=${4:-50000000}
NUM_SEEDS=${5:-20}

NUM_ENVS=${NUM_ENVS:-64}
NUM_STEPS=${NUM_STEPS:-1024}
TOP_N=${TOP_N:-3}
EVAL_EPISODES=${EVAL_EPISODES:-10}

mkdir -p /scratch/shahradm/slurm_logs

echo "========================================================================"
echo "Final Eval"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Env: $ENV_ID"
echo "Script: $SCRIPT_PATH"
echo "Config dir: $CONFIG_DIR"
echo "Top N: $TOP_N"
echo "Seeds: $NUM_SEEDS"
echo "Timesteps: $TOTAL_TIMESTEPS"
echo "========================================================================"

# Load environment (uv)
source .env

# Force offline W&B tracking for final runs
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-/scratch/shahradm/wandb}

# Pass any extra fixed args using: EXTRA_ARGS="--fixed-arg key=value --fixed-arg key2=value2"
# Default to enabling W&B tracking offline for all runs
if [ -z "${EXTRA_ARGS:-}" ]; then
  EXTRA_ARGS="--fixed-arg track=True --fixed-arg wandb-project-name=final_eval_offline"
fi

python cleanrl_utils/final_eval_generic.py \
  --script-path $SCRIPT_PATH \
  --env-id $ENV_ID \
  --config-dir $CONFIG_DIR \
  --top-n $TOP_N \
  --num-seeds $NUM_SEEDS \
  --total-timesteps $TOTAL_TIMESTEPS \
  --num-envs $NUM_ENVS \
  --num-steps $NUM_STEPS \
  --num-gpus 4 \
  --eval-episodes $EVAL_EPISODES \
  $EXTRA_ARGS

echo "========================================================================"
echo "Final eval completed"
echo "========================================================================"