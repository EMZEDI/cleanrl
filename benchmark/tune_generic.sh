#!/bin/bash
#SBATCH --job-name=tune_generic
#SBATCH --nodes=60
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --account=aip-rrabba
#SBATCH --mem=0
#SBATCH --time=02:59:00
#SBATCH --output=/scratch/shahradm/slurm_logs/tune_generic_%j.out
#SBATCH --error=/scratch/shahradm/slurm_logs/tune_generic_%j.err

# ================================================================================
# Generic large-scale tuning launcher
#
# Usage:
#   sbatch benchmark/tune_generic.sh <env_id> <search_space> <script_path> [timesteps] [num_trials]
#
# Examples:
#   sbatch benchmark/tune_generic.sh Humanoid-v4 ppo cleanrl/ppo_humanoid_sparse.py
#   sbatch benchmark/tune_generic.sh Humanoid-v4 dart cleanrl/dart_humanoid_sparse_opt.py
#   sbatch benchmark/tune_generic.sh Humanoid-v4 ppo_large_critic cleanrl/ppo_humanoid_sparse_large_critic.py
# ================================================================================

ENV_ID=${1:?"env_id required"}
SEARCH_SPACE=${2:?"search_space required (ppo|dart|ppo_large_critic)"}
SCRIPT_PATH=${3:?"script_path required"}
TOTAL_TIMESTEPS=${4:-50000000}
NUM_TRIALS=${5:-500}

NUM_SEEDS=${NUM_SEEDS:-1}
TRIALS_PER_GPU=${TRIALS_PER_GPU:-3}
NUM_ENVS=${NUM_ENVS:-64}
NUM_STEPS=${NUM_STEPS:-1024}

STUDY_NAME=${STUDY_NAME:-"${SEARCH_SPACE}_${ENV_ID}_$((TOTAL_TIMESTEPS/1000000))M"}
STORAGE=${STORAGE:-"sqlite:////scratch/shahradm/optuna_humanoid.db"}

mkdir -p /scratch/shahradm/slurm_logs
mkdir -p /scratch/shahradm/optuna_results

echo "========================================================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Env: $ENV_ID"
echo "Search space: $SEARCH_SPACE"
echo "Script: $SCRIPT_PATH"
echo "Study name: $STUDY_NAME"
echo "Storage: $STORAGE"
echo "Total trials: $NUM_TRIALS"
echo "Timesteps per trial: $TOTAL_TIMESTEPS"
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: 4"
echo "Trials per GPU: $TRIALS_PER_GPU"
echo "Total GPUs in job: $(($SLURM_NNODES * 4))"
echo "Total workers in job: $(($SLURM_NNODES * 4 * $TRIALS_PER_GPU))"
echo "Time limit: 3 hours (optimized for fast scheduling)"
echo "========================================================================"

# Load environment (uv)
source .env

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "srun sanity check (hostname lines should equal SLURM_NNODES=$SLURM_NNODES): $(srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 hostname | wc -l)"

# Pass any extra fixed args using: EXTRA_ARGS="--fixed-arg key=value --fixed-arg key2=value2"

srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 python cleanrl_utils/tune_generic.py \
  --script-path $SCRIPT_PATH \
  --env-id $ENV_ID \
  --search-space $SEARCH_SPACE \
  --study-name $STUDY_NAME \
  --storage $STORAGE \
  --num-gpus 4 \
  --trials-per-gpu $TRIALS_PER_GPU \
  --num-trials $NUM_TRIALS \
  --total-timesteps $TOTAL_TIMESTEPS \
  --num-envs $NUM_ENVS \
  --num-steps $NUM_STEPS \
  --num-seeds $NUM_SEEDS \
  $EXTRA_ARGS

echo "========================================================================"
echo "Node $SLURM_NODELIST completed tuning"
echo "========================================================================"