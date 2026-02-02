#!/bin/bash
#SBATCH --job-name=tune_humanoid
#SBATCH --nodes=60
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --account=aip-rrabba
#SBATCH --mem=0
#SBATCH --time=02:59:00
#SBATCH --output=/scratch/shahradm/slurm_logs/tune_humanoid_%j.out
#SBATCH --error=/scratch/shahradm/slurm_logs/tune_humanoid_%j.err

# ================================================================================
# Massive-scale hyperparameter tuning for PPO vs DART on Humanoid-v4
# 
# Resources: 80 nodes × 4 GPUs × 3 trials/GPU = 960 parallel workers
# Strategy: Exploit aggressive pruning in 3-hour window
#   - Most trials pruned at 5M steps (~20 min) → ~8,640 trial starts
#   - ~30% survive to 50M steps (~2.5 hrs) → ~2,880 completions
#   - Can easily get 500+ quality trials per algorithm in single run!
# 
# Usage:
#   sbatch benchmark/tune_humanoid_820gpu.sh ppo     # Tune PPO
#   sbatch benchmark/tune_humanoid_820gpu.sh dart    # Tune DART
# ================================================================================

# Get algorithm from command line argument (ppo or dart)
ALGORITHM=${1:-ppo}

# Configuration
STUDY_NAME="${ALGORITHM}_humanoid_50M"
STORAGE="sqlite:////scratch/shahradm/optuna_humanoid.db"
NUM_TRIALS=500
TOTAL_TIMESTEPS=50000000
NUM_SEEDS=1
TRIALS_PER_GPU=3  # Run 3 trials per GPU (L40s has 48GB, ~12GB per trial)

# Create directories
mkdir -p /scratch/shahradm/slurm_logs
mkdir -p /scratch/shahradm/optuna_results

# Print job info
echo "========================================================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Algorithm: $ALGORITHM"
echo "Study name: $STUDY_NAME"
echo "Storage: $STORAGE"
echo "Total trials: $NUM_TRIALS"
echo "Timesteps per trial: $TOTAL_TIMESTEPS"
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: 4"
echo "Trials per GPU: $TRIALS_PER_GPU"
echo "Total workers per node: $((4 * $TRIALS_PER_GPU))"
echo "Total GPUs in job: $(($SLURM_NNODES * 4))"
echo "Total workers in job: $(($SLURM_NNODES * 4 * $TRIALS_PER_GPU))"
echo "Time limit: 3 hours (optimized for fast scheduling)"
echo "Strategy: Aggressive pruning → ~9K trial starts → ~3K completions"
echo "========================================================================"

# Load environment (uv)
source .env

# Set environment variables for performance
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Run tuning script on all nodes (launches multiple workers per GPU)
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 python cleanrl_utils/tune_ppo_dart_humanoid.py \
    --algorithm $ALGORITHM \
    --study-name $STUDY_NAME \
    --storage $STORAGE \
    --num-gpus 4 \
    --trials-per-gpu $TRIALS_PER_GPU \
    --num-trials $NUM_TRIALS \
    --total-timesteps $TOTAL_TIMESTEPS \
    --num-seeds $NUM_SEEDS

echo "========================================================================"
echo "Node $SLURM_NODELIST completed tuning for $ALGORITHM"
echo "========================================================================"
